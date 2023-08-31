/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>
#include "merlin/group_lock.cuh"

using namespace nv::merlin;
using namespace std::chrono_literals;

// Test the basic functionality of the group_shared_mutex
TEST(GroupSharedMutexTest, BasicFunctionality) {
  group_shared_mutex mutex;
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  {
    // Multiple reads can acquire the lock simultaneously
    read_shared_lock read1(mutex);
    ASSERT_EQ(mutex.read_count(), 1);
    read_shared_lock read2(mutex);
    ASSERT_EQ(mutex.read_count(), 2);
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  {
    // A update is blocked by the reads
    update_shared_lock update(mutex, std::defer_lock);
    EXPECT_FALSE(update.owns_lock());
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 0);
    update.lock();
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 1);
    EXPECT_TRUE(update.owns_lock());
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  // A unique lock is also blocked by the reads
  {
    update_read_lock unique(mutex, std::defer_lock);
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 0);
    EXPECT_FALSE(unique.owns_lock());
    unique.lock();
    EXPECT_TRUE(unique.owns_lock());
    ASSERT_EQ(mutex.read_count(), 1);
    ASSERT_EQ(mutex.update_count(), 1);

    EXPECT_DEATH(unique.lock(), "trying to lock twice!");
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);
}

TEST(GroupSharedMutexTest, AdvancedFunctionalitySingleStream) {
  group_shared_mutex mutex;
  bool multiple_read = false;
  bool multiple_update = false;

  // Test multiple reads
  std::vector<std::thread> reads;
  for (int i = 0; i < 50; ++i) {
    reads.emplace_back([&]() {
      read_shared_lock read(mutex);
      EXPECT_TRUE(mutex.read_count() > 0);
      if (mutex.read_count() > 1) multiple_read = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.update_count(), 0);
    });
  }

  // Test multiple updates
  std::vector<std::thread> updates;
  for (int i = 0; i < 50; ++i) {
    updates.emplace_back([&]() {
      update_shared_lock update(mutex);
      EXPECT_TRUE(mutex.update_count() > 0);
      if (mutex.update_count() > 1) multiple_update = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.read_count(), 0);
    });
  }

  // Test multiple uniques
  std::vector<std::thread> uniques;
  for (int i = 0; i < 50; ++i) {
    uniques.emplace_back([&]() {
      update_read_lock unique(mutex);
      ASSERT_EQ(mutex.read_count(), 1);
      ASSERT_EQ(mutex.update_count(), 1);
      std::this_thread::sleep_for(100ms);
    });
  }

  for (auto& th : reads) {
    th.join();
  }

  for (auto& th : updates) {
    th.join();
  }

  for (auto& th : uniques) {
    th.join();
  }

  EXPECT_TRUE(multiple_update);
  EXPECT_TRUE(multiple_read);
}

TEST(GroupSharedMutexTest, AdvancedFunctionalityMultiStream) {
  group_shared_mutex mutex;
  bool multiple_read = false;
  bool multiple_update = false;

  // Test multiple reads
  std::vector<std::thread> reads;
  for (int i = 0; i < 50; ++i) {
    reads.emplace_back([&]() {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      read_shared_lock read(mutex);
      EXPECT_TRUE(mutex.read_count() > 0);
      if (mutex.read_count() > 1) multiple_read = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.update_count(), 0);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    });
  }

  // Test multiple updates
  std::vector<std::thread> updates;
  for (int i = 0; i < 50; ++i) {
    updates.emplace_back([&]() {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      update_shared_lock update(mutex);
      EXPECT_TRUE(mutex.update_count() > 0);
      if (mutex.update_count() > 1) multiple_update = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.read_count(), 0);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    });
  }

  // Test multiple uniques
  std::vector<std::thread> uniques;
  for (int i = 0; i < 50; ++i) {
    uniques.emplace_back([&]() {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      update_read_lock unique(mutex);
      ASSERT_EQ(mutex.read_count(), 1);
      ASSERT_EQ(mutex.read_count(), 1);
      std::this_thread::sleep_for(100ms);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    });
  }

  for (auto& th : reads) {
    th.join();
  }

  for (auto& th : updates) {
    th.join();
  }

  for (auto& th : uniques) {
    th.join();
  }

  EXPECT_TRUE(multiple_update);
  EXPECT_TRUE(multiple_read);
}
