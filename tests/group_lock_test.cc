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

#include "merlin/group_lock.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>

using namespace nv::merlin;
using namespace std::chrono_literals;

// Test the basic functionality of the group_shared_mutex
TEST(GroupSharedMutexTest, BasicFunctionality) {
  group_shared_mutex mutex;
  ASSERT_EQ(mutex.reader_count(), 0);
  ASSERT_EQ(mutex.writer_count(), 0);

  {
    // Multiple readers can acquire the lock simultaneously
    reader_shared_lock reader1(mutex);
    ASSERT_EQ(mutex.reader_count(), 1);
    reader_shared_lock reader2(mutex);
    ASSERT_EQ(mutex.reader_count(), 2);
  }
  ASSERT_EQ(mutex.reader_count(), 0);
  ASSERT_EQ(mutex.writer_count(), 0);

  {
    // A writer is blocked by the readers
    writer_shared_lock writer(mutex, std::defer_lock);
    EXPECT_FALSE(writer.owns_lock());
    ASSERT_EQ(mutex.reader_count(), 0);
    ASSERT_EQ(mutex.writer_count(), 0);
    writer.lock();
    ASSERT_EQ(mutex.reader_count(), 0);
    ASSERT_EQ(mutex.writer_count(), 1);
    EXPECT_TRUE(writer.owns_lock());
  }
  ASSERT_EQ(mutex.reader_count(), 0);
  ASSERT_EQ(mutex.writer_count(), 0);

  // A unique lock is also blocked by the readers
  {
    write_read_lock unique(mutex, std::defer_lock);
    ASSERT_EQ(mutex.reader_count(), 0);
    ASSERT_EQ(mutex.writer_count(), 0);
    EXPECT_FALSE(unique.owns_lock());
    unique.lock();
    EXPECT_TRUE(unique.owns_lock());
    ASSERT_EQ(mutex.reader_count(), 1);
    ASSERT_EQ(mutex.writer_count(), 1);

    EXPECT_DEATH(unique.lock(), "trying to lock twice!");
  }
  ASSERT_EQ(mutex.reader_count(), 0);
  ASSERT_EQ(mutex.writer_count(), 0);
}

TEST(GroupSharedMutexTest, AdvancedFunctionality) {
  group_shared_mutex mutex;
  bool multiple_reader = false;
  bool multiple_writer = false;

  // Test multiple readers
  std::vector<std::thread> readers;
  for (int i = 0; i < 50; ++i) {
    readers.emplace_back([&]() {
      reader_shared_lock reader(mutex);
      EXPECT_TRUE(mutex.reader_count() > 0);
      if (mutex.reader_count() > 1) multiple_reader = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.writer_count(), 0);
    });
  }

  // Test multiple writers
  std::vector<std::thread> writers;
  for (int i = 0; i < 50; ++i) {
    writers.emplace_back([&]() {
      writer_shared_lock writer(mutex);
      EXPECT_TRUE(mutex.writer_count() > 0);
      if (mutex.writer_count() > 1) multiple_writer = true;
      std::this_thread::sleep_for(1000ms);
      ASSERT_EQ(mutex.reader_count(), 0);
    });
  }

  // Test multiple uniques
  std::vector<std::thread> uniques;
  for (int i = 0; i < 50; ++i) {
    uniques.emplace_back([&]() {
      write_read_lock unique(mutex);
      ASSERT_EQ(mutex.reader_count(), 1);
      ASSERT_EQ(mutex.reader_count(), 1);
      std::this_thread::sleep_for(100ms);
    });
  }

  for (auto& th : readers) {
    th.join();
  }

  for (auto& th : writers) {
    th.join();
  }

  for (auto& th : uniques) {
    th.join();
  }

  EXPECT_TRUE(multiple_writer);
  EXPECT_TRUE(multiple_reader);
}
