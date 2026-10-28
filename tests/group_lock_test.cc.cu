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
#include <atomic>
#include <chrono>
#include <exception>
#include <system_error>
#include <thread>
#include <vector>
#include "merlin/group_lock.cuh"

using namespace nv::merlin;
using namespace std::chrono_literals;

namespace {

__global__ void phase_producer(int* values, int* observed, int row,
                               bool read_only) {
  const int offset = row * 64 + threadIdx.x;
  const auto started = clock64();
  while (clock64() - started < 1000000ULL) {
  }
  if (read_only) {
    observed[offset] = values[offset];
  } else {
    values[offset] = 101 + row;
  }
}

__global__ void phase_consumer(int* values, int* observed, int count,
                               bool read_only) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count) {
    observed[offset] = values[offset];
    if (!read_only) values[offset] = 777;
  }
}

void phase_lock(group_shared_mutex& mutex, char role) {
  if (role == 'R') mutex.lock_read();
  if (role == 'U') mutex.lock_update();
  if (role == 'I') mutex.lock_update_read();
}

void phase_unlock(group_shared_mutex& mutex, char role, cudaStream_t stream) {
  if (role == 'R') mutex.unlock_read(stream);
  if (role == 'U') mutex.unlock_update(stream);
  if (role == 'I') mutex.unlock_update_read(stream);
}

}  // namespace

TEST(GroupSharedMutexTest, NonBlockingPhasePublicationAndReleaseSequences) {
  for (const auto roles : {"RU", "RI", "UR", "UI", "IR", "IU", "II"}) {
    for (const bool explicit_events : {false, true}) {
      SCOPED_TRACE(roles);
      SCOPED_TRACE(explicit_events);
      group_shared_mutex mutex;
      CUDA_CHECK(cudaStreamSynchronize(0));
      const int producers = roles[0] == 'I' ? 1 : 2;
      const int count = producers * 64;
      std::vector<int> values(count, 7);
      int* device_values = nullptr;
      int* device_reads = nullptr;
      int* device_observed = nullptr;
      CUDA_CHECK(cudaMalloc(&device_values, count * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&device_reads, count * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&device_observed, count * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(device_values, values.data(), count * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(device_reads, 0xff, count * sizeof(int)));
      CUDA_CHECK(cudaMemset(device_observed, 0xff, count * sizeof(int)));
      std::vector<cudaStream_t> streams(producers);
      std::vector<cudaEvent_t> events(producers);
      for (int producer = 0; producer < producers; ++producer) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[producer],
                                             cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&events[producer],
                                            cudaEventDisableTiming));
      }
      cudaStream_t consumer;
      CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
      int device = 0;
      CUDA_CHECK(cudaGetDevice(&device));
      std::atomic<int> ready{0};
      std::atomic<bool> start{false};
      std::vector<std::exception_ptr> failures(producers);
      std::vector<std::thread> threads;
      for (int producer = 0; producer < producers; ++producer) {
        threads.emplace_back([&, producer] {
          bool announced = false;
          try {
            CUDA_CHECK(cudaSetDevice(device));
            phase_lock(mutex, roles[0]);
            ready.fetch_add(1);
            announced = true;
            while (!start.load()) std::this_thread::yield();
            phase_producer<<<1, 64, 0, streams[producer]>>>(
                device_values, device_reads, producer, roles[0] == 'R');
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(events[producer], streams[producer]));
            phase_unlock(mutex, roles[0], streams[producer]);
          } catch (...) {
            failures[producer] = std::current_exception();
            if (!announced) ready.fetch_add(1);
          }
        });
      }
      while (ready.load() != producers) std::this_thread::yield();
      start.store(true);
      for (auto& thread : threads) thread.join();
      for (auto failure : failures) {
        if (failure) std::rethrow_exception(failure);
      }
      phase_lock(mutex, roles[1]);
      if (explicit_events) {
        for (auto event : events) {
          CUDA_CHECK(cudaStreamWaitEvent(consumer, event, 0));
        }
      }
      phase_consumer<<<producers, 64, 0, consumer>>>(
          device_values, device_observed, count, roles[1] == 'R');
      CUDA_CHECK(cudaGetLastError());
      phase_unlock(mutex, roles[1], consumer);
      CUDA_CHECK(cudaStreamSynchronize(consumer));
      for (auto stream : streams) CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy(values.data(), device_observed, count * sizeof(int),
                            cudaMemcpyDeviceToHost));
      for (int offset = 0; offset < count; ++offset) {
        EXPECT_EQ(values[offset], roles[0] == 'R' ? 7 : 101 + offset / 64);
      }
      if (roles[0] == 'R') {
        CUDA_CHECK(cudaMemcpy(values.data(), device_reads, count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (auto value : values) EXPECT_EQ(value, 7);
      }
      for (auto event : events) CUDA_CHECK(cudaEventDestroy(event));
      for (auto stream : streams) CUDA_CHECK(cudaStreamDestroy(stream));
      CUDA_CHECK(cudaStreamDestroy(consumer));
      CUDA_CHECK(cudaFree(device_values));
      CUDA_CHECK(cudaFree(device_reads));
      CUDA_CHECK(cudaFree(device_observed));
    }
  }
}

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
