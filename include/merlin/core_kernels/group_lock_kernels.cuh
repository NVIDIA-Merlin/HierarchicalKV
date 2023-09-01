/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda/std/semaphore>

namespace nv {
namespace merlin {
namespace group_lock {

__global__ void init_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count,
    cuda::atomic<int, cuda::thread_scope_device>* reader_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  new (writer_count) cuda::atomic<int, cuda::thread_scope_device>{0};
  new (reader_count) cuda::atomic<int, cuda::thread_scope_device>{0};
  new (unique_flag) cuda::atomic<bool, cuda::thread_scope_device>{false};
}
__global__ void lock_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count,
    cuda::atomic<int, cuda::thread_scope_device>* reader_count) {
  for (;;) {
    while (writer_count->load(cuda::std::memory_order_relaxed)) {
    }
    reader_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (writer_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    reader_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* reader_count) {
  reader_count->fetch_sub(1, cuda::std::memory_order_relaxed);
}

__global__ void lock_write_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count,
    cuda::atomic<int, cuda::thread_scope_device>* reader_count) {
  for (;;) {
    while (reader_count->load(cuda::std::memory_order_relaxed)) {
    }
    writer_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (reader_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    writer_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_write_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count) {
  writer_count->fetch_sub(1, cuda::std::memory_order_relaxed);
}

__global__ void lock_write_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count,
    cuda::atomic<int, cuda::thread_scope_device>* reader_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  /* Lock unique flag */
  bool expected = false;
  while (!unique_flag->compare_exchange_weak(expected, true,
                                             cuda::std::memory_order_relaxed)) {
    expected = false;
  }

  /* Ban writer */
  for (;;) {
    while (writer_count->load(cuda::std::memory_order_relaxed)) {
    }
    reader_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (writer_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    reader_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }

  /* Ban reader */
  for (;;) {
    while (reader_count->load(cuda::std::memory_order_relaxed) > 1) {
    }
    writer_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (reader_count->load(cuda::std::memory_order_relaxed) == 1) {
      break;
    }
    writer_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_write_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* writer_count,
    cuda::atomic<int, cuda::thread_scope_device>* reader_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  reader_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  writer_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  unique_flag->store(false, cuda::std::memory_order_relaxed);
}

__global__ void writer_count_kernel(
    int* counter, cuda::atomic<int, cuda::thread_scope_device>* writer_count) {
  *counter = writer_count->load(cuda::std::memory_order_relaxed);
}

__global__ void reader_count_kernel(
    int* counter, cuda::atomic<int, cuda::thread_scope_device>* reader_count) {
  *counter = reader_count->load(cuda::std::memory_order_relaxed);
}

}  // namespace group_lock
}  // namespace merlin
}  // namespace nv