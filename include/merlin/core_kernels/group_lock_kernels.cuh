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
    cuda::atomic<int, cuda::thread_scope_device>* update_count,
    cuda::atomic<int, cuda::thread_scope_device>* read_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  new (update_count) cuda::atomic<int, cuda::thread_scope_device>{0};
  new (read_count) cuda::atomic<int, cuda::thread_scope_device>{0};
  new (unique_flag) cuda::atomic<bool, cuda::thread_scope_device>{false};
}
__global__ void lock_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* update_count,
    cuda::atomic<int, cuda::thread_scope_device>* read_count) {
  for (;;) {
    while (update_count->load(cuda::std::memory_order_relaxed)) {
    }
    read_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (update_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    read_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* read_count) {
  read_count->fetch_sub(1, cuda::std::memory_order_relaxed);
}

__global__ void lock_update_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* update_count,
    cuda::atomic<int, cuda::thread_scope_device>* read_count) {
  for (;;) {
    while (read_count->load(cuda::std::memory_order_relaxed)) {
    }
    update_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (read_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    update_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_update_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* update_count) {
  update_count->fetch_sub(1, cuda::std::memory_order_relaxed);
}

__global__ void lock_update_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* update_count,
    cuda::atomic<int, cuda::thread_scope_device>* read_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  /* Lock unique flag */
  bool expected = false;
  while (!unique_flag->compare_exchange_weak(expected, true,
                                             cuda::std::memory_order_relaxed)) {
    expected = false;
  }

  /* Ban update */
  for (;;) {
    while (update_count->load(cuda::std::memory_order_relaxed)) {
    }
    read_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (update_count->load(cuda::std::memory_order_relaxed) == 0) {
      break;
    }
    read_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }

  /* Ban read */
  for (;;) {
    while (read_count->load(cuda::std::memory_order_relaxed) > 1) {
    }
    update_count->fetch_add(1, cuda::std::memory_order_relaxed);
    if (read_count->load(cuda::std::memory_order_relaxed) == 1) {
      break;
    }
    update_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  }
}

__global__ void unlock_update_read_kernel(
    cuda::atomic<int, cuda::thread_scope_device>* update_count,
    cuda::atomic<int, cuda::thread_scope_device>* read_count,
    cuda::atomic<bool, cuda::thread_scope_device>* unique_flag) {
  read_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  update_count->fetch_sub(1, cuda::std::memory_order_relaxed);
  unique_flag->store(false, cuda::std::memory_order_relaxed);
}

__global__ void update_count_kernel(
    int* counter, cuda::atomic<int, cuda::thread_scope_device>* update_count) {
  *counter = update_count->load(cuda::std::memory_order_relaxed);
}

__global__ void read_count_kernel(
    int* counter, cuda::atomic<int, cuda::thread_scope_device>* read_count) {
  *counter = read_count->load(cuda::std::memory_order_relaxed);
}

}  // namespace group_lock
}  // namespace merlin
}  // namespace nv