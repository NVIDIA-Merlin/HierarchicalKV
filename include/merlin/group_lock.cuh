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
#include <atomic>
#include <cassert>
#include <mutex>
#include <system_error>
#include <thread>
#include "merlin/core_kernels/group_lock_kernels.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

/*
 * Implementing a triple-group, mutex and relative lock guard for better E2E
 * performance:
 * - There are three roles: `inserter`, `updater`, and `reader`.
 * - Allow only one inserter to be executed concurrently.  (like
 * `insert_or_assign` 'insert_and_evict`, `find_or_insert` etc.).
 * - Allow multiple updaters to be executed concurrently. (like `assign`, etc.)
 * The CUDA kernels guarantee the data consistency in this situation.
 * - Allow multiple readers to be executed concurrently. (like `find` 'size`
 * etc.)
 * - Not allow inserter, readers and updaters to run concurrently
 * - The `update_read_lock` is exclusive and used for special APIs (like
 * `reserve` `erase` `clear` etc.)
 */
class group_shared_mutex {
 public:
  group_shared_mutex(const group_shared_mutex&) = delete;
  group_shared_mutex& operator=(const group_shared_mutex&) = delete;

  group_shared_mutex() noexcept
      : h_update_count_(0), h_read_count_(0), h_unique_flag_(false) {
    CUDA_CHECK(
        cudaMalloc(&d_update_count_,
                   sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUDA_CHECK(cudaMalloc(
        &d_read_count_, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUDA_CHECK(
        cudaMalloc(&d_unique_flag_,
                   sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));
    group_lock::init_kernel<<<1, 1, 0>>>(d_update_count_, d_read_count_,
                                         d_unique_flag_);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~group_shared_mutex() noexcept {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_update_count_));
    CUDA_CHECK(cudaFree(d_read_count_));
    CUDA_CHECK(cudaFree(d_unique_flag_));
  }

  void lock_read() {
    for (;;) {
      while (h_update_count_.load(std::memory_order_acquire)) {
      }
      h_read_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_update_count_.load(std::memory_order_acquire) == 0) {
        {
          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));
          group_lock::lock_read_kernel<<<1, 1, 0, stream>>>(d_update_count_,
                                                            d_read_count_);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
        break;
      }
      h_read_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_read(cudaStream_t stream) {
    { group_lock::unlock_read_kernel<<<1, 1, 0, stream>>>(d_read_count_); }
    h_read_count_.fetch_sub(1, std::memory_order_release);
  }

  void lock_update() {
    for (;;) {
      while (h_read_count_.load(std::memory_order_acquire)) {
      }
      h_update_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_read_count_.load(std::memory_order_acquire) == 0) {
        {
          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));
          group_lock::lock_update_kernel<<<1, 1, 0, stream>>>(d_update_count_,
                                                              d_read_count_);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
        break;
      }
      h_update_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_update(cudaStream_t stream) {
    { group_lock::unlock_update_kernel<<<1, 1, 0, stream>>>(d_update_count_); }
    h_update_count_.fetch_sub(1, std::memory_order_release);
  }

  void lock_update_read() {
    /* Lock unique flag */
    bool expected = false;
    while (!h_unique_flag_.compare_exchange_weak(expected, true,
                                                 std::memory_order_acq_rel)) {
      expected = false;
    }

    /* Ban update */
    for (;;) {
      while (h_update_count_.load(std::memory_order_acquire)) {
      }
      h_read_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_update_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      h_read_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    /* Ban read */
    for (;;) {
      while (h_read_count_.load(std::memory_order_acquire) > 1) {
      }
      h_update_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_read_count_.load(std::memory_order_acquire) == 1) {
        break;
      }
      h_update_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      group_lock::lock_update_read_kernel<<<1, 1, 0, stream>>>(
          d_update_count_, d_read_count_, d_unique_flag_);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }

  void unlock_update_read(cudaStream_t stream) {
    {
      group_lock::unlock_update_read_kernel<<<1, 1, 0, stream>>>(
          d_update_count_, d_read_count_, d_unique_flag_);
    }
    h_read_count_.fetch_sub(1, std::memory_order_release);
    h_update_count_.fetch_sub(1, std::memory_order_release);
    h_unique_flag_.store(false, std::memory_order_release);
  }

  int update_count() noexcept {
    int count = 0;
    int* d_count;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    group_lock::update_count_kernel<<<1, 1, 0, stream>>>(d_count,
                                                         d_update_count_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return count;
  }

  int read_count() noexcept {
    int count = 0;
    int* d_count;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    group_lock::read_count_kernel<<<1, 1, 0, stream>>>(d_count, d_read_count_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return count;
  }

 private:
  std::atomic<int> h_update_count_;
  std::atomic<int> h_read_count_;
  std::atomic<bool> h_unique_flag_;

  cuda::atomic<int, cuda::thread_scope_device>* d_update_count_;
  cuda::atomic<int, cuda::thread_scope_device>* d_read_count_;
  cuda::atomic<bool, cuda::thread_scope_device>* d_unique_flag_;
};

class read_shared_lock {
 public:
  read_shared_lock(const read_shared_lock&) = delete;
  read_shared_lock(read_shared_lock&&) = delete;

  read_shared_lock& operator=(const read_shared_lock&) = delete;
  read_shared_lock& operator=(read_shared_lock&&) = delete;

  explicit read_shared_lock(group_shared_mutex& mutex, cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit read_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                            cudaStream_t stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~read_shared_lock() {
    if (owns_) {
      mutex_->unlock_read(stream_);
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_read();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  cudaStream_t stream_;
};

class update_shared_lock {
 public:
  update_shared_lock(const update_shared_lock&) = delete;
  update_shared_lock(update_shared_lock&&) = delete;

  update_shared_lock& operator=(const update_shared_lock&) = delete;
  update_shared_lock& operator=(update_shared_lock&&) = delete;

  explicit update_shared_lock(group_shared_mutex& mutex,
                              cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_update();
    owns_ = true;
    stream_ = stream;
  }

  explicit update_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                              cudaStream_t stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~update_shared_lock() {
    if (owns_) {
      mutex_->unlock_update(stream_);
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_update();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  cudaStream_t stream_;
};

class update_read_lock {
 public:
  update_read_lock(const update_read_lock&) = delete;
  update_read_lock(update_read_lock&&) = delete;

  update_read_lock& operator=(const update_read_lock&) = delete;
  update_read_lock& operator=(update_read_lock&&) = delete;

  explicit update_read_lock(group_shared_mutex& mutex, cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_update_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit update_read_lock(group_shared_mutex& mutex, std::defer_lock_t,
                            cudaStream_t stream = 0) noexcept
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~update_read_lock() {
    if (owns_) {
      mutex_->unlock_update_read(stream_);
    }
  }

  void lock() {
    assert(!owns_ && "[update_read_lock] trying to lock twice!");
    mutex_->lock_update_read();
    owns_ = true;
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  cudaStream_t stream_;
};

using insert_unique_lock = update_read_lock;

}  // namespace merlin
}  // namespace nv