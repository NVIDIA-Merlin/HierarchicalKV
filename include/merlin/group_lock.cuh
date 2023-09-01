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
 * Implementing a group mutex and relative lock guard for better E2E
 * performance:
 * - Allow multiple writers (like `insert_or_assign` `assign` `insert_and_evict`
 * etc.) The CUDA kernels guarantee the data consistency in this situation.
 * - Allow multiple readers (like `find` 'size` etc.)
 * - Not allow readers and writers to run concurrently
 * - The `write_read_lock` is used for special APIs (like `reserve` `erase`
 * `clear` etc.)
 */
class group_shared_mutex {
 public:
  group_shared_mutex(const group_shared_mutex&) = delete;
  group_shared_mutex& operator=(const group_shared_mutex&) = delete;

  group_shared_mutex() noexcept
      : h_writer_count_(0), h_reader_count_(0), h_unique_flag_(false) {
    CUDA_CHECK(
        cudaMalloc(&d_writer_count_,
                   sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUDA_CHECK(
        cudaMalloc(&d_reader_count_,
                   sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUDA_CHECK(
        cudaMalloc(&d_unique_flag_,
                   sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));
    group_lock::init_kernel<<<1, 1, 0>>>(d_writer_count_, d_reader_count_,
                                         d_unique_flag_);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~group_shared_mutex() noexcept {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_writer_count_));
    CUDA_CHECK(cudaFree(d_reader_count_));
    CUDA_CHECK(cudaFree(d_unique_flag_));
  }

  void lock_read() {
    for (;;) {
      while (h_writer_count_.load(std::memory_order_acquire)) {
      }
      h_reader_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_writer_count_.load(std::memory_order_acquire) == 0) {
        {
          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));
          group_lock::lock_read_kernel<<<1, 1, 0, stream>>>(d_writer_count_,
                                                            d_reader_count_);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
        break;
      }
      h_reader_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_read(cudaStream_t stream) {
    { group_lock::unlock_read_kernel<<<1, 1, 0, stream>>>(d_reader_count_); }
    h_reader_count_.fetch_sub(1, std::memory_order_release);
  }

  void lock_write() {
    for (;;) {
      while (h_reader_count_.load(std::memory_order_acquire)) {
      }
      h_writer_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_reader_count_.load(std::memory_order_acquire) == 0) {
        {
          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));
          group_lock::lock_write_kernel<<<1, 1, 0, stream>>>(d_writer_count_,
                                                             d_reader_count_);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
        break;
      }
      h_writer_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_write(cudaStream_t stream) {
    { group_lock::unlock_write_kernel<<<1, 1, 0, stream>>>(d_writer_count_); }
    h_writer_count_.fetch_sub(1, std::memory_order_release);
  }

  void lock_write_read() {
    /* Lock unique flag */
    bool expected = false;
    while (!h_unique_flag_.compare_exchange_weak(expected, true,
                                                 std::memory_order_acq_rel)) {
      expected = false;
    }

    /* Ban writer */
    for (;;) {
      while (h_writer_count_.load(std::memory_order_acquire)) {
      }
      h_reader_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_writer_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      h_reader_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    /* Ban reader */
    for (;;) {
      while (h_reader_count_.load(std::memory_order_acquire) > 1) {
      }
      h_writer_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_reader_count_.load(std::memory_order_acquire) == 1) {
        break;
      }
      h_writer_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      group_lock::lock_write_read_kernel<<<1, 1, 0, stream>>>(
          d_writer_count_, d_reader_count_, d_unique_flag_);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }

  void unlock_write_read(cudaStream_t stream) {
    {
      group_lock::unlock_write_read_kernel<<<1, 1, 0, stream>>>(
          d_writer_count_, d_reader_count_, d_unique_flag_);
    }
    h_reader_count_.fetch_sub(1, std::memory_order_release);
    h_writer_count_.fetch_sub(1, std::memory_order_release);
    h_unique_flag_.store(false, std::memory_order_release);
  }

  int writer_count() noexcept {
    int count = 0;
    int* d_count;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    group_lock::writer_count_kernel<<<1, 1, 0, stream>>>(d_count,
                                                         d_writer_count_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return count;
  }

  int reader_count() noexcept {
    int count = 0;
    int* d_count;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    group_lock::reader_count_kernel<<<1, 1, 0, stream>>>(d_count,
                                                         d_reader_count_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return count;
  }

 private:
  std::atomic<int> h_writer_count_;
  std::atomic<int> h_reader_count_;
  std::atomic<bool> h_unique_flag_;

  cuda::atomic<int, cuda::thread_scope_device>* d_writer_count_;
  cuda::atomic<int, cuda::thread_scope_device>* d_reader_count_;
  cuda::atomic<bool, cuda::thread_scope_device>* d_unique_flag_;
};

class reader_shared_lock {
 public:
  reader_shared_lock(const reader_shared_lock&) = delete;
  reader_shared_lock(reader_shared_lock&&) = delete;

  reader_shared_lock& operator=(const reader_shared_lock&) = delete;
  reader_shared_lock& operator=(reader_shared_lock&&) = delete;

  explicit reader_shared_lock(group_shared_mutex& mutex,
                              cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit reader_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                              cudaStream_t stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~reader_shared_lock() {
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

class writer_shared_lock {
 public:
  writer_shared_lock(const writer_shared_lock&) = delete;
  writer_shared_lock(writer_shared_lock&&) = delete;

  writer_shared_lock& operator=(const writer_shared_lock&) = delete;
  writer_shared_lock& operator=(writer_shared_lock&&) = delete;

  explicit writer_shared_lock(group_shared_mutex& mutex,
                              cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_write();
    owns_ = true;
    stream_ = stream;
  }

  explicit writer_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                              cudaStream_t stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~writer_shared_lock() {
    if (owns_) {
      mutex_->unlock_write(stream_);
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_write();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  cudaStream_t stream_;
};

class write_read_lock {
 public:
  write_read_lock(const write_read_lock&) = delete;
  write_read_lock(write_read_lock&&) = delete;

  write_read_lock& operator=(const write_read_lock&) = delete;
  write_read_lock& operator=(write_read_lock&&) = delete;

  explicit write_read_lock(group_shared_mutex& mutex, cudaStream_t stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_write_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit write_read_lock(group_shared_mutex& mutex, std::defer_lock_t,
                           cudaStream_t stream = 0) noexcept
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~write_read_lock() {
    if (owns_) {
      mutex_->unlock_write_read(stream_);
    }
  }

  void lock() {
    assert(!owns_ && "[write_read_lock] trying to lock twice!");
    mutex_->lock_write_read();
    owns_ = true;
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  cudaStream_t stream_;
};

}  // namespace merlin
}  // namespace nv