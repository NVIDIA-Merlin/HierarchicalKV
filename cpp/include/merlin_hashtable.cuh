/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "merlin/core_kernels.cuh"
#include "merlin/initializers.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

template <class K, class V, class BaseV, class T, class M, size_t DIM>
class HashTable {
 public:
  using Table = nv::merlin::Table<K, V, M, DIM>;
  using Initializer = nv::merlin::initializers::Initializer<T>;
  using Zeros = nv::merlin::initializers::Zeros<T>;

 public:
  explicit HashTable(uint64_t max_size,
                     const Initializer *initializer = nullptr,
                     bool master = true)
      : capacity_{max_size} {
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    initializer_ = std::make_shared<Initializer>(
        (initializer != nullptr) ? *initializer : Zeros());
    create_table<K, V, M, DIM>(&table_, capacity_);
  }
  ~HashTable() { destroy_table<K, V, M, DIM>(&table_); }
  HashTable(const HashTable &) = delete;
  HashTable &operator=(const HashTable &) = delete;

  void upsert(const K *d_keys, const BaseV *d_vals, const M *d_metas,
              size_t len, cudaStream_t stream,
              bool allow_duplicate_keys = true) {
    if (len == 0) {
      return;
    }

    V **d_dst;
    int *d_src_offset;
    int *d_bucket_offset;
    bool *d_status;
    CUDA_CHECK(cudaMalloc(&d_dst, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_dst, 0, len * sizeof(V *)));
    CUDA_CHECK(cudaMalloc(&d_src_offset, len * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_src_offset, 0, len * sizeof(int)));
    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaMalloc(&d_bucket_offset, len * sizeof(int)));
      CUDA_CHECK(cudaMemset(d_bucket_offset, 0, len * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_status, len * sizeof(bool)));
      CUDA_CHECK(cudaMemset(d_status, 0, len * sizeof(bool)));
    }

    // Determine bucket insert locations.
    if (!allow_duplicate_keys) {
      {
        const int N = len * table_->buckets_size;
        const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

        lookup_for_upsert_kernel<K, V, M, DIM>
            <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_status,
                                                    d_bucket_offset, N);
      }

      {
        const int N = len;
        const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
        upsert_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
            table_, d_keys, d_metas, d_dst, d_src_offset, d_status,
            d_bucket_offset, len);
      }

    } else {
      const int N = len;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      upsert_allow_duplicate_keys_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_metas,
                                                  d_dst, d_src_offset, len);
    }

    {
      const int N = len;
      CUDA_CHECK(cudaStreamSynchronize(stream));
      thrust::device_ptr<uint64_t> d_dst_ptr((uint64_t *)(d_dst));
      thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);
      thrust::sort_by_key(d_dst_ptr, d_dst_ptr + N, d_src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy provided data to the bucket.
    {
      const int N = len * DIM;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      write_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          (const V *)d_vals, d_dst, d_src_offset, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src_offset));

    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaFree(d_bucket_offset));
      CUDA_CHECK(cudaFree(d_status));
    }
  }

  void upsert(const K *d_keys, const BaseV *d_vals, size_t len,
              cudaStream_t stream, bool allow_duplicate_keys = true) {
    // TODO(jamesrong): split when len is too huge.
    if (len == 0) {
      return;
    }

    V **d_dst;
    int *d_src_offset;
    int *d_bucket_offset;
    bool *d_status;
    CUDA_CHECK(cudaMalloc(&d_dst, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_dst, 0, len * sizeof(V *)));
    CUDA_CHECK(cudaMalloc(&d_src_offset, len * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_src_offset, 0, len * sizeof(int)));

    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaMalloc(&d_bucket_offset, len * sizeof(int)));
      CUDA_CHECK(cudaMemset(d_bucket_offset, 0, len * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_status, len * sizeof(bool)));
      CUDA_CHECK(cudaMemset(d_status, 0, len * sizeof(bool)));
    }

    // Determine bucket insert locations.
    if (!allow_duplicate_keys) {
      {
        const int N = len * table_->buckets_size;
        const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
        lookup_for_upsert_kernel<K, V, M, DIM>
            <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_status,
                                                    d_bucket_offset, N);
      }

      {
        const int N = len;
        const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
        upsert_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
            table_, d_keys, d_dst, d_src_offset, d_status, d_bucket_offset,
            len);
      }

    } else {
      const int N = len;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      upsert_allow_duplicate_keys_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_dst,
                                                  d_src_offset, len);
    }

    {
      const int N = len;

      CUDA_CHECK(cudaStreamSynchronize(stream));
      thrust::device_ptr<uint64_t> d_dst_ptr((uint64_t *)(d_dst));
      thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);
      thrust::sort_by_key(d_dst_ptr, d_dst_ptr + N, d_src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy provided data to the bucket.
    {
      const int N = len * DIM;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      write_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          (const V *)d_vals, d_dst, d_src_offset, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src_offset));

    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaFree(d_bucket_offset));
      CUDA_CHECK(cudaFree(d_status));
    }
  }

  void accum(const K *d_keys, const BaseV *d_vals_or_deltas,
             const bool *d_exists, size_t len, cudaStream_t stream,
             bool allow_duplicate_keys = true) {
    if (len == 0) {
      return;
    }

    V **d_dst;
    int *d_src_offset;
    int *d_bucket_offset;
    bool *d_status;

    CUDA_CHECK(cudaMalloc(&d_dst, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_dst, 0, len * sizeof(V *)));
    CUDA_CHECK(cudaMalloc(&d_src_offset, len * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_src_offset, 0, len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_status, len * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_status, 0, len * sizeof(bool)));

    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaMalloc(&d_bucket_offset, len * sizeof(int)));
      CUDA_CHECK(cudaMemset(d_bucket_offset, 0, len * sizeof(int)));
    }

    if (!allow_duplicate_keys) {
      {
        const int N = len * table_->buckets_size;
        const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

        lookup_for_upsert_kernel<K, V, M, DIM>
            <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_status,
                                                    d_bucket_offset, N);
      }

      {
        const int N = len;
        const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
        accum_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
            table_, d_keys, d_dst, d_exists, d_src_offset, d_status,
            d_bucket_offset, len);
      }

    } else {
      const int N = len;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      accum_allow_duplicate_keys_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(
              table_, d_keys, d_dst, d_exists, d_status, d_src_offset, len);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
      const int N = len;
      thrust::device_ptr<uint64_t> d_dst_ptr((uint64_t *)(d_dst));
      thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);
      thrust::sort_by_key(d_dst_ptr, d_dst_ptr + N, d_src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    {
      const int N = len * DIM;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      write_with_accum_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>((const V *)d_vals_or_deltas,
                                                  d_dst, d_exists, d_status,
                                                  d_src_offset, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src_offset));

    if (!allow_duplicate_keys) {
      CUDA_CHECK(cudaFree(d_bucket_offset));
      CUDA_CHECK(cudaFree(d_status));
    }
  }

  void get(const K *d_keys, BaseV *d_vals, bool *d_status, size_t len,
           BaseV *d_def_val, cudaStream_t stream,
           bool full_size_default) const {
    if (len == 0) {
      return;
    }
    V **d_src;
    CUDA_CHECK(cudaMalloc(&d_src, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_src, 0, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset((void *)d_status, 0, len * sizeof(bool)));

    // Determine bucket locations for reading.
    {
      const int N = len * table_->buckets_size;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

      lookup_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          table_, d_keys, d_src, d_status, N);
    }

    // Copy data from bucket to the pointer to d_vals.
    {
      const int N = len * DIM;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
      read_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          d_src, (V *)d_vals, d_status, (V *)d_def_val, N, full_size_default);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));
  }

  void get(const K *d_keys, BaseV *d_vals, M *d_metas, bool *d_status,
           size_t len, BaseV *d_def_val, cudaStream_t stream,
           bool full_size_default) const {
    if (len == 0) {
      return;
    }
    V **d_src;
    CUDA_CHECK(cudaMalloc(&d_src, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_src, 0, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset((void *)d_status, 0, len * sizeof(bool)));

    // Determine bucket locations for reading.
    {
      const int N = len * table_->buckets_size;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
      lookup_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          table_, d_keys, d_src, d_metas, d_status, N);
    }

    // Copy data from bucket to the pointer to d_vals.
    {
      const int N = len * DIM;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
      read_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
          d_src, (V *)d_vals, d_status, (V *)d_def_val, N, full_size_default);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));
  }

  /* when missing, this `get` will return values by using initializer.*/
  void get(const K *d_keys, BaseV *d_vals, size_t len,
           cudaStream_t stream) const {
    if (len == 0) {
      return;
    }
    V **d_src;
    CUDA_CHECK(cudaMalloc(&d_src, len * sizeof(V *)));
    CUDA_CHECK(cudaMemset(d_src, 0, len * sizeof(V *)));

    initializer_->initialize((T *)d_vals, len * sizeof(V), stream);

    // Determine bucket locations for reading.
    {
      const int N = len * table_->buckets_size;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
      lookup_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_src, N);
    }

    // Copy data from bucket to the pointer to d_vals.
    {
      const int N = len * DIM;
      const int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
      read_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_src, (V *)d_vals, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));
  }

  size_t get_size(cudaStream_t stream) const {
    size_t h_size = 0;
    size_t *d_size;

    CUDA_CHECK(cudaMalloc((void **)&d_size, sizeof(size_t)));
    CUDA_CHECK(cudaMemset(d_size, 0, sizeof(size_t)));

    {
      const int N = table_->buckets_num * table_->buckets_size;
      const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
      size_kernel<K, V, M, DIM>
          <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_size, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(
        cudaMemcpy(&h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_size));
    return h_size;
  }

  size_t get_capacity() const { return static_cast<size_t>(capacity_); }

  void clear(cudaStream_t stream) {
    const int N = table_->buckets_num * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    clear_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void remove(const K *d_keys, size_t len, cudaStream_t stream) {
    const int N = len * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    remove_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K *d_key, BaseV *d_val, const size_t offset,
            const size_t search_length, size_t *d_dump_counter,
            cudaStream_t stream) const {
    CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

    // M_LANGER: Unsafe implicit type conversion.
    const size_t block_size =
        std::min(shared_mem_size_ * 0.5 / (sizeof(K) + sizeof(V)), 1024.0);
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the K-V size may be too large!");
    const size_t shared_size = sizeof(K) * block_size + sizeof(V) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump_kernel<K, V, M, DIM><<<grid_size, block_size, shared_size, stream>>>(
        table_, d_key, (V *)d_val, offset, search_length, d_dump_counter);
    CudaCheckError();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K *d_key, BaseV *d_val, M *d_metas, const size_t offset,
            const size_t search_length, size_t *d_dump_counter,
            cudaStream_t stream) const {
    CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

    // M_LANGER: Unsafe implicit type conversion.
    const size_t block_size = std::min(
        shared_mem_size_ * 0.5 / (sizeof(K) + sizeof(V) + sizeof(M)), 1024.0);
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the K-V size may be too large!");
    const size_t shared_size =
        ((sizeof(K) + sizeof(V) + sizeof(M))) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump_kernel<K, V, M, DIM><<<grid_size, block_size, shared_size, stream>>>(
        table_, d_key, (V *)d_val, (M *)d_metas, offset, search_length,
        d_dump_counter);
    CudaCheckError();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

 private:
  static const int BLOCK_SIZE_ = 1024;
  const uint64_t capacity_;
  size_t shared_mem_size_;
  Table *table_;
  std::shared_ptr<Initializer> initializer_;
};

}  // namespace merlin
}  // namespace nv
