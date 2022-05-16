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

#include "merlin/concurrent_ordered_map.cuh"
#include "merlin/util.cuh"

namespace nv {
namespace merlin {

template <class K, class V, class BaseV, class M, size_t DIM>
class HashTable {
 public:
  using Table = nv::merlin::Table<K, V, M, DIM>;

 public:
  HashTable(uint64_t max_size, bool master = true) {
    init_size_ = max_size;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    shared_mem_size = deviceProp.sharedMemPerBlock;
    create_table<K, V, M, DIM>(&table_, init_size_);
  }
  ~HashTable() { destroy_table<K, V, M, DIM>(&table_); }
  HashTable(const HashTable &) = delete;
  HashTable &operator=(const HashTable &) = delete;

  void upsert(const K *d_keys, const BaseV *d_vals, const M *d_metas,
              size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }

    V **d_dst;
    cudaMalloc(&d_dst, len * sizeof(V *));
    cudaMemset(d_dst, 0, len * sizeof(V *));

    int N = len;
    int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    upsert_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, d_metas, d_dst, len);

    N = len * DIM;
    grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    write_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>((const V *)d_vals, d_dst, N);

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void upsert(const K *d_keys, const BaseV *d_vals, size_t len,
              cudaStream_t stream) {
    if (len == 0) {
      return;
    }

    V **d_dst;
    cudaMalloc(&d_dst, len * sizeof(V *));
    cudaMemset(d_dst, 0, len * sizeof(V *));

    int N = len;
    int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    upsert_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_dst, len);

    N = len * DIM;
    grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    write_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>((const V *)d_vals, d_dst, N);

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

    int N = len * table_->buckets_size;
    int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

    lookup_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, d_src, d_status, N);

    N = len * DIM;
    grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
    read_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        d_src, (V *)d_vals, d_status, (V *)d_def_val, N, full_size_default);

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

    int N = len * table_->buckets_size;
    int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

    lookup_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, d_src, d_metas, d_status, N);

    N = len * DIM;
    grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
    read_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        d_src, (V *)d_vals, d_status, (V *)d_def_val, N, full_size_default);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));
  }

  size_t get_size(cudaStream_t stream) const {
    size_t h_size = 0;
    size_t *d_size;

    const int N = table_->table_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    CUDA_CHECK(cudaMallocManaged((void **)&d_size, sizeof(size_t)));
    CUDA_CHECK(cudaMemset(d_size, 0, sizeof(size_t)));

    size_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_size, N);

    CUDA_CHECK(
        cudaMemcpy(&h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_size));
    return h_size;
  }

  size_t get_capacity() const { return (size_t)init_size_; }

  void clear(cudaStream_t stream) {
    const int N = table_->table_size * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    clear_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, N);
  }

  void remove(const K *d_keys, size_t len, cudaStream_t stream) {
    const int N = len * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    remove_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, N);
  }

  void dump(K *d_key, BaseV *d_val, const size_t offset,
            const size_t search_length, size_t *d_dump_counter,
            cudaStream_t stream) const {
    CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
    size_t block_size = shared_mem_size * 0.5 / (sizeof(K) + sizeof(V));
    block_size = block_size <= 1024 ? block_size : 1024;
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the K-V size may be too large!");
    size_t shared_size = sizeof(K) * block_size + sizeof(V) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump_kernel<<<grid_size, block_size, 0, stream>>>(
        table_, d_key, (V *)d_val, offset, search_length, d_dump_counter);
  }

  void get(const K *d_keys, BaseV *d_vals, bool *d_status, size_t len,
           const int initializer, cudaStream_t stream) const {}

  void accum(const K *d_keys, const BaseV *d_vals_or_deltas,
             const bool *d_exists, size_t len, cudaStream_t stream) {}

  void update(const K *d_keys, const BaseV *d_grad, const int optimizer,
              float lr, size_t len, cudaStream_t stream) {}

 private:
  static const int BLOCK_SIZE_ = 1024;
  uint64_t init_size_;
  size_t shared_mem_size;
  Table *table_;
};

}  // namespace merlin
}  // namespace nv