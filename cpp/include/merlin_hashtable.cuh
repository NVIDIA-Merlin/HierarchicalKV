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
  HashTable(uint64_t max_size, const Initializer *initializer = nullptr,
            bool master = true) {
    capacity_ = max_size;
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
    // TODO(jamesrong): split when len is too huge.
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

    CUDA_CHECK(cudaStreamSynchronize(stream));
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

  /* when missing, this `get` will return values by using initializer.*/
  void get(const K *d_keys, BaseV *d_vals, size_t len,
           cudaStream_t stream) const {
    if (len == 0) {
      return;
    }
    V **d_src;
    CUDA_CHECK(cudaMalloc(&d_src, len * sizeof(V *)));
    CUDA_CHECK(cudaMemsetAsync(d_src, 0, len * sizeof(V *), stream));

    int N = len * table_->buckets_size;
    int grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

    initializer_->initialize((T *)d_vals, len * sizeof(V), stream);

    lookup_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_src, N);

    N = len * DIM;
    grid_size = (N + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;
    read_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_src, (V *)d_vals, N);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));
  }

  size_t get_size(cudaStream_t stream) const {
    size_t h_size = 0;
    size_t *d_size;

    const int N = table_->buckets_num * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    CUDA_CHECK(cudaMalloc((void **)&d_size, sizeof(size_t)));
    CUDA_CHECK(cudaMemset(d_size, 0, sizeof(size_t)));

    size_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_size, N);

    CUDA_CHECK(
        cudaMemcpy(&h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_size));
    return h_size;
  }

  size_t get_capacity() const { return (size_t)capacity_; }

  void clear(cudaStream_t stream) {
    const int N = table_->buckets_num * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    clear_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, N);
  }

  void remove(const K *d_keys, size_t len, cudaStream_t stream) {
    const int N = len * table_->buckets_size;
    const int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    remove_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K *d_key, BaseV *d_val, const size_t offset,
            const size_t search_length, size_t *d_dump_counter,
            cudaStream_t stream) const {
    CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
    size_t block_size = shared_mem_size_ * 0.5 / (sizeof(K) + sizeof(V));
    block_size = block_size <= 1024 ? block_size : 1024;
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the K-V size may be too large!");
    size_t shared_size = sizeof(K) * block_size + sizeof(V) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump_kernel<<<grid_size, block_size, shared_size, stream>>>(
        table_, d_key, (V *)d_val, offset, search_length, d_dump_counter);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K *d_key, BaseV *d_val, M *d_metas, const size_t offset,
            const size_t search_length, size_t *d_dump_counter,
            cudaStream_t stream) const {
    CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
    size_t block_size =
        shared_mem_size_ * 0.5 / (sizeof(K) + sizeof(V) + sizeof(M));
    block_size = block_size <= 1024 ? block_size : 1024;
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the K-V size may be too large!");
    size_t shared_size = sizeof(K) * block_size + sizeof(V) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump_kernel<<<grid_size, block_size, shared_size, stream>>>(
        table_, d_key, (V *)d_val, (M *)d_metas, offset, search_length,
        d_dump_counter);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void accum(const K *d_keys, const BaseV *d_vals_or_deltas,
             const bool *d_exists, size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }

    V **d_dst;
    cudaMalloc(&d_dst, len * sizeof(V *));
    cudaMemset(d_dst, 0, len * sizeof(V *));

    int N = len;
    int grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    accum_kernel<K, V, M, DIM><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, d_vals_or_deltas, d_exists, len);

    N = len * DIM;
    grid_size = (N - 1) / BLOCK_SIZE_ + 1;
    write_kernel<K, V, M, DIM>
        <<<grid_size, BLOCK_SIZE_, 0, stream>>>((const V *)d_vals, d_dst, N);

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

 private:
  static const int BLOCK_SIZE_ = 1024;
  uint64_t capacity_;
  size_t shared_mem_size_;
  Table *table_;
  std::shared_ptr<Initializer> initializer_;
};

}  // namespace merlin
}  // namespace nv