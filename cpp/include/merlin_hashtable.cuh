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
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <mutex>

#include "merlin/core_kernels.cuh"
#include "merlin/initializers.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

/**
 * The basic value type in Merlin-KV.
 * Any user data should be represented as a specific type of Vector in order to
 * be processed by Merlin_KV.
 *
 * @tparam V type of the Vector's elements type, which should be basic types of
 * C++/CUDA.
 * @tparam DIM dimension of the vector.
 */
template <class V, size_t DIM>
struct Vector {
  V value[DIM];
};

/**
 * Merlin HashTable is a concurrent and hierarchical HashTable powered by GPUs
 * which can use both HBM, HMEM and SSD(WIP) as storage for Key-Values.
 *
 * @note Supports concurrent upsert, but not concurrent upsert and find now.
 * The upsert means insert or update if already exists.
 *
 * @note There is no API eviction and the eviction will happen automatically
 * when table is almost full. We introduce a `meta` concept to help implement
 * it. The keys with minimum meta will be evicted with priority. So the meta
 * usually is the timestamp or times of the key occurrence. The user can also
 * customize the meaning of the value that is equivalent to customizing the
 * eviction policy.
 *
 * @tparam Key type of the key
 * @tparam V type of the Value's item type, which should be basic types of
 * C++/CUDA.
 * @tparam M type of the meta and must be uint64_t in this release.
 * @tparam DIM dimension of the vector
 *
 * @todo:
 *  - Support dynamic rehashing
 *  - Support SSD/NVMe device as part of storage
 */
template <class Key, class V, class M, size_t DIM>
class HashTable {
 public:
  using key_type = Key;
  using Vector = Vector<V, DIM>;
  using value_type = Vector;
  using size_type = size_t;
  using Table = nv::merlin::Table<Key, Vector, M, DIM>;
  using Initializer = nv::merlin::initializers::Initializer<V>;
  using Zeros = nv::merlin::initializers::Zeros<V>;
  using Pred = Predict<Key, M>;

 public:
  /**
   * @brief Construct a new merlin::HashTable.
   *
   * @param init_size The initial capacity.
   * @param max_size The maximum capacity.
   * @param max_hbm_for_vectors Max HBM allocated for vectors, by bytes.
   * @param max_load_factor The table automatically increases the number of
   * buckets if the load factor exceeds this threshold.
   * @param bucket_max_size The length of each buckets.
   * @param initializer Initializer used when getting a key fail.
   * @param primary No used.
   */
  explicit HashTable(size_type init_size,
                     size_type max_size = std::numeric_limits<uint64_t>::max(),
                     size_type max_hbm_for_vectors = 0,
                     float max_load_factor = 0.75,
                     size_type bucket_max_size = 128,
                     const Initializer *initializer = nullptr,
                     bool primary = true, int block_size = 1024)
      : init_size_(init_size),
        max_size_(max_size),
        max_hbm_for_vectors_(max_hbm_for_vectors),
        max_load_factor_(max_load_factor),
        bucket_max_size_(bucket_max_size),
        primary_(primary) {
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    initializer_ = std::make_shared<Initializer>(
        (initializer != nullptr) ? *initializer : Zeros());
    create_table<Key, Vector, M, DIM>(&table_, init_size_, max_size_,
                                      max_hbm_for_vectors_, bucket_max_size_,
                                      primary_);
    block_size_ = SAFE_GET_BLOCK_SIZE(block_size);
    reach_max_size_ = false;
    CudaCheckError();
  }

  /**
   * @brief Frees the resources of the table and destroys the table object.
   */
  ~HashTable() { destroy_table<Key, Vector, M, DIM>(&table_); }
  HashTable(const HashTable &) = delete;
  HashTable &operator=(const HashTable &) = delete;

  /**
   * @brief Attempts to insert key-vectors pairs into the table.
   * If one key already exists, the vector of the key will be updated.
   *
   * @note When the table is already full, the LRU policy will be applied: the
   * key-vector pairs to be replaced is least recently inserted.
   *
   * @param keys The keys to be inserted on GPU accessible memory.
   * @param vectors The vectors to be inserted on GPU accessible memory.
   * @param len Number of Key-Value pairs to be upsert.
   * @param allow_duplicated_keys Flag of if allow the @p keys contains
   * duplicate keys. If false, the caller should guarantee the @p keys
   * has no duplicated keys, and the performance will be better.
   * @param stream The CUDA stream used to execute the operation.
   */
  void insert_or_assign(const Key *keys, const V *vectors, size_t len,
                        bool allow_duplicated_keys = true,
                        cudaStream_t stream = 0) {
    // TODO(jamesrong): split when len is too huge.
    if (len == 0) {
      return;
    }

    if (!reach_max_size_ && load_factor() > max_load_factor_) {
      reserve(capacity() * 2);
    }

    Vector **dst;
    int *src_offset;
    CUDA_CHECK(cudaMallocAsync(&dst, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMallocAsync(&src_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(src_offset, 0, len * sizeof(int), stream));

    // Determine bucket insert locations.
    if (allow_duplicated_keys) {
      const size_t N = len;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      upsert_allow_duplicated_keys_kernel<Key, Vector, M, DIM>
          <<<grid_size, block_size_, 0, stream>>>(table_, keys, dst, src_offset,
                                                  len);
    } else {
      int *bucket_offset;
      bool *found;
      CUDA_CHECK(cudaMallocAsync(&bucket_offset, len * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(bucket_offset, 0, len * sizeof(int), stream));
      CUDA_CHECK(cudaMallocAsync(&found, len * sizeof(bool), stream));
      CUDA_CHECK(cudaMemsetAsync(found, 0, len * sizeof(bool), stream));

      {
        const size_t N = len * table_->bucket_max_size;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
        lookup_for_upsert_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(table_, keys, found,
                                                    bucket_offset, N);
      }

      {
        const size_t N = len;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
        upsert_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(
                table_, keys, dst, src_offset, found, bucket_offset, len);
      }

      CUDA_CHECK(cudaFreeAsync(bucket_offset, stream));
      CUDA_CHECK(cudaFreeAsync(found, stream));
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> dst_ptr(reinterpret_cast<uint64_t *>(dst));
      thrust::device_ptr<int> src_offset_ptr(src_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, dst_ptr, dst_ptr + N, src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy provided data to the bucket.
    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      write_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          reinterpret_cast<const Vector *>(vectors), dst, src_offset, N);
    }

    CUDA_CHECK(cudaFreeAsync(dst, stream));
    CUDA_CHECK(cudaFreeAsync(src_offset, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Attempts to insert key-value-meta tuples into the table.
   * If one key already exists, the value of the key will be updated.
   *
   * @note When the table is already full, the keys to be replaced
   * are the keys with minimum meta value. If the meta of the new key is
   * even less than minimum meta of the target bucket, the key will not
   * be inserted.
   *
   * @param keys The keys to be inserted on GPU accessible memory.
   * @param vectors The vectors to be inserted on GPU accessible memory.
   * @param metas The metas to be inserted on GPU accessible memory.
   * The metas usually are with uint64_t value which could stand for the
   * timestamp of the key inserted or the number of the key occurrences.
   *
   * @param len Number of Key-Value-Meta tuples to be upsert.
   * @param allow_duplicated_keys Flag of if allow the @p keys contains
   * duplicate keys. If false, the caller should guarantee the @p keys
   * has no duplicate keys, and the performance will be better.
   * @param stream The CUDA stream used to execute the operation.
   */
  void insert_or_assign(const Key *keys, const V *vectors, const M *metas,
                        size_t len, bool allow_duplicated_keys = true,
                        cudaStream_t stream = 0) {
    if (len == 0) {
      return;
    }

    if (!reach_max_size_ && load_factor() > max_load_factor_) {
      reserve(capacity() * 2);
    }

    Vector **d_dst;
    int *d_src_offset;
    CUDA_CHECK(cudaMallocAsync(&d_dst, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dst, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMallocAsync(&d_src_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_src_offset, 0, len * sizeof(int), stream));

    // Determine bucket insert locations.
    if (allow_duplicated_keys) {
      const size_t N = len;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      upsert_allow_duplicated_keys_kernel<Key, Vector, M, DIM>
          <<<grid_size, block_size_, 0, stream>>>(table_, keys, metas, d_dst,
                                                  d_src_offset, len);
    } else {
      int *d_bucket_offset;
      bool *d_status;
      CUDA_CHECK(cudaMallocAsync(&d_bucket_offset, len * sizeof(int), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_bucket_offset, 0, len * sizeof(int), stream));
      CUDA_CHECK(cudaMallocAsync(&d_status, len * sizeof(bool), stream));
      CUDA_CHECK(cudaMemsetAsync(d_status, 0, len * sizeof(bool), stream));

      {
        const size_t N = len * table_->bucket_max_size;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
        lookup_for_upsert_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(table_, keys, d_status,
                                                    d_bucket_offset, N);
      }

      {
        const size_t N = len;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
        upsert_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(table_, keys, metas, d_dst,
                                                    d_src_offset, d_status,
                                                    d_bucket_offset, len);
      }

      CUDA_CHECK(cudaFreeAsync(d_bucket_offset, stream));
      CUDA_CHECK(cudaFreeAsync(d_status, stream));
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> d_dst_ptr(
          reinterpret_cast<uint64_t *>(d_dst));
      thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, d_dst_ptr, d_dst_ptr + N, d_src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy provided data to the bucket.
    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      write_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          reinterpret_cast<const Vector *>(vectors), d_dst, d_src_offset, N);
    }

    CUDA_CHECK(cudaFreeAsync(d_dst, stream));
    CUDA_CHECK(cudaFreeAsync(d_src_offset, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * Searches for each key in @p keys in the table.
   * If the key is found and corresponding exist in @p exists is true,
   * the @p vectors_or_deltas will be treated as delta against to the old
   * value, and the delta will be add to the old value of the key.
   * If the key is not found and corresponding exist in @p exists is false,
   * the @p vectors_or_deltas will be treated as a new value and the key-value
   * pair will be updated into the table directly.
   *
   * @note Specially when the key is found and exist is false, or the key is not
   * found and exist is true, nothing will be changed and this accum will be
   * ignored, for we assume these situations occur while the key was modified
   * or removed by other processes just now.
   *
   * @param keys The keys to be inserted on GPU accessible memory.
   * @param vals_or_deltas The vectors to be inserted on GPU accessible memory.
   * @param exists if the key exists when last find in this process.
   * @param len Number of Key-Value pairs to be processed.
   * @param allow_duplicated_keys Flag of if allow the @p keys contains
   * duplicate keys. If false, the caller should guarantee the @p keys
   * has no duplicate keys, and the performance will be better.
   * @param stream The CUDA stream used to execute the operation
   *
   * @todo support accum with metas.
   */
  void accum(const Key *keys, const V *vals_or_deltas, const bool *exists,
             size_t len, bool allow_duplicated_keys = true,
             cudaStream_t stream = 0) {
    if (len == 0) {
      return;
    }

    Vector **dst;
    int *src_offset;
    bool *found;
    CUDA_CHECK(cudaMallocAsync(&dst, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMallocAsync(&src_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(src_offset, 0, len * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&found, len * sizeof(bool), stream));
    CUDA_CHECK(cudaMemsetAsync(found, 0, len * sizeof(bool), stream));

    if (allow_duplicated_keys) {
      const size_t N = len;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      accum_allow_duplicated_keys_kernel<Key, Vector, M, DIM>
          <<<grid_size, block_size_, 0, stream>>>(table_, keys, dst, exists,
                                                  found, src_offset, len);

    } else {
      int *bucket_offset;
      CUDA_CHECK(cudaMallocAsync(&bucket_offset, len * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(bucket_offset, 0, len * sizeof(int), stream));

      {
        const size_t N = len * table_->bucket_max_size;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);

        lookup_for_upsert_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(table_, keys, found,
                                                    bucket_offset, N);
      }

      {
        const size_t N = len;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
        accum_kernel<Key, Vector, M, DIM>
            <<<grid_size, block_size_, 0, stream>>>(table_, keys, dst, exists,
                                                    src_offset, found,
                                                    bucket_offset, len);
      }

      CUDA_CHECK(cudaFreeAsync(bucket_offset, stream));
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> dst_ptr(reinterpret_cast<uint64_t *>(dst));
      thrust::device_ptr<int> src_offset_ptr(src_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, dst_ptr, dst_ptr + N, src_offset_ptr,
                          thrust::less<uint64_t>());
    }

    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      write_with_accum_kernel<Key, Vector, M, DIM>
          <<<grid_size, block_size_, 0, stream>>>(
              reinterpret_cast<const Vector *>(vals_or_deltas), dst, exists,
              found, src_offset, N);
    }

    CUDA_CHECK(cudaFreeAsync(dst, stream));
    CUDA_CHECK(cudaFreeAsync(src_offset, stream));
    CUDA_CHECK(cudaFreeAsync(found, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Searches the table for the specified keys.
   *
   * @note When a key is missing, a default value in @p default_vectors will
   * returned. Specially, if @p full_size_default is true, @p default_vectors
   * will be treated as a V array with @p len * DIM, at this situation,
   * each keys will have a different default value, or if
   * @p full_size_default is true, the @p default_vectors only contains one
   * default and each keys will share it when missed.
   *
   * @param keys The keys to be searched on GPU accessible memory.
   * @param vectors The vectors to be searched on GPU accessible memory.
   * @param found The status indicates if the keys are found on GPU accessible
   * memory.
   * @param len Number of Key-Value pairs to be searched.
   * @param default_vectors The default vectors for each keys on GPU accessible
   * memory. If the keys are missing, the vectors in it will be returned.
   * @param full_size_default true if the default_vectors contains the same size
   * default vectors with @p keys.
   * @param stream The CUDA stream used to execute the operation.
   */
  void find(const Key *keys, V *vectors, bool *found, size_t len,
            const V *default_vectors, bool full_size_default,
            cudaStream_t stream = 0) const {
    if (len == 0) {
      return;
    }

    Vector **src;
    int *dst_offset;
    CUDA_CHECK(cudaMallocAsync(&src, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(found, 0, len * sizeof(bool), stream));
    CUDA_CHECK(cudaMallocAsync(&dst_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, len * sizeof(int), stream));

    // Determine bucket locations for reading.
    {
      const size_t N = len * table_->bucket_max_size;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);

      lookup_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          table_, keys, src, found, dst_offset, N);
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> src_ptr(reinterpret_cast<uint64_t *>(src));
      thrust::device_ptr<int> dst_offset_ptr(dst_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, src_ptr, src_ptr + N, dst_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy data from bucket to the pointer to vectors.
    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      read_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          src, reinterpret_cast<Vector *>(vectors), found,
          reinterpret_cast<const Vector *>(default_vectors), dst_offset, N,
          full_size_default);
    }

    CUDA_CHECK(cudaFreeAsync(src, stream));
    CUDA_CHECK(cudaFreeAsync(dst_offset, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Searches the table for the specified keys.
   *
   * @note When a key is missing, a default value in @p default_vectors will
   * returned. Specially, if @p full_size_default is true, @p default_vectors
   * will be treated as a V array with @p len * DIM, at this situation,
   * each keys will have a different default value, or if
   * @p full_size_default is false, the @p default_vectors only contains one
   * default vector and each keys will share it when missed.
   *
   * @param keys The keys to be searched on GPU accessible memory.
   * @param vectors The vectors to be searched on GPU accessible memory.
   * @param metas The metas to be searched on GPU accessible memory.
   * @param found The status indicates if the keys are found on GPU accessible
   * memory.
   * @param len Number of Key-Value-Meta tuples to be searched.
   * @param default_vectors The default vectors for each keys on GPU accessible
   * memory. If the keys are missing, the vectors in it will be returned.
   * @param full_size_default true if the default_vectors contains the same size
   * default vectors with @p keys. duplicate keys. If false, the caller should
   * guarantee the @p keys has no duplicated keys, and the performance will be
   * better.
   * @param stream The CUDA stream used to execute the operation.
   */
  void find(const Key *keys, V *vectors, M *metas, bool *found, size_type len,
            const V *default_vectors, bool full_size_default,
            cudaStream_t stream = 0) const {
    if (len == 0) {
      return;
    }

    Vector **src;
    int *dst_offset;
    CUDA_CHECK(cudaMallocAsync(&src, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(found, 0, len * sizeof(bool), stream));
    CUDA_CHECK(cudaMallocAsync(&dst_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, len * sizeof(int), stream));

    // Determine bucket locations for reading.
    {
      const size_t N = len * table_->bucket_max_size;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      lookup_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          table_, keys, src, metas, found, dst_offset, N);
      CudaCheckError();
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> src_ptr(reinterpret_cast<uint64_t *>(src));
      thrust::device_ptr<int> dst_offset_ptr(dst_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, src_ptr, src_ptr + N, dst_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy data from bucket to the pointer to vectors.
    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      read_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          src, reinterpret_cast<Vector *>(vectors), found,
          reinterpret_cast<const Vector *>(default_vectors), dst_offset, N,
          full_size_default);
    }

    CUDA_CHECK(cudaFreeAsync(src, stream));
    CUDA_CHECK(cudaFreeAsync(dst_offset, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Searches the table for the specified keys.
   *
   * @note When a key is missing, its vector will be initialized by
   * this->initializer_ and returned.
   *
   * @param keys The keys to be searched on GPU accessible memory.
   * @param vectors The vectors to be searched on GPU accessible memory.
   * @param len Number of Key-Value pairs to be searched.
   * @param stream The CUDA stream used to execute the operation.
   */
  void find(const Key *keys, V *vectors, size_type len,
            cudaStream_t stream = 0) const {
    if (len == 0) {
      return;
    }

    Vector **src;
    int *dst_offset;
    CUDA_CHECK(cudaMallocAsync(&src, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0, len * sizeof(Vector *), stream));
    CUDA_CHECK(cudaMallocAsync(&dst_offset, len * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, len * sizeof(int), stream));

    initializer_->initialize(vectors, len * sizeof(V), stream);

    // Determine bucket locations for reading.
    {
      const size_t N = len * table_->bucket_max_size;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      lookup_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          table_, keys, src, dst_offset, N);
    }

    {
      static_assert(
          sizeof(V *) == sizeof(uint64_t),
          "[merlin-kv] illegal conversation. V pointer must be 64 bit!");

      const size_t N = len;
      thrust::device_ptr<uint64_t> src_ptr(reinterpret_cast<uint64_t *>(src));
      thrust::device_ptr<int> dst_offset_ptr(dst_offset);

#if THRUST_VERSION >= 101600
      auto policy = thrust::cuda::par_nosync.on(stream);
#else
      auto policy = thrust::cuda::par.on(stream);
#endif
      thrust::sort_by_key(policy, src_ptr, src_ptr + N, dst_offset_ptr,
                          thrust::less<uint64_t>());
    }

    // Copy data from bucket to the pointer to vectors.
    {
      const size_t N = len * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      read_kernel<Key, Vector, M, DIM><<<grid_size, block_size_, 0, stream>>>(
          src, reinterpret_cast<Vector *>(vectors), dst_offset, N);
    }

    CUDA_CHECK(cudaFreeAsync(src, stream));
    CUDA_CHECK(cudaFreeAsync(dst_offset, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Get the table size.
   *
   * @param stream The CUDA stream used to execute the operation.
   * @return The table size
   */
  size_type size(cudaStream_t stream = 0) const {
    size_t h_size = 0;
    size_type N = table_->buckets_num;
    thrust::device_ptr<int> size_ptr(table_->buckets_size);

#if THRUST_VERSION >= 101600
    auto policy = thrust::cuda::par_nosync.on(stream);
#else
    auto policy = thrust::cuda::par.on(stream);
#endif
    h_size = thrust::reduce(policy, size_ptr, size_ptr + N, (int)0,
                            thrust::plus<int>());
    CudaCheckError();
    return h_size;
  }

  /**
   * @brief Returns the maximum number of elements the table.
   *
   * @return The table max size
   */
  size_type max_size() const noexcept {
    return static_cast<size_t>(bucket_max_size_ * table_->buckets_num);
  }

  /**
   * @brief Checks if the table has no elements.
   *
   * @param stream The CUDA stream used to execute the operation.
   * @return true if the table is empty, false otherwise
   */
  bool empty(cudaStream_t stream = 0) const { return size(stream) == 0; }

  /**
   * @brief Get the table capacity.
   *
   * @note The capacity is requested by the caller and the value may be
   * less than the actual capacity of the table because the table keeps
   * the capacity to be the power of 2 for performance consideration.
   *
   * @return The table capacity
   */
  size_type capacity() const { return table_->capacity; }

  /**
   * @brief Remove all of the elements in the table with no release object.
   */
  void clear(cudaStream_t stream = 0) {
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
    clear_kernel<Key, Vector, M, DIM>
        <<<grid_size, block_size_, 0, stream>>>(table_, N);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
  }

  /**
   * @brief Removes specified elements from the table.
   *
   * @param keys The keys to be removed on GPU accessible memory.
   * @param len Number of Key to be removed.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of elements removed
   */
  size_t erase(const Key *keys, size_type len, cudaStream_t stream = 0) {
    const size_t N = len * table_->bucket_max_size;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
    size_t count = 0;
    size_t *d_count;
    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

    remove_kernel<Key, Vector, M, DIM>
        <<<grid_size, block_size_, 0, stream>>>(table_, keys, d_count, N);

    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return count;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * table.
   *
   * @param pred predicate that returns true if the element should be erased.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of elements removed
   */
  size_t erase_if(Pred &pred, cudaStream_t stream = 0) {
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
    size_t count = 0;
    size_t *d_count;
    Pred h_pred;

    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    remove_kernel<Key, Vector, M, DIM>
        <<<grid_size, block_size_, 0, stream>>>(table_, h_pred, d_count, N);

    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return count;
  }

  /**
   * @brief Export a certain numer of the Key-value pairs in the table.
   *
   * @param keys The keys to be dumped on GPU accessible memory.
   * @param vectors The vectors to be dumped on GPU accessible memory.
   * @param offset Number of Key to be removed.
   * @param max_num Maximum number of dumped pairs.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return the number of items dumped.
   *
   * @throw CudaException If the K-V size is too large for GPU shared memory.
   * Reducing the @ p max_num is needed at this time.
   */
  size_type dump(Key *keys, V *vectors, const size_type offset,
                 const size_type max_num, cudaStream_t stream = 0) const {
    size_type h_counter = 0;
    size_type *d_counter;

    CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

    const size_t block_size =
        std::min(shared_mem_size_ / 2 / (sizeof(Key) + sizeof(Vector)), 1024UL);

    MERLIN_CHECK(block_size > 0,
                 "[merlin-kv] block_size <= 0, the K-V size may be too large!");
    const size_t shared_size =
        sizeof(Key) * block_size + sizeof(Vector) * block_size;
    const int grid_size = (max_num - 1) / (block_size) + 1;

    dump_kernel<Key, Vector, M, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, reinterpret_cast<Vector *>(vectors), offset, max_num,
            d_counter);

    CUDA_CHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_counter, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return h_counter;
  }

  /**
   * @brief Export all of the key-value-meta tuples in the table.
   *
   * @param keys The keys to be dumped on GPU accessible memory.
   * @param vectors The vectors to be dumped on GPU accessible memory.
   * @param metas The metas to be dumped on GPU accessible memory.
   * @param offset Number of Key to be removed.
   * @param max_num Maximum number of dumped pairs which helps caller
   * control the memory consuming on each calling.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return the number of items dumped.
   *
   * @throw CudaException If the K-V size is too large for GPU shared memory.
   * Reducing the @ p max_num is needed at this time.
   */
  size_type dump(Key *keys, V *vectors, M *metas, const size_type offset,
                 const size_type max_num, cudaStream_t stream = 0) const {
    size_type h_counter = 0;
    size_type *d_counter;

    CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

    const size_t block_size = std::min(
        shared_mem_size_ / 2 / (sizeof(Key) + sizeof(Vector) + sizeof(M)),
        1024UL);
    MERLIN_CHECK(block_size > 0,
                 "[merlin-kv] block_size <= 0, the K-V size may be too large!");
    const size_t shared_size =
        ((sizeof(Key) + sizeof(Vector) + sizeof(M))) * block_size;
    const int grid_size = (max_num - 1) / (block_size) + 1;

    dump_kernel<Key, Vector, M, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, reinterpret_cast<Vector *>(vectors), metas, offset,
            max_num, d_counter);
    CUDA_CHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_counter, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return h_counter;
  }

  /**
   * @brief Sets the number of buckets to the number needed to accomodate at
   * least count elements without exceeding maximum load factor and rehashes the
   * table, i.e. puts the elements into appropriate buckets considering that
   * total number of buckets has changed.
   *
   * @note If the count or double of the count is greater or equal than
   * max_size_, the reserve will not happen.
   *
   * @param count new capacity of the table.
   * @param stream The CUDA stream used to execute the operation.
   */
  void reserve(size_type count, cudaStream_t stream = 0) {
    if (reach_max_size_ || count > max_size_) {
      return;
    }

    while (capacity() < count && capacity() * 2 <= max_size_) {
      std::cout << "[merlin-kv] load_factor=" << load_factor()
                << ", reserve is being executed, "
                << "the capacity will increase from " << capacity() << " to "
                << capacity() * 2 << "." << std::endl;
      double_capacity(&table_);

      const size_t N = capacity() / 2;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size_);
      rehash_kernel<Key, Vector, M, DIM>
          <<<grid_size, block_size_, 0, stream>>>(table_, N);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    reach_max_size_ = (capacity() * 2 > max_size_);
    CudaCheckError();
  }

  /**
   * @brief Returns the maximum number of elements the table.
   *
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return The table max size
   */
  float load_factor(cudaStream_t stream = 0) const {
    return static_cast<float>((size(stream) * 1.0) / (capacity() * 1.0));
  };

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the container.
   */
  size_type bucket_count() const noexcept { return table_->buckets_num; }

  /**
   * @brief Returns the maximum number of buckets the table.
   *
   * @return Maximum number of buckets.
   */
  size_type max_bucket_count() const noexcept {
    return static_cast<size_t>(max_size_ / bucket_max_size_);
  }

  /**
   * @brief Returns the maximum number of buckets the table.
   *
   * @param n the index of the bucket to examine.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return The number of elements in the bucket @p n.
   */
  size_type bucket_size(size_type n, cudaStream_t stream = 0) const noexcept {
    size_type size = 0;
    if (n < table_->buckets_num) {
      CUDA_CHECK(cudaMemcpyAsync(&size, &(table_->buckets[n].size),
                                 sizeof(size_t), cudaMemcpyDeviceToHost,
                                 stream));
      CudaCheckError();
    }
    return size;
  }

 private:
  int block_size_;
  const size_type init_size_;
  const size_type max_size_;
  const size_type max_hbm_for_vectors_;
  bool reach_max_size_;
  float max_load_factor_;
  const size_type bucket_max_size_;
  std::shared_ptr<Initializer> initializer_;
  const bool primary_;
  size_t shared_mem_size_;
  Table *table_;
  //   std::mutex mtx_;
};

}  // namespace merlin
}  // namespace nv
