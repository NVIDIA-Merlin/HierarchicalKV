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

#include <assert.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <condition_variable>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <vector>
#include "merlin/core_kernels.cuh"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

/**
 * @brief Enumeration of the eviction strategies.
 *
 * @note The `meta` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. On `kLru`
 * mode, the `metas` parameter of the APIs should keep `nullptr`, the meta for
 * each key is assigned internally in LRU(Least Recently Used) policy. On
 * `kCustomized` mode, the `metas` should be provided by caller.
 *
 * @note Eviction occurs automatically when a bucket is full. The keys with the
 * minimum `meta` value are evicted first.
 *
 */
enum class EvictStrategy {
  kLru = 0,        ///< LRU mode.
  kCustomized = 1  ///< Customized mode.
};

/**
 * @brief The options struct of Merlin-KV.
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity of the hash table.
  size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
  size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
  size_t max_bucket_size = 128;    ///< The length of each bucket.
  float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
  int block_size = 1024;           ///< The default block size for CUDA kernels.
  int device_id = 0;               ///< The ID of device.
  EvictStrategy evict_strategy = EvictStrategy::kLru;  ///< The evict strategy.
  size_t max_batch_size =
      64 * 1024 * 1024;  ///< Maximum batch size, for batched operations (also
                         ///< the size of a workspace).
  size_t min_num_workspaces = 3;  ///< Number of workspaces to keep in reserve.
  size_t max_num_workspaces = 5;  ///< Maximum number of workspaces.
};

/**
 * @brief A customizable template function indicates which keys should be
 * erased from the hash table by returning `true`.
 *
 * @note The `erase_if` API traverses all of the items by this function and the
 * items that return `true` are removed.
 *
 *  Example:
 *
 *    ```
 *    template <class K, class M>
 *    __forceinline__ __device__ bool erase_if_pred(const K& key,
 *                                                  const M& meta,
 *                                                  const K& pattern,
 *                                                  const M& threshold) {
 *      return ((key & 0xFFFF000000000000 == pattern) &&
 *              (meta < threshold));
 *    }
 *    ```
 */
template <class K, class M>
using EraseIfPredict = bool (*)(
    const K& key,       ///< The traversed key in a hash table.
    const M& meta,      ///< The traversed meta in a hash table.
    const K& pattern,   ///< The key pattern to compare with the `key` argument.
    const M& threshold  ///< The threshold to compare with the `meta` argument.
);

/**
 * A Merlin-KV hash table is a concurrent and hierarchical hash table that is
 * powered by GPUs and can use HBM and host memory as storage for key-value
 * pairs. Support for SSD storage is a future consideration.
 *
 * The `meta` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. Eviction
 * occurs automatically when a bucket is full. The keys with the minimum `meta`
 * value are evicted first. In a customized eviction strategy, we recommend
 * using the timestamp or frequency of the key occurrence as the `meta` value
 * for each key. You can also assign a special value to the `meta` to
 * perform a customized eviction strategy.
 *
 * @note By default configuration, this class is thread-safe.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's item type.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam M The data type for `meta`.
 *           The currently supported data type is only `uint64_t`.
 * @tparam D The dimension of the vectors.
 *
 */
template <class K, class V, class M, size_t D>
class HashTable {
 public:
  /**
   * @brief The value type of a Merlin-KV hash table.
   */
  struct Vector {
    using value_type = V;
    static constexpr size_t DIM = D;
    value_type values[DIM];
  };

 public:
  using this_type = HashTable<K, V, M, D>;
  using size_type = size_t;
  static constexpr size_type DIM = D;
  using key_type = K;
  using value_type = V;
  using vector_type = Vector;
  using meta_type = M;
  using Pred = EraseIfPredict<key_type, meta_type>;

 private:
  using TableCore = nv::merlin::Table<key_type, vector_type, meta_type, DIM>;
  static constexpr unsigned int TILE_SIZE = 8;

 public:
  /**
   * @brief Default constructor for the hash table class.
   */
  HashTable(){};

  /**
   * @brief Frees the resources used by the hash table and destroys the hash
   * table object.
   */
  ~HashTable() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // Destroy workspaces.
    avail_ws_.clear();
    for (char* ptr : ws_) {
      CUDA_CHECK(cudaFree(ptr));
    }
    ws_.clear();

    // Erase table
    if (initialized_) {
      destroy_table<key_type, vector_type, meta_type, DIM>(&table_);
    }
  }

 private:
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  HashTable(HashTable&&) = delete;
  HashTable& operator=(HashTable&&) = delete;

 public:
  /**
   * @brief Initialize a merlin::HashTable.
   *
   * @param options The configuration options.
   */
 public:
  void init(const HashTableOptions options) {
    // Prevent double call.
    if (initialized_) {
      return;
    }
    options_ = options;

    // Construct table.
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaSetDevice(options_.device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    create_table<key_type, vector_type, meta_type, DIM>(
        &table_, options_.init_capacity, options_.max_capacity,
        options_.max_hbm_for_vectors, options_.max_bucket_size);
    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);
    reach_max_capacity_ = (options_.init_capacity * 2 > options_.max_capacity);
    initialized_ = true;

    // Preallocate workspaces.
    assert(options_.min_num_workspaces >= 1 &&
           options_.min_num_workspaces <= options_.max_num_workspaces);
    ws_buffer_size_ =
        std::max(sizeof(void*), sizeof(uint64_t)) * options_.max_batch_size;

    avail_ws_.reserve(options_.min_num_workspaces);
    while (avail_ws_.size() < options_.min_num_workspaces) {
      char* ptr;
      CUDA_CHECK(cudaMalloc(&ptr, ws_buffer_size_));
      ws_.push_back(ptr);
      avail_ws_.push_back(ptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-meta tuples into the hash table.
   * If the key already exists, the values and metas are assigned new values.
   *
   * If the target bucket is full, the keys with minimum meta will be
   * overwritten by new key unless the meta of the new key is even less than
   * minimum meta of the target bucket.
   *
   * @param n Number of key-value-meta tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param metas The metas to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void insert_or_assign(size_type n,
                        const key_type* keys,              // (n)
                        const value_type* values,          // (n, DIM)
                        const meta_type* metas = nullptr,  // (n)
                        cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    if (!reach_max_capacity_ && fast_load_factor() > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    check_evict_strategy(metas);

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    if (is_fast_mode()) {
      // Precalc some constants
      const size_t block_size = 128;
      const size_t N = n * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      if (metas == nullptr) {
        upsert_kernel_with_io<key_type, vector_type, meta_type, DIM, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                table_, keys, reinterpret_cast<const vector_type*>(values),
                table_->buckets, table_->buckets_size, table_->bucket_max_size,
                table_->buckets_num, N);
      } else {
        upsert_kernel_with_io<key_type, vector_type, meta_type, DIM, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                table_, keys, reinterpret_cast<const vector_type*>(values),
                metas, table_->buckets, table_->buckets_size,
                table_->bucket_max_size, table_->buckets_num, N);
      }
    } else {
      Workspace<2> ws(this, stream);
      vector_type** d_dst = reinterpret_cast<vector_type**>(ws[0]);
      int* d_src_offset = reinterpret_cast<int*>(ws[1]);

      for (size_t i = 0; i < n; i += options_.max_batch_size) {
        const size_t bs = std::min(n - i, options_.max_batch_size);

        CUDA_CHECK(
            cudaMemsetAsync(d_dst, 0, bs * sizeof(vector_type*), stream));
        CUDA_CHECK(cudaMemsetAsync(d_src_offset, 0, bs * sizeof(int), stream));

        {
          const size_t block_size = 128;
          const size_t N = bs * TILE_SIZE;
          const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

          if (metas == nullptr) {
            upsert_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
                <<<grid_size, block_size, 0, stream>>>(
                    table_, &keys[i], d_dst, table_->buckets,
                    table_->buckets_size, table_->bucket_max_size,
                    table_->buckets_num, d_src_offset, N);
          } else {
            upsert_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
                <<<grid_size, block_size, 0, stream>>>(
                    table_, &keys[i], d_dst, metas ? &metas[i] : nullptr,
                    table_->buckets, table_->buckets_size,
                    table_->bucket_max_size, table_->buckets_num, d_src_offset,
                    N);
          }
        }

        {
          static_assert(sizeof(value_type*) == sizeof(uint64_t),
                        "[merlin-kv] illegal conversation. value_type pointer "
                        "must be 64 bit!");

          thrust::device_ptr<uint64_t> d_dst_ptr(
              reinterpret_cast<uint64_t*>(d_dst));
          thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

#if THRUST_VERSION >= 101600
          auto policy = thrust::cuda::par_nosync.on(stream);
#else
          auto policy = thrust::cuda::par.on(stream);
#endif
          thrust::sort_by_key(policy, d_dst_ptr, d_dst_ptr + bs,
                              d_src_offset_ptr, thrust::less<uint64_t>());
        }

        {
          const size_t N = bs * DIM;
          const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);

          write_kernel<key_type, vector_type, meta_type, DIM>
              <<<grid_size, options_.block_size, 0, stream>>>(
                  reinterpret_cast<const vector_type*>(&values[i]), d_dst,
                  d_src_offset, N);
        }
      }
    }

    CudaCheckError();
  }

  /**
   * Searches for each key in @p keys in the hash table.
   * If the key is found and the corresponding value in @p accum_or_assigns is
   * `true`, the @p vectors_or_deltas is treated as a delta to the old
   * value, and the delta is added to the old value of the key.
   *
   * If the key is not found and the corresponding value in @p accum_or_assigns
   * is `false`, the @p vectors_or_deltas is treated as a new value and the
   * key-value pair is updated in the table directly.
   *
   * @note When the key is found and the value of @p accum_or_assigns is
   * `false`, or when the key is not found and the value of @p accum_or_assigns
   * is `true`, nothing is changed and this operation is ignored.
   * The algorithm assumes these situations occur while the key was modified or
   * removed by other processes just now.
   *
   * @param n The number of key-value-meta tuples to process.
   * @param keys The keys to insert on GPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on GPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param metas The metas to insert on GPU-accessible memory with shape (n).
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void accum_or_assign(size_type n,
                       const key_type* keys,               // (n)
                       const value_type* value_or_deltas,  // (n, DIM)
                       const bool* accum_or_assigns,       // (n)
                       const meta_type* metas = nullptr,   // (n)
                       cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    if (!reach_max_capacity_ && fast_load_factor() > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    check_evict_strategy(metas);

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    Workspace<3> ws(this, stream);
    vector_type** dst = reinterpret_cast<vector_type**>(ws[0]);
    int* src_offset = reinterpret_cast<int*>(ws[1]);
    bool* founds = reinterpret_cast<bool*>(ws[2]);

    for (size_t i = 0; i < n; i += options_.max_batch_size) {
      const size_t bs = std::min(n - i, options_.max_batch_size);

      CUDA_CHECK(cudaMemsetAsync(dst, 0, bs * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(src_offset, 0, bs * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(founds, 0, bs * sizeof(bool), stream));

      {
        const size_t block_size = 128;
        const size_t N = bs * TILE_SIZE;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        if (metas == nullptr) {
          accum_kernel<key_type, vector_type, meta_type, DIM>
              <<<grid_size, block_size, 0, stream>>>(
                  table_, &keys[i], dst, &accum_or_assigns[i], table_->buckets,
                  table_->buckets_size, table_->bucket_max_size,
                  table_->buckets_num, src_offset, founds, N);
        } else {
          accum_kernel<key_type, vector_type, meta_type, DIM>
              <<<grid_size, block_size, 0, stream>>>(
                  table_, &keys[i], dst, metas ? &metas[i] : nullptr,
                  &accum_or_assigns[i], table_->buckets, table_->buckets_size,
                  table_->bucket_max_size, table_->buckets_num, src_offset,
                  founds, N);
        }
      }

      if (!is_fast_mode()) {
        static_assert(
            sizeof(value_type*) == sizeof(uint64_t),
            "[merlin-kv] illegal conversation. value_type pointer must "
            "be 64 bit!");

        thrust::device_ptr<uint64_t> dst_ptr(reinterpret_cast<uint64_t*>(dst));
        thrust::device_ptr<int> src_offset_ptr(src_offset);

#if THRUST_VERSION >= 101600
        auto policy = thrust::cuda::par_nosync.on(stream);
#else
        auto policy = thrust::cuda::par.on(stream);
#endif
        thrust::sort_by_key(policy, dst_ptr, dst_ptr + bs, src_offset_ptr,
                            thrust::less<uint64_t>());
      }

      {
        const size_t N = bs * DIM;
        const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);

        write_with_accum_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, options_.block_size, 0, stream>>>(
                reinterpret_cast<const vector_type*>(&value_or_deltas[i]), dst,
                &accum_or_assigns[i], founds, src_offset, N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-meta tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find(size_type n,
            const key_type* keys,        // (n)
            value_type* values,          // (n, DIM)
            bool* founds,                // (n)
            meta_type* metas = nullptr,  // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    if (is_fast_mode()) {
      const size_t block_size = 128;
      const size_t N = n * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      lookup_kernel_with_io<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, reinterpret_cast<vector_type*>(values), metas,
              founds, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    } else {
      Workspace<2> ws(this, stream);
      vector_type** src = reinterpret_cast<vector_type**>(ws[0]);
      int* dst_offset = reinterpret_cast<int*>(ws[1]);

      for (size_t i = 0; i < n; i += options_.max_batch_size) {
        const size_t bs = std::min(n - i, options_.max_batch_size);

        CUDA_CHECK(cudaMemsetAsync(src, 0, bs * sizeof(vector_type*), stream));
        CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, bs * sizeof(int), stream));

        {
          const size_t block_size = 128;
          const size_t N = bs * TILE_SIZE;
          const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

          lookup_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
              <<<grid_size, block_size, 0, stream>>>(
                  table_, &keys[i], reinterpret_cast<vector_type**>(src),
                  metas ? &metas[i] : nullptr, &founds[i], table_->buckets,
                  table_->buckets_size, table_->bucket_max_size,
                  table_->buckets_num, dst_offset, N);
        }

        {
          static_assert(sizeof(value_type*) == sizeof(uint64_t),
                        "[merlin-kv] illegal conversation. value_type pointer "
                        "must be 64 bit!");

          thrust::device_ptr<uint64_t> src_ptr(
              reinterpret_cast<uint64_t*>(src));
          thrust::device_ptr<int> dst_offset_ptr(dst_offset);

#if THRUST_VERSION >= 101600
          auto policy = thrust::cuda::par_nosync.on(stream);
#else
          auto policy = thrust::cuda::par.on(stream);
#endif
          thrust::sort_by_key(policy, src_ptr, src_ptr + bs, dst_offset_ptr,
                              thrust::less<uint64_t>());
        }

        {
          const size_t N = bs * DIM;
          const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);

          read_kernel<key_type, vector_type, meta_type, DIM>
              <<<grid_size, options_.block_size, 0, stream>>>(
                  src, reinterpret_cast<vector_type*>(&values[i]), &founds[i],
                  dst_offset, N);
        }
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Removes specified elements from the hash table.
   *
   * @param n The number of keys to remove.
   * @param keys The keys to remove on GPU-accessible memory.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   */
  size_t erase(size_type n, const key_type* keys, cudaStream_t stream = 0) {
    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    Workspace<1> ws(this, stream);
    size_type* d_count = reinterpret_cast<size_type*>(ws[0]);

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

    for (size_t i = 0; i < n; i += options_.max_batch_size) {
      const size_t bs = std::min(n - i, options_.max_batch_size);

      const size_t block_size = 128;
      const size_t N = bs * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, &keys[i], d_count, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }

    size_type h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    return h_count;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * hash table.
   *
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class M>
   *    __forceinline__ __device__ bool erase_if_pred(const K& key,
   *                                                  const M& meta,
   *                                                  const K& pattern,
   *                                                  const M& threshold) {
   *      return ((key & 0x1 == pattern) && (meta < threshold));
   *    }
   *    ```
   *
   * @param pred The predicate function with type Pred that returns `true` if
   * the element should be erased.
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with meta_type
   * type.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   *
   */
  size_t erase_if(Pred& pred, const key_type& pattern,
                  const meta_type& threshold, cudaStream_t stream = 0) {
    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    Workspace<1> ws(this, stream);
    size_type* d_count = reinterpret_cast<size_type*>(ws[0]);

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

    Pred h_pred;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    {
      const size_t block_size = 256;
      const size_t N = table_->buckets_num;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, 0, stream>>>(
              table_, h_pred, pattern, threshold, d_count, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, N);
    }

    size_type h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    return h_count;
  }

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  void clear(cudaStream_t stream = 0) {
    // Precalc some constants.
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const size_t block_size = options_.block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    clear_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, 0, stream>>>(table_, N);

    CudaCheckError();
  }

 public:
  /**
   * @brief Exports a certain number of the key-value-meta tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key-value-meta tuple to export.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw CudaException If the key-value size is too large for GPU shared
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  size_type export_batch(size_type n, size_type offset,
                         key_type* keys,              // (n)
                         value_type* values,          // (n, DIM)
                         meta_type* metas = nullptr,  // (n)
                         cudaStream_t stream = 0) const {
    // Precalc some constants.
    const size_type meta_size = metas ? sizeof(meta_type) : 0;
    const size_t kvm_size = sizeof(key_type) + sizeof(vector_type) + meta_size;
    const size_t block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
    MERLIN_CHECK(
        block_size > 0,
        "[merlin-kv] block_size <= 0, the K-V-M size may be too large!");
    const size_t shared_size = kvm_size * block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    // Fetch temporary workspace.
    Workspace<1> ws(this, stream);
    size_type* d_count = reinterpret_cast<size_type*>(ws[0]);

    // Reset counter and dump kernel.
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

    dump_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, reinterpret_cast<vector_type*>(values), metas, offset,
            n, d_count);

    // Move result counter to host.
    size_type h_count;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    return h_count;
  }

 public:
  /**
   * @brief Indicates if the hash table has no elements.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return `true` if the table is empty and `false` otherwise.
   */
  bool empty(cudaStream_t stream = 0) const { return size(stream) == 0; }

  /**
   * @brief Returns the hash table size.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return The table size.
   */
  size_type size(cudaStream_t stream = 0) const {
    // Precalc constants.
    const size_type n = table_->buckets_num;
#if THRUST_VERSION >= 101600
    auto policy = thrust::cuda::par_nosync.on(stream);
#else
    auto policy = thrust::cuda::par.on(stream);
#endif

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }

    thrust::device_ptr<int> size_ptr(table_->buckets_size);
    const size_t h_size =
        thrust::reduce(policy, size_ptr, size_ptr + n, 0, thrust::plus<int>());

    CudaCheckError();
    return h_size;
  }

  /**
   * @brief Returns the hash table capacity.
   *
   * @note The value that is returned might be less than the actual capacity of
   * the hash table because the hash table currently keeps the capacity to be
   * a power of 2 for performance considerations.
   *
   * @return The table capacity.
   */
  size_type capacity() const { return table_->capacity; }

  /**
   * @brief Sets the number of buckets to the number that is needed to
   * accommodate at least @p new_capacity elements without exceeding the maximum
   * load factor. This method rehashes the hash table. Rehashing puts the
   * elements into the appropriate buckets considering that total number of
   * buckets has changed.
   *
   * @note If the value of @p new_capacity or double of @p new_capacity is
   * greater or equal than `options_.max_capacity`, the reserve does not perform
   * any change to the hash table.
   *
   * @param new_capacity The requested capacity for the hash table.
   * @param stream The CUDA stream that is used to execute the operation.
   */
  void reserve(size_type new_capacity, cudaStream_t stream = 0) {
    if (reach_max_capacity_ || new_capacity > options_.max_capacity) {
      return;
    }

    // Gain exclusive access to table.
    std::unique_lock<std::shared_timed_mutex> lock(table_mutex_);

    // Make sure any pending GPU calls have been processed.
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO(M_LANGER): Should resize to final capacity in one step?
    while (capacity() < new_capacity &&
           capacity() * 2 <= options_.max_capacity) {
      double_capacity(&table_);

      const size_t N = TILE_SIZE * table_->buckets_num / 2;
      const size_t block_size = 128;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      rehash_kernel_for_fast_mode<key_type, vector_type, meta_type, DIM,
                                  TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    reach_max_capacity_ = capacity() * 2 > options_.max_capacity;
    CudaCheckError();
  }

  /**
   * @brief Returns the average number of elements per slot, that is, size()
   * divided by capacity().
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The load factor
   */
  float load_factor(cudaStream_t stream = 0) const {
    return static_cast<float>((size(stream) * 1.0) / (capacity() * 1.0));
  }

  /**
   * @brief Save table to an abstract file.
   *
   * @param file An KVFile object defined the file format within filesystem.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys saved to file.
   */
  size_type save(KVFile<K, V, M, DIM>* file, cudaStream_t stream = 0) const {
    // Precalc some constants.
    const size_type N =
        ws_buffer_size_ /
        std::max(std::max(sizeof(key_type), sizeof(vector_type)),
                 sizeof(meta_type));
    assert(N > 0);

    const size_t kvm_size =
        sizeof(key_type) + sizeof(vector_type) + sizeof(meta_type);
    const size_t block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
    MERLIN_CHECK(
        block_size > 0,
        "[merlin-kv] block_size <= 0, the K-V-M size may be too large!");
    const size_t shared_size = kvm_size * block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }
    const size_type total_size = capacity();

    // Fetch temporary workspace.
    Workspace<4> ws(this, stream);
    size_type* d_count = reinterpret_cast<size_type*>(ws[0]);
    key_type* d_keys = reinterpret_cast<key_type*>(ws[1]);
    vector_type* d_vectors = reinterpret_cast<vector_type*>(ws[2]);
    meta_type* d_metas = reinterpret_cast<meta_type*>(ws[3]);

    // Grab enough host memory to hold batch data.
    auto h_keys = nv::merlin::make_unique_host<key_type>(N);
    auto h_values = nv::merlin::make_unique_host<V>(DIM * N);
    auto h_metas = nv::merlin::make_unique_host<meta_type>(N);

    // Step through table, dumping contents in batches.
    size_type total_count = 0;
    for (size_type offset = 0; offset < total_size; offset += N) {
      // Dump the next batch to workspace.
      CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

      dump_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, shared_size, stream>>>(
              table_, d_keys, d_vectors, d_metas, offset, N, d_count);

      size_type h_count;
      CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(size_type),
                                 cudaMemcpyDeviceToHost, stream));

      // Move workspace to host memory.
      CUDA_CHECK(cudaMemcpyAsync(h_keys.get(), d_keys,
                                 sizeof(key_type) * h_count,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_values.get(), d_vectors,
                                 sizeof(vector_type) * h_count,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_metas.get(), d_metas,
                                 sizeof(meta_type) * h_count,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Store permanently.
      file->Write(h_count, h_keys.get(), h_values.get(), h_metas.get());
      total_count += h_count;
    }

    return total_count;
  }

  /**
   * @brief Load file and restore table.
   *
   * @param file An KVFile object defined the file format within filesystem.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(KVFile<K, V, M, DIM>* file, cudaStream_t stream = 0) {
    // Precalc some constants.
    const size_type max_count =
        ws_buffer_size_ /
        std::max(std::max(sizeof(key_type), sizeof(vector_type)),
                 sizeof(meta_type));
    assert(max_count > 0);

    // Fetch temporary workspace.
    Workspace<3> ws(this, stream);
    key_type* d_keys = reinterpret_cast<key_type*>(ws[0]);
    V* d_values = reinterpret_cast<V*>(ws[1]);
    meta_type* d_metas = reinterpret_cast<meta_type*>(ws[2]);

    // Grab enough host memory to hold batch data.
    auto h_keys = nv::merlin::make_unique_host<key_type>(max_count);
    auto h_values = nv::merlin::make_unique_host<V>(DIM * max_count);
    auto h_metas = nv::merlin::make_unique_host<meta_type>(max_count);

    size_type total_count = 0;
    while (true) {
      // Read next batch.
      const size_type count =
          file->Read(max_count, h_keys.get(), h_values.get(), h_metas.get());
      if (count <= 0) {
        break;
      }

      // Move read data to device.
      CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys.get(), sizeof(key_type) * count,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_values, h_values.get(),
                                 sizeof(vector_type) * count,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas.get(),
                                 sizeof(meta_type) * count,
                                 cudaMemcpyHostToDevice, stream));

      insert_or_assign(count, d_keys, d_values, d_metas, stream);
      total_count += count;
    }

    return total_count;
  }

 private:
  inline bool is_fast_mode() const noexcept { return table_->is_pure_hbm; }

  /**
   * @brief Returns the load factor by sampling up to 1024 buckets.
   *
   * @note For performance consideration, the returned load factor is
   * inaccurate but within an error in 1% empirically which is enough for
   * capacity control. But it's not suitable for end-users.
   *
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return The evaluated load factor
   */
  inline float fast_load_factor(cudaStream_t stream = 0) const {
    // Unless we reached capacity, reallocation could happen.
    std::shared_lock<std::shared_timed_mutex> lock(table_mutex_);
    if (reach_max_capacity_) {
      lock.unlock();
    }
    size_type N = std::min(table_->buckets_num, 1024UL);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

#if THRUST_VERSION >= 101600
    auto policy = thrust::cuda::par_nosync.on(stream);
#else
    auto policy = thrust::cuda::par.on(stream);
#endif
    size_t h_size =
        thrust::reduce(policy, size_ptr, size_ptr + N, 0, thrust::plus<int>());

    CudaCheckError();
    return static_cast<float>((h_size * 1.0) /
                              (options_.max_bucket_size * N * 1.0));
  }

  inline void check_evict_strategy(const meta_type* metas) {
    if (options_.evict_strategy == EvictStrategy::kLru) {
      MERLIN_CHECK((metas == nullptr),
                   "the metas should not be specified when running on "
                   "LRU mode.");
    }

    if (options_.evict_strategy == EvictStrategy::kCustomized) {
      MERLIN_CHECK((metas != nullptr),
                   "the metas should be specified when running on "
                   "customized mode.")
    }
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  size_t shared_mem_size_ = 0;
  bool reach_max_capacity_ = false;
  bool initialized_ = false;
  mutable std::shared_timed_mutex table_mutex_;

  // Workspace management.
  template <size_t SIZE>
  class Workspace final {
   public:
    explicit Workspace(const this_type* parent, cudaStream_t stream)
        : parent_{parent}, stream_{stream} {
      parent_->claim_ws_(buffers_, stream);
    }

    Workspace(const Workspace&) = delete;
    Workspace(Workspace&&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace& operator=(Workspace&&) = delete;

    ~Workspace() {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      CudaCheckError();
      parent_->release_ws_(buffers_, stream_);
    }

    constexpr char* operator[](size_t i) {
      assert(i < SIZE);
      return buffers_[i];
    }

   private:
    const this_type* const parent_;
    cudaStream_t const stream_;
    char* buffers_[SIZE];
  };

  size_t ws_buffer_size_;
  mutable std::mutex ws_mutex_;
  mutable std::list<char*> ws_;
  mutable std::vector<char*> avail_ws_;
  mutable std::condition_variable ws_returned_;

  template <size_t SIZE>
  void claim_ws_(char* (&buffers)[SIZE], cudaStream_t stream) const {
    std::unique_lock<std::mutex> lock(ws_mutex_);

    // If have a prellocated workspace available.
    if (avail_ws_.size() >= SIZE) {
      for (size_t i = 0; i < SIZE; ++i) {
        buffers[i] = avail_ws_.back();
        avail_ws_.pop_back();
      }
    }
    // If workspace creation quota not yet reached.
    else if (ws_.size() + SIZE <= options_.max_num_workspaces) {
      for (size_t i = 0; i < SIZE; ++i) {
        char* ptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, ws_buffer_size_, stream));
        ws_.push_back(ptr);
        buffers[i] = ptr;
      }
    }
    // Creation quota reached. Wait for another thread to return a
    else {
      while (true) {
        ws_returned_.wait(lock, [&] { return avail_ws_.size() >= SIZE; });
        if (avail_ws_.size() < SIZE) {
          ws_returned_.notify_one();
          continue;
        }

        for (size_t i = 0; i < SIZE; ++i) {
          buffers[i] = avail_ws_.back();
          avail_ws_.pop_back();
        }
        break;
      }
    }
  }

  template <size_t SIZE>
  void release_ws_(char* (&buffers)[SIZE], cudaStream_t stream) const {
    std::lock_guard<std::mutex> lock(ws_mutex_);
    size_t i = 0;

    // Fill up available buffers until reach reserve capacity.
    bool has_returned_ws = false;
    for (; i < SIZE && avail_ws_.size() < options_.min_num_workspaces; ++i) {
      avail_ws_.emplace_back(buffers[i]);
      has_returned_ws = true;
    }

    // Discard overflow buffers.
    for (; i < SIZE; ++i) {
      char* ptr = buffers[i];
      for (auto it = ws_.begin(); it != ws_.end(); ++it) {
        if (*it == ptr) {
          CUDA_CHECK(cudaFreeAsync(ptr, stream));
        }
      }
    }

    // Give a waiting thread a chance to start.
    if (has_returned_ws) {
      ws_returned_.notify_one();
    }
  }
};

}  // namespace merlin
}  // namespace nv
