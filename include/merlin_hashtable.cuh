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
#include <shared_mutex>
#include <type_traits>
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
    if (initialized_) {
      return;
    }
    options_ = options;
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
   * @param n Number of key-value-meta tuples to inserted or assign.
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

    if (is_fast_mode()) {
      const size_t block_size = 128;
      const size_t N = n * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
      if (!reach_max_capacity_) {
        lock.lock();
      }

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
      vector_type** d_dst = nullptr;
      int* d_src_offset = nullptr;

      std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
      if (!reach_max_capacity_) {
        lock.lock();
      }

      CUDA_CHECK(cudaMallocAsync(&d_dst, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMallocAsync(&d_src_offset, n * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(d_src_offset, 0, n * sizeof(int), stream));

      {
        const size_t block_size = 128;
        const size_t N = n * TILE_SIZE;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
        if (metas == nullptr) {
          upsert_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
              <<<grid_size, block_size, 0, stream>>>(
                  table_, keys, d_dst, table_->buckets, table_->buckets_size,
                  table_->bucket_max_size, table_->buckets_num, d_src_offset,
                  N);
        } else {
          upsert_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
              <<<grid_size, block_size, 0, stream>>>(
                  table_, keys, d_dst, metas, table_->buckets,
                  table_->buckets_size, table_->bucket_max_size,
                  table_->buckets_num, d_src_offset, N);
        }
      }

      {
        static_assert(sizeof(value_type*) == sizeof(uint64_t),
                      "[merlin-kv] illegal conversation. value_type pointer "
                      "should be 64 bit!");

        const size_t N = n;
        thrust::device_ptr<uint64_t> d_dst_ptr(
            reinterpret_cast<uint64_t*>(d_dst));
        thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

#if THRUST_VERSION >= 101600
        auto policy = thrust::cuda::par_nosync.on(stream);
#else
        auto policy = thrust::cuda::par.on(stream);
#endif
        thrust::sort_by_key(policy, d_dst_ptr, d_dst_ptr + N, d_src_offset_ptr,
                            thrust::less<uint64_t>());
      }

      {
        const size_t N = n * DIM;
        const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);
        write_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, options_.block_size, 0, stream>>>(
                reinterpret_cast<const vector_type*>(values), d_dst,
                d_src_offset, N);
      }

      CUDA_CHECK(cudaFreeAsync(d_dst, stream));
      CUDA_CHECK(cudaFreeAsync(d_src_offset, stream));
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

    vector_type** dst;
    int* src_offset;
    bool* founds;

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    CUDA_CHECK(cudaMallocAsync(&dst, n * sizeof(vector_type*), stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0, n * sizeof(vector_type*), stream));
    CUDA_CHECK(cudaMallocAsync(&src_offset, n * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(src_offset, 0, n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&founds, n * sizeof(bool), stream));
    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    {
      const size_t block_size = 128;
      const size_t N = n * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      if (metas == nullptr) {
        accum_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, block_size, 0, stream>>>(
                table_, keys, dst, accum_or_assigns, table_->buckets,
                table_->buckets_size, table_->bucket_max_size,
                table_->buckets_num, src_offset, founds, N);
      } else {
        accum_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, block_size, 0, stream>>>(
                table_, keys, dst, metas, accum_or_assigns, table_->buckets,
                table_->buckets_size, table_->bucket_max_size,
                table_->buckets_num, src_offset, founds, N);
      }
    }

    if (!is_fast_mode()) {
      static_assert(sizeof(value_type*) == sizeof(uint64_t),
                    "[merlin-kv] illegal conversation. value_type pointer must "
                    "be 64 bit!");

      const size_t N = n;
      thrust::device_ptr<uint64_t> dst_ptr(reinterpret_cast<uint64_t*>(dst));
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
      const size_t N = n * DIM;
      const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);
      write_with_accum_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, options_.block_size, 0, stream>>>(
              reinterpret_cast<const vector_type*>(value_or_deltas), dst,
              accum_or_assigns, founds, src_offset, N);
    }

    CUDA_CHECK(cudaFreeAsync(dst, stream));
    CUDA_CHECK(cudaFreeAsync(src_offset, stream));
    CUDA_CHECK(cudaFreeAsync(founds, stream));

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
  void find(size_type n, const key_type* keys,  // (n)
            value_type* values,                 // (n, DIM)
            bool* founds,                       // (n)
            meta_type* metas = nullptr,         // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
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
      vector_type** src;
      int* dst_offset = nullptr;
      CUDA_CHECK(cudaMallocAsync(&src, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(src, 0, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMallocAsync(&dst_offset, n * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, n * sizeof(int), stream));

      {
        const size_t block_size = 128;
        const size_t N = n * TILE_SIZE;
        const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        lookup_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                table_, keys, reinterpret_cast<vector_type**>(src), metas,
                founds, table_->buckets, table_->buckets_size,
                table_->bucket_max_size, table_->buckets_num, dst_offset, N);
      }

      {
        static_assert(sizeof(value_type*) == sizeof(uint64_t),
                      "[merlin-kv] illegal conversation. value_type pointer "
                      "must be 64 bit!");

        const size_t N = n;
        thrust::device_ptr<uint64_t> src_ptr(reinterpret_cast<uint64_t*>(src));
        thrust::device_ptr<int> dst_offset_ptr(dst_offset);

#if THRUST_VERSION >= 101600
        auto policy = thrust::cuda::par_nosync.on(stream);
#else
        auto policy = thrust::cuda::par.on(stream);
#endif
        thrust::sort_by_key(policy, src_ptr, src_ptr + N, dst_offset_ptr,
                            thrust::less<uint64_t>());
      }

      {
        const size_t N = n * DIM;
        const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);
        read_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, options_.block_size, 0, stream>>>(
                src, reinterpret_cast<vector_type*>(values), founds, dst_offset,
                N);
      }

      CUDA_CHECK(cudaFreeAsync(src, stream));
      CUDA_CHECK(cudaFreeAsync(dst_offset, stream));
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
    const size_t block_size = 128;
    const size_t N = n * TILE_SIZE;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    size_t count = 0;
    size_t* d_count;

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

    remove_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
        <<<grid_size, block_size, 0, stream>>>(
            table_, keys, d_count, table_->buckets, table_->buckets_size,
            table_->bucket_max_size, table_->buckets_num, N);

    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return count;
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
    const size_t block_size = 256;
    const size_t N = table_->buckets_num;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    size_t count = 0;
    size_t* d_count;
    Pred h_pred;

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    remove_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, 0, stream>>>(
            table_, h_pred, pattern, threshold, d_count, table_->buckets,
            table_->buckets_size, table_->bucket_max_size, table_->buckets_num,
            N);

    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return count;
  }

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  void clear(cudaStream_t stream = 0) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);

    if (!reach_max_capacity_) {
      lock.lock();
    }

    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);

    clear_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, options_.block_size, 0, stream>>>(table_, N);

    CudaCheckError();
  }

 public:
  /**
   * @brief Exports a certain number of the key-value-meta tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to remove.
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
  void export_batch(size_type n, size_type offset, size_type* d_counter,
                    key_type* keys,              // (n)
                    value_type* values,          // (n, DIM)
                    meta_type* metas = nullptr,  // (n)
                    cudaStream_t stream = 0) const {
    if (offset >= table_->capacity) {
      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
      return;
    }
    n = std::min(table_->capacity - offset, n);
    size_type meta_size = (metas == nullptr ? 0 : sizeof(meta_type));

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    const size_t block_size =
        std::min(shared_mem_size_ / 2 /
                     (sizeof(key_type) + sizeof(vector_type) + meta_size),
                 1024UL);

    MERLIN_CHECK(
        (block_size > 0),
        "[merlin-kv] block_size <= 0, the K-V-M size may be too large!");
    const size_t shared_size =
        (sizeof(key_type) + sizeof(vector_type) + meta_size) * block_size;
    const int grid_size = (n - 1) / (block_size) + 1;

    dump_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, reinterpret_cast<vector_type*>(values), metas, offset,
            n, d_counter);
    CudaCheckError();
  }

  size_type export_batch(size_type n, size_type offset,
                         key_type* keys,              // (n)
                         value_type* values,          // (n, DIM)
                         meta_type* metas = nullptr,  // (n)
                         cudaStream_t stream = 0) const {
    size_type* d_counter = nullptr;
    size_type h_counter = 0;
    CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
    export_batch(n, offset, d_counter, keys, values, metas, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_counter;
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
    size_t h_size = 0;
    size_type N = table_->buckets_num;
    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

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

    {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::unique_lock<std::shared_timed_mutex> lock(mutex_);

      while (capacity() < new_capacity &&
             capacity() * 2 <= options_.max_capacity) {
        double_capacity(&table_);

        const size_t block_size = 128;
        const size_t N = TILE_SIZE * table_->buckets_num / 2;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
        rehash_kernel_for_fast_mode<key_type, vector_type, meta_type, DIM,
                                    TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                table_, table_->buckets, table_->buckets_size,
                table_->bucket_max_size, table_->buckets_num, N);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
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
   * @param file A BaseKVFile object defined the file format within filesystem.
   * @param buffer_size The size of buffer used for saving in bytes.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys saved to file.
   */
  size_type save(BaseKVFile<K, V, M, DIM>* file,
                 size_type buffer_size = 1048576,
                 cudaStream_t stream = 0) const {
    K* d_keys = nullptr;
    V* d_vectors = nullptr;
    M* d_metas = nullptr;
    K* h_keys = nullptr;
    V* h_vectors = nullptr;
    M* h_metas = nullptr;
    size_type* d_next_nkeys = nullptr;
    size_type nkeys = 0;
    size_type total_size = capacity();
    size_type pair_size = sizeof(K) + sizeof(M) + sizeof(V) * DIM;
    size_type batch_pairs_num =
        std::min((buffer_size + pair_size) / pair_size, total_size);

    CUDA_CHECK(cudaMallocAsync(&d_next_nkeys, sizeof(size_type), stream));
    CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(K) * batch_pairs_num, stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_vectors, sizeof(V) * batch_pairs_num * DIM, stream));
    CUDA_CHECK(cudaMallocAsync(&d_metas, sizeof(M) * batch_pairs_num, stream));
    CUDA_CHECK(cudaMemsetAsync(d_next_nkeys, 0, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_keys, 0, sizeof(K) * batch_pairs_num, stream));
    CUDA_CHECK(cudaMemsetAsync(d_vectors, 0, sizeof(V) * batch_pairs_num * DIM,
                               stream));
    CUDA_CHECK(
        cudaMemsetAsync(d_metas, 0, sizeof(M) * batch_pairs_num, stream));

    CUDA_CHECK(cudaMallocHost(&h_keys, sizeof(K) * batch_pairs_num));
    CUDA_CHECK(cudaMallocHost(&h_vectors, sizeof(V) * batch_pairs_num * DIM));
    CUDA_CHECK(cudaMallocHost(&h_metas, sizeof(M) * batch_pairs_num));

    export_batch(batch_pairs_num, 0, d_next_nkeys, d_keys, d_vectors, d_metas,
                 stream);
    CUDA_CHECK(cudaMemcpyAsync(&nkeys, d_next_nkeys, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(K) * nkeys,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_vectors, d_vectors, sizeof(V) * nkeys * DIM,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_metas, d_metas, sizeof(M) * nkeys,
                               cudaMemcpyDeviceToHost, stream));

    size_type counter = nkeys;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_type offset = batch_pairs_num; offset < total_size;
         offset += batch_pairs_num) {
      CUDA_CHECK(cudaMemsetAsync(d_next_nkeys, 0, sizeof(size_type), stream));
      export_batch(batch_pairs_num, offset, d_next_nkeys, d_keys, d_vectors,
                   d_metas, stream);
      file->write(nkeys, h_keys, h_vectors, h_metas);
      CUDA_CHECK(cudaMemcpyAsync(&nkeys, d_next_nkeys, sizeof(size_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(K) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_vectors, d_vectors, sizeof(V) * nkeys * DIM,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_metas, d_metas, sizeof(M) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
      counter += nkeys;
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (nkeys > 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(K) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_vectors, d_vectors, sizeof(V) * nkeys * DIM,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_metas, d_metas, sizeof(M) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      file->write(nkeys, h_keys, h_vectors, h_metas);
    }

    CUDA_FREE_POINTERS(stream, d_keys, d_vectors, d_metas, d_next_nkeys, h_keys,
                       h_vectors, h_metas);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counter;
  }

  /**
   * @brief Load file and restore table.
   *
   * @param file An BaseKVFile object defined the file format within filesystem.
   * @param buffer_size The size of buffer used for loading in bytes.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(BaseKVFile<K, V, M, DIM>* file,
                 size_type buffer_size = 1048576, cudaStream_t stream = 0) {
    K* d_keys = nullptr;
    V* d_vectors = nullptr;
    M* d_metas = nullptr;
    K* h_keys = nullptr;
    V* h_vectors = nullptr;
    M* h_metas = nullptr;
    size_type pair_size = sizeof(K) + sizeof(M) + sizeof(V) * DIM;
    size_type batch_pairs_num = (buffer_size + pair_size) / pair_size;

    CUDA_CHECK(cudaMallocHost(&h_keys, sizeof(K) * batch_pairs_num));
    CUDA_CHECK(cudaMallocHost(&h_vectors, sizeof(V) * batch_pairs_num * DIM));
    CUDA_CHECK(cudaMallocHost(&h_metas, sizeof(M) * batch_pairs_num));
    size_type nkeys = file->read(batch_pairs_num, h_keys, h_vectors, h_metas);
    size_type counts = nkeys;
    if (nkeys == 0) {
      CUDA_FREE_POINTERS(stream, h_keys, h_vectors, h_metas);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return 0;
    }
    CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(K) * batch_pairs_num, stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_vectors, sizeof(V) * batch_pairs_num * DIM, stream));
    CUDA_CHECK(cudaMallocAsync(&d_metas, sizeof(M) * batch_pairs_num, stream));

    do {
      CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, sizeof(K) * nkeys,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, sizeof(V) * nkeys * DIM,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, sizeof(M) * nkeys,
                                 cudaMemcpyHostToDevice, stream));
      insert_or_assign(nkeys, d_keys, d_vectors, d_metas, stream);
      nkeys = file->read(batch_pairs_num, h_keys, h_vectors, h_metas);
      counts += nkeys;
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } while (nkeys > 0);

    CUDA_FREE_POINTERS(stream, d_keys, d_vectors, d_metas, h_keys, h_vectors,
                       h_metas);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counts;
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
    size_t h_size = 0;

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }
    size_type N = std::min(table_->buckets_num, 1024UL);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

#if THRUST_VERSION >= 101600
    auto policy = thrust::cuda::par_nosync.on(stream);
#else
    auto policy = thrust::cuda::par.on(stream);
#endif
    h_size = thrust::reduce(policy, size_ptr, size_ptr + N, (int)0,
                            thrust::plus<int>());

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
  mutable std::shared_timed_mutex mutex_;
};

}  // namespace merlin
}  // namespace nv
