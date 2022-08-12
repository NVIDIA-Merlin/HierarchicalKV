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
#include <type_traits>
#include "merlin/core_kernels.cuh"
#include "merlin/initializers.cuh"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

/**
 * @brief The options struct of Merlin-KV.
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity.
  size_t max_capacity = 0;         ///< The maximum capacity.
  size_t max_hbm_for_vectors = 0;  ///< Max HBM allocated for vectors, by bytes.
  size_t max_bucket_size = 128;    ///< The length of each buckets.
  float max_load_factor = 0.75f;   ///< The max load factor before rehashing.
  int block_size = 1024;           ///< default block size for CUDA kernels.
  int device_id = 0;               ///< the id of device.
  bool primary = true;             ///< no used, reserved for future.
};

/**
 * @brief A function template is used as `erase_if` first input, which help
 * end-users implements customized and flexible erase(or evict) strategies.
 *
 * The erase_if will traverse all of the items by this function, the items which
 * return `true` will be removed.
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
using EraseIfPredict = bool (*)(const K& key,       ///< traversed key in table
                                const M& meta,      ///< traversed meta in table
                                const K& pattern,   ///< input key from caller
                                const M& threshold  ///< input meta from caller
);

/**
 * Merlin HashTable is a concurrent and hierarchical HashTable powered by GPUs
 * which can use both HBM, Host MEM and SSD(WIP) as storage for Key-Values.
 *
 * @note Supports concurrent insert_or_assign, but not concurrent
 * insert_or_assign and find now. The insert_or_assign means insert or update if
 * already exists.
 *
 *
 * @note The eviction will happen automatically when table is almost full. We
 * introduce the `meta` concept to help implement it. The keys with minimum meta
 * will be evicted first. We recommend using the timestamp or times
 * of the key occurrence as the meta value for each keys. The user can also
 * customize the meaning of the meta value that is equivalent to customize an
 * eviction strategy.
 *
 * @tparam K type of the key
 * @tparam V type of the Vector's item type, which should be basic types of
 * C++/CUDA.
 * @tparam M type of the meta and must be uint64_t in this release.
 * @tparam D dimension of the vectors
 *
 *
 */
template <class K, class V, class M, size_t D>
class HashTable {
 public:
  /**
   * @brief value type of Merlin-KV.
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

  /**
   * @brief Enumeration of the eviction strategies.
   */
  enum class EvictStrategy {
    kUndefined = 0,  ///< undefined.
    kLru = 1,        ///< kLru mode.
    kCustomized = 2  ///< Customized mode.
  };

 public:
  /**
   * @brief Default Construct a table object.
   */
  HashTable(){};

  /**
   * @brief Frees the resources of the table and destroys the table object.
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
    options_ = options;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaSetDevice(options_.device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    create_table<key_type, vector_type, meta_type, DIM>(
        &table_, options_.init_capacity, options_.max_capacity,
        options_.max_hbm_for_vectors, options_.max_bucket_size,
        options_.primary);
    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);
    reach_max_capacity_ = false;
    initialized_ = true;
    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-meta tuples into the table,
   * if key already exists, the values and metas will be assigned.
   *
   * @note If the target bucket is full, the keys with minimum meta will be
   * overwritten. If the meta of the new key is even less than minimum meta of
   * the target bucket, it will not be inserted.
   *
   * @param n Number of Key-Value-Meta tuples to be inserted or assigned.
   * @param keys The keys to be inserted on GPU accessible memory with shape
   * (n).
   * @param values The values to be inserted on GPU accessible memory with
   * shape (n, DIM).
   * @param metas The metas to be inserted on GPU accessible memory with shape
   * (n).
   *
   * @notice: The metas must be uint64_t value which could stand for the
   * timestamp of the key inserted or the number of the key occurrences. if
   * @p metas is nullptr, the LRU strategy will be applied.
   *
   * @param stream The CUDA stream used to execute the operation.
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

    if (!reach_max_capacity_ && load_factor() > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    evict_strategy_check(metas);

    if (is_fast_mode()) {
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
      vector_type** d_dst = nullptr;
      int* d_src_offset = nullptr;
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
                      "must be 64 bit!");

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
   * Searches for each key in @p keys in the table.
   * If the key is found and corresponding value in @p accum_or_assigns is true,
   * the @p vectors_or_deltas will be treated as delta against to the old
   * value, and the delta will be add to the old value of the key.
   *
   * If the key is not found and corresponding value in @p accum_or_assigns is
   * false, the @p vectors_or_deltas will be treated as a new value and the
   * key-value pair will be updated into the table directly.
   *
   * @note Specially when the key is found and value of @p accum_or_assigns is
   * false, or the key is not found and value of @p accum_or_assigns is true,
   * nothing will be changed and this operation will be ignored, for we assume
   * these situations occur while the key was modified or removed by other
   * processes just now.
   *
   * @param n Number of key_type-Value pairs to be processed.
   * @param keys The keys to be inserted on GPU accessible memory with shape
   * (n).
   * @param value_or_deltas The values or deltas to be inserted on GPU
   * accessible memory with shape (n, DIM).
   * @param accum_or_assigns Indicate the operation type with shape (n), true
   * means accum, false means assign.
   * @param metas The metas to be inserted on GPU accessible memory with shape
   * (n).
   * @param stream The CUDA stream used to execute the operation
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

    evict_strategy_check(metas);

    vector_type** dst;
    int* src_offset;
    bool* founds;
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
   * @brief Searches the table for the specified keys.
   *
   * @note When a key is missing, the value in @p values will not changed.
   *
   * @param n Number of Key-Value-Meta tuples to be searched.
   * @param keys The keys to be searched on GPU accessible memory with shape
   * (n).
   * @param values The values to be searched on GPU accessible memory with
   * shape (n, DIM).
   * @param founds The status indicates if the keys are found on GPU accessible
   * memory with shape (n).
   * @param metas The metas to be searched on GPU accessible memory with shape
   * (n).
   * @param stream The CUDA stream used to execute the operation.
   *
   */
  void find(size_type n, const key_type* keys, value_type* values, bool* founds,
            meta_type* metas = nullptr, cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
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
   * @brief Removes specified elements from the table.
   *
   * @param n Number of Key to be removed.
   * @param keys The keys to be removed on GPU accessible memory.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of elements removed
   */
  size_t erase(size_type n, const key_type* keys, cudaStream_t stream = 0) {
    const size_t block_size = 128;
    const size_t N = n * TILE_SIZE;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    size_t count = 0;
    size_t* d_count;
    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

    remove_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
        <<<grid_size, block_size, 0, stream>>>(
            table_, keys, d_count, table_->buckets, table_->buckets_size,
            table_->bucket_max_size, table_->buckets_num, N);

    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));
    CudaCheckError();
    return count;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * table.
   *
   * @param pred predicate that returns true if the element should be erased.
   * @param pattern the 3rd user-defined argument to @p pred with key_type type.
   * @param threshold the 4th user-defined argument to @p pred with meta_type
   * type.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @notice: pred should be a function defined like the Example:
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
   * @return Number of elements removed
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
    CudaCheckError();
    return count;
  }

  /**
   * @brief Remove all of the elements in the table with no release object.
   */
  void clear(cudaStream_t stream = 0) {
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const int grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);
    clear_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, options_.block_size, 0, stream>>>(table_, N);

    CudaCheckError();
  }

 public:
  /**
   * @brief Export a certain number of the key-value-meta tuples from the table.
   *
   * @param n Maximum number of exported pairs.
   * @param offset Number of Key to be removed.
   * @param keys The keys to be dumped on GPU accessible memory with shape (n).
   * @param values The values to be dumped on GPU accessible memory with shape
   * (n, DIM).
   * @param metas The metas to be searched on GPU accessible memory with shape
   * (n).
   * @param stream The CUDA stream used to execute the operation.
   *
   * @throw CudaException If the K-V size is too large for GPU shared memory.
   * Reducing the @ p max_num is needed at this time.
   */
  void export_batch(size_type n, size_type offset, size_type* d_counter,
                    key_type* keys,              // (n)
                    value_type* values,          // (n, DIM)
                    meta_type* metas = nullptr,  // (n)
                    cudaStream_t stream = 0) const {
    size_type meta_size = (metas == nullptr ? 0 : sizeof(meta_type));

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

 public:
  /**
   * @brief Checks if the table has no elements.
   *
   * @param stream The CUDA stream used to execute the operation.
   * @return true if the table is empty, false otherwise
   */
  bool empty(cudaStream_t stream = 0) const { return size(stream) == 0; }

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
   * @brief Get the table capacity.
   *
   * @note The capacity is requested by the caller and the value may be
   * less than the actual capacity of the table because the table keeps
   * the capacity to be the power of 2 for performance consideration in this
   * release.
   *
   * @return The table capacity
   */
  size_type capacity() const { return table_->capacity; }

  /**
   * @brief Sets the number of buckets to the number needed to accomodate at
   * least count elements without exceeding maximum load factor and rehashes the
   * table, i.e. puts the elements into appropriate buckets considering that
   * total number of buckets has changed.
   *
   * @note If the count or double of the count is greater or equal than
   * options_.max_capacity, the reserve will not happen.
   *
   * @param count new capacity of the table.
   * @param stream The CUDA stream used to execute the operation.
   */
  void reserve(size_type new_capacity, cudaStream_t stream = 0) {
    if (reach_max_capacity_ || new_capacity > options_.max_capacity) {
      return;
    }

    while (capacity() < new_capacity &&
           capacity() * 2 <= options_.max_capacity) {
      std::cout << "[merlin-kv] load_factor=" << load_factor()
                << ", reserve is being executed, "
                << "the capacity will increase from " << capacity() << " to "
                << capacity() * 2 << "." << std::endl;
      double_capacity(&table_);

      const size_t N = capacity() / 2;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, options_.block_size);
      rehash_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, options_.block_size, 0, stream>>>(table_, N);
    }
    reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
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
   * @brief Save table to an abstract file.
   *
   * @param file An KVFile object defined the file format within filesystem.
   * @param buffer_size The number of keys to store for running the save.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys saved to file.
   */
  size_type save_to_file(KVFile<K, V, M, DIM>* file,
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
    buffer_size = std::min(buffer_size, total_size);

    CUDA_CHECK(cudaMallocAsync(&d_next_nkeys, sizeof(size_type), stream));
    CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(K) * buffer_size, stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_vectors, sizeof(V) * buffer_size * DIM, stream));
    CUDA_CHECK(cudaMallocAsync(&d_metas, sizeof(M) * buffer_size, stream));
    CUDA_CHECK(cudaMemsetAsync(d_next_nkeys, 0, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_keys, 0, sizeof(K) * buffer_size, stream));
    CUDA_CHECK(
        cudaMemsetAsync(d_vectors, 0, sizeof(V) * buffer_size * DIM, stream));
    CUDA_CHECK(cudaMemsetAsync(d_metas, 0, sizeof(M) * buffer_size, stream));

    CUDA_CHECK(cudaMallocHost(&h_keys, sizeof(K) * buffer_size));
    CUDA_CHECK(cudaMallocHost(&h_vectors, sizeof(V) * buffer_size * DIM));
    CUDA_CHECK(cudaMallocHost(&h_metas, sizeof(M) * buffer_size));

    export_batch(buffer_size, 0, d_next_nkeys, d_keys, d_vectors, d_metas,
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

    for (size_type offset = buffer_size; offset < total_size;
         offset += buffer_size) {
      CUDA_CHECK(cudaMemsetAsync(d_next_nkeys, 0, sizeof(size_type), stream));
      export_batch(buffer_size, offset, d_next_nkeys, d_keys, d_vectors,
                   d_metas, stream);
      file->Write(h_keys, h_vectors, h_metas, nkeys);
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
      file->Write(h_keys, h_vectors, h_metas, nkeys);
    }

    cudaFreeAsync(d_keys, stream);
    cudaFreeAsync(d_vectors, stream);
    cudaFreeAsync(d_metas, stream);
    cudaFreeAsync(d_next_nkeys, stream);
    cudaFreeHost(h_keys);
    cudaFreeHost(h_vectors);
    cudaFreeHost(h_metas);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counter;
  }

  /**
   * @brief Load file and restore table.
   *
   * @param file An KVFile object defined the file format within filesystem.
   * @param buffer_size The number of keys to store for running the save.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load_from_file(KVFile<K, V, M, DIM>* file,
                           size_type buffer_size = 1048576,
                           cudaStream_t stream = 0) {
    K* d_keys = nullptr;
    V* d_vectors = nullptr;
    M* d_metas = nullptr;
    K* h_keys = nullptr;
    V* h_vectors = nullptr;
    M* h_metas = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_keys, sizeof(K) * buffer_size));
    CUDA_CHECK(cudaMallocHost(&h_vectors, sizeof(V) * buffer_size * DIM));
    CUDA_CHECK(cudaMallocHost(&h_metas, sizeof(M) * buffer_size));
    size_type nkeys = file->Read(h_keys, h_vectors, h_metas, buffer_size);
    size_type counts = nkeys;
    if (nkeys == 0) {
      // Add warn log.
      CUDA_CHECK(cudaFreeHost(h_keys));
      CUDA_CHECK(cudaFreeHost(h_vectors));
      CUDA_CHECK(cudaFreeHost(h_metas));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return counts;
    }
    CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(K) * buffer_size, stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_vectors, sizeof(V) * buffer_size * DIM, stream));
    CUDA_CHECK(cudaMallocAsync(&d_metas, sizeof(M) * buffer_size, stream));

    do {
      CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, sizeof(K) * nkeys,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, sizeof(V) * nkeys * DIM,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, sizeof(M) * nkeys,
                                 cudaMemcpyHostToDevice, stream));
      insert_or_assign(nkeys, d_keys, d_vectors, d_metas, stream);
      nkeys = file->Read(h_keys, h_vectors, h_metas, buffer_size);
      counts += nkeys;
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } while (nkeys > 0);
    CUDA_CHECK(cudaFreeAsync(d_keys, stream));
    CUDA_CHECK(cudaFreeAsync(d_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(d_metas, stream));
    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_vectors));
    CUDA_CHECK(cudaFreeHost(h_metas));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counts;
  }

 private:
  bool is_fast_mode() const noexcept { return table_->is_pure_hbm; }

  void evict_strategy_check(const meta_type* metas) {
    if (evict_strategy_ == EvictStrategy::kUndefined) {
      evict_strategy_ =
          metas == nullptr ? EvictStrategy::kLru : EvictStrategy::kCustomized;
    }

    if (evict_strategy_ == EvictStrategy::kLru) {
      MERLIN_CHECK((metas == nullptr),
                   "the metas should not be specified when already running on "
                   "LRU mode.");
    }

    if (evict_strategy_ == EvictStrategy::kCustomized) {
      MERLIN_CHECK((metas != nullptr),
                   "the metas should be specified when already running on "
                   "customized mode.")
    }
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  size_t shared_mem_size_ = 0;
  bool reach_max_capacity_ = false;
  bool initialized_ = false;
  EvictStrategy evict_strategy_ = EvictStrategy::kUndefined;
};

}  // namespace merlin
}  // namespace nv
