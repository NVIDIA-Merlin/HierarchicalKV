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
#include <condition_variable>
#include <list>
#include <mutex>
#include <type_traits>
#include "merlin/core_kernels.cuh"
#include "merlin/initializers.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

// TODO(kimi): why not use std::array<T, D>?
/**
 * @brief value type of Merlin-KV.
 *
 * @tparam T type of elements of the vector. It must be POD.
 * @tparam D dimension of the vector.
 */
template <typename T, size_t D>
struct Vector {
  static_assert(std::is_pod<T>::value, "T must be POD.");
  using value_type = T;
  static constexpr size_t DIM = D;
  // TODO(kimi): rename value to values?
  value_type value[DIM];
};

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
  size_t max_batch_size =
      64 * 1024 * 1024;   ///< Maximum batch size, for batched operations (also
                          ///< the size of a workspace).
  size_t min_num_ws = 3;  ///< Number of workspaces to keep in reserve.
  size_t max_num_ws = 5;  ///< Maximum number of workspaces.
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
  using this_type = HashTable<K, V, M, D>;
  using size_type = size_t;
  static constexpr size_type DIM = D;
  using key_type = K;
  using value_type = V;
  using vector_type = Vector<value_type, DIM>;
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
   * @brief
   *
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

    // Preallocate workspaces.
    assert(options_.min_num_ws >= 1 &&
           options_.min_num_ws <= options_.max_num_ws);

    avail_ws_.reserve(options_.min_num_ws);
    while (avail_ws_.size() < options_.min_num_ws) {
      ws_.emplace_back(options_.max_batch_size);
      avail_ws_.emplace_back(&ws_.back());
    }

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
   * @param num_items Number of Key-Value-Meta tuples to be inserted or
   * assigned.
   * @param keys The keys to be inserted on GPU accessible memory with shape
   * (num_items).
   * @param values The values to be inserted on GPU accessible memory with
   * shape (num_items, DIM).
   * @param metas The metas to be inserted on GPU accessible memory with shape
   * (num_items).
   *
   * @notice: The metas must be uint64_t value which could stand for the
   * timestamp of the key inserted or the number of the key occurrences. if
   * @p metas is nullptr, the LRU strategy will be applied.
   *
   * @param stream The CUDA stream used to execute the operation.
   *
   */
  void insert_or_assign(const size_type num_items,
                        const key_type* const keys,      // (num_items)
                        const value_type* const values,  // (num_items, DIM)
                        const meta_type* const metas = nullptr,  // (num_items)
                        cudaStream_t const stream = 0) {
    if (num_items == 0) {
      return;
    }

    if (!reach_max_capacity_ && load_factor() > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    evict_strategy_check(metas);

    if (is_fast_mode()) {
      const size_t block_size = 128;
      const size_t N = num_items * TILE_SIZE;
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
      vector_type** d_dst = ws[0]->vec;
      int* d_src_offset = ws[1]->i32;

      for (size_t i = 0; i < num_items; i += options_.max_batch_size) {
        const size_t n = std::min(num_items - i, options_.max_batch_size);

        CUDA_CHECK(cudaMemsetAsync(d_dst, 0, n * sizeof(vector_type*), stream));
        CUDA_CHECK(cudaMemsetAsync(d_src_offset, 0, n * sizeof(int), stream));

        {
          const size_t block_size = 128;
          const size_t N = n * TILE_SIZE;
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

          const size_t N = n;
          thrust::device_ptr<uint64_t> d_dst_ptr(
              reinterpret_cast<uint64_t*>(d_dst));
          thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

#if THRUST_VERSION >= 101600
          auto policy = thrust::cuda::par_nosync.on(stream);
#else
          auto policy = thrust::cuda::par.on(stream);
#endif
          thrust::sort_by_key(policy, d_dst_ptr, d_dst_ptr + N,
                              d_src_offset_ptr, thrust::less<uint64_t>());
        }

        {
          const size_t N = n * DIM;
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
   * @param num_items Number of key_type-Value pairs to be processed.
   * @param keys The keys to be inserted on GPU accessible memory with shape
   * (num_items).
   * @param value_or_deltas The values or deltas to be inserted on GPU
   * accessible memory with shape (num_items, DIM).
   * @param accum_or_assigns Indicate the operation type with shape (num_items),
   * true means accum, false means assign.
   * @param metas The metas to be inserted on GPU accessible memory with shape
   * (num_items).
   * @param stream The CUDA stream used to execute the operation
   *
   */
  void accum_or_assign(
      const size_type num_items,
      const key_type* const keys,               // (num_items)
      const value_type* const value_or_deltas,  // (num_items, DIM)
      const bool* const accum_or_assigns,       // (num_items)
      const meta_type* const metas = nullptr,   // (num_items)
      cudaStream_t const stream = 0) {
    if (num_items == 0) {
      return;
    }

    evict_strategy_check(metas);

    Workspace<3> ws(this, stream);
    vector_type** dst = ws[0]->vec;
    int* src_offset = ws[0]->i32;
    bool* founds = ws[0]->b8;

    for (size_t i = 0; i < num_items; i += options_.max_batch_size) {
      const size_t n = std::min(num_items - i, options_.max_batch_size);

      CUDA_CHECK(cudaMemsetAsync(dst, 0, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(src_offset, 0, n * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

      {
        const size_t block_size = 128;
        const size_t N = n * TILE_SIZE;
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
                reinterpret_cast<const vector_type*>(&value_or_deltas[i]), dst,
                &accum_or_assigns[i], founds, src_offset, N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the table for the specified keys.
   *
   * @note When a key is missing, the value in @p values will not changed.
   *
   * @param num_keys Number of Key-Value-Meta tuples to be searched.
   * @param keys The keys to be searched on GPU accessible memory with shape
   * (num_keys).
   * @param values The values to be searched on GPU accessible memory with
   * shape (num_keys, DIM).
   * @param founds The status indicates if the keys are found on GPU accessible
   * memory with shape (num_keys).
   * @param metas The metas to be searched on GPU accessible memory with shape
   * (num_keys).
   * @param stream The CUDA stream used to execute the operation.
   *
   */
  void find(const size_type num_keys,
            const key_type* const keys,        // (num_keys)
            value_type* const values,          // (num_keys, DIM)
            bool* const founds,                // (num_keys)
            meta_type* const metas = nullptr,  // (num_keys)
            cudaStream_t const stream = 0) const {
    if (num_keys == 0) {
      return;
    }

    // Clear found flags.
    CUDA_CHECK(cudaMemsetAsync(founds, 0, num_keys * sizeof(bool), stream));

    if (is_fast_mode()) {
      const size_t block_size = 128;
      const size_t N = num_keys * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      lookup_kernel_with_io<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, reinterpret_cast<vector_type*>(values), metas,
              founds, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    } else {
      Workspace<2> ws(this, stream);
      vector_type** src = ws[0]->vec;
      int* dst_offset = ws[1]->i32;

      for (size_t i = 0; i < num_keys; i += options_.max_batch_size) {
        const size_t n = std::min(num_keys - i, options_.max_batch_size);

        CUDA_CHECK(cudaMemsetAsync(src, 0, n * sizeof(vector_type*), stream));
        CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, n * sizeof(int), stream));

        {
          const size_t block_size = 128;
          const size_t N = n * TILE_SIZE;
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

          const size_t N = n;
          thrust::device_ptr<uint64_t> src_ptr(
              reinterpret_cast<uint64_t*>(src));
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
                  src, reinterpret_cast<vector_type*>(&values[i]), &founds[i],
                  dst_offset, N);
        }
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Removes specified elements from the table.
   *
   * @param num_keys Number of Key to be removed.
   * @param keys The keys to be removed on GPU accessible memory (num_keys).
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of elements removed.
   */
  size_t erase(const size_type num_keys,
               const key_type* const keys,  // (num_keys)
               cudaStream_t const stream = 0) {
    Workspace<1> ws(this, stream);
    size_t* d_count = ws[0]->size;

    for (size_t i = 0; i < num_keys; i += options_.max_batch_size) {
      const size_t n = std::min(num_keys - i, options_.max_batch_size);

      CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_t), stream));

      const size_t block_size = 128;
      const size_t N = num_keys * TILE_SIZE;
      const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      remove_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, &keys[i], d_count, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }

    size_t count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));

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
   * @return Number of elements removed.
   *
   */
  size_t erase_if(Pred& pred, const key_type& pattern,
                  const meta_type& threshold, cudaStream_t const stream = 0) {
    Workspace<1> ws(this, stream);
    size_t* d_count = ws[0]->size;

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

    size_t h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));

    CudaCheckError();
    return h_count;
  }

  /**
   * @brief Remove all of the elements in the table with no release object.
   */
  void clear(cudaStream_t const stream = 0) {
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
   * @return the number of items dumped.
   *
   * @throw CudaException If the K-V size is too large for GPU shared memory.
   * Reducing the @ p max_num is needed at this time.
   */
  size_type export_batch(const size_type num_items, const size_type offset,
                         key_type* const keys,              // (num_items)
                         value_type* const values,          // (num_items, DIM)
                         meta_type* const metas = nullptr,  // (num_items)
                         cudaStream_t const stream = 0) const {
    const size_type meta_size = (metas == nullptr ? 0 : sizeof(meta_type));
    const size_t block_size =
        std::min(shared_mem_size_ / 2 /
                     (sizeof(key_type) + sizeof(vector_type) + meta_size),
                 1024UL);

    Workspace<2> ws(this, stream);
    size_type* d_counter = ws[0]->size;

    for (size_t i = 0; i < num_items; i += options_.max_batch_size) {
      const size_t n = std::min(num_items - i, options_.max_batch_size);

      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

      MERLIN_CHECK(
          (block_size > 0),
          "[merlin-kv] block_size <= 0, the K-V-M size may be too large!");
      const size_t shared_size =
          (sizeof(key_type) + sizeof(vector_type) + meta_size) * block_size;
      const int grid_size = (n - 1) / (block_size) + 1;

      dump_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, shared_size, stream>>>(
              table_, &keys[i], reinterpret_cast<vector_type*>(&values[i]),
              metas ? &metas[i] : nullptr, offset, n, d_counter);
    }

    size_type h_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));

    CudaCheckError();
    return h_counter;
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
    const size_type N = table_->buckets_num;
    thrust::device_ptr<int> size_ptr(table_->buckets_size);

#if THRUST_VERSION >= 101600
    auto policy = thrust::cuda::par_nosync.on(stream);
#else
    auto policy = thrust::cuda::par.on(stream);
#endif
    const size_t h_size =
        thrust::reduce(policy, size_ptr, size_ptr + N, 0, thrust::plus<int>());

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

  // Workspace management.
  struct WorkspaceBuffer final {
    union {
      void* ptr;
      vector_type** vec;
      size_t* size;
      bool* b8;
      int* i32;
      // uint64_t u64;
    };

    WorkspaceBuffer(const size_t size) {
      const size_t item_size = std::max(sizeof(void*), sizeof(uint64_t));
      CUDA_CHECK(cudaMalloc(&ptr, item_size * size));
    }

    WorkspaceBuffer(const size_t size, cudaStream_t const stream) {
      const size_t item_size = std::max(sizeof(void*), sizeof(uint64_t));
      CUDA_CHECK(cudaMallocAsync(&vec, item_size * size, stream));
    }

    ~WorkspaceBuffer() { CUDA_CHECK(cudaFree(ptr)); }
  };
  static_assert(sizeof(WorkspaceBuffer) == sizeof(void*));

  template <size_t SIZE>
  class Workspace final {
   public:
    Workspace(const this_type* const parent, cudaStream_t const stream)
        : parent_{parent}, stream_{stream} {
      parent_->claim_ws_(*this, stream);
    }

    ~Workspace() {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      parent_->release_ws_(*this, stream_);
    }

    constexpr WorkspaceBuffer*& operator[](const size_t i) {
      assert(i < SIZE);
      return buffers_[i];
    }
    constexpr const WorkspaceBuffer*& operator[](const size_t i) const {
      assert(i < SIZE);
      return buffers_[i];
    }

   private:
    const this_type* const parent_;
    cudaStream_t const stream_;
    WorkspaceBuffer* buffers_[SIZE];
  };

  mutable std::mutex ws_mtx_;
  mutable std::list<WorkspaceBuffer> ws_;
  mutable std::vector<WorkspaceBuffer*> avail_ws_;
  mutable std::condition_variable ws_returned_;

  template <size_t SIZE>
  void claim_ws_(Workspace<SIZE>& ws, cudaStream_t const stream) const {
    std::unique_lock<std::mutex> lock(ws_mtx_);

    // If have a prellocated workspace available.
    if (avail_ws_.size() >= SIZE) {
      for (size_t i = 0; i < SIZE; i++) {
        ws[i] = avail_ws_.back();
        avail_ws_.pop_back();
      }
    }
    // If workspace creation quota not yet reached.
    else if (ws_.size() + SIZE <= options_.max_num_ws) {
      for (size_t i = 0; i < SIZE; i++) {
        ws_.emplace_back(options_.max_batch_size, stream);
        ws[i] = &ws_.back();
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

        for (size_t i = 0; i < SIZE; i++) {
          ws[i] = avail_ws_.back();
          avail_ws_.pop_back();
        }
        break;
      }
    }
  }

  template <size_t SIZE>
  void release_ws_(Workspace<SIZE>& ws, cudaStream_t const stream) const {
    std::lock_guard<std::mutex> lock(ws_mtx_);
    size_t i = 0;

    // Fill up available buffers until reach reserve capacity.
    bool has_returned_ws = false;
    for (; i < SIZE && avail_ws_.size() < options_.min_num_ws; i++) {
      avail_ws_.emplace_back(ws[i]);
      has_returned_ws = true;
    }

    // Discard remaining buffers.
    for (; i < SIZE; i++) {
      for (auto it = ws_.begin(); it != ws_.end(); it++) {
        if (&(*it) == ws[i]) {
          // TODO: Use stream to avoid stalling.
          ws_.erase(it);
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
