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
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include "merlin/core_kernels.cuh"
#include "merlin/flexible_buffer.cuh"
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
 * @brief The options struct of HierarchicalKV.
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity of the hash table.
  size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
  size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
  size_t max_bucket_size = 128;    ///< The length of each bucket.
  float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
  int block_size = 128;            ///< The default block size for CUDA kernels.
  int io_block_size = 1024;        ///< The block size for IO CUDA kernels.
  int device_id = 0;               ///< The ID of device.
  bool io_by_cpu = false;  ///< The flag indicating if the CPU handles IO.
  EvictStrategy evict_strategy = EvictStrategy::kLru;  ///< The evict strategy.
};

/**
 * @brief A customizable template function indicates which keys should be
 * erased from the hash table by returning `true`.
 *
 * @note The `erase_if` or `export_batch_if` API traverses all of the items by
 * this function and the items that return `true` are removed or exported.
 *
 *  Example for erase_if:
 *
 *    ```
 *    template <class K, class M>
 *    __forceinline__ __device__ bool erase_if_pred(const K& key,
 *                                                  M& meta,
 *                                                  const K& pattern,
 *                                                  const M& threshold) {
 *      return ((key & 0xFFFF000000000000 == pattern) &&
 *              (meta < threshold));
 *    }
 *    ```
 *
 *  Example for export_batch_if:
 *    ```
 *    template <class K, class M>
 *    __forceinline__ __device__ bool export_if_pred(const K& key,
 *                                                   M& meta,
 *                                                   const K& pattern,
 *                                                   const M& threshold) {
 *      return meta >= threshold;
 *    }
 *    ```
 */
template <class K, class M>
using EraseIfPredict = bool (*)(
    const K& key,       ///< The traversed key in a hash table.
    M& meta,            ///< The traversed meta in a hash table.
    const K& pattern,   ///< The key pattern to compare with the `key` argument.
    const M& threshold  ///< The threshold to compare with the `meta` argument.
);

/**
 * A HierarchicalKV hash table is a concurrent and hierarchical hash table that
 * is powered by GPUs and can use HBM and host memory as storage for key-value
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
   * @brief The value type of a HierarchicalKV hash table.
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
  static constexpr unsigned int TILE_SIZE = 4;

#if THRUST_VERSION >= 101600
  static constexpr auto thrust_par = thrust::cuda::par_nosync;
#else
  static constexpr auto thrust_par = thrust::cuda::par;
#endif

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
      initialized_ = false;
      destroy_table<key_type, vector_type, meta_type, DIM>(&table_);
      CUDA_CHECK(cudaFree(d_table_));
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
    MERLIN_CHECK((!(options_.io_by_cpu && options_.max_hbm_for_vectors != 0)),
                 "[HierarchicalKV] `io_by_cpu` should not be true when "
                 "`max_hbm_for_vectors` is not 0!");
    CUDA_CHECK(cudaMalloc((void**)&(d_table_), sizeof(TableCore)));
    CUDA_CHECK(
        cudaMemcpy(d_table_, table_, sizeof(TableCore), cudaMemcpyDefault));
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
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * metas anyway. If true, it does not check whether the metas confroms to
   * the evict strategy. If false, it requires the metas follow the evict
   * strategy of table.
   */
  void insert_or_assign(const size_type n,
                        const key_type* keys,              // (n)
                        const value_type* values,          // (n, DIM)
                        const meta_type* metas = nullptr,  // (n)
                        cudaStream_t stream = 0,
                        bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n) > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    if (is_fast_mode()) {
      using Selector =
          SelectUpsertKernelWithIO<key_type, vector_type, meta_type, DIM>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % 7) == 0) {
        load_factor = fast_load_factor();
      }

      Selector::execute_kernel(
          load_factor, options_.block_size, stream, n, d_table_, keys,
          reinterpret_cast<const vector_type*>(values), metas);
    } else {
      vector_type** d_dst = nullptr;
      int* d_src_offset = nullptr;

      CUDA_CHECK(cudaMallocAsync(&d_dst, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMallocAsync(&d_src_offset, n * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(d_src_offset, 0, n * sizeof(int), stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        upsert_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(d_table_, keys, d_dst, metas,
                                                   d_src_offset, N);
      }

      {
        thrust::device_ptr<uintptr_t> d_dst_ptr(
            reinterpret_cast<uintptr_t*>(d_dst));
        thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

        thrust::sort_by_key(thrust_par.on(stream), d_dst_ptr, d_dst_ptr + n,
                            d_src_offset_ptr, thrust::less<uintptr_t>());
      }

      if (options_.io_by_cpu) {
        static thread_local FlexPinnedBuffer<vector_type*> h_dst;
        static thread_local FlexPinnedBuffer<vector_type> h_values;
        static thread_local FlexPinnedBuffer<int> h_src_offset;

        vector_type** l_dst = h_dst.alloc_or_reuse(n);
        vector_type* l_values = h_values.alloc_or_reuse(n);
        int* l_src_offset = h_src_offset.alloc_or_reuse(n);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(l_dst, d_dst, n * sizeof(vector_type*),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(l_values, values, n * sizeof(vector_type),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(l_src_offset, d_src_offset, n * sizeof(int),
                              cudaMemcpyDeviceToHost));
        write_by_cpu<vector_type>(l_dst, l_values, l_src_offset, n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * DIM;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, block_size, 0, stream>>>(
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
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * metas anyway. If true, it does not check whether the metas confroms to
   * the evict strategy. If false, it requires the metas follow the evict
   * strategy of table.
   *
   */
  void accum_or_assign(const size_type n,
                       const key_type* keys,               // (n)
                       const value_type* value_or_deltas,  // (n, DIM)
                       const bool* accum_or_assigns,       // (n)
                       const meta_type* metas = nullptr,   // (n)
                       cudaStream_t stream = 0,
                       bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n) > options_.max_load_factor) {
      reserve(capacity() * 2);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

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
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      accum_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, dst, metas, accum_or_assigns, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, src_offset, founds, N);
    }

    if (!is_fast_mode()) {
      thrust::device_ptr<uintptr_t> dst_ptr(reinterpret_cast<uintptr_t*>(dst));
      thrust::device_ptr<int> src_offset_ptr(src_offset);

      thrust::sort_by_key(thrust_par.on(stream), dst_ptr, dst_ptr + n,
                          src_offset_ptr, thrust::less<uintptr_t>());
    }

    {
      const size_t block_size = options_.io_block_size;
      const size_t N = n * DIM;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      write_with_accum_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, 0, stream>>>(
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
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            bool* founds,                             // (n)
            meta_type* metas = nullptr,               // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    if (is_fast_mode()) {
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      lookup_kernel_with_io<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              d_table_, keys, reinterpret_cast<vector_type*>(values), metas,
              founds, N);
    } else {
      vector_type** src;
      int* dst_offset = nullptr;
      CUDA_CHECK(cudaMallocAsync(&src, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMemsetAsync(src, 0, n * sizeof(vector_type*), stream));
      CUDA_CHECK(cudaMallocAsync(&dst_offset, n * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(dst_offset, 0, n * sizeof(int), stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        lookup_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, keys, reinterpret_cast<vector_type**>(src), metas,
                founds, dst_offset, N);
      }

      {
        thrust::device_ptr<uintptr_t> src_ptr(
            reinterpret_cast<uintptr_t*>(src));
        thrust::device_ptr<int> dst_offset_ptr(dst_offset);

        thrust::sort_by_key(thrust_par.on(stream), src_ptr, src_ptr + n,
                            dst_offset_ptr, thrust::less<uintptr_t>());
      }

      {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * DIM;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_kernel<key_type, vector_type, meta_type, DIM>
            <<<grid_size, block_size, 0, stream>>>(
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
   */
  void erase(const size_type n, const key_type* keys, cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    std::unique_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    lock.lock();

    {
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, vector_type, meta_type, DIM, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaCheckError();
    return;
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
  size_type erase_if(const Pred& pred, const key_type& pattern,
                     const meta_type& threshold, cudaStream_t stream = 0) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    lock.lock();

    size_type* d_count;
    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

    Pred h_pred;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    {
      const size_t block_size = options_.block_size;
      const size_t N = table_->buckets_num;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, vector_type, meta_type, DIM>
          <<<grid_size, block_size, 0, stream>>>(
              table_, h_pred, pattern, threshold, d_count, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, N);
    }

    size_type count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
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

    const size_t block_size = options_.block_size;
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

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
   * @param offset The position of the key to remove.
   * @param counter Accumulates amount of successfully exported values.
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
  void export_batch(size_type n, const size_type offset,
                    size_type* counter,          // (1)
                    key_type* keys,              // (n)
                    value_type* values,          // (n, DIM)
                    meta_type* metas = nullptr,  // (n)
                    cudaStream_t stream = 0) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    if (offset >= table_->capacity) {
      CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(size_type), stream));
      return;
    }
    n = std::min(table_->capacity - offset, n);

    const size_t meta_size = metas ? sizeof(meta_type) : 0;
    const size_t kvm_size = sizeof(key_type) + sizeof(vector_type) + meta_size;
    const size_t block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
    MERLIN_CHECK(
        (block_size > 0),
        "[HierarchicalKV] block_size <= 0, the K-V-M size may be too large!");

    const size_t shared_size = kvm_size * block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    dump_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, reinterpret_cast<vector_type*>(values), metas, offset,
            n, counter);

    CudaCheckError();
  }

  size_type export_batch(const size_type n, const size_type offset,
                         key_type* keys,              // (n)
                         value_type* values,          // (n, DIM)
                         meta_type* metas = nullptr,  // (n)
                         cudaStream_t stream = 0) const {
    size_type* d_counter = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
    export_batch(n, offset, d_counter, keys, values, metas, stream);

    size_type counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&counter, d_counter, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_counter, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counter;
  }

  /**
   * @brief Exports a certain number of the key-value-meta tuples which match
   * specified condition from the hash table.
   *
   * @param n The maximum number of exported pairs.
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class M>
   *    __forceinline__ __device__ bool export_if_pred(const K& key,
   *                                                   M& meta,
   *                                                   const K& pattern,
   *                                                   const M& threshold) {
   *
   *      return meta > threshold;
   *    }
   *    ```
   *
   * @param pred The predicate function with type Pred that returns `true` if
   * the element should be exported.
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with meta_type
   * type.
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
  void export_batch_if(Pred& pred, const key_type& pattern,
                       const meta_type& threshold, size_type n,
                       const size_type offset, size_type* d_counter,
                       key_type* keys,              // (n)
                       value_type* values,          // (n, DIM)
                       meta_type* metas = nullptr,  // (n)
                       cudaStream_t stream = 0) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    if (offset >= table_->capacity) {
      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
      return;
    }
    n = std::min(table_->capacity - offset, n);

    const size_t meta_size = metas ? sizeof(meta_type) : 0;
    const size_t kvm_size = sizeof(key_type) + sizeof(vector_type) + meta_size;
    const size_t block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
    MERLIN_CHECK(
        block_size > 0,
        "[HierarchicalKV] block_size <= 0, the K-V-M size may be too large!");

    const size_t shared_size = kvm_size * block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    Pred h_pred;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    dump_kernel<key_type, vector_type, meta_type, DIM>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, h_pred, pattern, threshold, keys,
            reinterpret_cast<vector_type*>(values), metas, offset, n,
            d_counter);

    CudaCheckError();
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
    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }

    size_type h_size = 0;

    const size_type N = table_->buckets_num;
    const size_type step = static_cast<size_type>(
        std::numeric_limits<int>::max() / options_.max_bucket_size);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

    for (size_type start_i = 0; start_i < N; start_i += step) {
      size_type end_i = std::min(start_i + step, N);
      h_size += thrust::reduce(thrust_par.on(stream), size_ptr + start_i,
                               size_ptr + end_i, 0, thrust::plus<int>());
    }

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
  void reserve(const size_type new_capacity, cudaStream_t stream = 0) {
    if (reach_max_capacity_ || new_capacity > options_.max_capacity) {
      return;
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(mutex_);

      // Once we have exclusive access, make sure that pending GPU calls have
      // been processed.
      CUDA_CHECK(cudaDeviceSynchronize());

      while (capacity() < new_capacity &&
             capacity() * 2 <= options_.max_capacity) {
        double_capacity(&table_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(
            cudaMemcpy(d_table_, table_, sizeof(TableCore), cudaMemcpyDefault));

        const size_t block_size = options_.block_size;
        const size_t N = TILE_SIZE * table_->buckets_num / 2;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        rehash_kernel_for_fast_mode<key_type, vector_type, meta_type, DIM,
                                    TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(d_table_, N);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
    }
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
   * @brief Save keys, vectors, metas in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param buffer_size The size of buffer used for saving in bytes.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  size_type save(BaseKVFile<K, V, M, DIM>* file,
                 const size_type buffer_size = 1048576,
                 cudaStream_t stream = 0) const {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
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

    size_type counter = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_type offset = batch_pairs_num; offset < total_size;
         offset += batch_pairs_num) {
      CUDA_CHECK(cudaMemsetAsync(d_next_nkeys, 0, sizeof(size_type), stream));
      export_batch(batch_pairs_num, offset, d_next_nkeys, d_keys, d_vectors,
                   d_metas, stream);
      counter += file->write(nkeys, h_keys, h_vectors, h_metas);
      CUDA_CHECK(cudaMemcpyAsync(&nkeys, d_next_nkeys, sizeof(size_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(K) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_vectors, d_vectors, sizeof(V) * nkeys * DIM,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_metas, d_metas, sizeof(M) * nkeys,
                                 cudaMemcpyDeviceToHost, stream));
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
      counter += file->write(nkeys, h_keys, h_vectors, h_metas);
    }

    CUDA_FREE_POINTERS(stream, d_keys, d_vectors, d_metas, d_next_nkeys);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_FREE_POINTERS(stream, h_keys, h_vectors, h_metas);
    return counter;
  }

  /**
   * @brief Load keys, vectors, metas from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param buffer_size The size of buffer used for loading in bytes.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(BaseKVFile<K, V, M, DIM>* file,
                 const size_type buffer_size = 1048576,
                 cudaStream_t stream = 0) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
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
    size_type counter = nkeys;
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
      insert_or_assign(nkeys, d_keys, d_vectors, d_metas, stream, true);
      nkeys = file->read(batch_pairs_num, h_keys, h_vectors, h_metas);
      counter += nkeys;
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } while (nkeys > 0);

    CUDA_FREE_POINTERS(stream, d_keys, d_vectors, d_metas);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_FREE_POINTERS(stream, h_keys, h_vectors, h_metas);
    return counter;
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
   * @param delta A hypothetical upcoming change on table size.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return The evaluated load factor
   */
  inline float fast_load_factor(const size_type delta = 0,
                                cudaStream_t stream = 0) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_, std::defer_lock);
    if (!reach_max_capacity_) {
      lock.lock();
    }
    size_t N = std::min(table_->buckets_num, 1024UL);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

    int size = thrust::reduce(thrust_par.on(stream), size_ptr, size_ptr + N, 0,
                              thrust::plus<int>());

    CudaCheckError();
    return static_cast<float>((delta * 1.0) / (capacity() * 1.0) +
                              (size * 1.0) /
                                  (options_.max_bucket_size * N * 1.0));
  }

  inline void check_evict_strategy(const meta_type* metas) {
    if (options_.evict_strategy == EvictStrategy::kLru) {
      MERLIN_CHECK(metas == nullptr,
                   "the metas should not be specified when running on "
                   "LRU mode.");
    }

    if (options_.evict_strategy == EvictStrategy::kCustomized) {
      MERLIN_CHECK(metas != nullptr,
                   "the metas should be specified when running on "
                   "customized mode.");
    }
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  TableCore* d_table_ = nullptr;
  size_t shared_mem_size_ = 0;
  std::atomic_bool reach_max_capacity_{false};
  bool initialized_ = false;
  mutable std::shared_timed_mutex mutex_;
};

}  // namespace merlin
}  // namespace nv
