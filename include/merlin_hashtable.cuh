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
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include "merlin/allocator.cuh"
#include "merlin/array_kernels.cuh"
#include "merlin/core_kernels.cuh"
#include "merlin/flexible_buffer.cuh"
#include "merlin/group_lock.cuh"
#include "merlin/memory_pool.cuh"
#include "merlin/multi_vector.hpp"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"
#include "merlin_hashtable_base.hpp"

namespace nv {
namespace merlin {

/**
 * @brief The eviction strategies.
 *
 * @note The `Score` concept is introduced to define the importance of each key,
 * the larger, the more important, the less likely they will be evicted. On
 * `kLru` mode, the `scores` parameter of the APIs should keep `nullptr`, the
 * score for each key is assigned internally in LRU(Least Recently Used) policy.
 * On `kCustomized` mode, the `scores` should be provided by caller.
 *
 * @note Eviction occurs automatically when a bucket is full. The keys with the
 * minimum `score` value are evicted first.
 *
 * @note on `kLru`, Set the score to the Device clock in a nanosecond, which
 * could differ slightly from the host clock.
 *
 * @note For `kEpochLru` and `kEpochLfu`, the high 32bits would be set to
 * `global_epoch` while the low 32 bits is `timestamp` or `frequency`.
 *
 * @note on `kLfu`, Frequency increment provided by caller via the input
 * parameter of `scores` of `insert-like` APIs as the increment of frequency.
 * when the scores reaches to the max of `uint64_t`, it will not increase any
 * more.
 *
 * @note On `kEpochLru`, the high 32bits is the global epoch provided via the
 * input parameter of `global_epoch`, the low 32bits is equal to `(device_clock
 * >> 20) & 0xffffffff` with granularity close to 1 ms.
 *
 * @note On `kEpochLfu`, the high 32bits is the global epoch provided via the
 * input parameter of `global_epoch`, the low 32bits is the frequency, the
 * frequency will keep constant after reaching the max value of `0xffffffff`.
 *
 * @note On `kCustomized`, fully provided by the caller via the input parameter
 * of `scores` of `insert-like` APIs.
 *
 */
struct EvictStrategy {
  enum EvictStrategyEnum {
    kLru = 0,         ///< LRU mode.
    kLfu = 1,         ///< LFU mode.
    kEpochLru = 2,    ///< Epoch Lru mode.
    kEpochLfu = 3,    ///< Epoch Lfu mode.
    kCustomized = 4,  ///< Customized mode.
  };
};

/**
 * @brief The options struct of HierarchicalKV.
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity of the hash table.
  size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
  size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
  size_t max_bucket_size = 128;    ///< The length of each bucket.
  size_t dim = 64;                 ///< The dimension of the vectors.
  float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
  int block_size = 128;            ///< The default block size for CUDA kernels.
  int io_block_size = 1024;        ///< The block size for IO CUDA kernels.
  int device_id = -1;              ///< The ID of device.
  bool io_by_cpu = false;  ///< The flag indicating if the CPU handles IO.
  bool use_constant_memory = false;  ///< reserved
  /*
   * reserved_key_start_bit = 0, is the default behavior, HKV reserves
   * `0xFFFFFFFFFFFFFFFD`, `0xFFFFFFFFFFFFFFFE`, and `0xFFFFFFFFFFFFFFFF`  for
   * internal using. if the default one conflicted with your keys, change the
   * reserved_key_start_bit value to a numbers between 1 and 62,
   * reserved_key_start_bit = 1 means using the insignificant bits index 1 and 2
   * as the keys as the reserved keys and the index 0 bit is 0 and all the other
   * bits are 1, the new reserved keys are `FFFFFFFFFFFFFFFE`,
   * `0xFFFFFFFFFFFFFFFC`, `0xFFFFFFFFFFFFFFF8`, and `0xFFFFFFFFFFFFFFFA` the
   * console log prints the reserved keys during the table initialization.
   */
  int reserved_key_start_bit = 0;       ///< The binary index of reserved key.
  size_t num_of_buckets_per_alloc = 1;  ///< Number of buckets allocated in each
                                        ///< HBM allocation, must be power of 2.
  bool api_lock = true;  ///<  The flag indicating whether to lock the table
                         ///<  once enters the API.
  MemoryPoolOptions
      device_memory_pool;  ///< Configuration options for device memory pool.
  MemoryPoolOptions
      host_memory_pool;  ///< Configuration options for host memory pool.
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
 *    template <class K, class S>
 *    struct EraseIfPredFunctor {
 *      __forceinline__ __device__ bool operator()(const K& key,
 *                                                 S& score,
 *                                                 const K& pattern,
 *                                                 const S& threshold) {
 *        return ((key & 0xFFFF000000000000 == pattern) &&
 *                (score < threshold));
 *      }
 *    };
 *    ```
 *
 *  Example for export_batch_if:
 *    ```
 *    template <class K, class S>
 *    struct ExportIfPredFunctor {
 *      __forceinline__ __device__ bool operator()(const K& key,
 *                                                 S& score,
 *                                                 const K& pattern,
 *                                                 const S& threshold) {
 *        return score >= threshold;
 *      }
 *    };
 *    ```
 */
template <class K, class S>
using EraseIfPredict = bool (*)(
    const K& key,       ///< The traversed key in a hash table.
    S& score,           ///< The traversed score in a hash table.
    const K& pattern,   ///< The key pattern to compare with the `key` argument.
    const S& threshold  ///< The threshold to compare with the `score` argument.
);

#if THRUST_VERSION >= 101600
static constexpr auto& thrust_par = thrust::cuda::par_nosync;
#else
static constexpr auto& thrust_par = thrust::cuda::par;
#endif

/**
 * A HierarchicalKV hash table is a concurrent and hierarchical hash table that
 * is powered by GPUs and can use HBM and host memory as storage for key-value
 * pairs. Support for SSD storage is a future consideration.
 *
 * The `score` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. Eviction
 * occurs automatically when a bucket is full. The keys with the minimum `score`
 * value are evicted first. In a customized eviction strategy, we recommend
 * using the timestamp or frequency of the key occurrence as the `score` value
 * for each key. You can also assign a special value to the `score` to
 * perform a customized eviction strategy.
 *
 * @note By default configuration, this class is thread-safe.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's item type.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam S The data type for `score`.
 *           The currently supported data type is only `uint64_t`.
 *
 * @note ArchTag controls internal tuning (SM-specific pipeline config), not
 *       the actual compiled GPU binary architecture. We keep the default at
 *       Sm80 so that all kernels reuse the existing specializations, while
 *       nvcc still generates sm_100 code via -gencode flags.
 */
template <typename K, typename V, typename S = uint64_t,
          int Strategy = EvictStrategy::kLru, typename ArchTag = Sm80>
class HashTable : public HashTableBase<K, V, S> {
 public:
  using size_type = size_t;
  using key_type = K;
  using value_type = V;
  using score_type = S;
  static constexpr int evict_strategy = Strategy;

  using Pred = EraseIfPredict<key_type, score_type>;
  using allocator_type = BaseAllocator;

 private:
  using TableCore = nv::merlin::Table<key_type, value_type, score_type>;
  static constexpr unsigned int TILE_SIZE = 4;

  using DeviceMemoryPool = MemoryPool<DeviceAllocator<char>>;
  using HostMemoryPool = MemoryPool<HostAllocator<char>>;

 public:
  /**
   * @brief Default constructor for the hash table class.
   */
  HashTable() {
    static_assert((std::is_same<key_type, int64_t>::value ||
                   std::is_same<key_type, uint64_t>::value),
                  "The key_type must be int64_t or uint64_t.");

    static_assert(std::is_same<score_type, uint64_t>::value,
                  "The key_type must be uint64_t.");
  };

  /**
   * @brief Frees the resources used by the hash table and destroys the hash
   * table object.
   */
  ~HashTable() {
    if (initialized_) {
      CUDA_CHECK(cudaDeviceSynchronize());

      initialized_ = false;
      destroy_table<key_type, value_type, score_type>(&table_, allocator_);
      allocator_->free(MemoryType::Device, d_table_);
      dev_mem_pool_.reset();
      host_mem_pool_.reset();

      CUDA_CHECK(cudaDeviceSynchronize());
      if (default_allocator_ && allocator_ != nullptr) {
        delete allocator_;
      }
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
  void init(const HashTableOptions& options,
            allocator_type* allocator = nullptr) {
    if (initialized_) {
      return;
    }
    options_ = options;
    MERLIN_CHECK(options.reserved_key_start_bit >= 0 &&
                     options.reserved_key_start_bit <= MAX_RESERVED_KEY_BIT,
                 "options.reserved_key_start_bit should >= 0 and <= 62.");
    CUDA_CHECK(init_reserved_keys(options.reserved_key_start_bit));

    default_allocator_ = (allocator == nullptr);
    allocator_ = (allocator == nullptr) ? (new DefaultAllocator()) : allocator;

    thrust_allocator_.set_allocator(allocator_);

    if (options_.device_id >= 0) {
      CUDA_CHECK(cudaSetDevice(options_.device_id));
    } else {
      CUDA_CHECK(cudaGetDevice(&(options_.device_id)));
    }

    MERLIN_CHECK(ispow2(static_cast<uint32_t>(options_.max_bucket_size)),
                 "Bucket size should be the pow of 2");
    MERLIN_CHECK(
        ispow2(static_cast<uint32_t>(options_.num_of_buckets_per_alloc)),
        "Then `num_of_buckets_per_alloc` should be the pow of 2");
    MERLIN_CHECK(options_.init_capacity >= options_.num_of_buckets_per_alloc *
                                               options_.max_bucket_size,
                 "Then `num_of_buckets_per_alloc` must be equal or less than "
                 "initial required buckets number");

    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);

    MERLIN_CHECK(
        (((options_.max_bucket_size * (sizeof(key_type) + sizeof(score_type))) %
          128) == 0),
        "Storage size of keys and scores in one bucket should be the mutiple "
        "of cache line size");

    // Construct table.
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, options_.device_id));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    sm_cnt_ = deviceProp.multiProcessorCount;
    max_threads_per_block_ = deviceProp.maxThreadsPerBlock;
    create_table<key_type, value_type, score_type>(
        &table_, allocator_, options_.dim, options_.init_capacity,
        options_.max_capacity, options_.max_hbm_for_vectors,
        options_.max_bucket_size, options_.num_of_buckets_per_alloc);
    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);
    reach_max_capacity_ = (options_.init_capacity * 2 > options_.max_capacity);
    MERLIN_CHECK((!(options_.io_by_cpu && options_.max_hbm_for_vectors != 0)),
                 "[HierarchicalKV] `io_by_cpu` should not be true when "
                 "`max_hbm_for_vectors` is not 0!");
    allocator_->alloc(MemoryType::Device, (void**)&(d_table_),
                      sizeof(TableCore));

    sync_table_configuration();

    // Create memory pools.
    dev_mem_pool_ = std::make_unique<MemoryPool<DeviceAllocator<char>>>(
        options_.device_memory_pool, allocator_);
    host_mem_pool_ = std::make_unique<MemoryPool<HostAllocator<char>>>(
        options_.host_memory_pool, allocator_);

    CUDA_CHECK(cudaDeviceSynchronize());

    initialized_ = true;
    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores conforms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void insert_or_assign(const size_type n,
                        const key_type* keys,                // (n)
                        const value_type* values,            // (n, DIM)
                        const score_type* scores = nullptr,  // (n)
                        cudaStream_t stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) {
    if (ignore_evict_strategy) {
      insert_or_assign_impl<EvictStrategy::kCustomized>(
          n, keys, values, scores, stream, unique_key, ignore_evict_strategy);
    } else {
      insert_or_assign_impl<evict_strategy>(n, keys, values, scores, stream,
                                            unique_key, ignore_evict_strategy);
    }
  }

  template <int evict_strategy_>
  void insert_or_assign_impl(const size_type n,
                             const key_type* keys,      // (n)
                             const value_type* values,  // (n, DIM)
                             const score_type* scores,  // (n)
                             cudaStream_t stream, bool unique_key,
                             bool ignore_evict_strategy) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    if (is_fast_mode()) {
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }

      using Selector = KernelSelector_Upsert<key_type, value_type, score_type,
                                             evict_strategy_, ArchTag>;
      if (Selector::callable(unique_key,
                             static_cast<uint32_t>(options_.max_bucket_size),
                             static_cast<uint32_t>(options_.dim))) {
        typename Selector::Params kernelParams(
            load_factor, table_->buckets, table_->buckets_size,
            table_->buckets_num,
            static_cast<uint32_t>(options_.max_bucket_size),
            static_cast<uint32_t>(options_.dim), keys, values, scores, n,
            global_epoch_);
        Selector::select_kernel(kernelParams, stream);
      } else {
        using Selector = SelectUpsertKernelWithIO<key_type, value_type,
                                                  score_type, evict_strategy_>;
        Selector::execute_kernel(
            load_factor, options_.block_size, options_.max_bucket_size,
            table_->buckets_num, options_.dim, stream, n, d_table_,
            table_->buckets, keys, reinterpret_cast<const value_type*>(values),
            scores, global_epoch_);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, key_type*, uint8_t> mv(
          n, n, n, n, n, d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto d_dst = get_vector<0>(mv, temp_storage);
      auto d_src_offset = get_vector<1>(mv, temp_storage);
      auto d_dst_sorted = get_vector<2>(mv, temp_storage);
      auto d_src_offset_sorted = get_vector<3>(mv, temp_storage);
      auto keys_ptr = get_vector<4>(mv, temp_storage);
      auto d_sort_storage = get_vector<5>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          unique_key && options_.max_bucket_size >= MinBucketCapacityFilter &&
          !options_.io_by_cpu;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128;

        upsert_kernel_lock_key_hybrid<key_type, value_type, score_type,
                                      BLOCK_SIZE, evict_strategy>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_size, table_->buckets_num,
                options_.max_bucket_size, options_.dim, keys, d_dst, scores,
                keys_ptr, d_src_offset, n, global_epoch_);

      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        upsert_kernel<key_type, value_type, score_type, evict_strategy,
                      TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_table_, table_->buckets, options_.max_bucket_size,
            table_->buckets_num, options_.dim, keys, d_dst, scores,
            d_src_offset, global_epoch_, N);
      }

      sortOp.sort(n, reinterpret_cast<uintptr_t*>(d_dst),
                  reinterpret_cast<uintptr_t*>(d_dst_sorted), d_src_offset,
                  d_src_offset_sorted, stream);

      if (filter_condition) {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel_unlock_key<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(values, d_dst_sorted,
                                                   d_src_offset_sorted, dim(),
                                                   keys, keys_ptr, N);

      } else if (options_.io_by_cpu) {
        MultiVector<value_type*, int, value_type> mv1(n, n, n * dim());
        const size_type host_ws_size = mv1.total_size();
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto host_temp_storage = host_ws.get<uint8_t*>(0);
        auto h_dst_sorted = get_vector<0>(mv1, host_temp_storage);
        auto h_src_offset_sorted = get_vector<1>(mv1, host_temp_storage);
        auto h_values = get_vector<2>(mv1, host_temp_storage);

        CUDA_CHECK(cudaMemcpyAsync(h_dst_sorted, d_dst_sorted, mv1.offset(2),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, values,
                                   n * dim() * sizeof(value_type),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        write_by_cpu<value_type>(h_dst_sorted, h_values, h_src_offset_sorted,
                                 dim(), n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(
                values, d_dst_sorted, d_src_offset_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param d_evicted_counter The number of elements evicted on GPU-accessible
   * memory. @notice The caller should guarantee it is set to `0` before
   * calling.
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void insert_and_evict(const size_type n,
                        const key_type* keys,          // (n)
                        const value_type* values,      // (n, DIM)
                        const score_type* scores,      // (n)
                        key_type* evicted_keys,        // (n)
                        value_type* evicted_values,    // (n, DIM)
                        score_type* evicted_scores,    // (n)
                        size_type* d_evicted_counter,  // (1)
                        cudaStream_t stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    // TODO: Currently only need eviction when using HashTable as HBM cache.
    if (!is_fast_mode()) {
      throw std::runtime_error("Only allow insert_and_evict in pure HBM mode.");
    }

    static thread_local int step_counter = 0;
    static thread_local float load_factor = 0.0;

    if (((step_counter++) % kernel_select_interval_) == 0) {
      load_factor = fast_load_factor(0, stream, false);
    }

    using Selector =
        KernelSelector_UpsertAndEvict<key_type, value_type, score_type,
                                      evict_strategy, ArchTag>;
    if (Selector::callable(unique_key,
                           static_cast<uint32_t>(options_.max_bucket_size),
                           static_cast<uint32_t>(options_.dim))) {
      typename Selector::Params kernelParams(
          load_factor, table_->buckets, table_->buckets_size,
          table_->buckets_num, static_cast<uint32_t>(options_.max_bucket_size),
          static_cast<uint32_t>(options_.dim), keys, values, scores,
          evicted_keys, evicted_values, evicted_scores, n, d_evicted_counter,
          global_epoch_);
      Selector::select_kernel(kernelParams, stream);
    } else if (unique_key and options_.max_bucket_size % 16 == 0) {
      using KernelLauncher =
          InsertAndEvictKernelLauncher<key_type, value_type, score_type,
                                       evict_strategy>;
      typename KernelLauncher::Params kernelParams(
          load_factor, table_->buckets, table_->buckets_size,
          table_->buckets_num, static_cast<uint32_t>(options_.max_bucket_size),
          static_cast<uint32_t>(options_.dim), keys, values, scores,
          evicted_keys, evicted_values, evicted_scores, n, d_evicted_counter,
          global_epoch_);
      KernelLauncher::launch_kernel(kernelParams, stream);
    } else {
      // always use max tile to avoid data-deps as possible.
      const int TILE_SIZE = 32;
      size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
      const size_type dev_ws_size =
          n * (sizeof(key_type) + sizeof(score_type)) +
          n_offsets * sizeof(int64_t) + n * dim() * sizeof(value_type) +
          n * sizeof(bool);

      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto tmp_evict_keys{dev_ws.get<key_type*>(0)};
      auto tmp_evict_scores = reinterpret_cast<score_type*>(tmp_evict_keys + n);
      auto d_offsets = reinterpret_cast<int64_t*>(tmp_evict_scores + n);
      auto tmp_evict_values =
          reinterpret_cast<value_type*>(d_offsets + n_offsets);
      auto d_masks = reinterpret_cast<bool*>(tmp_evict_values + n * dim());

      CUDA_CHECK(
          cudaMemsetAsync(d_offsets, 0, n_offsets * sizeof(int64_t), stream));
      CUDA_CHECK(cudaMemsetAsync(d_masks, 0, n * sizeof(bool), stream));

      size_type block_size = options_.block_size;
      size_type grid_size = SAFE_GET_GRID_SIZE(n, block_size);
      CUDA_CHECK(memset64Async(tmp_evict_keys, EMPTY_KEY_CPU, n, stream));
      using Selector =
          SelectUpsertAndEvictKernelWithIO<key_type, value_type, score_type,
                                           evict_strategy>;
      Selector::execute_kernel(
          load_factor, options_.block_size, options_.max_bucket_size,
          table_->buckets_num, options_.dim, stream, n, d_table_,
          table_->buckets, keys, values, scores, tmp_evict_keys,
          tmp_evict_values, tmp_evict_scores, global_epoch_);
      keys_not_empty<K>
          <<<grid_size, block_size, 0, stream>>>(tmp_evict_keys, d_masks, n);

      gpu_cell_count<int64_t, TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
          d_masks, d_offsets, n, d_evicted_counter);

      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_offsets, d_offsets, n_offsets, stream);
      auto dev_ws1{dev_mem_pool_->get_workspace<1>(temp_storage_bytes, stream)};
      d_temp_storage = dev_ws1.get<void*>(0);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_offsets, d_offsets, n_offsets, stream);

      compact_key_value_score_kernel<K, V, S, int64_t, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              d_masks, n, d_offsets, tmp_evict_keys, tmp_evict_values,
              tmp_evict_scores, evicted_keys, evicted_values, evicted_scores,
              dim());
    }
    return;
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   *
   * @return The number of elements evicted.
   */
  size_type insert_and_evict(const size_type n,
                             const key_type* keys,        // (n)
                             const value_type* values,    // (n, DIM)
                             const score_type* scores,    // (n)
                             key_type* evicted_keys,      // (n)
                             value_type* evicted_values,  // (n, DIM)
                             score_type* evicted_scores,  // (n)
                             cudaStream_t stream = 0, bool unique_key = true,
                             bool ignore_evict_strategy = false) {
    if (n == 0) {
      return 0;
    }
    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    size_type* d_evicted_counter{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(
        cudaMemsetAsync(d_evicted_counter, 0, sizeof(size_type), stream));
    insert_and_evict(n, keys, values, scores, evicted_keys, evicted_values,
                     evicted_scores, d_evicted_counter, stream, unique_key,
                     ignore_evict_strategy);

    size_type h_evicted_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_evicted_counter, d_evicted_counter,
                               sizeof(size_type), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return h_evicted_counter;
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
   * @param n The number of key-value-score tuples to process.
   * @param keys The keys to insert on GPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on GPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param scores The scores to insert on GPU-accessible memory with shape (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void accum_or_assign(const size_type n,
                       const key_type* keys,                // (n)
                       const value_type* value_or_deltas,   // (n, DIM)
                       const bool* accum_or_assigns,        // (n)
                       const score_type* scores = nullptr,  // (n)
                       cudaStream_t stream = 0,
                       bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    if (is_fast_mode()) {
      using Selector =
          SelectAccumOrAssignKernelWithIO<key_type, value_type, score_type,
                                          evict_strategy>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      Selector::execute_kernel(
          load_factor, options_.block_size, options_.max_bucket_size,
          table_->buckets_num, dim(), stream, n, d_table_, keys,
          value_or_deltas, scores, accum_or_assigns, global_epoch_);

    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, bool, uint8_t> mv(
          n, n, n, n, n, d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto dst = get_vector<0>(mv, temp_storage);
      auto src_offset = get_vector<1>(mv, temp_storage);
      auto dst_sorted = get_vector<2>(mv, temp_storage);
      auto src_offset_sorted = get_vector<3>(mv, temp_storage);
      auto founds = get_vector<4>(mv, temp_storage);
      auto d_sort_storage = get_vector<5>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(dst, 0, dev_ws_size, stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        accum_or_assign_kernel<key_type, value_type, score_type, evict_strategy,
                               TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_table_, options_.max_bucket_size, table_->buckets_num, dim(),
            keys, dst, scores, accum_or_assigns, src_offset, founds,
            global_epoch_, N);
      }

      sortOp.sort(n, reinterpret_cast<uintptr_t*>(dst),
                  reinterpret_cast<uintptr_t*>(dst_sorted), src_offset,
                  src_offset_sorted, stream);

      {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_with_accum_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(value_or_deltas, dst_sorted,
                                                   accum_or_assigns, founds,
                                                   src_offset_sorted, dim(), N);
      }
    }
    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   * When a key is missing, the value in @p values and @p scores will be
   * inserted.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type* values,                       // (n * DIM)
                      score_type* scores = nullptr,             // (n)
                      cudaStream_t stream = 0, bool unique_key = true,
                      bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    if (is_fast_mode()) {
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }

      using Selector =
          KernelSelector_FindOrInsert<key_type, value_type, score_type,
                                      evict_strategy, ArchTag>;
      if (Selector::callable(unique_key,
                             static_cast<uint32_t>(options_.max_bucket_size),
                             static_cast<uint32_t>(options_.dim))) {
        typename Selector::Params kernelParams(
            load_factor, table_->buckets, table_->buckets_size,
            table_->buckets_num,
            static_cast<uint32_t>(options_.max_bucket_size),
            static_cast<uint32_t>(options_.dim), keys, values, scores, n,
            global_epoch_);
        Selector::select_kernel(kernelParams, stream);
      } else {
        using Selector =
            SelectFindOrInsertKernelWithIO<key_type, value_type, score_type,
                                           evict_strategy>;
        Selector::execute_kernel(
            load_factor, options_.block_size, options_.max_bucket_size,
            table_->buckets_num, options_.dim, stream, n, d_table_,
            table_->buckets, keys, values, scores, global_epoch_);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, bool, key_type*, uint8_t>
          mv(n, n, n, n, n, n, d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto d_table_value_addrs = get_vector<0>(mv, temp_storage);
      auto param_key_index = get_vector<1>(mv, temp_storage);
      auto d_table_value_addrs_sorted = get_vector<2>(mv, temp_storage);
      auto param_key_index_sorted = get_vector<3>(mv, temp_storage);
      auto founds = get_vector<4>(mv, temp_storage);
      auto keys_ptr = get_vector<5>(mv, temp_storage);
      auto d_sort_storage = get_vector<6>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(d_table_value_addrs, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          unique_key && options_.max_bucket_size >= MinBucketCapacityFilter &&
          !options_.io_by_cpu;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128;

        find_or_insert_kernel_lock_key_hybrid<key_type, value_type, score_type,
                                              BLOCK_SIZE, evict_strategy>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_size, table_->buckets_num,
                options_.max_bucket_size, options_.dim, keys,
                d_table_value_addrs, scores, keys_ptr, param_key_index, founds,
                n, global_epoch_);

      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        find_or_insert_kernel<key_type, value_type, score_type, evict_strategy,
                              TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_table_, table_->buckets, options_.max_bucket_size,
            table_->buckets_num, options_.dim, keys, d_table_value_addrs,
            scores, founds, param_key_index, global_epoch_, N);
      }

      sortOp.sort(n, reinterpret_cast<uintptr_t*>(d_table_value_addrs),
                  reinterpret_cast<uintptr_t*>(d_table_value_addrs_sorted),
                  param_key_index, param_key_index_sorted, stream);

      if (filter_condition) {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_or_write_kernel_unlock_key<key_type, value_type, score_type, V>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_value_addrs_sorted, values, founds,
                param_key_index_sorted, keys_ptr, keys, dim(), N);

      } else if (options_.io_by_cpu) {
        MultiVector<value_type*, int, bool, value_type> mv1(n, n, n, n * dim());
        const size_type host_ws_size = mv1.total_size();
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto host_temp_storage = host_ws.get<uint8_t*>(0);
        auto h_table_value_addrs_sorted = get_vector<0>(mv1, host_temp_storage);
        auto h_param_key_index_sorted = get_vector<1>(mv1, host_temp_storage);
        auto h_founds = get_vector<2>(mv1, host_temp_storage);
        auto h_param_values = get_vector<3>(mv1, host_temp_storage);

        CUDA_CHECK(cudaMemcpyAsync(h_table_value_addrs_sorted,
                                   d_table_value_addrs_sorted, mv1.offset(3),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_param_values, values,
                                   n * sizeof(value_type) * dim(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        read_or_write_by_cpu<value_type>(
            h_table_value_addrs_sorted, h_param_values,
            h_param_key_index_sorted, h_founds, dim(), n);
        CUDA_CHECK(cudaMemcpyAsync(values, h_param_values,
                                   n * sizeof(value_type) * dim(),
                                   cudaMemcpyHostToDevice, stream));
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_or_write_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_value_addrs_sorted, values, founds,
                param_key_index_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values. When a key is missing, the value in @p values and @p scores
   * will be inserted.
   *
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values  The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   * @param locked_key_ptrs If it isn't nullptr then the keys in the table will
   * be locked, and key's address will write to locked_key_ptrs. Using
   * unlock_keys to unlock these keys.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type** values,                      // (n)
                      bool* founds,                             // (n)
                      score_type* scores = nullptr,             // (n)
                      cudaStream_t stream = 0, bool unique_key = true,
                      bool ignore_evict_strategy = false,
                      key_type** locked_key_ptrs = nullptr) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);

    if (locked_key_ptrs != nullptr) {
      if (!unique_key || options_.max_bucket_size < MinBucketCapacityFilter) {
        throw std::invalid_argument(
            "unique_key should be true and max_bucket_size should be larger.");
      }

      constexpr uint32_t BLOCK_SIZE = 128U;
      find_or_insert_ptr_kernel_lock_key<key_type, value_type, score_type,
                                         BLOCK_SIZE, evict_strategy>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_->buckets, table_->buckets_size, table_->buckets_num,
              options_.max_bucket_size, options_.dim, keys, values, scores,
              locked_key_ptrs, n, founds, global_epoch_);
      CudaCheckError();
      return;
    }

    if (unique_key && options_.max_bucket_size >= MinBucketCapacityFilter) {
      constexpr uint32_t BLOCK_SIZE = 128U;

      const size_type dev_ws_size{n * sizeof(key_type**)};
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto keys_ptr{dev_ws.get<key_type**>(0)};
      CUDA_CHECK(cudaMemsetAsync(keys_ptr, 0, dev_ws_size, stream));

      find_or_insert_ptr_kernel_lock_key<key_type, value_type, score_type,
                                         BLOCK_SIZE, evict_strategy>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_->buckets, table_->buckets_size, table_->buckets_num,
              options_.max_bucket_size, options_.dim, keys, values, scores,
              keys_ptr, n, founds, global_epoch_);

      find_or_insert_ptr_kernel_unlock_key<key_type>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              keys, keys_ptr, n);
    } else {
      using Selector = SelectFindOrInsertPtrKernel<key_type, value_type,
                                                   score_type, evict_strategy>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      Selector::execute_kernel(
          load_factor, options_.block_size, options_.max_bucket_size,
          table_->buckets_num, options_.dim, stream, n, d_table_,
          table_->buckets, keys, values, scores, founds, global_epoch_);
    }

    CudaCheckError();
  }

  /**
   * @brief
   * This function will lock the keys in the table and unexisted keys will be
   * ignored.
   *
   * @param n The number of keys in the table to be locked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param success The status that indicates if the lock operation is
   * succeed.
   * @param stream The CUDA stream that is used to execute the operation.
   * @param scores The scores of the input keys will set to scores if provided.
   *
   */
  void lock_keys(const size_type n,
                 key_type const* keys,        // (n)
                 key_type** locked_key_ptrs,  // (n)
                 bool* success = nullptr,     // (n)
                 cudaStream_t stream = 0, score_type const* scores = nullptr) {
    if (n == 0) {
      return;
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);
    if (options_.max_bucket_size < MinBucketCapacityFilter) {
      throw std::runtime_error(
          "Not support lock_keys API because the bucket capacity is too "
          "small.");
    }
    constexpr uint32_t BLOCK_SIZE = 128U;
    lock_kernel_with_filter<key_type, value_type, score_type, evict_strategy>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table_->buckets, table_->buckets_num, options_.max_bucket_size,
            options_.dim, keys, locked_key_ptrs, success, scores, global_epoch_,
            n);
    CudaCheckError();
  }

  /**
   * @brief Using pointers to address the keys in the hash table and set them
   * to target keys.
   * This function will unlock the keys in the table which are locked by
   * the previous call to find_or_insert.
   *
   * @param n The number of keys in the table to be unlocked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param success The status that indicates if the unlock operation is
   * succeed.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void unlock_keys(const size_type n, key_type** locked_key_ptrs,  // (n)
                   const key_type* keys,                           // (n)
                   bool* success = nullptr,                        // (n)
                   cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    std::unique_ptr<insert_unique_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<insert_unique_lock>(mutex_, stream);
    }

    constexpr uint32_t BLOCK_SIZE = 128U;
    /// TODO: check the key belongs to the bucket.
    unlock_keys_kernel<key_type>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            n, locked_key_ptrs, keys, success);
  }

  /**
   * @brief Assign new key-value-score tuples into the hash table.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign(const size_type n,
              const key_type* keys,                // (n)
              const value_type* values,            // (n, DIM)
              const score_type* scores = nullptr,  // (n)
              cudaStream_t stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    check_evict_strategy(scores);

    std::unique_ptr<update_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_shared_lock>(mutex_, stream);
    }

    if (is_fast_mode()) {
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      using Selector = KernelSelector_Update<key_type, value_type, score_type,
                                             evict_strategy, ArchTag>;
      if (Selector::callable(unique_key,
                             static_cast<uint32_t>(options_.max_bucket_size),
                             static_cast<uint32_t>(options_.dim))) {
        typename Selector::Params kernelParams(
            load_factor, table_->buckets, table_->buckets_num,
            static_cast<uint32_t>(options_.max_bucket_size),
            static_cast<uint32_t>(options_.dim), keys, values, scores, n,
            global_epoch_);
        Selector::select_kernel(kernelParams, stream);
      } else {
        using Selector = SelectUpdateKernelWithIO<key_type, value_type,
                                                  score_type, evict_strategy>;
        Selector::execute_kernel(
            load_factor, options_.block_size, options_.max_bucket_size,
            table_->buckets_num, options_.dim, stream, n, d_table_,
            table_->buckets, keys, values, scores, global_epoch_);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, key_type*, uint8_t> mv(
          n, n, n, n, n, d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto d_dst = get_vector<0>(mv, temp_storage);
      auto d_src_offset = get_vector<1>(mv, temp_storage);
      auto d_dst_sorted = get_vector<2>(mv, temp_storage);
      auto d_src_offset_sorted = get_vector<3>(mv, temp_storage);
      auto keys_ptr = get_vector<4>(mv, temp_storage);
      auto d_sort_storage = get_vector<5>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          options_.max_bucket_size >= MinBucketCapacityFilter &&
          !options_.io_by_cpu && unique_key;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128U;

        tlp_update_kernel_hybrid<key_type, value_type, score_type,
                                 evict_strategy>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_num, options_.max_bucket_size,
                options_.dim, keys, d_dst, scores, keys_ptr, d_src_offset,
                global_epoch_, n);

      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        update_kernel<key_type, value_type, score_type, evict_strategy,
                      TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_table_, table_->buckets, options_.max_bucket_size,
            table_->buckets_num, options_.dim, keys, d_dst, scores,
            d_src_offset, global_epoch_, N);
      }

      sortOp.sort(n, reinterpret_cast<uintptr_t*>(d_dst),
                  reinterpret_cast<uintptr_t*>(d_dst_sorted), d_src_offset,
                  d_src_offset_sorted, stream);

      if (filter_condition) {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel_unlock_key<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(values, d_dst_sorted,
                                                   d_src_offset_sorted, dim(),
                                                   keys, keys_ptr, N);

      } else if (options_.io_by_cpu) {
        MultiVector<value_type*, int, value_type> mv1(n, n, n * dim());
        const size_type host_ws_size = mv1.total_size();
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto host_temp_storage = host_ws.get<uint8_t*>(0);
        auto h_dst_sorted = get_vector<0>(mv1, host_temp_storage);
        auto h_src_offset_sorted = get_vector<1>(mv1, host_temp_storage);
        auto h_values = get_vector<2>(mv1, host_temp_storage);

        CUDA_CHECK(cudaMemcpyAsync(h_dst_sorted, d_dst_sorted, mv1.offset(2),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, values,
                                   n * dim() * sizeof(value_type),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        write_by_cpu<value_type>(h_dst_sorted, h_values, h_src_offset_sorted,
                                 dim(), n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(
                values, d_dst_sorted, d_src_offset_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Assign new scores for keys.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-score pairs to assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign_scores(const size_type n,
                     const key_type* keys,                // (n)
                     const score_type* scores = nullptr,  // (n)
                     cudaStream_t stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    check_evict_strategy(scores);

    {
      std::unique_ptr<update_shared_lock> lock_ptr;
      if (options_.api_lock) {
        lock_ptr = std::make_unique<update_shared_lock>(mutex_, stream);
      }
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      using Selector = KernelSelector_UpdateScore<key_type, value_type,
                                                  score_type, evict_strategy>;
      if (Selector::callable(unique_key,
                             static_cast<uint32_t>(options_.max_bucket_size))) {
        typename Selector::Params kernelParams(
            load_factor, table_->buckets, table_->buckets_num,
            static_cast<uint32_t>(options_.max_bucket_size), keys, scores, n,
            global_epoch_);
        Selector::select_kernel(kernelParams, stream);
      } else {
        using Selector = SelectUpdateScoreKernel<key_type, value_type,
                                                 score_type, evict_strategy>;
        Selector::execute_kernel(load_factor, options_.block_size,
                                 options_.max_bucket_size, table_->buckets_num,
                                 stream, n, d_table_, table_->buckets, keys,
                                 scores, global_epoch_);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Alias of `assign_scores`.
   */
  void assign(const size_type n,
              const key_type* keys,                // (n)
              const score_type* scores = nullptr,  // (n)
              cudaStream_t stream = 0, bool unique_key = true) {
    assign_scores(n, keys, scores, stream, unique_key);
  }

  /**
   * @brief Assign new values for each keys .
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value pairs to assign.
   * @param keys The keys need to be operated, which must be on GPU-accessible
   * memory with shape (n).
   * @param values The values need to be updated, which must be on
   * GPU-accessible memory with shape (n, DIM).
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign_values(const size_type n,
                     const key_type* keys,      // (n)
                     const value_type* values,  // (n, DIM)
                     cudaStream_t stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    std::unique_ptr<update_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_shared_lock>(mutex_, stream);
    }

    if (is_fast_mode()) {
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      using Selector = KernelSelector_UpdateValues<key_type, value_type,
                                                   score_type, ArchTag>;
      if (Selector::callable(unique_key,
                             static_cast<uint32_t>(options_.max_bucket_size),
                             static_cast<uint32_t>(options_.dim))) {
        typename Selector::Params kernelParams(
            load_factor, table_->buckets, table_->buckets_num,
            static_cast<uint32_t>(options_.max_bucket_size),
            static_cast<uint32_t>(options_.dim), keys, values, n);
        Selector::select_kernel(kernelParams, stream);
      } else {
        using Selector =
            SelectUpdateValuesKernelWithIO<key_type, value_type, score_type>;
        Selector::execute_kernel(load_factor, options_.block_size,
                                 options_.max_bucket_size, table_->buckets_num,
                                 options_.dim, stream, n, d_table_,
                                 table_->buckets, keys, values);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, key_type*, uint8_t> mv(
          n, n, n, n, n, d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto d_dst = get_vector<0>(mv, temp_storage);
      auto d_src_offset = get_vector<1>(mv, temp_storage);
      auto d_dst_sorted = get_vector<2>(mv, temp_storage);
      auto d_src_offset_sorted = get_vector<3>(mv, temp_storage);
      auto keys_ptr = get_vector<4>(mv, temp_storage);
      auto d_sort_storage = get_vector<5>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          options_.max_bucket_size >= MinBucketCapacityFilter &&
          !options_.io_by_cpu && unique_key;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128U;

        tlp_update_values_kernel_hybrid<key_type, value_type, score_type>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_num, options_.max_bucket_size,
                options_.dim, keys, d_dst, keys_ptr, d_src_offset, n);

      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        update_values_kernel<key_type, value_type, score_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, table_->buckets, options_.max_bucket_size,
                table_->buckets_num, options_.dim, keys, d_dst, d_src_offset,
                N);
      }

      sortOp.sort(n, reinterpret_cast<uintptr_t*>(d_dst),
                  reinterpret_cast<uintptr_t*>(d_dst_sorted), d_src_offset,
                  d_src_offset_sorted, stream);

      if (filter_condition) {
        const size_t block_size = options_.io_block_size;
        uint64_t total_value_size = sizeof(value_type) * dim();
        if (total_value_size % 16 == 0) {
          using VecV = byte16;
          uint64_t vec_dim = total_value_size / sizeof(VecV);
          const size_t N = n * vec_dim;
          const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

          write_kernel_unlock_key<key_type, VecV, score_type>
              <<<grid_size, block_size, 0, stream>>>(
                  reinterpret_cast<const VecV*>(values),
                  reinterpret_cast<VecV**>(d_dst_sorted), d_src_offset_sorted,
                  vec_dim, keys, keys_ptr, N);
        } else {
          const size_t N = n * dim();
          const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

          write_kernel_unlock_key<key_type, value_type, score_type>
              <<<grid_size, block_size, 0, stream>>>(values, d_dst_sorted,
                                                     d_src_offset_sorted, dim(),
                                                     keys, keys_ptr, N);
        }
      } else if (options_.io_by_cpu) {
        MultiVector<value_type*, int, value_type> mv1(n, n, n * dim());
        const size_type host_ws_size = mv1.total_size();
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto host_temp_storage = host_ws.get<uint8_t*>(0);
        auto h_dst_sorted = get_vector<0>(mv1, host_temp_storage);
        auto h_src_offset_sorted = get_vector<1>(mv1, host_temp_storage);
        auto h_values = get_vector<2>(mv1, host_temp_storage);

        CUDA_CHECK(cudaMemcpyAsync(h_dst_sorted, d_dst_sorted, mv1.offset(2),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, values,
                                   n * dim() * sizeof(value_type),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        write_by_cpu<value_type>(h_dst_sorted, h_values, h_src_offset_sorted,
                                 dim(), n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(
                values, d_dst_sorted, d_src_offset_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            bool* founds,                             // (n)
            score_type* scores = nullptr,             // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    const uint32_t value_size = dim() * sizeof(V);

    if (is_fast_mode()) {
      using Selector = SelectPipelineLookupKernelWithIO<key_type, value_type,
                                                        score_type, ArchTag>;
      const uint32_t pipeline_max_size = Selector::max_value_size();
      // Pipeline lookup kernel only supports "bucket_size = 128".
      if (options_.max_bucket_size == 128 && value_size <= pipeline_max_size) {
        LookupKernelParams<key_type, value_type, score_type> lookupParams(
            table_->buckets, table_->buckets_num, static_cast<uint32_t>(dim()),
            keys, values, scores, founds, n);
        Selector::select_kernel(lookupParams, stream);
      } else {
        using Selector =
            SelectLookupKernelWithIO<key_type, value_type, score_type>;
        static thread_local int step_counter = 0;
        static thread_local float load_factor = 0.0;

        if (((step_counter++) % kernel_select_interval_) == 0) {
          load_factor = fast_load_factor(0, stream, false);
        }
        Selector::execute_kernel(load_factor, options_.block_size,
                                 options_.max_bucket_size, table_->buckets_num,
                                 options_.dim, stream, n, d_table_,
                                 table_->buckets, keys, values, scores, founds);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, uint8_t> mv(n, n, n, n,
                                                                  d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto src = get_vector<0>(mv, temp_storage);
      auto dst_offset = get_vector<1>(mv, temp_storage);
      auto src_sorted = get_vector<2>(mv, temp_storage);
      auto dst_offset_sorted = get_vector<3>(mv, temp_storage);
      auto d_sort_storage = get_vector<4>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(src, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          options_.max_bucket_size >= MinBucketCapacityFilter;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128U;

        tlp_lookup_kernel_hybrid<key_type, value_type, score_type>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_num, options_.max_bucket_size,
                options_.dim, keys, src, scores, dst_offset, founds, n);
      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        lookup_kernel<key_type, value_type, score_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, table_->buckets, options_.max_bucket_size,
                table_->buckets_num, options_.dim, keys, src, scores, founds,
                dst_offset, N);
      }

      if (values != nullptr) {
        sortOp.sort(n, reinterpret_cast<uintptr_t*>(src),
                    reinterpret_cast<uintptr_t*>(src_sorted), dst_offset,
                    dst_offset_sorted, stream);

        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(src_sorted, values, founds,
                                                   dst_offset_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When the searched keys are not hit, missed keys/indices/size can be
   * obtained.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param missed_keys The missed keys to search on GPU-accessible memory with
   * shape (n).
   * @param missed_indices The missed indices to search on GPU-accessible memory
   * with shape (n).
   * @param missed_size The size of `missed_keys` and `missed_indices`.
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            key_type* missed_keys,                    // (n)
            int* missed_indices,                      // (n)
            int* missed_size,                         // scalar
            score_type* scores = nullptr,             // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    CUDA_CHECK(cudaMemsetAsync(missed_size, 0, sizeof(*missed_size), stream));

    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    const uint32_t value_size = options_.dim * sizeof(V);

    if (is_fast_mode()) {
      using Selector = SelectPipelineLookupKernelWithIO<key_type, value_type,
                                                        score_type, ArchTag>;
      const uint32_t pipeline_max_size = Selector::max_value_size();
      // Pipeline lookup kernel only supports "bucket_size = 128".
      if (options_.max_bucket_size == 128 && value_size <= pipeline_max_size) {
        LookupKernelParamsV2<key_type, value_type, score_type> lookupParams(
            table_->buckets, table_->buckets_num, static_cast<uint32_t>(dim()),
            keys, values, scores, missed_keys, missed_indices, missed_size, n);
        Selector::select_kernel(lookupParams, stream);
      } else {
        using Selector =
            SelectLookupKernelWithIOV2<key_type, value_type, score_type>;
        static thread_local int step_counter = 0;
        static thread_local float load_factor = 0.0;

        if (((step_counter++) % kernel_select_interval_) == 0) {
          load_factor = fast_load_factor(0, stream, false);
        }
        Selector::execute_kernel(load_factor, options_.block_size,
                                 options_.max_bucket_size, table_->buckets_num,
                                 options_.dim, stream, n, d_table_,
                                 table_->buckets, keys, values, scores,
                                 missed_keys, missed_indices, missed_size);
      }
    } else {
      auto sortOp = SortPairOp<uintptr_t, int>();
      auto d_sort_bytes = sortOp.get_storage_bytes(n, stream);

      MultiVector<value_type*, int, value_type*, int, uint8_t> mv(n, n, n, n,
                                                                  d_sort_bytes);
      const size_type dev_ws_size = mv.total_size();
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto temp_storage = dev_ws.get<uint8_t*>(0);
      auto src = get_vector<0>(mv, temp_storage);
      auto dst_offset = get_vector<1>(mv, temp_storage);
      auto src_sorted = get_vector<2>(mv, temp_storage);
      auto dst_offset_sorted = get_vector<3>(mv, temp_storage);
      auto d_sort_storage = get_vector<4>(mv, temp_storage);
      sortOp.set_storage(reinterpret_cast<void*>(d_sort_storage));

      CUDA_CHECK(cudaMemsetAsync(src, 0, dev_ws_size, stream));

      constexpr uint32_t MinBucketCapacityFilter =
          sizeof(VecD_Load) / sizeof(D);

      bool filter_condition =
          options_.max_bucket_size >= MinBucketCapacityFilter;

      if (filter_condition) {
        constexpr uint32_t BLOCK_SIZE = 128U;

        tlp_lookup_kernel_hybrid<key_type, value_type, score_type>
            <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_->buckets, table_->buckets_num, options_.max_bucket_size,
                options_.dim, keys, src, scores, dst_offset, missed_keys,
                missed_indices, missed_size, n);
      } else {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        lookup_kernel<key_type, value_type, score_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, table_->buckets, options_.max_bucket_size,
                table_->buckets_num, options_.dim, keys, src, scores,
                missed_keys, missed_indices, missed_size, dst_offset, N);
      }

      if (values != nullptr) {
        sortOp.sort(n, reinterpret_cast<uintptr_t*>(src),
                    reinterpret_cast<uintptr_t*>(src_sorted), dst_offset,
                    dst_offset_sorted, stream);

        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_kernel<key_type, value_type, score_type>
            <<<grid_size, block_size, 0, stream>>>(src_sorted, values,
                                                   dst_offset_sorted, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type** values,                      // (n)
            bool* founds,                             // (n)
            score_type* scores = nullptr,             // (n)
            cudaStream_t stream = 0, bool unique_key = true) const {
    if (n == 0) {
      return;
    }

    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);
    if (unique_key && options_.max_bucket_size >= MinBucketCapacityFilter) {
      constexpr uint32_t BLOCK_SIZE = 128U;
      tlp_lookup_ptr_kernel_with_filter<key_type, value_type, score_type,
                                        evict_strategy>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_->buckets, table_->buckets_num, options_.max_bucket_size,
              options_.dim, keys, values, scores, founds, n, false,
              global_epoch_);
    } else {
      using Selector = SelectLookupPtrKernel<key_type, value_type, score_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }

      Selector::execute_kernel(load_factor, options_.block_size,
                               options_.max_bucket_size, table_->buckets_num,
                               options_.dim, stream, n, d_table_,
                               table_->buckets, keys, values, scores, founds);
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values, and will update the scores.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find_and_update(const size_type n, const key_type* keys,  // (n)
                       value_type** values,                      // (n)
                       bool* founds,                             // (n)
                       score_type* scores = nullptr,             // (n)
                       cudaStream_t stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    check_evict_strategy(scores);

    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);
    if (unique_key && options_.max_bucket_size >= MinBucketCapacityFilter) {
      constexpr uint32_t BLOCK_SIZE = 128U;
      tlp_lookup_ptr_kernel_with_filter<key_type, value_type, score_type,
                                        evict_strategy>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_->buckets, table_->buckets_num, options_.max_bucket_size,
              options_.dim, keys, values, scores, founds, n, true,
              global_epoch_);
    } else {
      throw std::runtime_error(
          "Not support update score when keys are not unique or bucket "
          "capacity is small.");
    }

    CudaCheckError();
  }

  /**
   * @brief Checks if there are elements with key equivalent to `keys` in the
   * table.
   *
   * @param n The number of `keys` to check.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param founds The result that indicates if the keys are found, and should
   * be allocated by caller on GPU-accessible memory with shape (n).
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void contains(const size_type n, const key_type* keys,  // (n)
                bool* founds,                             // (n)
                cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    if (options_.max_bucket_size == 128) {
      // Pipeline lookup kernel only supports "bucket_size = 128".
      using Selector = SelectPipelineContainsKernel<key_type, value_type,
                                                    score_type, ArchTag>;
      ContainsKernelParams<key_type, value_type, score_type> containsParams(
          table_->buckets, table_->buckets_num, static_cast<uint32_t>(dim()),
          keys, founds, n);
      Selector::select_kernel(containsParams, stream);
    } else {
      using Selector = SelectContainsKernel<key_type, value_type, score_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      Selector::execute_kernel(load_factor, options_.block_size,
                               options_.max_bucket_size, table_->buckets_num,
                               options_.dim, stream, n, d_table_,
                               table_->buckets, keys, founds);
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

    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    {
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, value_type, score_type, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              d_table_, keys, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }

    CudaCheckError();
    return;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * hash table.
   *
   * @tparam PredFunctor The predicate template <typename K, typename S>
   * function with operator signature (bool*)(const K&, const S&, const K&,
   * const threshold) that returns `true` if the element should be erased. The
   * value for @p pred should be a function with type `Pred` defined like the
   * following example:
   *
   *    ```
   *    template <class K, class S>
   *    struct EraseIfPredFunctor {
   *      __forceinline__ __device__ bool operator()(const K& key,
   *                                                 S& score,
   *                                                 const K& pattern,
   *                                                 const S& threshold) {
   *        return ((key & 0x1 == pattern) && (score < threshold));
   *      }
   *    };
   *    ```
   *
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with
   * score_type type.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   *
   */
  template <template <typename, typename> class PredFunctor>
  size_type erase_if(const key_type& pattern, const score_type& threshold,
                     cudaStream_t stream = 0) {
    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_count{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

    {
      const size_t block_size = options_.block_size;
      const size_t N = table_->buckets_num;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, value_type, score_type, PredFunctor>
          <<<grid_size, block_size, 0, stream>>>(
              d_table_, pattern, threshold, d_count, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, N);
    }

    size_type count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaCheckError();
    return count;
  }

  /**
   * @brief Erase the key-value-score tuples which match @tparam PredFunctor.
   * @param pred A functor with template <K, V, S> defined an operator with
   * signature:  __device__ (bool*)(const K&, const V*, const S&, const
   * cg::thread_block_tile<GroupSize>&).
   *  @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   */

  template <typename PredFunctor>
  size_type erase_if_v2(PredFunctor& pred, cudaStream_t stream = 0) {
    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_count{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

    {
      /// Search_length should be multiple of GroupSize for communication.
      uint64_t dim = table_->dim;
      uint64_t n = options_.max_capacity;
      auto kernel = [&] {
        if (dim >= 32 && n % 32 == 0) {
          return remove_kernel_v2<key_type, value_type, score_type, PredFunctor,
                                  32>;
        } else if (dim >= 16 && n % 16 == 0) {
          return remove_kernel_v2<key_type, value_type, score_type, PredFunctor,
                                  16>;
        } else if (dim >= 8 && n % 8 == 0) {
          return remove_kernel_v2<key_type, value_type, score_type, PredFunctor,
                                  8>;
        }
        return remove_kernel_v2<key_type, value_type, score_type, PredFunctor,
                                1>;
      }();
      uint64_t block_size = 128UL;
      uint64_t grid_size =
          std::min(sm_cnt_ * max_threads_per_block_ / block_size,
                   SAFE_GET_GRID_SIZE(n, block_size));
      kernel<<<grid_size, block_size, 0, stream>>>(
          n, 0, pred, table_->buckets, table_->buckets_size,
          table_->bucket_max_size, table_->dim, d_count);
    }

    size_type count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaCheckError();
    return count;
  }

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  void clear(cudaStream_t stream = 0) {
    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    const size_t block_size = options_.block_size;
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

    clear_kernel<key_type, value_type, score_type>
        <<<grid_size, block_size, 0, stream>>>(d_table_, table_->buckets, N);

    CudaCheckError();
  }

 public:
  /**
   * @brief Exports a certain number of the key-value-score tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter Accumulates amount of successfully exported values.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
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
                    size_type* d_counter,          // (1)
                    key_type* keys,                // (n)
                    value_type* values,            // (n, DIM)
                    score_type* scores = nullptr,  // (n)
                    cudaStream_t stream = 0) const {
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
    if (offset >= table_->capacity) {
      return;
    }
    n = std::min(table_->capacity - offset, n);

    size_type shared_size;
    size_type block_size;
    std::tie(shared_size, block_size) =
        dump_kernel_shared_memory_size<K, V, S>(shared_mem_size_);

    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    dump_kernel<key_type, value_type, score_type>
        <<<grid_size, block_size, shared_size, stream>>>(
            d_table_, table_->buckets, keys, values, scores, offset, n,
            d_counter);

    CudaCheckError();
  }

  size_type export_batch(const size_type n, const size_type offset,
                         key_type* keys,                // (n)
                         value_type* values,            // (n, DIM)
                         score_type* scores = nullptr,  // (n)
                         cudaStream_t stream = 0) const {
    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_counter{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
    export_batch(n, offset, d_counter, keys, values, scores, stream);

    size_type counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&counter, d_counter, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counter;
  }

  /**
   * @brief Exports a certain number of the key-value-score tuples which match
   *
   * @tparam PredFunctor A functor with template <K, S> defined an operator
   * with signature:  __device__ (bool*)(const K&, S&, const K&, const S&).
   * specified condition from the hash table.
   *
   * @param n The maximum number of exported pairs.
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class S>
   *    struct ExportIfPredFunctor {
   *      __forceinline__ __device__ bool operator()(const K& key,
   *                                                 S& score,
   *                                                 const K& pattern,
   *                                                 const S& threshold) {
   *        return score >= threshold;
   *      }
   *    };
   *    ```
   *
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with
   * score_type type.
   * @param offset The position of the key to search.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
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
  template <template <typename, typename> class PredFunctor>
  void export_batch_if(const key_type& pattern, const score_type& threshold,
                       size_type n, const size_type offset,
                       size_type* d_counter,
                       key_type* keys,                // (n)
                       value_type* values,            // (n, DIM)
                       score_type* scores = nullptr,  // (n)
                       cudaStream_t stream = 0) const {
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

    if (offset >= table_->capacity) {
      return;
    }
    n = std::min(table_->capacity - offset, n);
    if (n == 0) {
      return;
    }

    bool match_fast_cond = true;
    const size_t value_size = sizeof(V) * dim();
    auto check_tile_size = [&](int tile_size) {
      return options_.max_bucket_size % tile_size == 0 &&
             options_.max_bucket_size >= tile_size && offset % tile_size == 0 &&
             n % tile_size == 0;
    };
    auto select_tile_size = [&](auto vec) {
      using VecV = decltype(vec);
      size_t vec_dim = value_size / sizeof(VecV);
      if (vec_dim >= 32 && check_tile_size(32)) {
        return dump_kernel_v2<key_type, value_type, score_type, VecV,
                              PredFunctor, 32>;
      } else if (vec_dim >= 16 && check_tile_size(16)) {
        return dump_kernel_v2<key_type, value_type, score_type, VecV,
                              PredFunctor, 16>;
      } else if (vec_dim >= 8 && check_tile_size(8)) {
        return dump_kernel_v2<key_type, value_type, score_type, VecV,
                              PredFunctor, 8>;
      }
      match_fast_cond = false;
      return dump_kernel<key_type, value_type, score_type, PredFunctor>;
    };
    auto kernel = [&] {
      if (value_size >= sizeof(float4) * 8 &&
          value_size % sizeof(float4) == 0) {
        return select_tile_size(float4{});
      } else if (value_size >= sizeof(float2) * 8 &&
                 value_size % sizeof(float2) == 0) {
        return select_tile_size(float2{});
      } else if (value_size >= sizeof(float) * 8 &&
                 value_size % sizeof(float) == 0) {
        return select_tile_size(float{});
      } else if (value_size >= sizeof(uint16_t) * 8 &&
                 value_size % sizeof(uint16_t) == 0) {
        return select_tile_size(uint16_t{});
      }
      return select_tile_size(V{});
    }();
    size_t grid_size = 0, block_size = 0, shared_size = 0;
    if (match_fast_cond) {
      block_size = options_.block_size;
      grid_size = std::min(sm_cnt_ * max_threads_per_block_ / block_size,
                           SAFE_GET_GRID_SIZE(n, block_size));
    } else {
      const size_t score_size = scores ? sizeof(score_type) : 0;
      const size_t kvm_size =
          sizeof(key_type) + sizeof(value_type) * dim() + score_size;
      block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
      MERLIN_CHECK(
          block_size > 0,
          "[HierarchicalKV] block_size <= 0, the K-V-S size may be too large!");

      shared_size = kvm_size * block_size;
      grid_size = SAFE_GET_GRID_SIZE(n, block_size);
    }
    kernel<<<grid_size, block_size, shared_size, stream>>>(
        d_table_, table_->buckets, pattern, threshold, keys, values, scores,
        offset, n, d_counter);

    CudaCheckError();
  }

  /**
   * @brief Exports a certain number of key-value-score tuples that match a
   * given predicate.
   *
   * @tparam PredFunctor A functor type with a template signature `<K, V, S>`.
   * It should define an operator with the signature:
   * `__device__ bool operator()(const K&, const V*, const S&,
   * cg::thread_block_tile<GroupSize>&)`.
   *
   * @param pred A functor of type `PredFunctor` that defines the predicate for
   * filtering tuples.
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter The number of elements dumped which is on device.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape (n,
   * DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return void
   *
   */

  template <typename PredFunctor>
  void export_batch_if_v2(PredFunctor& pred, size_type n,
                          const size_type offset, size_type* d_counter,
                          key_type* keys,                // (n)
                          value_type* values,            // (n, DIM)
                          score_type* scores = nullptr,  // (n)
                          cudaStream_t stream = 0) const {
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

    if (offset >= table_->capacity) {
      return;
    }
    n = std::min(table_->capacity - offset, n);
    if (n == 0) {
      return;
    }

    /// Search_length should be multiple of GroupSize for communication.
    uint64_t dim = table_->dim;
    auto kernel = [&] {
      if (dim >= 32 && n % 32 == 0) {
        return dump_kernel<key_type, value_type, score_type, PredFunctor, 32>;
      } else if (dim >= 16 && n % 16 == 0) {
        return dump_kernel<key_type, value_type, score_type, PredFunctor, 16>;
      } else if (dim >= 8 && n % 8 == 0) {
        return dump_kernel<key_type, value_type, score_type, PredFunctor, 8>;
      }
      return dump_kernel<key_type, value_type, score_type, PredFunctor, 1>;
    }();
    uint64_t block_size = 128UL;
    uint64_t grid_size = std::min(sm_cnt_ * max_threads_per_block_ / block_size,
                                  SAFE_GET_GRID_SIZE(n, block_size));
    kernel<<<grid_size, block_size, 0, stream>>>(
        n, offset, pred, table_->buckets, table_->bucket_max_size, dim, keys,
        values, scores, d_counter);

    CudaCheckError();
  }

  /**
   * @brief Applies the given function to items in the range [first, last) in
   * the table.
   *
   * @tparam ExecutionFunc A functor type with a template signature `<K, V, S>`.
   * It should define an operator with the signature:
   * `__device__ void operator()(const K&, V*, S*,
   * cg::thread_block_tile<GroupSize>&)`.
   *
   * @param first The first element to which the function object will be
   * applied.
   * @param last The last element(excluding) to which the function object will
   * be applied.
   * @param f A functor of type `ExecutionFunc` that defines the predicate for
   * filtering tuples. signature:  __device__ (bool*)(const K&, const V*, const
   * S&, const cg::tiled_partition<GroupSize>&).
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return void
   *
   */

  template <typename ExecutionFunc>
  void for_each(const size_type first, const size_type last, ExecutionFunc& f,
                cudaStream_t stream = 0) {
    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    if (first >= table_->capacity or last > table_->capacity or first >= last) {
      return;
    }
    uint64_t n = last - first;

    /// Search_length should be multiple of GroupSize for communication.
    uint64_t dim = table_->dim;
    auto kernel = [&] {
      if (dim >= 32 && n % 32 == 0) {
        return traverse_kernel<key_type, value_type, score_type, ExecutionFunc,
                               32>;
      } else if (dim >= 16 && n % 16 == 0) {
        return traverse_kernel<key_type, value_type, score_type, ExecutionFunc,
                               16>;
      } else if (dim >= 8 && n % 8 == 0) {
        return traverse_kernel<key_type, value_type, score_type, ExecutionFunc,
                               8>;
      }
      return traverse_kernel<key_type, value_type, score_type, ExecutionFunc,
                             1>;
    }();
    uint64_t block_size = 128UL;
    uint64_t grid_size = std::min(sm_cnt_ * max_threads_per_block_ / block_size,
                                  SAFE_GET_GRID_SIZE(n, block_size));
    kernel<<<grid_size, block_size, 0, stream>>>(n, first, f, table_->buckets,
                                                 table_->bucket_max_size, dim);

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
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }

    const size_type N = table_->buckets_num;

    auto sumOp = SumOp<int, int64_t>();
    auto d_sum_bytes = sumOp.get_storage_bytes(N, stream);

    MultiVector<int64_t, uint8_t> mv(1, d_sum_bytes);
    const size_type dev_ws_size = mv.total_size();
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto temp_storage = dev_ws.get<uint8_t*>(0);
    auto d_total_size = get_vector<0>(mv, temp_storage);
    auto d_sum_storage = get_vector<1>(mv, temp_storage);
    sumOp.set_storage(reinterpret_cast<void*>(d_sum_storage));
    sumOp.sum(N, table_->buckets_size, d_total_size, stream);

    int64_t h_total_size = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_total_size, d_total_size, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaCheckError();
    return static_cast<size_type>(h_total_size);
  }

  /**
   * @brief Returns the number of keys if meet PredFunctor.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return The table size match condiction of PredFunctor.
   */
  template <template <typename, typename> class PredFunctor>
  void size_if(const key_type& pattern, const score_type& threshold,
               size_type* d_counter, cudaStream_t stream = 0) const {
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));

    size_t grid_size = SAFE_GET_GRID_SIZE(capacity(), options_.block_size);
    grid_size = std::min(grid_size,
                         static_cast<size_t>(sm_cnt_ * max_threads_per_block_ /
                                             options_.block_size));
    size_if_kernel<key_type, value_type, score_type, PredFunctor>
        <<<grid_size, options_.block_size, 0, stream>>>(
            d_table_, table_->buckets, pattern, threshold, d_counter);
    CudaCheckError();
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
      reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
      return;
    }

    {
      std::unique_ptr<update_read_lock> lock_ptr;
      if (options_.api_lock) {
        lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
      }

      // Once we have exclusive access, make sure that pending GPU calls have
      // been processed.
      CUDA_CHECK(cudaDeviceSynchronize());

      while (capacity() < new_capacity &&
             capacity() * 2 <= options_.max_capacity) {
        double_capacity<key_type, value_type, score_type>(&table_, allocator_);
        CUDA_CHECK(cudaDeviceSynchronize());
        sync_table_configuration();

        const size_t block_size = options_.block_size;
        const size_t N = TILE_SIZE * table_->buckets_num / 2;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        rehash_kernel_for_fast_mode<key_type, value_type, score_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(d_table_, table_->buckets,
                                                   N);
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
   * @brief Set max_capacity of the table.
   *
   * @param new_max_capacity The new expecting max_capacity. It must be power
   * of 2. Otherwise it will raise an error.
   */
  void set_max_capacity(size_type new_max_capacity) {
    if (!is_power(2, new_max_capacity)) {
      throw std::invalid_argument(
          "None power-of-2 new_max_capacity is not supported.");
    }

    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_);
    }

    if (new_max_capacity < capacity()) {
      return;
    }
    if (reach_max_capacity_) {
      reach_max_capacity_ = false;
    }
    options_.max_capacity = new_max_capacity;
  }

  /**
   * @brief Returns the dimension of the vectors.
   *
   * @return The dimension of the vectors.
   */
  size_type dim() const noexcept { return options_.dim; }

  /**
   * @brief Returns The length of each bucket.
   *
   * @return The length of each bucket.
   */
  size_type max_bucket_size() const noexcept {
    return options_.max_bucket_size;
  }

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the table.
   */
  size_type bucket_count() const noexcept { return table_->buckets_num; }

  /**
   * @brief Save keys, vectors, scores in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param max_workspace_size Saving is conducted in chunks. This value denotes
   * the maximum amount of temporary memory to use when dumping the table.
   * Larger values *can* lead to higher performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  size_type save(BaseKVFile<K, V, S>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 cudaStream_t stream = 0) const {
    const size_type tuple_size{sizeof(key_type) + sizeof(score_type) +
                               sizeof(value_type) * dim()};
    MERLIN_CHECK(max_workspace_size >= tuple_size,
                 "[HierarchicalKV] max_workspace_size is smaller than a single "
                 "`key + scoredata + value` tuple! Please set a larger value!");

    size_type shared_size;
    size_type block_size;
    std::tie(shared_size, block_size) =
        dump_kernel_shared_memory_size<K, V, S>(shared_mem_size_);

    // Request exclusive access (to make sure capacity won't change anymore).
    std::unique_ptr<update_read_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<update_read_lock>(mutex_, stream);
    }

    const size_type total_size{capacity()};
    const size_type n{std::min(max_workspace_size / tuple_size, total_size)};
    const size_type grid_size{SAFE_GET_GRID_SIZE(n, block_size)};

    // Grab temporary device and host memory.
    const size_type host_ws_size{n * tuple_size};
    auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_scores{reinterpret_cast<score_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_scores + n)};

    const size_type dev_ws_size{sizeof(size_type) + host_ws_size};
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto d_count{dev_ws.get<size_type*>(0)};
    auto d_keys{reinterpret_cast<key_type*>(d_count + 1)};
    auto d_scores{reinterpret_cast<score_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_scores + n)};

    // Step through table, dumping contents in batches.
    size_type total_count{0};
    for (size_type i{0}; i < total_size; i += n) {
      // Dump the next batch to workspace, and then write it to the file.
      CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

      dump_kernel<key_type, value_type, score_type>
          <<<grid_size, block_size, shared_size, stream>>>(
              d_table_, table_->buckets, d_keys, d_values, d_scores, i,
              std::min(total_size - i, n), d_count);

      size_type count;
      CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      if (count == n) {
        CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, host_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(key_type) * count,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_scores, d_scores,
                                   sizeof(score_type) * count,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, d_values,
                                   sizeof(value_type) * dim() * count,
                                   cudaMemcpyDeviceToHost, stream));
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      file->write(count, dim(), h_keys, h_values, h_scores);
      total_count += count;
    }

    return total_count;
  }

  /**
   * @brief Load keys, vectors, scores from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param max_workspace_size Loading is conducted in chunks. This value
   * denotes the maximum size of such chunks. Larger values *can* lead to higher
   * performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(BaseKVFile<K, V, S>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 cudaStream_t stream = 0) {
    const size_type tuple_size{sizeof(key_type) + sizeof(score_type) +
                               sizeof(value_type) * dim()};
    MERLIN_CHECK(max_workspace_size >= tuple_size,
                 "[HierarchicalKV] max_workspace_size is smaller than a single "
                 "`key + score + value` tuple! Please set a larger value!");

    const size_type n{max_workspace_size / tuple_size};
    const size_type ws_size{n * tuple_size};

    // Grab enough host memory to hold batch data.
    auto host_ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_scores{reinterpret_cast<score_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_scores + n)};

    // Attempt a first read.
    size_type count{file->read(n, dim(), h_keys, h_values, h_scores)};
    if (count == 0) {
      return 0;
    }

    // Grab equal amount of device memory as temporary storage.
    auto dev_ws{dev_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto d_keys{dev_ws.get<key_type*>(0)};
    auto d_scores{reinterpret_cast<score_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_scores + n)};

    size_type total_count{0};
    do {
      if (count == n) {
        CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, ws_size,
                                   cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, sizeof(key_type) * count,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_scores, h_scores,
                                   sizeof(score_type) * count,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_values, h_values,
                                   sizeof(value_type) * dim() * count,
                                   cudaMemcpyHostToDevice, stream));
      }

      set_global_epoch(static_cast<S>(IGNORED_GLOBAL_EPOCH));
      insert_or_assign(count, d_keys, d_values, d_scores, stream, true, true);
      total_count += count;

      // Read next batch.
      CUDA_CHECK(cudaStreamSynchronize(stream));
      count = file->read(n, dim(), h_keys, h_values, h_scores);
    } while (count > 0);

    return total_count;
  }

  void set_global_epoch(const uint64_t epoch) { global_epoch_ = epoch; }

 private:
  bool is_power(size_t base, size_t n) {
    if (base < 2) {
      throw std::invalid_argument("is_power with zero base.");
    }
    while (n > 1) {
      if (n % base != 0) {
        return false;
      }
      n /= base;
    }
    return true;
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
   * @param need_lock If lock is needed.
   *
   * @return The evaluated load factor
   */
  inline float fast_load_factor(const size_type delta = 0,
                                cudaStream_t stream = 0,
                                const bool need_lock = true) const {
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr =
          std::make_unique<read_shared_lock>(mutex_, std::defer_lock, stream);
      if (need_lock) {
        lock_ptr->lock();
      }
    }

    size_t N = std::min(table_->buckets_num, 1024UL);

    auto sumOp = SumOp<int, int64_t>();
    auto d_sum_bytes = sumOp.get_storage_bytes(N, stream);

    MultiVector<int64_t, uint8_t> mv(1, d_sum_bytes);
    const size_type dev_ws_size = mv.total_size();
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto temp_storage = dev_ws.get<uint8_t*>(0);
    auto d_total_size = get_vector<0>(mv, temp_storage);
    auto d_sum_storage = get_vector<1>(mv, temp_storage);
    sumOp.set_storage(reinterpret_cast<void*>(d_sum_storage));
    sumOp.sum(N, table_->buckets_size, d_total_size, stream);

    int64_t h_total_size = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_total_size, d_total_size, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return static_cast<float>((delta * 1.0) / (capacity() * 1.0) +
                              (h_total_size * 1.0) /
                                  (options_.max_bucket_size * N * 1.0));
  }

  inline void check_evict_strategy(const score_type* scores) {
    if (evict_strategy == EvictStrategy::kLru ||
        evict_strategy == EvictStrategy::kEpochLru) {
      MERLIN_CHECK(scores == nullptr,
                   "the scores should not be specified when running on "
                   "LRU or Epoch LRU mode.");
    }

    if (evict_strategy == EvictStrategy::kLfu ||
        evict_strategy == EvictStrategy::kEpochLfu) {
      MERLIN_CHECK(scores != nullptr,
                   "the scores should be specified when running on "
                   "LFU or Epoch LFU mode.");
    }

    if (evict_strategy == EvictStrategy::kCustomized) {
      MERLIN_CHECK(scores != nullptr,
                   "the scores should be specified when running on "
                   "customized mode.");
    }

    if ((evict_strategy == EvictStrategy::kEpochLru ||
         evict_strategy == EvictStrategy::kEpochLfu)) {
      MERLIN_CHECK(
          global_epoch_ != static_cast<S>(IGNORED_GLOBAL_EPOCH),
          "the global_epoch is invalid and should be assigned by calling "
          "`set_global_epoch` when running on "
          "Epoch LRU or Epoch LFU mode.");
    }
  }

  /**
   * @brief Synchronize the TableCore struct to replicas.
   *
   * @note For performance consideration, synchronize the TableCore struct to
   * its replicas in constant memory and device memory when it's changed.
   */
  inline void sync_table_configuration() {
    CUDA_CHECK(
        cudaMemcpy(d_table_, table_, sizeof(TableCore), cudaMemcpyDefault));
  }

 public:
  // Expose device buckets and layout for read-only lookup kernels
  inline nv::merlin::Bucket<K, V, S>* device_buckets() const {
    return table_ ? table_->buckets : nullptr;
  }
  inline size_t device_bucket_count() const {
    return table_ ? table_->buckets_num : 0;
  }
  inline size_t device_bucket_max_size() const {
    return table_ ? table_->bucket_max_size : 0;
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  TableCore* d_table_ = nullptr;
  size_t shared_mem_size_ = 0;
  int sm_cnt_ = 0;
  int max_threads_per_block_ = 0;
  std::atomic_bool reach_max_capacity_{false};
  bool initialized_ = false;
  mutable group_shared_mutex mutex_;
  const unsigned int kernel_select_interval_ = 7;
  std::unique_ptr<DeviceMemoryPool> dev_mem_pool_;
  std::unique_ptr<HostMemoryPool> host_mem_pool_;
  allocator_type* allocator_;
  ThrustAllocator<uint8_t> thrust_allocator_;
  bool default_allocator_ = true;
  std::atomic<uint64_t> global_epoch_{
      static_cast<uint64_t>(IGNORED_GLOBAL_EPOCH)};
};

}  // namespace merlin
}  // namespace nv
