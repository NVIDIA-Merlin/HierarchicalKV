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

#include <stddef.h>
#include <cstdint>
#include <cuda/std/semaphore>
#include "debug.hpp"

namespace nv {
namespace merlin {

/**
 * Shorthand for a Key-Value-score tuple.
 */
template <class K, class V, class S>
struct KVM {
  K key;
  V* value;
  S score;
};

// Storage size.
using byte16 = uint4;
using byte8 = uint2;
using byte4 = uint32_t;
using byte2 = uint16_t;
using byte = uint8_t;

// Digest.
using D = byte;
constexpr uint64_t DEFAULT_EMPTY_KEY = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t DEFAULT_RECLAIM_KEY = UINT64_C(0xFFFFFFFFFFFFFFFE);
constexpr uint64_t DEFAULT_LOCKED_KEY = UINT64_C(0xFFFFFFFFFFFFFFFD);

constexpr uint64_t DEFAULT_RESERVED_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr uint64_t DEFAULT_VACANT_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFE);

constexpr uint64_t MAX_SCORE = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t EMPTY_SCORE = UINT64_C(0);
constexpr uint64_t IGNORED_GLOBAL_EPOCH = UINT64_C(0xFFFFFFFFFFFFFFFF);

uint64_t EMPTY_KEY_CPU = DEFAULT_EMPTY_KEY;
__constant__ uint64_t EMPTY_KEY = DEFAULT_EMPTY_KEY;
__constant__ uint64_t RECLAIM_KEY = DEFAULT_RECLAIM_KEY;
__constant__ uint64_t LOCKED_KEY = DEFAULT_LOCKED_KEY;

__constant__ uint64_t RESERVED_KEY_MASK_1 = DEFAULT_RESERVED_KEY_MASK;
__constant__ uint64_t RESERVED_KEY_MASK_2 = DEFAULT_RESERVED_KEY_MASK;
__constant__ uint64_t VACANT_KEY_MASK_1 = DEFAULT_VACANT_KEY_MASK;
__constant__ uint64_t VACANT_KEY_MASK_2 = DEFAULT_VACANT_KEY_MASK;

constexpr int MAX_RESERVED_KEY_BIT = 62;

template <class K>
__forceinline__ __device__ bool IS_RESERVED_KEY(K key) {
  return (RESERVED_KEY_MASK_1 & key) == RESERVED_KEY_MASK_2;
}

template <class K>
__forceinline__ __device__ bool IS_VACANT_KEY(K key) {
  return (VACANT_KEY_MASK_1 & key) == VACANT_KEY_MASK_2;
}

cudaError_t init_reserved_keys(int index) {
  if (index < 1 || index > MAX_RESERVED_KEY_BIT) {
    // index = 0 is the default,
    // index = 62 is the maximum index can be set for reserved keys.
    return cudaSuccess;
  }
  uint64_t reservedKeyMask1 = ~(UINT64_C(3) << index);
  uint64_t reservedKeyMask2 = reservedKeyMask1 & ~UINT64_C(1);
  uint64_t vacantKeyMask1 = ~(UINT64_C(1) << index);
  uint64_t vacantKeyMask2 = vacantKeyMask1 & ~UINT64_C(1);

  uint64_t emptyKey = reservedKeyMask2 | (UINT64_C(3) << index);
  uint64_t reclaimKey = vacantKeyMask2;
  uint64_t lockedKey = emptyKey & ~(UINT64_C(2) << index);
  EMPTY_KEY_CPU = emptyKey;

  CUDA_CHECK(cudaMemcpyToSymbol(EMPTY_KEY, &emptyKey, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(RECLAIM_KEY, &reclaimKey, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(LOCKED_KEY, &lockedKey, sizeof(uint64_t)));

  CUDA_CHECK(cudaMemcpyToSymbol(RESERVED_KEY_MASK_1, &reservedKeyMask1,
                                sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(RESERVED_KEY_MASK_2, &reservedKeyMask2,
                                sizeof(uint64_t)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(VACANT_KEY_MASK_1, &vacantKeyMask1, sizeof(uint64_t)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(VACANT_KEY_MASK_2, &vacantKeyMask2, sizeof(uint64_t)));
  return cudaGetLastError();
}

template <class K>
using AtomicKey = cuda::atomic<K, cuda::thread_scope_device>;

template <class S>
using AtomicScore = cuda::atomic<S, cuda::thread_scope_device>;

template <class T>
using AtomicPos = cuda::atomic<T, cuda::thread_scope_device>;

template <class K, class V, class S>
struct Bucket {
  AtomicKey<K>* keys_;
  /// TODO: compute the pointer of scores and digests using bucket_max_size
  AtomicScore<S>* scores_;
  /// @brief not visible to users
  D* digests_;
  V* vectors;  // Pinned memory or HBM

  __forceinline__ __device__ D* digests(int index) const {
    return digests_ + index;
  }

  __forceinline__ __device__ AtomicKey<K>* keys(int index) const {
    return keys_ + index;
  }

  __forceinline__ __device__ AtomicScore<S>* scores(int index) const {
    return scores_ + index;
  }

  __forceinline__ __device__ K** keys_addr() {
    return reinterpret_cast<K**>(&keys_);
  }

  static __forceinline__ __device__ AtomicKey<K>* keys(K* keys,
                                                       uint32_t offset) {
    return reinterpret_cast<AtomicKey<K>*>(keys) + offset;
  }

  static __forceinline__ __device__ D* digests(K* keys,
                                               uint32_t bucket_capacity,
                                               uint32_t offset) {
    bucket_capacity = umax(bucket_capacity, 128);
    return reinterpret_cast<D*>(keys) - bucket_capacity + offset;
  }

  static __forceinline__ __device__ S* scores(K* keys, uint32_t bucket_capacity,
                                              uint32_t offset) {
    return reinterpret_cast<S*>(keys + bucket_capacity) + offset;
  }
};

template <cuda::thread_scope Scope, class T = int>
class Lock {
  mutable cuda::atomic<T, Scope> _lock;

 public:
  __device__ Lock() : _lock{1} {}

  template <typename CG>
  __forceinline__ __device__ void acquire(CG const& g,
                                          unsigned long long lane = 0) const {
    if (g.thread_rank() == lane) {
      T expected = 1;
      while (!_lock.compare_exchange_weak(expected, 2,
                                          cuda::std::memory_order_acquire)) {
        expected = 1;
      }
    }
    g.sync();
  }

  template <typename CG>
  __forceinline__ __device__ void release(CG const& g,
                                          unsigned long long lane = 0) const {
    g.sync();
    if (g.thread_rank() == lane) {
      _lock.store(1, cuda::std::memory_order_release);
    }
  }
};

using Mutex = Lock<cuda::thread_scope_device>;

template <class K, class V, class S>
struct Table {
  Bucket<K, V, S>* buckets;
  Mutex* locks;                 // mutex for write buckets
  int* buckets_size;            // size of each buckets.
  V** slices;                   // Handles of the HBM/ HMEM slices.
  size_t dim;                   // Dimension of the `vectors`.
  size_t bytes_per_slice;       // Size by byte of one slice.
  size_t num_of_memory_slices;  // Number of vectors memory slices.
  size_t capacity = 134217728;  // Initial capacity.
  size_t max_size =
      std::numeric_limits<uint64_t>::max();  // Up limit of the table capacity.
  size_t buckets_num;                        // Number of the buckets.
  size_t bucket_max_size = 128;              // Volume of each buckets.
  size_t max_hbm_for_vectors = 0;            // Max HBM allocated for vectors
  size_t remaining_hbm_for_vectors = 0;  // Remaining HBM allocated for vectors
  size_t num_of_buckets_per_alloc = 1;   // Number of buckets allocated in each
                                         // HBM allocation, must be power of 2.
  bool is_pure_hbm = true;               // unused
  bool primary = true;                   // unused
  int slots_offset = 0;                  // unused
  int slots_number = 0;                  // unused
  int device_id = 0;                     // Device id
  int tile_size;
};

template <class K, class S>
using EraseIfPredictInternal =
    bool (*)(const K& key,       ///< iterated key in table
             S& score,           ///< iterated score in table
             const K& pattern,   ///< input key from caller
             const S& threshold  ///< input score from caller
    );

/**
 * An abstract class provides interface between the nv::merlin::HashTable
 * and a file, which enables the table to save to the file or load from
 * the file, by overriding the `read` and `write` method.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam S The data type for `score`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class S>
class BaseKVFile {
 public:
  virtual ~BaseKVFile() {}

  /**
   * Read from file and fill into the keys, values, and scores buffer.
   * When calling save/load method from table, it can assume that the
   * received buffer of keys, vectors, and scores are automatically
   * pre-allocated.
   *
   * @param n The number of KV pairs expect to read. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The pointer to received buffer for keys.
   * @param vectors The pointer to received buffer for vectors.
   * @param scores The pointer to received buffer for scores.
   *
   * @return Number of KV pairs have been successfully read.
   */
  virtual size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
                      S* scores) = 0;

  /**
   * Write keys, values, scores from table to the file. It defines
   * an abstract method to get batch of KV pairs and write them into
   * file.
   *
   * @param n The number of KV pairs to be written. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The keys will be written to file.
   * @param vectors The vectors of values will be written to file.
   * @param scores The scores will be written to file.
   *
   * @return Number of KV pairs have been successfully written.
   */
  virtual size_t write(const size_t n, const size_t dim, const K* keys,
                       const V* vectors, const S* scores) = 0;
};

enum class OccupyResult {
  INITIAL,         ///< Initial status
  CONTINUE,        ///< Insert did not succeed, continue trying to insert
  OCCUPIED_EMPTY,  ///< New pair inserted successfully
  OCCUPIED_RECLAIMED,
  DUPLICATE,  ///< Insert did not succeed, key is already present
  EVICT,      ///< Insert succeeded by evicting one key with minimum score.
  REFUSED,    ///< Insert did not succeed, insert score is too low.
  ILLEGAL,    ///< Illegal state, and don't need to do anything.
};

enum class OverrideResult {
  INITIAL,   ///< Initial status
  CONTINUE,  ///< Override did not succeed, continue trying to override
  SUCCESS,   ///< Override successfully
  REFUSED,   ///< Override is refused.
};

struct Sm70 {
  static int const kComputeCapability = 70;
};
struct Sm72 {
  static int const kComputeCapability = 72;
};
struct Sm75 {
  static int const kComputeCapability = 75;
};
struct Sm80 {
  static int const kComputeCapability = 80;
};
struct Sm86 {
  static int const kComputeCapability = 86;
};

struct Sm90 {
  static int const kComputeCapability = 90;
};

/* This struct is mainly for keeping the code readable, it should be strictly
 * consistent with `EvictStrategy::EvictStrategyEnum`.
 */
struct EvictStrategyInternal {
  constexpr static int kLru = 0;         ///< LRU mode.
  constexpr static int kLfu = 1;         ///< LFU mode.
  constexpr static int kEpochLru = 2;    ///< Epoch + LRU mode.
  constexpr static int kEpochLfu = 3;    ///< Epoch + LFU mode.
  constexpr static int kCustomized = 4;  ///< Customized mode.
};

}  // namespace merlin
}  // namespace nv
