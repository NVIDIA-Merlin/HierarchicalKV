/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>
#include "merlin/core_kernels/kernel_utils.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_hashtable_base.hpp"

namespace nv {
namespace merlin {
namespace device {

/**
 * @brief Lightweight device-side view of a hash table.
 *
 * This view exposes only the metadata and bucket pointer needed by device
 * read-only APIs.
 */
template <typename K, typename V, typename S>
struct HashTableDeviceView {
  using base_type = HashTableBase<K, V, S>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using score_type = typename base_type::score_type;
  using bucket_type = typename base_type::bucket_type;

  // Read-only view. Non-const to match find_without_lock_readonly_no_sync.
  bucket_type* buckets;
  size_t bucket_count;
  size_t bucket_max_size;
};

/**
 * @brief Compute bucket index and aligned start for a key.
 *
 * @tparam K Key type.
 * @tparam V Value type.
 * @tparam S Score type.
 * @tparam TILE_SIZE Alignment granularity for the start position.
 * @param view Device-side view of the table.
 * @param key Key to hash and locate.
 * @param bucket_idx Output bucket index.
 * @param aligned_start Output aligned start position in the bucket.
 * @return Always returns true. The caller must ensure view metadata is valid.
 */
template <typename K, typename V, typename S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ bool compute_bucket_index_and_aligned_start(
    const HashTableDeviceView<K, V, S>& view,
    const typename HashTableDeviceView<K, V, S>::key_type key,
    uint32_t* bucket_idx, uint32_t* aligned_start) {
  static_assert((TILE_SIZE & (TILE_SIZE - 1)) == 0,
                "TILE_SIZE must be power of two.");
  // bucket_max_size is guaranteed to be power-of-two.
  const uint32_t bucket_capacity = static_cast<uint32_t>(view.bucket_max_size);
  const uint64_t total_slots =
      static_cast<uint64_t>(view.bucket_count) * bucket_capacity;
  const uint64_t hashed_key = static_cast<uint64_t>(Murmur3HashDevice(key));
  const uint64_t global_idx = hashed_key % total_slots;
  const uint32_t bucket_shift =
      static_cast<uint32_t>(__ffs(bucket_capacity) - 1);
  *bucket_idx = static_cast<uint32_t>(global_idx >> bucket_shift);
  uint32_t start_idx =
      static_cast<uint32_t>(global_idx) & (bucket_capacity - 1);
  *aligned_start = align_to<TILE_SIZE>(start_idx);
  return true;
}

/**
 * @brief Read-only lookup without tile synchronization in a known bucket.
 *
 * @tparam K Key type.
 * @tparam V Value type.
 * @tparam S Score type.
 * @tparam TILE_SIZE Tile size used for probing.
 * @param bucket Bucket pointer for the target bucket.
 * @param key Key to search in the bucket.
 * @param aligned_start Aligned start position in the bucket.
 * @param bucket_max_size Bucket capacity (must be power-of-two).
 * @param rank Thread rank inside the tile.
 * @note The table must be immutable during lookup, and keys in the same batch
 * must be non-overlapping.
 * @return Position inside the bucket if found, otherwise `-1`.
 */
template <typename K, typename V, typename S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ int find_readonly_no_sync_in_bucket(
    typename HashTableDeviceView<K, V, S>::bucket_type* bucket,
    const typename HashTableDeviceView<K, V, S>::key_type key,
    const uint32_t aligned_start, const uint32_t bucket_max_size,
    const uint32_t rank) {
  return find_without_lock_readonly_no_sync<K, V, S, TILE_SIZE>(
      bucket, key, aligned_start, bucket_max_size, rank);
}

/**
 * @brief Read-only lookup without tile synchronization.
 *
 * @tparam K Key type.
 * @tparam V Value type.
 * @tparam S Score type.
 * @tparam TILE_SIZE Tile size used for probing.
 * @param view Device-side view of the table.
 * @param key Key to search in the table.
 * @param rank Thread rank inside the tile.
 * @note The table must be immutable during lookup, and keys in the same batch
 * must be non-overlapping.
 * @return Position inside the bucket if found, otherwise `-1`.
 */
template <typename K, typename V, typename S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ int find_readonly_no_sync(
    const HashTableDeviceView<K, V, S>& view,
    const typename HashTableDeviceView<K, V, S>::key_type key,
    const uint32_t rank) {
  uint32_t bucket_idx = 0;
  uint32_t aligned_start = 0;
  if (!compute_bucket_index_and_aligned_start<K, V, S, TILE_SIZE>(
          view, key, &bucket_idx, &aligned_start)) {
    return -1;
  }
  return find_readonly_no_sync_in_bucket<K, V, S, TILE_SIZE>(
      view.buckets + bucket_idx, key, aligned_start,
      static_cast<uint32_t>(view.bucket_max_size), rank);
}

/**
 * @brief Create a device-side view from a host table instance.
 *
 * @param table Host-side hash table.
 * @return Device-side view that can be used in kernels.
 */
template <typename K, typename V, typename S, int Strategy, typename ArchTag>
__host__ inline HashTableDeviceView<K, V, S> make_device_view(
    const HashTable<K, V, S, Strategy, ArchTag>& table) {
  HashTableDeviceView<K, V, S> view{};
  view.buckets = table.device_buckets();
  view.bucket_count = table.device_bucket_count();
  view.bucket_max_size = table.device_bucket_max_size();
  return view;
}

}  // namespace device
}  // namespace merlin
}  // namespace nv
