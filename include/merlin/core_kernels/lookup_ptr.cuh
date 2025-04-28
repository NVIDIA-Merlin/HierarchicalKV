/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kernel_utils.cuh"

namespace nv {
namespace merlin {

// Use 1 thread to deal with a KV-pair, including copying value.
template <typename K, typename V, typename S>
__global__ void tlp_lookup_ptr_kernel_with_filter(
    Bucket<K, V, S>* __restrict__ buckets, const uint64_t buckets_num,
    uint32_t bucket_capacity, const uint32_t dim, const K* __restrict__ keys,
    V** __restrict values, S* __restrict scores, bool* __restrict founds,
    uint64_t n) {
  using BUCKET = Bucket<K, V, S>;
  // Load `STRIDE` digests every time.
  constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  V* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  uint32_t key_pos = {0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = reinterpret_cast<V*>(bucket->vectors);
    } else {
      occupy_result = OccupyResult::ILLEGAL;
      goto WRITE_BACK;
    }
  } else {
    return;
  }

  // One more loop to handle empty keys.
  for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
    VecD_Load digests_vec = *(reinterpret_cast<VecD_Load*>(digests_ptr));
    VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                digests_vec.w};

    for (int i = 0; i < 4; i++) {
      VecD_Comp probe_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      int cmp_result = __vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        // CUDA uses little endian,
        // and the lowest byte in register stores in the lowest address.
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        auto current_key = bucket_keys_ptr[possible_pos];
        score = *BUCKET::scores(bucket_keys_ptr, bucket_capacity, possible_pos);
        if (current_key == key) {
          key_pos = possible_pos;
          occupy_result = OccupyResult::DUPLICATE;
          goto WRITE_BACK;
        }
      } while (true);
      VecD_Comp empty_digests_ = empty_digests<K>();
      cmp_result = __vcmpeq4(probe_digests, empty_digests_);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = bucket_keys_ptr[possible_pos];
        if (current_key == static_cast<K>(EMPTY_KEY)) {
          occupy_result = OccupyResult::OCCUPIED_EMPTY;
          goto WRITE_BACK;
        }
      } while (true);
    }
  }

WRITE_BACK:
  bool found_ = occupy_result == OccupyResult::DUPLICATE;
  if (founds) {
    founds[kv_idx] = found_;
  }
  if (found_) {
    if (scores) {
      scores[kv_idx] = score;
    }
    values[kv_idx] = bucket_values_ptr + key_pos * dim;
  } else {
    values[kv_idx] = nullptr;
  }
}

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_ptr_kernel(const Table<K, V, S>* __restrict table,
                                  Bucket<K, V, S>* buckets,
                                  const size_t bucket_max_size,
                                  const size_t buckets_num, const size_t dim,
                                  const K* __restrict keys,
                                  V** __restrict values, S* __restrict scores,
                                  bool* __restrict found, size_t N) {
  int* buckets_size = table->buckets_size;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    OccupyResult occupy_result{OccupyResult::INITIAL};
    int key_pos = -1;
    int src_lane = -1;
    Bucket<K, V, S>* bucket{nullptr};
    if (!IS_RESERVED_KEY<K>(find_key)) {
      size_t bkt_idx = 0;
      size_t start_idx = 0;

      bucket = get_key_position<K>(buckets, find_key, bkt_idx, start_idx,
                                   buckets_num, bucket_max_size);

      const int bucket_size = buckets_size[bkt_idx];
      if (bucket_size >= bucket_max_size) {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
      }

      occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
          g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    if (rank == src_lane) {
      bool found_ = occupy_result == OccupyResult::DUPLICATE;
      if (found != nullptr) {
        *(found + key_idx) = found_;
      }
      if (found_) {
        values[key_idx] = bucket->vectors + key_pos * dim;
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
      } else {
        values[key_idx] = nullptr;
      }
    }
  }
}

template <typename K, typename V, typename S>
struct SelectLookupPtrKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, const K* __restrict keys,
                             V** __restrict values, S* __restrict scores,
                             bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, found, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv