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

// Use 1 thread to deal with a KV-pair.
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          uint32_t BLOCK_SIZE = 128, int Strategy = -1>
__global__ void find_or_insert_ptr_kernel_lock_key(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, V** __restrict__ value_ptrs,
    S* __restrict__ scores, K** __restrict__ key_ptrs, uint64_t n,
    bool* __restrict__ founds, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  V* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  int32_t* bucket_size_ptr{nullptr};
  uint32_t key_pos = {0};
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      bucket_size_ptr = buckets_size + bkt_idx;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_size = *bucket_size_ptr;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = reinterpret_cast<V*>(bucket->vectors);
    } else {
      key_ptrs[kv_idx] = nullptr;
      return;
    }
  } else {
    return;
  }

  // Load `STRIDE` digests every time.
  constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);
  // One more loop to handle empty keys.
  for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
    if (occupy_result != OccupyResult::INITIAL) break;

    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
    VecD_Load digests_vec = *(reinterpret_cast<VecD_Load*>(digests_ptr));
    VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                digests_vec.w};

    for (int i = 0; i < 4; i++) {
      VecD_Comp probe_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      bool result = false;
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
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = key;
        // Modifications to the bucket will not before this instruction.
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::DUPLICATE;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                         kv_idx, score, bucket_capacity,
                                         get_digest<K>(key), false);
        break;
      } else if (bucket_size == bucket_capacity) {
        continue;
      }
      VecD_Comp empty_digests_ = empty_digests<K>();
      cmp_result = __vcmpeq4(probe_digests, empty_digests_);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = static_cast<K>(EMPTY_KEY);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::OCCUPIED_EMPTY;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                         kv_idx, score, bucket_capacity,
                                         get_digest<K>(key), true);

        atomicAdd(bucket_size_ptr, 1);
        break;
      }
    }
  }

  while (occupy_result == OccupyResult::INITIAL) {
    S* bucket_scores_ptr = BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
    S min_score = static_cast<S>(MAX_SCORE);
    int min_pos = -1;
#pragma unroll
    for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
      __pipeline_memcpy_async(sm_bucket_scores[tx] + j, bucket_scores_ptr + j,
                              sizeof(S) * Load_LEN_S);
    }
    __pipeline_commit();
    for (int i = 0; i < bucket_capacity; i += STRIDE_S) {
      if (i < bucket_capacity - STRIDE_S) {
#pragma unroll
        for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
          __pipeline_memcpy_async(
              sm_bucket_scores[tx] + diff_buf(i / STRIDE_S) * STRIDE_S + j,
              bucket_scores_ptr + i + STRIDE_S + j, sizeof(S) * Load_LEN_S);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      S temp_scores[Load_LEN_S];
      S* src = sm_bucket_scores[tx] + same_buf(i / STRIDE_S) * STRIDE_S;
#pragma unroll
      for (int k = 0; k < STRIDE_S; k += Load_LEN_S) {
        *reinterpret_cast<byte16*>(temp_scores) =
            *reinterpret_cast<byte16*>(src + k);
#pragma unroll
        for (int j = 0; j < Load_LEN_S; j += 1) {
          S temp_score = temp_scores[j];
          if (temp_score < min_score) {
            auto verify_key_ptr = BUCKET::keys(bucket_keys_ptr, i + k + j);
            auto verify_key =
                verify_key_ptr->load(cuda::std::memory_order_relaxed);
            if (verify_key != static_cast<K>(LOCKED_KEY) &&
                verify_key != static_cast<K>(EMPTY_KEY)) {
              min_score = temp_score;
              min_pos = i + k + j;
            }
          }
        }
      }
    }

    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (score <= min_score) {
      occupy_result = OccupyResult::REFUSED;
      break;
    }
    auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos);
    auto expected_key = min_score_key->load(cuda::std::memory_order_relaxed);
    if (expected_key != static_cast<K>(LOCKED_KEY) &&
        expected_key != static_cast<K>(EMPTY_KEY)) {
      bool result = min_score_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      if (result) {
        S* min_score_ptr =
            BUCKET::scores(bucket_keys_ptr, bucket_capacity, min_pos);
        auto verify_score_ptr =
            reinterpret_cast<AtomicScore<S>*>(min_score_ptr);
        auto verify_score =
            verify_score_ptr->load(cuda::std::memory_order_relaxed);
        if (verify_score <= min_score) {
          key_pos = min_pos;
          ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                           kv_idx, score, bucket_capacity,
                                           get_digest<K>(key), true);

          if (expected_key == static_cast<K>(RECLAIM_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
            atomicAdd(bucket_size_ptr, 1);
          } else {
            occupy_result = OccupyResult::EVICT;
          }
        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }

  if (kv_idx < n) {
    if (occupy_result == OccupyResult::REFUSED) {
      value_ptrs[kv_idx] = nullptr;
      key_ptrs[kv_idx] = nullptr;
    } else {
      value_ptrs[kv_idx] = bucket_values_ptr + key_pos * dim;
      auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
      key_ptrs[kv_idx] = reinterpret_cast<K*>(key_address);
    }
    founds[kv_idx] = occupy_result == OccupyResult::DUPLICATE;
  }
}

template <typename K>
__global__ void find_or_insert_ptr_kernel_unlock_key(const K* __restrict__ keys,
                                                     K** __restrict__ key_ptrs,
                                                     uint64_t n) {
  int kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
  K key;
  K* key_ptr{nullptr};
  if (kv_idx < n) {
    key = keys[kv_idx];
    key_ptr = key_ptrs[kv_idx];
    if (key_ptr) {
      *key_ptr = key;
    }
  }
}

/* find or insert with the end-user specified score.
 */
template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void find_ptr_or_insert_kernel(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
    const K* __restrict keys, V** __restrict vectors, S* __restrict scores,
    bool* __restrict found, const S global_epoch, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY<K>(find_or_insert_key)) continue;

    const S find_or_insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(buckets, find_or_insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (g.thread_rank() == src_lane) {
      if (occupy_result != OccupyResult::REFUSED) {
        ScoreFunctor::update(bucket, key_pos, scores, key_idx,
                             find_or_insert_score,
                             occupy_result != OccupyResult::DUPLICATE);
        bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
        (bucket->keys(key_pos))
            ->store(find_or_insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
      } else {
        *(vectors + key_idx) = nullptr;
      }
      *(found + key_idx) = occupy_result == OccupyResult::DUPLICATE;
    }
  }
}

template <typename K, typename V, typename S, int Strategy>
struct SelectFindOrInsertPtrKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, const K* __restrict keys,
                             V** __restrict values, S* __restrict scores,
                             bool* __restrict found, const S global_epoch) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, found, global_epoch, N);
    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, found, global_epoch, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, found, global_epoch, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv