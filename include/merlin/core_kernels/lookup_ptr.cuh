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
template <typename K, typename V, typename S, int Strategy>
__global__ void tlp_lookup_ptr_kernel_with_filter(
    Bucket<K, V, S>* __restrict__ buckets, const uint64_t buckets_num,
    uint32_t bucket_capacity, const uint32_t dim, const K* __restrict__ keys,
    V** __restrict values, S* __restrict scores, bool* __restrict founds,
    uint64_t n, bool update_score, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
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
    if (update_score) {
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    }
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
        if (update_score) {
          auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
          K expected_key = key;
          // Modifications to the bucket will not before this instruction.
          bool result = current_key->compare_exchange_strong(
              expected_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
          if (result) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                             kv_idx, score, bucket_capacity,
                                             get_digest<K>(key), false);
            current_key->store(key, cuda::std::memory_order_release);
            score = *BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
            goto WRITE_BACK;
          }
        } else {
          auto current_key = bucket_keys_ptr[possible_pos];
          score =
              *BUCKET::scores(bucket_keys_ptr, bucket_capacity, possible_pos);
          if (current_key == key) {
            key_pos = possible_pos;
            occupy_result = OccupyResult::DUPLICATE;
            goto WRITE_BACK;
          }
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

// Pipelined pointer-return lookup: reuses the cooperative 32-thread digest scan
// from lookup_kernel_with_io_pipeline_v1 (the value-copy find kernel) but skips
// the value copy stages entirely, writing only V* pointers.  This ensures find*
// throughput at lambda=1.0 is always >= find throughput.
template <typename K, typename V, typename S>
__global__ void lookup_ptr_kernel_with_pipeline(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values, S* __restrict scores,
    bool* __restrict founds, size_t n) {
  constexpr int GROUP_SIZE = 32;
  constexpr int RESERVE = 16;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr int DIGEST_SPAN = BUCKET_SIZE / 4;

  __shared__ int sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr[BLOCK_SIZE];
  __shared__ V* sm_values_ptr[BLOCK_SIZE];
  // Reuse
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];

  // Initialization
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int groupID = threadIdx.x / GROUP_SIZE;
  int rank = g.thread_rank();
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;
  if (key_idx_base >= n) return;
  int loop_num =
      (n - key_idx_base) < GROUP_SIZE ? (n - key_idx_base) : GROUP_SIZE;
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    K target_key = keys[key_idx_base + rank];
    sm_target_keys[idx_block] = target_key;
    const K hashed_key = Murmur3HashDevice(target_key);
    const uint8_t target_digest = static_cast<uint8_t>(hashed_key >> 32);
    sm_target_digests[idx_block] = static_cast<uint32_t>(target_digest);
    int global_idx = hashed_key % (buckets_num * BUCKET_SIZE);
    int bkt_idx = global_idx / BUCKET_SIZE;
    Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __pipeline_memcpy_async(sm_keys_ptr + idx_block, bucket->keys_addr(),
                            sizeof(K*));
    __pipeline_commit();
    __pipeline_memcpy_async(sm_values_ptr + idx_block, &(bucket->vectors),
                            sizeof(V*));
  }
  __pipeline_wait_prior(0);

  // Pipeline loading: prefetch digests for the first key
  uint8_t* digests_ptr =
      reinterpret_cast<uint8_t*>(sm_keys_ptr[groupID * GROUP_SIZE]) -
      BUCKET_SIZE;
  __pipeline_memcpy_async(sm_probing_digests[0] + groupID * DIGEST_SPAN + rank,
                          digests_ptr + rank * 4, sizeof(uint32_t));
  __pipeline_commit();
  __pipeline_commit();

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    /* Step1: prefetch all digests in one bucket */
    if ((i + 1) < loop_num) {
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(sm_keys_ptr[key_idx_block + 1]) -
          BUCKET_SIZE;
      __pipeline_memcpy_async(
          sm_probing_digests[diff_buf(i)] + groupID * DIGEST_SPAN + rank,
          digests_ptr + rank * 4, sizeof(uint32_t));
    }
    __pipeline_commit();

    /* Step2: check digests and load possible keys */
    uint32_t target_digest = sm_target_digests[key_idx_block];
    uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(2);
    uint32_t probing_digests =
        sm_probing_digests[same_buf(i)][groupID * DIGEST_SPAN + rank];
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests);
    uint32_t find_result = 0;
    if ((find_result_ & 0x01) != 0) find_result |= 0x01;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x02;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x04;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x08;
    int find_number = __popc(find_result);
    int group_base = 0;
    if (find_number > 0) {
      group_base = atomicAdd(sm_counts + key_idx_block, find_number);
    }
    bool gt_reserve = (group_base + find_number) > RESERVE;
    int gt_vote = g.ballot(gt_reserve);
    K* key_ptr = sm_keys_ptr[key_idx_block];
    if (gt_vote == 0) {
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = rank * 4 + digest_idx;
          sm_possible_pos[same_buf(i)][groupID * RESERVE + group_base] =
              key_pos;
          __pipeline_memcpy_async(
              sm_possible_keys[same_buf(i)] + (groupID * RESERVE + group_base),
              key_ptr + key_pos, sizeof(K));
          group_base += 1;
        } else {
          break;
        }
      } while (true);
    } else {
      K target_key = sm_target_keys[key_idx_block];
      sm_counts[key_idx_block] = 0;
      int found_vote = 0;
      bool found = false;
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = rank * 4 + digest_idx;
          K possible_key = key_ptr[key_pos];
          if (possible_key == target_key) {
            found = true;
            sm_counts[key_idx_block] = 1;
            sm_possible_pos[same_buf(i)][groupID * RESERVE] = key_pos;
            sm_possible_keys[same_buf(i)][groupID * RESERVE] = possible_key;
          }
        }
        found_vote = g.ballot(found);
        if (found_vote) {
          break;
        }
        found_vote = digest_idx >= 0;
      } while (g.any(found_vote));
    }
    __pipeline_commit();

    /* Step3: check possible keys, write back pointer immediately */
    if (i > 0) {
      int prev_idx_block = key_idx_block - 1;
      K target_key = sm_target_keys[prev_idx_block];
      int possible_num = sm_counts[prev_idx_block];
      sm_founds[prev_idx_block] = 0;
      __pipeline_wait_prior(2);
      int key_pos = -1;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[diff_buf(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        sm_founds[prev_idx_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        // Write pointer directly (no value copy needed).
        if (rank == 0) {
          int key_idx_grid = blockIdx.x * blockDim.x + prev_idx_block;
          values[key_idx_grid] =
              sm_values_ptr[prev_idx_block] + target_pos * dim;
          if (scores) {
            S* score_ptr =
                reinterpret_cast<S*>(sm_keys_ptr[prev_idx_block] + BUCKET_SIZE);
            scores[key_idx_grid] = score_ptr[target_pos];
          }
        }
      }
    }
  }  // End loop

  /* Pipeline emptying: process the last key */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    __pipeline_wait_prior(0);
    int key_pos = -1;
    bool found_flag = false;
    if (rank < possible_num) {
      key_pos = sm_possible_pos[diff_buf(loop_num)][groupID * RESERVE + rank];
      K possible_key =
          sm_possible_keys[diff_buf(loop_num)][groupID * RESERVE + rank];
      if (target_key == possible_key) {
        found_flag = true;
      }
    }
    int found_vote = g.ballot(found_flag);
    if (found_vote) {
      sm_founds[key_idx_block] = 1;
      int src_lane = __ffs(found_vote) - 1;
      int target_pos = g.shfl(key_pos, src_lane);
      if (rank == 0) {
        int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
        values[key_idx_grid] = sm_values_ptr[key_idx_block] + target_pos * dim;
        if (scores) {
          S* score_ptr =
              reinterpret_cast<S*>(sm_keys_ptr[key_idx_block] + BUCKET_SIZE);
          scores[key_idx_grid] = score_ptr[target_pos];
        }
      }
    }
  }

  // Write found flags and nullptr for misses.
  if (rank < loop_num) {
    int key_idx_block = groupID * GROUP_SIZE + rank;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    bool found_ = sm_founds[key_idx_block] > 0;
    if (founds) founds[key_idx_grid] = found_;
    if (!found_) values[key_idx_grid] = nullptr;
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