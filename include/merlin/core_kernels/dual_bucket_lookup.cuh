/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "dual_bucket_utils.cuh"
#include "kernel_utils.cuh"

namespace nv {
namespace merlin {

/**
 * Dual-bucket pipeline lookup kernel (sequential two-bucket search).
 *
 * For each key, computes (b1, b2) via high/low 32-bit split of Murmur3 hash.
 * First probes b1; if not found, probes b2.
 * Uses dual_bucket_digest (bit[56:63]) to avoid digest collision with b2
 * addressing.
 *
 * Architecture: Based on lookup_kernel_with_io_pipeline_v1 with 32 threads
 * per key, 128-thread blocks, 128-slot buckets. 4-stage IO pipeline
 * (prefetch digests -> digest match + key load -> key verify + value prefetch
 * -> value writeback).
 */
template <class K, class V, class S, class VecV,
          typename CopyScore = CopyScoreEmpty<S, K, 128>,
          typename CopyValue = CopyValueTwoGroup<VecV, 32>,
          typename FoundFunctor = FoundFunctorV1<K>, int VALUE_BUF = 56>
__global__ void dual_bucket_pipeline_lookup_kernel_with_io(
    Bucket<K, V, S>* buckets, const int32_t* __restrict__ buckets_size,
    const size_t buckets_num, const int dim, const K* __restrict keys,
    VecV* __restrict values, S* __restrict scores, FoundFunctor found_functor,
    size_t n) {
  constexpr int GROUP_SIZE = 32;
  constexpr int RESERVE = 16;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr int DIGEST_SPAN = BUCKET_SIZE / 4;

  using BUCKET = Bucket<K, V, S>;

  // Shared memory declarations.
  __shared__ int sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr1[BLOCK_SIZE];       // b1 bucket keys ptr
  __shared__ K* sm_keys_ptr2[BLOCK_SIZE];       // b2 bucket keys ptr
  __shared__ VecV* sm_values_ptr1[BLOCK_SIZE];  // b1 values ptr
  __shared__ VecV* sm_values_ptr2[BLOCK_SIZE];  // b2 values ptr
  __shared__ S sm_target_scores[BLOCK_SIZE];
  // Reuse sm_target_digests
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];
  __shared__ VecV sm_vector[2][GROUP_NUM][VALUE_BUF];

  // Initialization.
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int groupID = threadIdx.x / GROUP_SIZE;
  int rank = g.thread_rank();
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;
  if (key_idx_base >= n) return;
  int loop_num =
      (n - key_idx_base) < GROUP_SIZE ? (n - key_idx_base) : GROUP_SIZE;

  // Phase 1: Initialize per-key data (hash, digest, bucket pointers).
  // Save digest in register to avoid recomputing Murmur3 hash in Pass 2
  // (sm_target_digests is aliased with sm_counts/sm_founds and gets
  // corrupted during Pass 1).
  uint32_t reg_target_digest = 0;
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    K target_key = keys[key_idx_base + rank];
    sm_target_keys[idx_block] = target_key;
    const K hashed_key = Murmur3HashDevice(target_key);

    // Dual-bucket digest: bit[56:63]
    const uint8_t target_digest =
        static_cast<uint8_t>(static_cast<uint64_t>(hashed_key) >> 56);
    reg_target_digest = static_cast<uint32_t>(target_digest);
    sm_target_digests[idx_block] = reg_target_digest;

    // Dual-bucket positions (centralized in dual_bucket_utils.cuh).
    size_t bkt_idx1, bkt_idx2;
    get_dual_bucket_indices<K>(hashed_key, buckets_num, bkt_idx1, bkt_idx2);

    BUCKET* bucket1 = buckets + bkt_idx1;
    BUCKET* bucket2 = buckets + bkt_idx2;
    sm_keys_ptr1[idx_block] = reinterpret_cast<K*>(bucket1->keys(0));
    sm_keys_ptr2[idx_block] = reinterpret_cast<K*>(bucket2->keys(0));
    __pipeline_memcpy_async(sm_values_ptr1 + idx_block, &(bucket1->vectors),
                            sizeof(VecV*));
    __pipeline_commit();
    __pipeline_memcpy_async(sm_values_ptr2 + idx_block, &(bucket2->vectors),
                            sizeof(VecV*));
  }
  __pipeline_wait_prior(0);

  // Helper lambda-like function to run pipeline lookup on one bucket.
  // We process keys sequentially through the pipeline for one bucket,
  // then process missed keys through the second bucket.

  // --- PASS 1: Search bucket b1 ---
  // Pipeline loading for b1.
  {
    uint8_t* digests_ptr =
        reinterpret_cast<uint8_t*>(sm_keys_ptr1[groupID * GROUP_SIZE]) -
        BUCKET_SIZE;
    __pipeline_memcpy_async(
        sm_probing_digests[0] + groupID * DIGEST_SPAN + rank,
        digests_ptr + rank * 4, sizeof(uint32_t));
  }
  __pipeline_commit();
  __pipeline_commit();
  __pipeline_commit();

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    // Step1: prefetch digests for next key's b1 bucket.
    if ((i + 1) < loop_num) {
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(sm_keys_ptr1[key_idx_block + 1]) -
          BUCKET_SIZE;
      __pipeline_memcpy_async(
          sm_probing_digests[diff_buf(i)] + groupID * DIGEST_SPAN + rank,
          digests_ptr + rank * 4, sizeof(uint32_t));
    }
    __pipeline_commit();

    // Step2: check digests and load possible keys.
    uint32_t target_digest = sm_target_digests[key_idx_block];
    uint32_t target_digests_vec =
        __byte_perm(target_digest, target_digest, 0x0000);
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(3);
    uint32_t probing_digests =
        sm_probing_digests[same_buf(i)][groupID * DIGEST_SPAN + rank];
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests_vec);
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
    K* key_ptr = sm_keys_ptr1[key_idx_block];
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
        if (found_vote) break;
        found_vote = digest_idx >= 0;
      } while (g.any(found_vote));
    }
    __pipeline_commit();

    // Step3: verify keys, prefetch values.
    if (i > 0) {
      int prev_block = groupID * GROUP_SIZE + i - 1;
      K target_key = sm_target_keys[prev_block];
      int possible_num = sm_counts[prev_block];
      sm_founds[prev_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr1, prev_block);
      VecV* value_ptr = sm_values_ptr1[prev_block];
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[diff_buf(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
          CopyScore::ldg_sts(sm_target_scores + prev_block,
                             score_ptr + key_pos);
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        VecV* v_dst = sm_vector[diff_buf(i)][groupID];
        sm_founds[prev_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        VecV* v_src = value_ptr + target_pos * dim;
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    // Step4: write back value and score.
    if (i > 1) {
      int wb_block = groupID * GROUP_SIZE + i - 2;
      int key_idx_grid = blockIdx.x * blockDim.x + wb_block;
      VecV* v_src = sm_vector[same_buf(i)][groupID];
      VecV* v_dst = values + key_idx_grid * dim;
      int found_flag = sm_founds[wb_block];
      __pipeline_wait_prior(3);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + wb_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    }
  }

  // Pipeline emptying for b1: step3 for last key.
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr1, key_idx_block);
    VecV* value_ptr = sm_values_ptr1[key_idx_block];
    __pipeline_wait_prior(1);
    int key_pos;
    bool found_flag = false;
    if (rank < possible_num) {
      key_pos = sm_possible_pos[diff_buf(loop_num)][groupID * RESERVE + rank];
      K possible_key =
          sm_possible_keys[diff_buf(loop_num)][groupID * RESERVE + rank];
      if (target_key == possible_key) {
        found_flag = true;
        CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                           score_ptr + key_pos);
      }
    }
    int found_vote = g.ballot(found_flag);
    if (found_vote) {
      sm_founds[key_idx_block] = 1;
      int src_lane = __ffs(found_vote) - 1;
      int target_pos = g.shfl(key_pos, src_lane);
      VecV* v_src = value_ptr + target_pos * dim;
      VecV* v_dst = sm_vector[diff_buf(loop_num)][groupID];
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  // Pipeline emptying: step4 for second-to-last key.
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    VecV* v_src = sm_vector[same_buf(loop_num)][groupID];
    VecV* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(1);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  // Pipeline emptying: step4 for last key.
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    VecV* v_src = sm_vector[same_buf(loop_num + 1)][groupID];
    VecV* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(0);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  // Finalize b1 pass and record found status.
  // Keys found in b1 are marked. Unfound keys need b2 search.
  if (rank < loop_num) {
    int key_idx_block = groupID * GROUP_SIZE + rank;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    // Only write found for b1 hits; b2 pass will handle misses.
    if (sm_founds[key_idx_block] > 0) {
      found_functor(key_idx_grid, sm_target_keys[key_idx_block], true);
    }
  }

  // --- PASS 2: Search bucket b2 for keys not found in b1 ---
  // Count unfound keys. If all found in b1, skip b2 entirely.
  int any_unfound = 0;
  if (rank < loop_num) {
    int key_idx_block = groupID * GROUP_SIZE + rank;
    if (sm_founds[key_idx_block] == 0) {
      any_unfound = 1;
    }
  }
  any_unfound = g.any(any_unfound);
  if (!any_unfound) return;

  // Save b1 found flags (sm_founds will be reused).
  // We use a simple approach: store per-thread found flag in register.
  int b1_found = 0;
  if (rank < loop_num) {
    b1_found = sm_founds[groupID * GROUP_SIZE + rank];
  }

  // Restore digests from registers saved during Phase 1 init.
  // sm_target_digests was aliased with sm_counts/sm_founds and corrupted
  // during Pass 1.  Using the register avoids recomputing Murmur3 hash.
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    sm_target_digests[idx_block] = reg_target_digest;
  }
  __syncwarp();

  // Pipeline loading for b2.
  {
    uint8_t* digests_ptr =
        reinterpret_cast<uint8_t*>(sm_keys_ptr2[groupID * GROUP_SIZE]) -
        BUCKET_SIZE;
    __pipeline_memcpy_async(
        sm_probing_digests[0] + groupID * DIGEST_SPAN + rank,
        digests_ptr + rank * 4, sizeof(uint32_t));
  }
  __pipeline_commit();
  __pipeline_commit();
  __pipeline_commit();

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;
    // Check if this key was already found in b1.
    int skip = g.shfl(b1_found, i);

    // Step1: prefetch digests for next key's b2 bucket.
    if ((i + 1) < loop_num) {
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(sm_keys_ptr2[key_idx_block + 1]) -
          BUCKET_SIZE;
      __pipeline_memcpy_async(
          sm_probing_digests[diff_buf(i)] + groupID * DIGEST_SPAN + rank,
          digests_ptr + rank * 4, sizeof(uint32_t));
    }
    __pipeline_commit();

    // Step2: check digests and load possible keys (skip if found in b1).
    // Read digest BEFORE zeroing sm_counts (they alias sm_target_digests).
    uint32_t target_digest = sm_target_digests[key_idx_block];
    sm_counts[key_idx_block] = 0;
    if (!skip) {
      uint32_t target_digests_vec =
          __byte_perm(target_digest, target_digest, 0x0000);
      __pipeline_wait_prior(3);
      uint32_t probing_digests =
          sm_probing_digests[same_buf(i)][groupID * DIGEST_SPAN + rank];
      uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests_vec);
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
      K* key_ptr = sm_keys_ptr2[key_idx_block];
      if (gt_vote == 0) {
        do {
          int digest_idx = __ffs(find_result) - 1;
          if (digest_idx >= 0) {
            find_result &= (find_result - 1);
            int key_pos = rank * 4 + digest_idx;
            sm_possible_pos[same_buf(i)][groupID * RESERVE + group_base] =
                key_pos;
            __pipeline_memcpy_async(sm_possible_keys[same_buf(i)] +
                                        (groupID * RESERVE + group_base),
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
          if (found_vote) break;
          found_vote = digest_idx >= 0;
        } while (g.any(found_vote));
      }
    } else {
      __pipeline_wait_prior(3);
    }
    __pipeline_commit();

    // Step3: verify keys and prefetch values from b2.
    if (i > 0) {
      int prev_block = groupID * GROUP_SIZE + i - 1;
      int prev_skip = g.shfl(b1_found, i - 1);
      if (!prev_skip) {
        K target_key = sm_target_keys[prev_block];
        // Read count BEFORE zeroing (sm_counts aliases sm_founds).
        int possible_num = sm_counts[prev_block];
        sm_founds[prev_block] = 0;
        S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr2, prev_block);
        VecV* value_ptr = sm_values_ptr2[prev_block];
        __pipeline_wait_prior(3);
        int key_pos;
        bool found_flag = false;
        if (rank < possible_num) {
          K possible_key =
              sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
          key_pos = sm_possible_pos[diff_buf(i)][groupID * RESERVE + rank];
          if (possible_key == target_key) {
            found_flag = true;
            CopyScore::ldg_sts(sm_target_scores + prev_block,
                               score_ptr + key_pos);
          }
        }
        int found_vote = g.ballot(found_flag);
        if (found_vote) {
          VecV* v_dst = sm_vector[diff_buf(i)][groupID];
          sm_founds[prev_block] = 1;
          int src_lane = __ffs(found_vote) - 1;
          int target_pos = g.shfl(key_pos, src_lane);
          VecV* v_src = value_ptr + target_pos * dim;
          CopyValue::ldg_sts(rank, v_dst, v_src, dim);
        }
      } else {
        __pipeline_wait_prior(3);
      }
    }
    __pipeline_commit();

    // Step4: write back values from b2.
    if (i > 1) {
      int wb_block = groupID * GROUP_SIZE + i - 2;
      int prev_skip = g.shfl(b1_found, i - 2);
      if (!prev_skip) {
        int key_idx_grid = blockIdx.x * blockDim.x + wb_block;
        VecV* v_src = sm_vector[same_buf(i)][groupID];
        VecV* v_dst = values + key_idx_grid * dim;
        int found_flag = sm_founds[wb_block];
        __pipeline_wait_prior(3);
        if (found_flag > 0) {
          S score_ = CopyScore::lgs(sm_target_scores + wb_block);
          CopyValue::lds_stg(rank, v_dst, v_src, dim);
          CopyScore::stg(scores + key_idx_grid, score_);
        }
      } else {
        __pipeline_wait_prior(3);
      }
    }
  }

  // Pipeline emptying for b2: step3 for last key.
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    int last_skip = g.shfl(b1_found, loop_num - 1);
    if (!last_skip) {
      K target_key = sm_target_keys[key_idx_block];
      // Read count BEFORE zeroing (sm_counts aliases sm_founds).
      int possible_num = sm_counts[key_idx_block];
      sm_founds[key_idx_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr2, key_idx_block);
      VecV* value_ptr = sm_values_ptr2[key_idx_block];
      __pipeline_wait_prior(1);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        key_pos = sm_possible_pos[diff_buf(loop_num)][groupID * RESERVE + rank];
        K possible_key =
            sm_possible_keys[diff_buf(loop_num)][groupID * RESERVE + rank];
        if (target_key == possible_key) {
          found_flag = true;
          CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                             score_ptr + key_pos);
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        sm_founds[key_idx_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        VecV* v_src = value_ptr + target_pos * dim;
        VecV* v_dst = sm_vector[diff_buf(loop_num)][groupID];
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    } else {
      __pipeline_wait_prior(1);
    }
  }
  __pipeline_commit();

  // Pipeline emptying: step4 for second-to-last key.
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int prev_skip = g.shfl(b1_found, loop_num - 2);
    if (!prev_skip) {
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      VecV* v_src = sm_vector[same_buf(loop_num)][groupID];
      VecV* v_dst = values + key_idx_grid * dim;
      int found_flag = sm_founds[key_idx_block];
      __pipeline_wait_prior(1);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    } else {
      __pipeline_wait_prior(1);
    }
  }

  // Pipeline emptying: step4 for last key.
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int last_skip = g.shfl(b1_found, loop_num - 1);
    if (!last_skip) {
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      VecV* v_src = sm_vector[same_buf(loop_num + 1)][groupID];
      VecV* v_dst = values + key_idx_grid * dim;
      int found_flag = sm_founds[key_idx_block];
      __pipeline_wait_prior(0);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    } else {
      __pipeline_wait_prior(0);
    }
  }

  // Finalize b2 pass: report found for keys found in b2.
  if (rank < loop_num) {
    int key_idx_block = groupID * GROUP_SIZE + rank;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    if (b1_found == 0) {
      // Key was not found in b1; report b2 result.
      found_functor(key_idx_grid, sm_target_keys[key_idx_block],
                    sm_founds[key_idx_block] > 0);
    }
  }
}

// --- Kernel Launchers ---

template <typename K, typename V, typename S, typename CopyScore, typename VecV,
          uint32_t ValueBufSize>
struct LaunchDualBucketLookupV1 {
  template <template <typename, typename, typename> typename LookupKernelParams>
  static void launch_kernel(LookupKernelParams<K, V, S>& params,
                            const int32_t* buckets_size, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int GROUP_SIZE = 32;
    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    constexpr uint32_t VecSize = ValueBufSize / sizeof(VecV);
    if (params.dim > (GROUP_SIZE * 2)) {
      using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
      dual_bucket_pipeline_lookup_kernel_with_io<
          K, V, S, VecV, CopyScore, CopyValue, decltype(params.found_functor),
          VecSize>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, buckets_size, params.buckets_num, params.dim,
              params.keys, reinterpret_cast<VecV*>(params.values),
              params.scores, params.found_functor, params.n);
    } else if (params.dim > GROUP_SIZE) {
      using CopyValue = CopyValueTwoGroup<VecV, GROUP_SIZE>;
      dual_bucket_pipeline_lookup_kernel_with_io<
          K, V, S, VecV, CopyScore, CopyValue, decltype(params.found_functor),
          VecSize>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, buckets_size, params.buckets_num, params.dim,
              params.keys, reinterpret_cast<VecV*>(params.values),
              params.scores, params.found_functor, params.n);
    } else {
      using CopyValue = CopyValueOneGroup<VecV, GROUP_SIZE>;
      dual_bucket_pipeline_lookup_kernel_with_io<
          K, V, S, VecV, CopyScore, CopyValue, decltype(params.found_functor),
          VecSize>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, buckets_size, params.buckets_num, params.dim,
              params.keys, reinterpret_cast<VecV*>(params.values),
              params.scores, params.found_functor, params.n);
    }
  }
};

// --- Kernel Selector ---

template <typename K, typename V, typename S = uint64_t,
          typename ArchTag = Sm80>
struct SelectDualBucketLookupKernel {
  using ValueBufConfig = LookupValueBufConfig<ArchTag>;

  static inline uint32_t max_value_size() {
    return ValueBufConfig::size_pipeline_v1;
  }

  template <template <typename, typename, typename> typename LookupKernelParams>
  static void select_kernel(LookupKernelParams<K, V, S>& params,
                            const int32_t* buckets_size, cudaStream_t& stream) {
    constexpr int BUCKET_SIZE = 128;
    constexpr uint32_t buf_size_v1 = ValueBufConfig::size_pipeline_v1;

    uint32_t total_value_size = static_cast<uint32_t>(params.dim * sizeof(V));

    // For dual-bucket lookup, we use v1 kernel (32 threads/key) only.
    if (params.scores == nullptr) {
      using CopyScore = CopyScoreEmpty<S, K, BUCKET_SIZE>;
      if (total_value_size % sizeof(float4) == 0) {
        using VecV = float4;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(float2) == 0) {
        using VecV = float2;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(float) == 0) {
        using VecV = float;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(uint16_t) == 0) {
        using VecV = uint16_t;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else {
        using VecV = uint8_t;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      }
    } else {
      using CopyScore = CopyScoreByPassCache<S, K, BUCKET_SIZE>;
      if (total_value_size % sizeof(float4) == 0) {
        using VecV = float4;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(float2) == 0) {
        using VecV = float2;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(float) == 0) {
        using VecV = float;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else if (total_value_size % sizeof(uint16_t) == 0) {
        using VecV = uint16_t;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      } else {
        using VecV = uint8_t;
        LaunchDualBucketLookupV1<K, V, S, CopyScore, VecV,
                                 buf_size_v1>::launch_kernel(params,
                                                             buckets_size,
                                                             stream);
      }
    }
  }
};

}  // namespace merlin
}  // namespace nv
