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

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
struct LookupKernelParams {
  LookupKernelParams(Bucket<K, V, S>* __restrict buckets_, size_t buckets_num_,
                     uint32_t dim_, const K* __restrict keys_,
                     V* __restrict values_, S* __restrict scores_,
                     bool* __restrict founds_, size_t n_)
      : buckets(buckets_),
        buckets_num(buckets_num_),
        dim(dim_),
        keys(keys_),
        values(values_),
        scores(scores_),
        founds(founds_),
        n(n_) {}
  Bucket<K, V, S>* __restrict buckets;
  size_t buckets_num;
  uint32_t dim;
  const K* __restrict keys;
  V* __restrict values;
  S* __restrict scores;
  bool* __restrict founds;
  size_t n;
};

// Using 32 threads to deal with one key
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename CopyScore = CopyScoreEmpty<S, K, 128>,
          typename CopyValue = CopyValueTwoGroup<float, float4, 32>,
          int DIM_BUF = 224>
__global__ void lookup_kernel_with_io_pipeline_v1(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
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
  S* sm_target_scores = reinterpret_cast<S*>(sm_target_keys);
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];
  __shared__ V sm_vector[2][GROUP_NUM][DIM_BUF];

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

  // Pipeline loading
  uint8_t* digests_ptr =
      reinterpret_cast<uint8_t*>(sm_keys_ptr[groupID * GROUP_SIZE]) -
      BUCKET_SIZE;
  __pipeline_memcpy_async(sm_probing_digests[0] + groupID * DIGEST_SPAN + rank,
                          digests_ptr + rank * 4, sizeof(uint32_t));
  __pipeline_commit();
  __pipeline_commit();  // padding
  __pipeline_commit();  // padding

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
    __pipeline_wait_prior(3);
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

    /* Step3: check possible keys, and prefecth the value and score */
    if (i > 0) {
      key_idx_block -= 1;
      K target_key = sm_target_keys[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      sm_founds[key_idx_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
      V* value_ptr = sm_values_ptr[key_idx_block];
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[diff_buf(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
          CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                             score_ptr + key_pos);
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        V* v_dst = sm_vector[diff_buf(i)][groupID];
        sm_founds[key_idx_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        V* v_src = value_ptr + target_pos * dim;
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    /* Step4: write back value and score */
    if (i > 1) {
      key_idx_block -= 1;
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      V* v_src = sm_vector[same_buf(i)][groupID];
      V* v_dst = values + key_idx_grid * dim;
      int found_flag = sm_founds[key_idx_block];
      __pipeline_wait_prior(3);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        founds[key_idx_grid] = true;
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
    V* value_ptr = sm_values_ptr[key_idx_block];
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
      V* v_src = value_ptr + target_pos * dim;
      V* v_dst = sm_vector[diff_buf(loop_num)][groupID];
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  /* Pipeline emptying: step4, i = loop_num */
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_src = sm_vector[same_buf(loop_num)][groupID];
    V* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(1);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  /* Pipeline emptying: step4, i = loop_num + 1 */
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_src = sm_vector[same_buf(loop_num + 1)][groupID];
    V* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(0);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }
}  // End function

// Using 16 threads to deal with one key
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename CopyScore = CopyScoreEmpty<S, K, 128>,
          typename CopyValue = CopyValueTwoGroup<float, float4, 16>,
          int DIM_BUF = 128>
__global__ void lookup_kernel_with_io_pipeline_v2(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
    bool* __restrict founds, size_t n) {
  constexpr int GROUP_SIZE = 16;
  constexpr int RESERVE = 8;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr int DIGEST_SPAN = BUCKET_SIZE / 4;

  __shared__ int sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr[BLOCK_SIZE];
  __shared__ V* sm_values_ptr[BLOCK_SIZE];
  // Reuse
  S* sm_target_scores = reinterpret_cast<S*>(sm_target_keys);
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];
  __shared__ V sm_vector[2][GROUP_NUM][DIM_BUF];

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

  // Pipeline loading
  uint8_t* digests_ptr =
      reinterpret_cast<uint8_t*>(sm_keys_ptr[groupID * GROUP_SIZE]) -
      BUCKET_SIZE;
  __pipeline_memcpy_async(
      sm_probing_digests[0] + groupID * DIGEST_SPAN + rank * 2,
      digests_ptr + rank * 8, sizeof(uint2));
  __pipeline_commit();
  __pipeline_commit();  // padding
  __pipeline_commit();  // padding

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    /* Step1: prefetch all digests in one bucket */
    if ((i + 1) < loop_num) {
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(sm_keys_ptr[key_idx_block + 1]) -
          BUCKET_SIZE;
      __pipeline_memcpy_async(
          sm_probing_digests[diff_buf(i)] + groupID * DIGEST_SPAN + rank * 2,
          digests_ptr + rank * 8, sizeof(uint2));
    }
    __pipeline_commit();

    /* Step2: check digests and load possible keys */
    uint32_t target_digest = sm_target_digests[key_idx_block];
    uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(3);
    uint32_t probing_digests =
        sm_probing_digests[same_buf(i)][groupID * DIGEST_SPAN + rank];
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests);
    uint32_t find_result = 0;
    if ((find_result_ & 0x01) != 0) find_result |= 0x01;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x02;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x04;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x08;
    probing_digests = sm_probing_digests[same_buf(i)][groupID * DIGEST_SPAN +
                                                      rank + GROUP_SIZE];
    find_result_ = __vcmpeq4(probing_digests, target_digests);
    if ((find_result_ & 0x01) != 0) find_result |= 0x10;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x20;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x40;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x80;
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
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
          sm_possible_pos[same_buf(i)][groupID * RESERVE + group_base] =
              key_pos;
          __pipeline_memcpy_async(
              sm_possible_keys[same_buf(i)] + groupID * RESERVE + group_base,
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
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
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

    /* Step3: check possible keys, and prefecth the value and score */
    if (i > 0) {
      key_idx_block -= 1;
      K target_key = sm_target_keys[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      sm_founds[key_idx_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
      V* value_ptr = sm_values_ptr[key_idx_block];
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[diff_buf(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
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
        V* v_src = value_ptr + target_pos * dim;
        V* v_dst = sm_vector[diff_buf(i)][groupID];
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    /* Step4: write back value and score */
    if (i > 1) {
      key_idx_block -= 1;
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      int found_flag = sm_founds[key_idx_block];
      V* v_src = sm_vector[same_buf(i)][groupID];
      V* v_dst = values + key_idx_grid * dim;
      __pipeline_wait_prior(3);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        founds[key_idx_grid] = true;
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
    V* value_ptr = sm_values_ptr[key_idx_block];
    __pipeline_wait_prior(1);
    int key_pos;
    bool found_flag = false;
    if (rank < possible_num) {
      key_pos = sm_possible_pos[diff_buf(loop_num)][groupID * RESERVE + rank];
      K possible_key =
          sm_possible_keys[diff_buf(loop_num)][groupID * RESERVE + rank];
      if (possible_key == target_key) {
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
      V* v_src = value_ptr + target_pos * dim;
      V* v_dst = sm_vector[diff_buf(loop_num)][groupID];
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  /* Pipeline emptying: step4, i = loop_num */
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_dst = values + key_idx_grid * dim;
    V* v_src = sm_vector[same_buf(loop_num)][groupID];
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(1);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  /* Pipeline emptying: step4, i = loop_num + 1 */
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_dst = values + key_idx_grid * dim;
    V* v_src = sm_vector[same_buf(loop_num + 1)][groupID];
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(0);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }
}  // End function

template <typename K, typename V, typename S, typename CopyScore,
          typename ValueUnit, uint32_t value_dim>
struct SelectLookupCopyValueV1 {
  static void launch_kernel(const LookupKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    // Using 32 threads to deal with one key
    constexpr int GROUP_SIZE = 32;
    constexpr int UNIT_DIM = sizeof(ValueUnit) / sizeof(float);

    if (params.dim > (GROUP_SIZE * UNIT_DIM * 2)) {
      using CopyValue = CopyValueMultipleGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    } else if (params.dim > (GROUP_SIZE * UNIT_DIM)) {
      using CopyValue = CopyValueTwoGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    } else {
      using CopyValue = CopyValueOneGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    }
  }
};

template <typename K, typename V, typename S, typename CopyScore,
          typename ValueUnit, uint32_t value_dim>
struct SelectLookupCopyValueV2 {
  static void launch_kernel(const LookupKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    // Using 16 threads to deal with one key
    constexpr int GROUP_SIZE = 16;
    constexpr int UNIT_DIM = sizeof(ValueUnit) / sizeof(float);

    if (params.dim > (GROUP_SIZE * UNIT_DIM * 2)) {
      using CopyValue = CopyValueMultipleGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    } else if (params.dim > (GROUP_SIZE * UNIT_DIM)) {
      using CopyValue = CopyValueTwoGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    } else {
      using CopyValue = CopyValueOneGroup<float, ValueUnit, GROUP_SIZE>;
      lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue,
                                        value_dim>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_num, params.dim, params.keys,
              params.values, params.scores, params.founds, params.n);
    }
  }
};

template <typename ArchTag>
struct LookupValueDimConfig;

/// TODO: support more arch
template <>
struct LookupValueDimConfig<Sm80> {
  static constexpr uint32_t pipeline_v1_dim = 224;
  static constexpr uint32_t pipeline_v2_dim = 128;
};

template <typename K, typename V, typename S = uint64_t,
          typename ArchTag = Sm80>
struct SelectLookupKernelWithIOPipeline {
  using ValueDimConfig = LookupValueDimConfig<ArchTag>;

  static inline uint32_t max_value_size() {
    return ValueDimConfig::pipeline_v1_dim * sizeof(float);
  }

  static void select_kernel(LookupKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    constexpr int BUCKET_SIZE = 128;
    constexpr size_t ValueGranularity = sizeof(float);
    constexpr uint32_t v1_value_dim = ValueDimConfig::pipeline_v1_dim;
    constexpr uint32_t v2_value_dim = ValueDimConfig::pipeline_v2_dim;

    params.dim =
        static_cast<uint32_t>(params.dim * sizeof(V) / ValueGranularity);

    if (params.scores == nullptr) {
      using CopyScore = CopyScoreEmpty<S, K, BUCKET_SIZE>;
      if (params.dim > v2_value_dim) {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float4,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float2,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV1<K, V, S, CopyScore, float,
                                  v1_value_dim>::launch_kernel(params, stream);
        }
      } else {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float4,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float2,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV2<K, V, S, CopyScore, float,
                                  v2_value_dim>::launch_kernel(params, stream);
        }
      }
    } else {
      using CopyScore = CopyScoreByPassCache<S, K, BUCKET_SIZE>;
      if (params.dim > v2_value_dim) {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float4,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float2,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV1<K, V, S, CopyScore, float,
                                  v1_value_dim>::launch_kernel(params, stream);
        }
      } else {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float4,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float2,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV2<K, V, S, CopyScore, float,
                                  v2_value_dim>::launch_kernel(params, stream);
        }
      }
    }
  }  // End function
};

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel_with_io(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V* __restrict values, S* __restrict scores, bool* __restrict found,
    size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, S>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    V* find_value = values + key_idx * dim;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim, find_value,
                                dim);
      if (rank == src_lane) {
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}

template <typename K, typename V, typename S>
struct SelectLookupKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             const K* __restrict keys, V* __restrict values,
                             S* __restrict scores, bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, found, N);
    }
    return;
  }
};

/* lookup kernel.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel(const Table<K, V, S>* __restrict table,
                              const size_t bucket_max_size,
                              const size_t buckets_num, const size_t dim,
                              const K* __restrict keys, V** __restrict values,
                              S* __restrict scores, bool* __restrict found,
                              int* __restrict dst_offset, size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, S>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    if (dst_offset != nullptr && rank == 0) {
      *(dst_offset + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (rank == src_lane) {
        *(values + key_idx) = (bucket->vectors + key_pos * dim);
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    } else {
      if (rank == 0) {
        *(values + key_idx) = nullptr;
      }
    }
  }
}

}  // namespace merlin
}  // namespace nv