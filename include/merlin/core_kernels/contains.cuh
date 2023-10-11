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
struct ContainsKernelParams {
  ContainsKernelParams(Bucket<K, V, S>* __restrict buckets_,
                       size_t buckets_num_, uint32_t dim_,
                       const K* __restrict keys_, bool* __restrict founds_,
                       size_t n_)
      : buckets(buckets_),
        buckets_num(buckets_num_),
        dim(dim_),
        keys(keys_),
        founds(founds_),
        n(n_) {}
  Bucket<K, V, S>* __restrict buckets;
  size_t buckets_num;
  uint32_t dim;
  const K* __restrict keys;
  bool* __restrict founds;
  size_t n;
};

// Using 32 threads to deal with one key
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__global__ void contains_kernel_pipeline(Bucket<K, V, S>* buckets,
                                         const size_t buckets_num,
                                         const int dim,
                                         const K* __restrict keys,
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
  // Reuse
  int* sm_counts = sm_target_digests;

  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];

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
  }
  __pipeline_wait_prior(0);

  // Pipeline loading
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
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      K target_key = sm_target_keys[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      __pipeline_wait_prior(2);
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[diff_buf(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
        }
      }
      int found_vote = g.ballot(found_flag);
      founds[key_idx_grid] = (found_vote > 0);
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    __pipeline_wait_prior(0);
    bool found_flag = false;
    if (rank < possible_num) {
      K possible_key =
          sm_possible_keys[diff_buf(loop_num)][groupID * RESERVE + rank];
      if (target_key == possible_key) {
        found_flag = true;
      }
    }
    int found_vote = g.ballot(found_flag);
    founds[key_idx_grid] = (found_vote > 0);
  }

}  // End function

template <typename K, typename V, typename S>
struct LaunchPipelineContains {
  static void launch_kernel(ContainsKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    // Using 32 threads to deal with one key
    contains_kernel_pipeline<K, V, S>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            params.buckets, params.buckets_num, params.dim, params.keys,
            params.founds, params.n);
  }
};

template <typename K, typename V, typename S = uint64_t,
          typename ArchTag = Sm80>
struct SelectPipelineContainsKernel {
  static void select_kernel(ContainsKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    LaunchPipelineContains<K, V, S>::launch_kernel(params, stream);
  }
};

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void contains_kernel(const Table<K, V, S>* __restrict table,
                                Bucket<K, V, S>* buckets,
                                const size_t bucket_max_size,
                                const size_t buckets_num, const size_t dim,
                                const K* __restrict keys,
                                bool* __restrict found, size_t N) {
  int* buckets_size = table->buckets_size;

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

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (rank == src_lane) {
      *(found + key_idx) = (occupy_result == OccupyResult::DUPLICATE);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectContainsKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, const K* __restrict keys,
                             bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      contains_kernel<K, V, S, tile_size><<<grid_size, block_size, 0, stream>>>(
          table, buckets, bucket_max_size, buckets_num, dim, keys, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      contains_kernel<K, V, S, tile_size><<<grid_size, block_size, 0, stream>>>(
          table, buckets, bucket_max_size, buckets_num, dim, keys, found, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv