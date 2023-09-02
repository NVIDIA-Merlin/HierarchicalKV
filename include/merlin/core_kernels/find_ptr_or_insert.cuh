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

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

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

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        *(found + key_idx) = true;
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
      }
    } else {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        *(found + key_idx) = false;
        ScoreFunctor::update(bucket, key_pos, scores, key_idx,
                             find_or_insert_score, true);
      }
    }

    if (g.thread_rank() == src_lane) {
      bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
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