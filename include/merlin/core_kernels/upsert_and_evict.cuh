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

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores,
    K* __restrict evicted_keys, V* __restrict evicted_values,
    S* __restrict evicted_scores, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const S insert_score =
        scores != nullptr ? scores[key_idx] : static_cast<S>(MAX_SCORE);
    const V* insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      }
      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      copy_vector<V, TILE_SIZE>(g, insert_value, evicted_values + key_idx * dim,
                                dim);
      continue;
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == src_lane) {
        evicted_keys[key_idx] = evicted_key;
      }
      if (scores != nullptr) {
        evicted_scores[key_idx] = scores[key_idx];
      }
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                evicted_values + key_idx * dim, dim);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_score(bucket, key_pos, scores, key_idx);
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectUpsertAndEvictKernelWithIO {
  static void execute_kernel(
      const float& load_factor, const int& block_size,
      const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
      cudaStream_t& stream, const size_t& n,
      const Table<K, V, S>* __restrict table, const K* __restrict keys,
      const V* __restrict values, const S* __restrict scores,
      K* __restrict evicted_keys, V* __restrict evicted_values,
      S* __restrict evicted_scores) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);

    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);

    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv