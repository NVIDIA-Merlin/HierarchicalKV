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

/*
 * update with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void update_kernel_with_io(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    const V* update_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, update_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, S, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, update_value,
                                bucket->vectors + key_pos * dim, dim);
      if (src_lane == g.thread_rank()) {
        update_score(bucket, key_pos, scores, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectUpdateKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             const K* __restrict keys,
                             const V* __restrict values,
                             const S* __restrict scores) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, N);
    }
    return;
  }
};

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void update_kernel(const Table<K, V, S>* __restrict table,
                              const size_t bucket_max_size,
                              const size_t buckets_num, const size_t dim,
                              const K* __restrict keys, V** __restrict vectors,
                              const S* __restrict scores,
                              int* __restrict src_offset, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, update_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    *(src_offset + key_idx) = key_idx;

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, S, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (g.thread_rank() == src_lane) {
      if (occupy_result == OccupyResult::DUPLICATE) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_score(bucket, key_pos, scores, key_idx);
      } else {
        *(vectors + key_idx) = nullptr;
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

}  // namespace merlin
}  // namespace nv