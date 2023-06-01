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

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_ptr_kernel(const Table<K, V, S>* __restrict table,
                                  const size_t bucket_max_size,
                                  const size_t buckets_num, const size_t dim,
                                  const K* __restrict keys,
                                  V** __restrict values, S* __restrict scores,
                                  bool* __restrict found, size_t N) {
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

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (rank == src_lane) {
        values[key_idx] = bucket->vectors + key_pos * dim;
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
struct SelectLookupPtrKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             const K* __restrict keys, V** __restrict values,
                             S* __restrict scores, bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, found, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv