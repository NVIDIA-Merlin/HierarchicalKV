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

template <class V, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ void accum_or_assign_vector(
    cg::thread_block_tile<TILE_SIZE> const& g, const V* delta_or_val, V* dst,
    const bool is_accum, const size_t dim) {
  for (auto i = g.thread_rank(); i < dim; i += g.size()) {
    if (is_accum) {
      dst[i] += delta_or_val[i];
    } else {
      dst[i] = delta_or_val[i];
    }
  }
}

/* Write the values of delta_or_val into the table. If the key[i] is already in
   the table indicted be @exists[i], a @delta_or_val[i] will be added to the the
   existing value. if the key not exists, the value @val_or_delta[i] will be
   assigned to the address @dst[i].

   `delta_or_val`: will be treated as val and accumlating should be executed.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `existed`: If the keys existed before this kernel is executed.
   `status`: The existence status for each key when the kernel is being
   executed.

   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class S>
__global__ void write_with_accum_kernel(const V* __restrict delta_or_val,
                                        V** __restrict dst,
                                        const bool* __restrict existed,
                                        const bool* __restrict status,
                                        const int* __restrict src_offset,
                                        const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;

    if (dst[vec_index] != nullptr &&
        existed[src_offset[vec_index]] == status[src_offset[vec_index]]) {
      if (status[src_offset[vec_index]]) {
        dst[vec_index][dim_index] +=
            delta_or_val[src_offset[vec_index] * dim + dim_index];
      } else {
        dst[vec_index][dim_index] =
            delta_or_val[src_offset[vec_index] * dim + dim_index];
      }
    }
  }
}

/*
 * update with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void accum_or_assign_kernel_with_io(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict value_or_deltas, const S* __restrict scores,
    const bool* __restrict accum_or_assigns, const S global_epoch,
    const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const S insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);

    const V* insert_value = value_or_deltas + key_idx * dim;
    const bool is_accum = accum_or_assigns[key_idx];

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
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((is_accum && occupy_result != OccupyResult::DUPLICATE) ||
        (!is_accum && occupy_result == OccupyResult::DUPLICATE)) {
      if (g.thread_rank() == src_lane) {
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          evicted_key = static_cast<K>(EMPTY_KEY);
        }
        if (occupy_result == OccupyResult::OCCUPIED_RECLAIMED) {
          evicted_key = static_cast<K>(RECLAIM_KEY);
        }
        if (occupy_result == OccupyResult::DUPLICATE) {
          evicted_key = insert_key;
        }
        (bucket->keys(key_pos))
            ->store(evicted_key, ScoreFunctor::UNLOCK_MEM_ORDER);
      }
      g.sync();
      continue;
    }
    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    accum_or_assign_vector<V, TILE_SIZE>(
        g, insert_value, bucket->vectors + key_pos * dim, is_accum, dim);

    if (g.thread_rank() == src_lane) {
      ScoreFunctor::update(bucket, key_pos, scores, key_idx, insert_score,
                           (occupy_result != OccupyResult::DUPLICATE));
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
    }
  }
}

template <typename K, typename V, typename S, int Strategy>
struct SelectAccumOrAssignKernelWithIO {
  static void execute_kernel(
      const float& load_factor, const int& block_size,
      const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
      cudaStream_t& stream, const size_t& n,
      const Table<K, V, S>* __restrict table, const K* __restrict keys,
      const V* __restrict value_or_deltas, const S* __restrict scores,
      const bool* __restrict accum_or_assigns, const S global_epoch) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      accum_or_assign_kernel_with_io<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, value_or_deltas,
              scores, accum_or_assigns, global_epoch, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      accum_or_assign_kernel_with_io<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, value_or_deltas,
              scores, accum_or_assigns, global_epoch, N);
    }
    return;
  }
};

template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void accum_or_assign_kernel(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V** __restrict value_or_deltas, const S* __restrict scores,
    const bool* __restrict accum_or_assigns, int* __restrict src_offset,
    bool* __restrict founds, const S global_epoch, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const S insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);

    const bool is_accum = accum_or_assigns[key_idx];

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    if (g.thread_rank() == 0) {
      *(src_offset + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((is_accum && occupy_result != OccupyResult::DUPLICATE) ||
        (!is_accum && occupy_result == OccupyResult::DUPLICATE)) {
      if (g.thread_rank() == src_lane) {
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          evicted_key = static_cast<K>(EMPTY_KEY);
        }
        if (occupy_result == OccupyResult::OCCUPIED_RECLAIMED) {
          evicted_key = static_cast<K>(RECLAIM_KEY);
        }
        if (occupy_result == OccupyResult::DUPLICATE) {
          evicted_key = insert_key;
        }

        (bucket->keys(key_pos))
            ->store(evicted_key, ScoreFunctor::UNLOCK_MEM_ORDER);
      }
      g.sync();
      continue;
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (g.thread_rank() == src_lane) {
      *(value_or_deltas + key_idx) = (bucket->vectors + key_pos * dim);
      *(founds + key_idx) = is_accum;
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      ScoreFunctor::update(bucket, key_pos, scores, key_idx, insert_score,
                           (occupy_result != OccupyResult::DUPLICATE));
      (bucket->keys(key_pos))
          ->store(insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
    }
  }
}

}  // namespace merlin
}  // namespace nv