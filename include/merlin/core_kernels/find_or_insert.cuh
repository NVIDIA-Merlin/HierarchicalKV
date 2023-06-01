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
 * find or insert with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel_with_io(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V* __restrict values, S* __restrict scores, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

    const S find_or_insert_score =
        scores != nullptr ? scores[key_idx] : static_cast<S>(MAX_SCORE);
    V* find_or_insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, find_or_insert_key, bkt_idx,
                            start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE>(
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
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                find_or_insert_value, dim);
      if (scores != nullptr && g.thread_rank() == src_lane) {
        *(scores + key_idx) =
            bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
      }
    } else {
      copy_vector<V, TILE_SIZE>(g, find_or_insert_value,
                                bucket->vectors + key_pos * dim, dim);
      if (g.thread_rank() == src_lane) {
        update_score(bucket, key_pos, scores, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectFindOrInsertKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             const K* __restrict keys, V* __restrict values,
                             S* __restrict scores) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 scores, N);
    }
    return;
  }
};

/* find or insert with the end-user specified score.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V** __restrict vectors, S* __restrict scores, bool* __restrict found,
    int* __restrict keys_index, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

    const S find_or_insert_score =
        scores != nullptr ? scores[key_idx] : static_cast<S>(MAX_SCORE);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, find_or_insert_key, bkt_idx,
                            start_idx, buckets_num, bucket_max_size);

    if (g.thread_rank() == 0) {
      *(keys_index + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE>(
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

        if (found != nullptr) {
          *(found + key_idx) = true;
        }

        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
      }
    } else {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_score(bucket, key_pos, scores, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

/* Read the data from address of table_value_addrs to corresponding position
  in param_value if mask[i] is true, otherwise write data to table_value_addrs
  form param_value,
  usually called by find_or_insert kernel.

  `table_value_addrs`: A pointer of pointer of V which should be on HBM,
        but each value (a pointer of V) could point to a
        memory on HBM or HMEM.
  `param_value`: A continue memory pointer with Vector
        which should be HBM.
  `mask`: One for each `param_value`. If true, reading from table_value_addrs,
          or false writing table_value_addrs from  param_value.
  `param_key_index`: N values from address of table_value_addrs are mapped to
        param_values according to param_key_index.
  `dim`: the dim of value.
  `N`: The number of vectors needed to be read.
*/
template <class K, class V, class S>
__global__ void read_or_write_kernel(V** __restrict table_value_addrs,
                                     V* __restrict param_values,
                                     const bool* mask,
                                     const int* __restrict param_key_index,
                                     const size_t dim, const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_key_index =
        param_key_index != nullptr ? param_key_index[vec_index] : vec_index;

    /// if found, read the value form table, otherwise write it
    if (table_value_addrs[vec_index] != nullptr) {
      /// find
      if (mask[real_key_index]) {
        param_values[real_key_index * dim + dim_index] =
            table_value_addrs[vec_index][dim_index];
      }
      /// insert
      else {
        table_value_addrs[vec_index][dim_index] =
            param_values[real_key_index * dim + dim_index];
      }
    }
  }
}

/* If founds[i] = true, read data from corresponding address of
 * table_value_addrs and write to param_values; if founds[i] = false, write data
 * from param_values to corresponding address of table_value_addrs. usually
 * called by find_or_insert kernel.
 */
template <class V>
void read_or_write_by_cpu(V** __restrict table_value_addrs,
                          V* __restrict param_values,
                          const int* __restrict offset, const bool* founds,
                          size_t dim, int N, int n_worker = 16) {
  std::vector<std::thread> thds;
  if (n_worker < 1) n_worker = 1;

  auto functor = [founds, dim](V** __restrict table_value_addrs,
                               V* __restrict param_values,
                               const int* __restrict offset, int handled_size,
                               int trunk_size) -> void {
    for (int i = handled_size; i < handled_size + trunk_size; i++) {
      if (table_value_addrs[i] != nullptr) {
        if (founds[offset[i]]) {
          memcpy(param_values + offset[i] * dim, table_value_addrs[i],
                 sizeof(V) * dim);
        } else {
          memcpy(table_value_addrs[i], param_values + offset[i] * dim,
                 sizeof(V) * dim);
        }
      }
    }
  };

  int32_t trunk_size_floor = N / n_worker;
  int32_t trunk_size_remain = N % n_worker;
  int32_t n_worker_used = trunk_size_floor == 0 ? trunk_size_remain : n_worker;

  size_t handled_size = 0;
  for (int i = 0; i < n_worker_used; i++) {
    int32_t cur_trunk_size = trunk_size_floor;
    if (trunk_size_remain != 0) {
      cur_trunk_size += 1;
      trunk_size_remain--;
    }
    thds.push_back(std::thread(functor, table_value_addrs, param_values, offset,
                               handled_size, cur_trunk_size));
    handled_size += cur_trunk_size;
  }

  for (int i = 0; i < n_worker_used; i++) {
    thds[i].join();
  }
}

}  // namespace merlin
}  // namespace nv