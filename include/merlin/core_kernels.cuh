/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thread>
#include <vector>
#include "types.cuh"
#include "utils.cuh"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace nv {
namespace merlin {

template <class M>
__global__ void create_locks(M* __restrict mutex, const size_t start,
                             const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    new (mutex + start + tid) M(1);
  }
}

template <class M>
__global__ void release_locks(M* __restrict mutex, const size_t start,
                              const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    (mutex + start + tid)->~M();
  }
}

/* 2GB per slice by default.*/
constexpr size_t kDefaultBytesPerSlice = (8ul << 30);

/* Initialize the buckets with index from start to end. */
template <class K, class V, class M, size_t DIM>
void initialize_buckets(Table<K, V, M, DIM>** table, const size_t start,
                        const size_t end) {
  /* As testing results show us, when the number of buckets is greater than
   * the 4 million the performance will drop significantly, we believe the
   * to many pinned memory allocation causes this issue, so we change the
   * strategy to allocate some memory slices whose size is not greater than
   * 64GB, and put the buckets pointer point to the slices.
   */
  MERLIN_CHECK(start < end,
               "initialize_buckets, start should be less than end!");
  size_t buckets_num = end - start;
  const size_t total_size_of_vectors =
      buckets_num * (*table)->bucket_max_size * sizeof(V);
  const size_t num_of_memory_slices =
      1 + (total_size_of_vectors - 1) / (*table)->bytes_per_slice;
  size_t num_of_buckets_in_one_slice =
      (*table)->bytes_per_slice / ((*table)->bucket_max_size * sizeof(V));
  size_t num_of_allocated_buckets = 0;

  realloc_managed<V**>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V*),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V*));

  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == (*table)->num_of_memory_slices + num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size =
        num_of_buckets_in_one_slice * (*table)->bucket_max_size * sizeof(V);
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      CUDA_CHECK(cudaMalloc(&((*table)->slices[i]), slice_real_size));
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      (*table)->is_pure_hbm = false;
      CUDA_CHECK(
          cudaMallocHost(&((*table)->slices[i]), slice_real_size,
                         cudaHostAllocMapped | cudaHostAllocWriteCombined));
    }
    for (int j = 0; j < num_of_buckets_in_one_slice; j++) {
      (*table)->buckets[start + num_of_allocated_buckets + j].vectors =
          (*table)->slices[i] + j * (*table)->bucket_max_size;
    }
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }

  (*table)->num_of_memory_slices += num_of_memory_slices;
  for (int i = start; i < end; i++) {
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].keys),
                          (*table)->bucket_max_size * sizeof(K)));
    CUDA_CHECK(cudaMemset((*table)->buckets[i].keys, 0xFF,
                          (*table)->bucket_max_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].metas),
                          (*table)->bucket_max_size * sizeof(Meta<M>)));
  }

  {
    const size_t block_size = 512;
    const size_t N = (*table)->buckets_num;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_locks<Mutex><<<grid_size, block_size>>>((*table)->locks, start, end);
  }
  CudaCheckError();
}

/* Initialize a Table struct.

   K: The key type
   V: The value type which should be static array type and C++ class
      with customized construct is not supported.
   M: The meta type, the meta will be used to store the timestamp
      or occurrence frequency or any thing for eviction.
   DIM: Vector dimension.
*/
template <class K, class V, class M, size_t DIM>
void create_table(Table<K, V, M, DIM>** table,
                  const size_t init_size = 134217728,
                  const size_t max_size = std::numeric_limits<size_t>::max(),
                  const size_t max_hbm_for_vectors = 0,
                  const size_t bucket_max_size = 128,
                  const size_t tile_size = 32, const bool primary = true,
                  const size_t bytes_per_slice = kDefaultBytesPerSlice) {
  CUDA_CHECK(cudaMallocManaged((void**)table, sizeof(Table<K, V, M, DIM>)));
  CUDA_CHECK(cudaMemset(*table, 0, sizeof(Table<K, V, M, DIM>)));
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->bytes_per_slice = bytes_per_slice;
  (*table)->max_size = max_size;
  (*table)->tile_size = tile_size;
  (*table)->is_pure_hbm = true;

  (*table)->buckets_num = 1;
  while ((*table)->buckets_num * (*table)->bucket_max_size < init_size) {
    (*table)->buckets_num *= 2;
  }
  (*table)->capacity = (*table)->buckets_num * (*table)->bucket_max_size;
  (*table)->max_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->remaining_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->primary = primary;

  CUDA_CHECK(cudaMalloc((void**)&((*table)->locks),
                        (*table)->buckets_num * sizeof(Mutex)));
  CUDA_CHECK(
      cudaMemset((*table)->locks, 0, (*table)->buckets_num * sizeof(Mutex)));

  CUDA_CHECK(cudaMalloc((void**)&((*table)->buckets_size),
                        (*table)->buckets_num * sizeof(int)));
  CUDA_CHECK(cudaMemset((*table)->buckets_size, 0,
                        (*table)->buckets_num * sizeof(int)));

  CUDA_CHECK(
      cudaMallocManaged((void**)&((*table)->buckets),
                        (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>)));
  CUDA_CHECK(cudaMemset((*table)->buckets, 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>)));

  initialize_buckets<K, V, M, DIM>(table, 0, (*table)->buckets_num);
  CudaCheckError();
}

/* Double the capacity on storage, must be followed by calling the
 * rehash_kernel. */
template <class K, class V, class M, size_t DIM>
void double_capacity(Table<K, V, M, DIM>** table) {
  realloc<Mutex*>(&((*table)->locks), (*table)->buckets_num * sizeof(Mutex),
                  (*table)->buckets_num * sizeof(Mutex) * 2);
  realloc<int*>(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
                (*table)->buckets_num * sizeof(int) * 2);

  realloc_managed<Bucket<K, V, M, DIM>*>(
      &((*table)->buckets),
      (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>),
      (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>) * 2);

  initialize_buckets<K, V, M, DIM>(table, (*table)->buckets_num,
                                   (*table)->buckets_num * 2);

  (*table)->capacity *= 2;
  (*table)->buckets_num *= 2;
}

/* free all of the resource of a Table. */
template <class K, class V, class M, size_t DIM>
void destroy_table(Table<K, V, M, DIM>** table) {
  for (int i = 0; i < (*table)->buckets_num; i++) {
    CUDA_CHECK(cudaFree((*table)->buckets[i].keys));
    CUDA_CHECK(cudaFree((*table)->buckets[i].metas));
  }

  for (int i = 0; i < (*table)->num_of_memory_slices; i++) {
    if (is_on_device((*table)->slices[i])) {
      CUDA_CHECK(cudaFree((*table)->slices[i]));
    } else {
      CUDA_CHECK(cudaFreeHost((*table)->slices[i]));
    }
  }
  {
    const size_t block_size = 512;
    const size_t N = (*table)->buckets_num;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    release_locks<Mutex>
        <<<grid_size, block_size>>>((*table)->locks, 0, (*table)->buckets_num);
  }
  CUDA_CHECK(cudaFree((*table)->slices));
  CUDA_CHECK(cudaFree((*table)->buckets_size));
  CUDA_CHECK(cudaFree((*table)->buckets));
  CUDA_CHECK(cudaFree((*table)->locks));
  CUDA_CHECK(cudaFree(*table));
  CudaCheckError();
}

template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__forceinline__ __device__ void defragmentation_for_rehash(
    Bucket<K, V, M, DIM>* __restrict bucket, uint32_t remove_pos,
    const size_t bucket_max_size, const size_t buckets_num) {
  uint32_t key_idx;
  size_t global_idx = 0;
  size_t start_idx = 0;
  K find_key;
  K hashed_key;

  uint32_t empty_pos = remove_pos;

  int i = 1;
  while (i < bucket_max_size) {
    key_idx = (remove_pos + i) & (bucket_max_size - 1);
    find_key = *(bucket->keys + key_idx);
    if (find_key == EMPTY_KEY) {
      break;
    }
    hashed_key = Murmur3HashDevice(find_key);
    global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    start_idx = global_idx % bucket_max_size;

    if ((start_idx <= empty_pos && empty_pos < key_idx) ||
        (key_idx < start_idx && start_idx <= empty_pos) ||
        (empty_pos <= key_idx && key_idx < start_idx)) {
      *(bucket->keys + empty_pos) = *(bucket->keys + key_idx);
      bucket->metas[empty_pos].val = bucket->metas[key_idx].val;
      for (int j = 0; j < DIM; j++) {
        bucket->vectors[empty_pos].values[j] =
            bucket->vectors[key_idx].values[j];
      }
      *(bucket->keys + key_idx) = EMPTY_KEY;
      empty_pos = key_idx;
      remove_pos = key_idx;
      i = 1;
    } else {
      i++;
    }
  }
}

template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__forceinline__ __device__ void refresh_bucket_meta(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M, DIM>* bucket,
    const size_t bucket_max_size) {
  M min_val = MAX_META;
  int min_pos = 0;

  for (int i = g.thread_rank(); i < bucket_max_size; i += TILE_SIZE) {
    if (bucket->keys[i] == EMPTY_KEY) {
      continue;
    }
    if (bucket->metas[i].val < min_val) {
      min_pos = i;
      min_val = bucket->metas[i].val;
    }
  }
  M global_min_val = cg::reduce(g, min_val, cg::less<M>());
  if (min_val == global_min_val) {
    bucket->min_pos = min_pos;
    bucket->min_meta = min_val;
  }
}

template <class V, size_t DIM, uint32_t TILE_SIZE = 8>
__forceinline__ __device__ void copy_vector(cg::thread_block_tile<TILE_SIZE> g,
                                            const V* src, V* dst) {
  for (auto i = g.thread_rank(); i < DIM; i += g.size()) {
    dst->values[i] = src->values[i];
  }
}

/* Write the N data from src to each address in *dst by using CPU threads,
 * usually called by upsert kernel.
 *
 * @note: In some machines with AMD CPUs, the `write_kernel` has low performance
 * thru PCI-E, so we try to use the `memcpy` on CPU threads for writing work to
 * reach better performance.
 */
template <class V>
void write_by_cpu(V** __restrict dst, const V* __restrict src,
                  const int* __restrict offset, int N, int n_worker = 16) {
  std::vector<std::thread> thds;
  if (n_worker < 1) n_worker = 1;

  auto functor = [](V** __restrict dst, const V* __restrict src,
                    const int* __restrict offset, int handled_size,
                    int trunk_size) -> void {
    for (int i = handled_size; i < handled_size + trunk_size; i++) {
      memcpy(dst[i], src + offset[i], sizeof(V));
    }
  };

  size_t trunk_size = N / n_worker;
  size_t handled_size = 0;
  for (int i = 0; i < n_worker - 1; i++) {
    thds.push_back(
        std::thread(functor, dst, src, offset, handled_size, trunk_size));
    handled_size += trunk_size;
  }

  size_t remaining = N - handled_size;
  thds.push_back(
      std::thread(functor, dst, src, offset, handled_size, remaining));

  for (int i = 0; i < n_worker; i++) {
    thds[i].join();
  }
}

template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__forceinline__ __device__ void move_key_to_new_bucket(
    cg::thread_block_tile<TILE_SIZE> g, int rank, const K& key, const M& meta,
    const V* __restrict vector, Bucket<K, V, M, DIM>* __restrict new_bucket,
    const size_t new_bkt_idx, const size_t new_start_idx,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num) {
  uint32_t key_pos;
  unsigned empty_vote;
  int local_size;
  int src_lane;

#pragma unroll
  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    size_t key_offset =
        (new_start_idx + tile_offset + rank) & (bucket_max_size - 1);
    K current_key = *(new_bucket->keys + key_offset);
    empty_vote = g.ballot(current_key == EMPTY_KEY);
    if (empty_vote) {
      src_lane = __ffs(empty_vote) - 1;
      key_pos =
          (new_start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
      local_size = buckets_size[new_bkt_idx];
      if (rank == src_lane) {
        new_bucket->keys[key_pos] = key;
        new_bucket->metas[key_pos].val = meta;
        buckets_size[new_bkt_idx]++;
      }
      local_size = g.shfl(local_size, src_lane);
      if (local_size >= bucket_max_size) {
        refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, new_bucket,
                                                     bucket_max_size);
      }
      copy_vector<V, DIM, TILE_SIZE>(g, vector, new_bucket->vectors + key_pos);
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void rehash_kernel_for_fast_mode(
    const Table<K, V, M, DIM>* __restrict table,
    Bucket<K, V, M, DIM>* __restrict buckets, int* __restrict buckets_size,
    const size_t bucket_max_size, const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t global_idx;
  uint32_t start_idx = 0;
  K target_key = 0;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t / TILE_SIZE;
    Bucket<K, V, M, DIM>* bucket = (buckets + bkt_idx);

    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    uint32_t key_idx = 0;
    while (key_idx < bucket_max_size) {
      key_idx = g.shfl(key_idx, 0);
      target_key = bucket->keys[key_idx];
      if (target_key != EMPTY_KEY) {
        K hashed_key = Murmur3HashDevice(target_key);
        global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
        uint32_t new_bkt_idx = global_idx / bucket_max_size;
        if (new_bkt_idx != bkt_idx) {
          start_idx = global_idx % bucket_max_size;
          move_key_to_new_bucket<K, V, M, DIM, TILE_SIZE>(
              g, rank, target_key, bucket->metas[key_idx].val,
              (bucket->vectors + key_idx), buckets + new_bkt_idx, new_bkt_idx,
              start_idx, buckets_size, bucket_max_size, buckets_num);
          if (rank == 0) {
            bucket->keys[key_idx] = EMPTY_KEY;
            buckets_size[bkt_idx]--;
            defragmentation_for_rehash<K, V, M, DIM, TILE_SIZE>(
                bucket, key_idx, bucket_max_size, buckets_num / 2);
            key_idx = 0;
          }
        } else {
          key_idx++;
        }
      } else {
        key_idx++;
      }
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Write the N data from src to each address in *dst,
   usually called by upsert kernel.

   `src`: A continuous memory pointer with Vector
          which can be HBM.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: Number of vectors that need to be written.
*/
template <class K, class V, class M, size_t DIM>
__global__ void write_kernel(const V* __restrict src, V** __restrict dst,
                             const int* __restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr) {
      if (src_offset != nullptr) {
        (*(dst[vec_index])).values[dim_index] =
            src[src_offset[vec_index]].values[dim_index];
      } else {
        (*(dst[vec_index])).values[dim_index] =
            src[vec_index].values[dim_index];
      }
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
template <class K, class V, class M, size_t DIM>
__global__ void write_with_accum_kernel(const V* __restrict delta_or_val,
                                        V** __restrict dst,
                                        const bool* __restrict existed,
                                        const bool* __restrict status,
                                        const int* __restrict src_offset,
                                        size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr &&
        existed[src_offset[vec_index]] == status[src_offset[vec_index]]) {
      if (status[src_offset[vec_index]]) {
        (*(dst[vec_index])).values[dim_index] +=
            delta_or_val[src_offset[vec_index]].values[dim_index];
      } else {
        (*(dst[vec_index])).values[dim_index] =
            delta_or_val[src_offset[vec_index]].values[dim_index];
      }
    }
  }
}

/* Add a @delta[i] to the the value saved in the address @dst[i].

   `delta`: a delta value which should be add to.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class M, size_t DIM>
__global__ void write_with_accum_kernel(const V* __restrict delta,
                                        V** __restrict dst,
                                        const int* __restrict src_offset,
                                        size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr) {
      (*(dst[vec_index])).values[dim_index] +=
          delta[src_offset[vec_index]].values[dim_index];
    }
  }
}

/* Read the N data from src to each address in *dst,
   usually called by upsert kernel.

   `src`: A pointer of pointer of V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `dst`: A continue memory pointer with Vector
          which should be HBM.
   `mask`: One for each `dst`. If true, reading from src,
           or false reading from default_val.
   `default_val`: Default value with shape (1, DIM) or (N, DIM)
   `N`: The number of vectors needed to be read.
   'full_size_default':
      If true, the d_def_val will be treated as
      a full size default value which shape must be (N, DIM).
*/
template <class K, class V, class M, size_t DIM>
__global__ void read_kernel(const V* const* __restrict src, V* __restrict dst,
                            const bool* mask, const int* __restrict dst_offset,
                            size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;

    /// Copy selected values and fill in default value for all others.
    if (mask[real_dst_offset] && src[vec_index] != nullptr) {
      dst[real_dst_offset].values[dim_index] =
          src[vec_index]->values[dim_index];
    }
  }
}

/* Read the N data from src to each address in *dst,
 *  usually called by upsert kernel.
 *
 *  `src`: A pointer of pointer of V which should be on HBM,
 *         but each value (a pointer of V) could point to a
 *         memory on HBM or HMEM.
 *  `dst`: A continue memory pointer with Vector
 *         which should be HBM.
 *  `N`: Number of vectors needed to be read.
 */
template <class K, class V, class M, size_t DIM>
__global__ void read_kernel(V** __restrict src, V* __restrict dst,
                            const int* __restrict dst_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;
    int dim_index = t % DIM;
    if (src[vec_index] != nullptr) {
      dst[real_dst_offset].values[dim_index] =
          src[vec_index]->values[dim_index];
    }
  }
}

/* Upsert with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void upsert_kernel_with_io(
    const Table<K, V, M, DIM>* __restrict table, const K* __restrict keys,
    const V* __restrict values, const M* __restrict metas,
    Bucket<K, V, M, DIM>* __restrict buckets, int* __restrict buckets_size,
    const size_t bucket_max_size, const size_t buckets_num, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    unsigned found_or_empty_vote = 0;
    unsigned reclaim_vote = 0;
    unsigned reclaim_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    uint32_t tile_offset = 0;
    size_t key_offset = 0;
    K current_key = 0;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      reclaim_vote = g.ballot(current_key == RECLAIM_KEY);
      if (found_or_empty_vote || reclaim_vote) {
        if (found_or_empty_vote) {
          src_lane = __ffs(found_or_empty_vote) - 1;
        } else {
          src_lane = __ffs(reclaim_vote) - 1;
        }
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          bucket->keys[key_pos] = insert_key;
          bucket->metas[key_pos].val = metas[key_idx];
          if (current_key == EMPTY_KEY || reclaim_vote) {
            buckets_size[bkt_idx]++;
            local_size++;
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        copy_vector<V, DIM, TILE_SIZE>(g, values + key_idx,
                                       bucket->vectors + key_pos);
        tile_offset += TILE_SIZE;
        break;
      }
    }

    // When insert to reclaimed position, continue the loop for erase duplicated
    // key.
    if (!found_or_empty_vote && reclaim_vote) {
      for (; tile_offset < bucket_max_size; tile_offset += TILE_SIZE) {
        key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
        current_key = *(bucket->keys + key_offset);
        reclaim_or_empty_vote =
            g.ballot(insert_key == current_key || current_key == EMPTY_KEY);
        if (reclaim_or_empty_vote) {
          src_lane = __ffs(reclaim_or_empty_vote) - 1;
          key_pos =
              (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
          if (rank == src_lane && current_key == insert_key) {
            bucket->keys[key_pos] = RECLAIM_KEY;
          }
          break;
        }
      }
    }

    if (!found_or_empty_vote && !reclaim_vote) {
      src_lane = (bucket->min_pos % TILE_SIZE);
      if (rank == src_lane) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        bucket->metas[key_pos].val = metas[key_idx];
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
      key_pos = g.shfl(key_pos, src_lane);
      copy_vector<V, DIM, TILE_SIZE>(g, values + key_idx,
                                     bucket->vectors + key_pos);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Upsert with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void upsert_kernel_with_io(
    const Table<K, V, M, DIM>* __restrict table, const K* __restrict keys,
    const V* __restrict values, Bucket<K, V, M, DIM>* __restrict buckets,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    unsigned found_or_empty_vote = 0;
    unsigned reclaim_vote = 0;
    unsigned reclaim_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    uint32_t tile_offset = 0;
    size_t key_offset = 0;
    K current_key = 0;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      reclaim_vote = g.ballot(current_key == RECLAIM_KEY);
      if (found_or_empty_vote || reclaim_vote) {
        if (found_or_empty_vote) {
          src_lane = __ffs(found_or_empty_vote) - 1;
        } else {
          src_lane = __ffs(reclaim_vote) - 1;
        }
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          bucket->keys[key_pos] = insert_key;
          M cur_meta = bucket->cur_meta + 1;
          bucket->cur_meta = cur_meta;
          bucket->metas[key_pos].val = cur_meta;
          if (current_key == EMPTY_KEY || current_key == RECLAIM_KEY) {
            buckets_size[bkt_idx]++;
            local_size++;
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        copy_vector<V, DIM, TILE_SIZE>(g, values + key_idx,
                                       bucket->vectors + key_pos);
        tile_offset += TILE_SIZE;
        break;
      }
    }

    // When insert to reclaimed position, continue the loop for erase duplicated
    // key.
    if (!found_or_empty_vote && reclaim_vote) {
      for (; tile_offset < bucket_max_size; tile_offset += TILE_SIZE) {
        key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
        current_key = *(bucket->keys + key_offset);
        reclaim_or_empty_vote =
            g.ballot(insert_key == current_key || current_key == EMPTY_KEY);
        if (reclaim_or_empty_vote) {
          src_lane = __ffs(reclaim_or_empty_vote) - 1;
          key_pos =
              (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
          if (rank == src_lane && current_key == insert_key) {
            bucket->keys[key_pos] = RECLAIM_KEY;
          }
          break;
        }
      }
    }

    if (!found_or_empty_vote && !reclaim_vote) {
      src_lane = (bucket->min_pos % TILE_SIZE);
      if (rank == src_lane) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        M cur_meta = bucket->cur_meta + 1;
        bucket->cur_meta = cur_meta;
        bucket->metas[key_pos].val = cur_meta;
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
      key_pos = g.shfl(key_pos, src_lane);
      copy_vector<V, DIM, TILE_SIZE>(g, values + key_idx,
                                     bucket->vectors + key_pos);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Upsert with the end-user specified meta.
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void upsert_kernel(const Table<K, V, M, DIM>* __restrict table,
                              const K* __restrict keys, V** __restrict vectors,
                              const M* __restrict metas,
                              Bucket<K, V, M, DIM>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num,
                              int* __restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    unsigned found_or_empty_vote = 0;
    unsigned reclaim_vote = 0;
    unsigned reclaim_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    uint32_t tile_offset = 0;
    size_t key_offset = 0;
    K current_key = 0;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      reclaim_vote = g.ballot(current_key == RECLAIM_KEY);
      if (found_or_empty_vote || reclaim_vote) {
        if (found_or_empty_vote) {
          src_lane = __ffs(found_or_empty_vote) - 1;
        } else {
          src_lane = __ffs(reclaim_vote) - 1;
        }
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          bucket->keys[key_pos] = insert_key;
          if (current_key == EMPTY_KEY || current_key == RECLAIM_KEY) {
            buckets_size[bkt_idx]++;
            local_size++;
          }
          *(vectors + key_idx) = (bucket->vectors + key_pos);
          bucket->metas[key_pos].val = metas[key_idx];
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        tile_offset += TILE_SIZE;
        break;
      }
    }

    // When insert to reclaimed position, continue the loop for erase duplicated
    // key.
    if (!found_or_empty_vote && reclaim_vote) {
      for (; tile_offset < bucket_max_size; tile_offset += TILE_SIZE) {
        key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
        current_key = *(bucket->keys + key_offset);
        reclaim_or_empty_vote =
            g.ballot(insert_key == current_key || current_key == EMPTY_KEY);
        if (reclaim_or_empty_vote) {
          src_lane = __ffs(reclaim_or_empty_vote) - 1;
          key_pos =
              (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
          if (rank == src_lane && current_key == insert_key) {
            bucket->keys[key_pos] = RECLAIM_KEY;
          }
          break;
        }
      }
    }

    if (!found_or_empty_vote && !reclaim_vote) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        bucket->metas[key_pos].val = metas[key_idx];
        *(vectors + key_idx) = (bucket->vectors + key_pos);
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Upsert with no user specified meta.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `size_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void upsert_kernel(const Table<K, V, M, DIM>* __restrict table,
                              const K* __restrict keys, V** __restrict vectors,
                              Bucket<K, V, M, DIM>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num,
                              int* __restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    unsigned found_or_empty_vote = 0;
    unsigned reclaim_vote = 0;
    unsigned reclaim_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    uint32_t tile_offset = 0;
    size_t key_offset = 0;
    K current_key = 0;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      reclaim_vote = g.ballot(current_key == RECLAIM_KEY);
      if (found_or_empty_vote || reclaim_vote) {
        if (found_or_empty_vote) {
          src_lane = __ffs(found_or_empty_vote) - 1;
        } else {
          src_lane = __ffs(reclaim_vote) - 1;
        }
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          bucket->keys[key_pos] = insert_key;
          if (current_key == EMPTY_KEY || current_key == RECLAIM_KEY) {
            buckets_size[bkt_idx]++;
            local_size++;
          }
          *(vectors + key_idx) = (bucket->vectors + key_pos);
          M cur_meta = bucket->cur_meta + 1;
          bucket->cur_meta = cur_meta;
          bucket->metas[key_pos].val = cur_meta;
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        tile_offset += TILE_SIZE;
        break;
      }
    }

    // When insert to reclaimed position, continue the loop for erase duplicated
    // key.
    if (!found_or_empty_vote && reclaim_vote) {
      for (; tile_offset < bucket_max_size; tile_offset += TILE_SIZE) {
        key_offset = (start_idx + tile_offset + rank) & (bucket_max_size - 1);
        current_key = *(bucket->keys + key_offset);
        reclaim_or_empty_vote =
            g.ballot(insert_key == current_key || current_key == EMPTY_KEY);
        if (reclaim_or_empty_vote) {
          src_lane = __ffs(reclaim_or_empty_vote) - 1;
          key_pos =
              (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
          if (rank == src_lane && current_key == insert_key) {
            bucket->keys[key_pos] = RECLAIM_KEY;
          }
          break;
        }
      }
    }

    if (!found_or_empty_vote && !reclaim_vote) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        *(vectors + key_idx) = (bucket->vectors + key_pos);
        M cur_meta = bucket->cur_meta + 1;
        bucket->cur_meta = cur_meta;
        bucket->metas[key_pos].val = cur_meta;
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Upsert with no user specified meta and return if the keys already exists.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `size_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void accum_kernel(
    const Table<K, V, M, DIM>* __restrict table, const K* __restrict keys,
    V** __restrict vectors, const bool* __restrict existed,
    Bucket<K, V, M, DIM>* __restrict buckets, int* __restrict buckets_size,
    const size_t bucket_max_size, const size_t buckets_num,
    int* __restrict src_offset, bool* __restrict status, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    bool local_found = false;
    unsigned found_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

#pragma unroll
    for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      size_t key_offset =
          (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      K current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          if (current_key == insert_key) {
            local_found = true;
            *(status + key_idx) = local_found;
          }
          if (local_found == existed[key_idx]) {
            bucket->keys[key_pos] = insert_key;
            if (!local_found) {
              buckets_size[bkt_idx]++;
              local_size++;
            }
            *(vectors + key_idx) = (bucket->vectors + key_pos);
            M cur_meta = bucket->cur_meta + 1;
            bucket->cur_meta = cur_meta;
            bucket->metas[key_pos].val = cur_meta;
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        break;
      }
    }
    if (!found_or_empty_vote) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        *(vectors + key_idx) = (bucket->vectors + key_pos);
        M cur_meta = bucket->cur_meta + 1;
        bucket->cur_meta = cur_meta;
        bucket->metas[key_pos].val = cur_meta;
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Accum kernel with customized metas.
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void accum_kernel(
    const Table<K, V, M, DIM>* __restrict table, const K* __restrict keys,
    V** __restrict vectors, const M* __restrict metas,
    const bool* __restrict existed, Bucket<K, V, M, DIM>* __restrict buckets,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, int* __restrict src_offset,
    bool* __restrict status, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    bool local_found = false;
    unsigned found_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);
    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

#pragma unroll
    for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      size_t key_offset =
          (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      K current_key = *(bucket->keys + key_offset);
      found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          if (current_key == insert_key) {
            local_found = true;
            *(status + key_idx) = local_found;
          }
          if (local_found == existed[key_idx]) {
            bucket->keys[key_pos] = insert_key;
            if (!local_found) {
              buckets_size[bkt_idx]++;
              local_size++;
            }
            *(vectors + key_idx) = (bucket->vectors + key_pos);
            bucket->metas[key_pos].val = metas[key_idx];
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket,
                                                       bucket_max_size);
        }
        break;
      }
    }
    if (!found_or_empty_vote && metas[key_idx] > bucket->min_meta &&
        !existed[key_idx]) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        *(bucket->keys + key_pos) = insert_key;
        *(vectors + key_idx) = (bucket->vectors + key_pos);
        bucket->metas[key_pos].val = metas[key_idx];
      }
      refresh_bucket_meta<K, V, M, DIM, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void lookup_kernel_with_io(
    const Table<K, V, M, DIM>* __restrict table, const K* __restrict keys,
    V* __restrict values, M* __restrict metas, bool* __restrict found,
    Bucket<K, V, M, DIM>* __restrict buckets, int* __restrict buckets_size,
    const size_t bucket_max_size, const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;
    int key_pos = -1;
    bool local_found = false;

    K find_key = keys[key_idx];
    uint32_t hashed_key = Murmur3HashDevice(find_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

    uint32_t tile_offset = 0;
    uint32_t key_offset = 0;
    K current_key = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) % bucket_max_size;
      current_key = *(bucket->keys + key_offset);
      auto const found_or_empty_vote =
          g.ballot(find_key == current_key || current_key == EMPTY_KEY);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        if (src_lane == rank) {
          local_found = (current_key == find_key);
        }
        local_found = g.shfl(local_found, src_lane);
        break;
      }
    }

    if (rank == 0) {
      if (metas != nullptr && local_found) {
        *(metas + key_idx) = bucket->metas[key_pos].val;
      }
      if (found != nullptr) {
        *(found + key_idx) = local_found;
      }
    }

    if (local_found) {
      copy_vector<V, DIM, TILE_SIZE>(g, bucket->vectors + key_pos,
                                     values + key_idx);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* lookup kernel.
 */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void lookup_kernel(const Table<K, V, M, DIM>* __restrict table,
                              const K* __restrict keys, V** __restrict vectors,
                              M* __restrict metas, bool* __restrict found,
                              Bucket<K, V, M, DIM>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num,
                              int* __restrict dst_offset, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;
    int key_pos = -1;
    bool local_found = false;

    K find_key = keys[key_idx];
    uint32_t hashed_key = Murmur3HashDevice(find_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

    uint32_t tile_offset = 0;
    uint32_t key_offset = 0;
    K current_key = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) % bucket_max_size;
      current_key = *(bucket->keys + key_offset);
      auto const found_or_empty_vote =
          g.ballot(find_key == current_key || current_key == EMPTY_KEY);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        if (src_lane == rank) {
          local_found = (current_key == find_key);
        }
        local_found = g.shfl(local_found, src_lane);
        break;
      }
    }

    if (rank == 0) {
      *(vectors + key_idx) =
          local_found ? (bucket->vectors + key_pos) : nullptr;
      if (metas != nullptr && local_found) {
        *(metas + key_idx) = bucket->metas[key_pos].val;
      }
      if (found != nullptr) {
        *(found + key_idx) = local_found;
      }
      if (dst_offset != nullptr) {
        *(dst_offset + key_idx) = key_idx;
      }
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class M, size_t DIM>
__global__ void clear_kernel(Table<K, V, M, DIM>* __restrict table, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, M, DIM>* bucket = &(table->buckets[bkt_idx]);

    bucket->keys[key_idx] = EMPTY_KEY;
    if (key_idx == 0) {
      table->buckets_size[bkt_idx] = 0;
    }
  }
}

/* Remove specified keys. */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 8>
__global__ void remove_kernel(const Table<K, V, M, DIM>* __restrict table,
                              const K* __restrict keys,
                              size_t* __restrict count,
                              Bucket<K, V, M, DIM>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;
    int key_pos = -1;
    bool local_found = false;

    K find_key = keys[key_idx];
    uint32_t hashed_key = Murmur3HashDevice(find_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

    uint32_t tile_offset = 0;
    uint32_t key_offset = 0;
    K current_key = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_offset = (start_idx + tile_offset + rank) % bucket_max_size;
      current_key = *(bucket->keys + key_offset);
      auto const found_or_empty_vote =
          g.ballot(find_key == current_key || current_key == EMPTY_KEY);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        if (src_lane == rank) {
          local_found = (current_key == find_key);
          if (local_found) {
            atomicAdd(count, 1);
            *(bucket->keys + key_pos) = RECLAIM_KEY;
            buckets_size[bkt_idx]--;
          }
        }
        break;
      }
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Remove specified keys which match the Predict. */
template <class K, class V, class M, size_t DIM, uint32_t TILE_SIZE = 1>
__global__ void remove_kernel(const Table<K, V, M, DIM>* __restrict table,
                              const EraseIfPredictInternal<K, M> pred,
                              const K pattern, const M threshold,
                              size_t* __restrict count,
                              Bucket<K, V, M, DIM>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t;
    uint32_t key_pos = 0;

    Bucket<K, V, M, DIM>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);

    K current_key = 0;
    uint32_t key_offset = 0;
    while (key_offset < bucket_max_size) {
      current_key = *(bucket->keys + key_offset);
      if (current_key != EMPTY_KEY) {
        if (pred(current_key, bucket->metas[key_offset].val, pattern,
                 threshold)) {
          atomicAdd(count, 1);
          key_pos = key_offset;
          *(bucket->keys + key_pos) = RECLAIM_KEY;
          buckets_size[bkt_idx]--;
        } else {
          key_offset++;
        }
      } else {
        key_offset++;
      }
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Dump with meta. */
template <class K, class V, class M, size_t DIM>
__global__ void dump_kernel(const Table<K, V, M, DIM>* __restrict table,
                            K* d_key, V* __restrict d_val, M* __restrict d_meta,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  K* smem = (K*)s;
  K* block_result_key = smem;
  V* block_result_val = (V*)&(smem[blockDim.x]);
  M* block_result_meta = (M*)&(block_result_val[blockDim.x]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;
  const size_t bucket_max_size = table->bucket_max_size;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / bucket_max_size;
    int key_idx = (tid + offset) % bucket_max_size;
    Bucket<K, V, M, DIM>* bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_idx] != EMPTY_KEY) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = bucket->keys[key_idx];
      for (int i = 0; i < DIM; i++) {
        atomicExch(&(block_result_val[local_index].values[i]),
                   bucket->vectors[key_idx].values[i]);
      }
      if (d_meta != nullptr) {
        block_result_meta[local_index] = bucket->metas[key_idx].val;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    for (int i = 0; i < DIM; i++) {
      d_val[global_acc + threadIdx.x].values[i] =
          block_result_val[threadIdx.x].values[i];
    }
    if (d_meta != nullptr) {
      d_meta[global_acc + threadIdx.x] = block_result_meta[threadIdx.x];
    }
  }
}

/* Dump with meta. */
template <class K, class V, class M, size_t DIM>
__global__ void dump_kernel(const Table<K, V, M, DIM>* __restrict table,
                            const EraseIfPredictInternal<K, M> pred,
                            const K pattern, const M threshold, K* d_key,
                            V* __restrict d_val, M* __restrict d_meta,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  K* smem = (K*)s;
  K* block_result_key = smem;
  V* block_result_val = (V*)&(smem[blockDim.x]);
  M* block_result_meta = (M*)&(block_result_val[blockDim.x]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;
  const size_t bucket_max_size = table->bucket_max_size;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / bucket_max_size;
    int key_idx = (tid + offset) % bucket_max_size;
    Bucket<K, V, M, DIM>* bucket = &(table->buckets[bkt_idx]);

    K key = bucket->keys[key_idx];
    M meta = bucket->metas[key_idx].val;

    if (key != EMPTY_KEY && pred(key, meta, pattern, threshold)) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = bucket->keys[key_idx];
      for (int i = 0; i < DIM; i++) {
        atomicExch(&(block_result_val[local_index].values[i]),
                   bucket->vectors[key_idx].values[i]);
      }
      if (d_meta != nullptr) {
        block_result_meta[local_index] = bucket->metas[key_idx].val;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    for (int i = 0; i < DIM; i++) {
      d_val[global_acc + threadIdx.x].values[i] =
          block_result_val[threadIdx.x].values[i];
    }
    if (d_meta != nullptr) {
      d_meta[global_acc + threadIdx.x] = block_result_meta[threadIdx.x];
    }
  }
}

}  // namespace merlin
}  // namespace nv
