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

#include "merlin/core_kernels/find_or_insert.cuh"
#include "merlin/core_kernels/find_ptr_or_insert.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
#include "merlin/core_kernels/lookup.cuh"
#include "merlin/core_kernels/lookup_ptr.cuh"
#include "merlin/core_kernels/update.cuh"
#include "merlin/core_kernels/upsert.cuh"
#include "merlin/core_kernels/upsert_and_evict.cuh"

namespace nv {
namespace merlin {

/* For improving performance consideration, allocating up to 64 table structures
 * in constant memory is supported. To close this function, please set
 * `TableOption::use_constant_memory` to `false`.
 */
constexpr int MAX_CONSTANT_TABLE = 64;
static std::mutex constant_table_mutex;
static uint64_t constant_table_flag = 0;

__constant__ char
    c_table_[sizeof(Table<uint64_t, float, uint64_t>) * MAX_CONSTANT_TABLE];

template <class T = uint64_t>
int allocate_constant_table() {
  std::lock_guard<std::mutex> guard(constant_table_mutex);
  if (constant_table_flag == std::numeric_limits<uint64_t>::max()) return -1;
  int table_index = 0;
  while (constant_table_flag & (1l << table_index)) {
    table_index++;
  }

  constant_table_flag = constant_table_flag | (1l << table_index);

  return table_index;
}

template <class T = uint64_t>
void release_constant_table(int table_index) {
  std::lock_guard<std::mutex> guard(constant_table_mutex);
  if (table_index < 0 || table_index >= MAX_CONSTANT_TABLE) return;
  constant_table_flag = constant_table_flag & (~(1l << table_index));
}

template <class S>
__global__ void create_locks(S* __restrict mutex, const size_t start,
                             const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    new (mutex + start + tid) S();
  }
}

template <class S>
__global__ void release_locks(S* __restrict mutex, const size_t start,
                              const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    (mutex + start + tid)->~S();
  }
}

template <class K, class V, class S>
__global__ void create_atomic_keys(Bucket<K, V, S>* __restrict buckets,
                                   const size_t start, const size_t end,
                                   const size_t bucket_max_size) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    for (size_t i = 0; i < bucket_max_size; i++)
      buckets[start + tid].digests(i)[0] = empty_digest<K>();
    for (size_t i = 0; i < bucket_max_size; i++)
      new (buckets[start + tid].keys(i))
          AtomicKey<K>{static_cast<K>(EMPTY_KEY)};
  }
}

template <class K, class V, class S>
__global__ void create_atomic_scores(Bucket<K, V, S>* __restrict buckets,
                                     const size_t start, const size_t end,
                                     const size_t bucket_max_size) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    for (size_t i = 0; i < bucket_max_size; i++) {
      new (buckets[start + tid].scores(i))
          AtomicScore<S>{static_cast<S>(EMPTY_SCORE)};
    }
    new (&(buckets[start + tid].min_score))
        AtomicScore<S>{static_cast<S>(EMPTY_SCORE)};
    new (&(buckets[start + tid].min_pos)) AtomicPos<int>{1};
  }
}

/* Initialize the buckets with index from start to end. */
template <class K, class V, class S>
void initialize_buckets(Table<K, V, S>** table, const size_t start,
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
      buckets_num * (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t num_of_memory_slices =
      1 + (total_size_of_vectors - 1) / (*table)->bytes_per_slice;
  size_t num_of_buckets_in_one_slice =
      (*table)->bytes_per_slice /
      ((*table)->bucket_max_size * sizeof(V) * (*table)->dim);
  size_t num_of_allocated_buckets = 0;

  realloc_managed<V**>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V*),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V*));

  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == (*table)->num_of_memory_slices + num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size = num_of_buckets_in_one_slice *
                             (*table)->bucket_max_size * sizeof(V) *
                             (*table)->dim;
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      CUDA_CHECK(cudaMalloc(&((*table)->slices[i]), slice_real_size));
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      (*table)->is_pure_hbm = false;
      CUDA_CHECK(cudaMallocHost(&((*table)->slices[i]), slice_real_size,
                                cudaHostAllocMapped));
    }
    for (int j = 0; j < num_of_buckets_in_one_slice; j++) {
      if ((*table)->is_pure_hbm) {
        (*table)->buckets[start + num_of_allocated_buckets + j].vectors =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
      } else {
        V* h_ptr =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
        CUDA_CHECK(cudaHostGetDevicePointer(
            &((*table)->buckets[start + num_of_allocated_buckets + j].vectors),
            h_ptr, 0));
      }
    }
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }

  (*table)->num_of_memory_slices += num_of_memory_slices;
  uint32_t bucket_max_size = static_cast<uint32_t>((*table)->bucket_max_size);
  size_t bucket_memory_size =
      bucket_max_size * (sizeof(AtomicKey<K>) + sizeof(AtomicScore<S>));
  // Align the digests according to the cache line size.
  uint32_t reserve_digest =
      ((bucket_max_size + 127) / 128) * 128 * sizeof(uint8_t);
  bucket_memory_size += reserve_digest;
  for (int i = start; i < end; i++) {
    CUDA_CHECK(
        cudaMalloc(&((*table)->buckets[i].digests_), bucket_memory_size));
    (*table)->buckets[i].keys_ =
        reinterpret_cast<AtomicKey<K>*>((*table)->buckets[i].digests_ + 128);
    (*table)->buckets[i].scores_ = reinterpret_cast<AtomicScore<S>*>(
        (*table)->buckets[i].keys_ + bucket_max_size);
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_locks<Mutex><<<grid_size, block_size>>>((*table)->locks, start, end);
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_atomic_keys<K, V, S><<<grid_size, block_size>>>(
        (*table)->buckets, start, end, (*table)->bucket_max_size);
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_atomic_scores<K, V, S><<<grid_size, block_size>>>(
        (*table)->buckets, start, end, (*table)->bucket_max_size);
  }
  CudaCheckError();
}

template <class K, class V, class S>
size_t get_slice_size(Table<K, V, S>** table) {
  const size_t min_slice_size =
      (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t max_table_size = (*table)->max_size * sizeof(V) * (*table)->dim;
  size_t slice_size = 0;

  if (max_table_size >= GB(128)) {
    slice_size = GB(16);
  } else if (max_table_size >= GB(16)) {
    slice_size = GB(2);
  } else if (max_table_size >= GB(2)) {
    slice_size = MB(128);
  } else if (max_table_size >= MB(128)) {
    slice_size = MB(16);
  } else if (max_table_size >= MB(16)) {
    slice_size = MB(1);
  } else {
    slice_size = min_slice_size;
  }

  return std::max(min_slice_size, slice_size);
}

/* Initialize a Table struct.

   K: The key type
   V: The value type which should be static array type and C++ class
      with customized construct is not supported.
   S: The score type, the score will be used to store the timestamp
      or occurrence frequency or any thing for eviction.
   DIM: Vector dimension.
*/
template <class K, class V, class S>
void create_table(Table<K, V, S>** table, const size_t dim,
                  const size_t init_size = 134217728,
                  const size_t max_size = std::numeric_limits<size_t>::max(),
                  const size_t max_hbm_for_vectors = 0,
                  const size_t bucket_max_size = 128,
                  const size_t tile_size = 32, const bool primary = true) {
  CUDA_CHECK(cudaMallocManaged((void**)table, sizeof(Table<K, V, S>)));
  CUDA_CHECK(cudaMemset(*table, 0, sizeof(Table<K, V, S>)));
  (*table)->dim = dim;
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->max_size = std::max(init_size, max_size);
  (*table)->tile_size = tile_size;
  (*table)->is_pure_hbm = true;
  (*table)->bytes_per_slice = get_slice_size<K, V, S>(table);

  // The bucket number will be the minimum needed for saving memory if no
  // rehash.
  if ((init_size * 2) > (*table)->max_size) {
    (*table)->buckets_num =
        1 + (((*table)->max_size - 1) / (*table)->bucket_max_size);
  } else {
    (*table)->buckets_num = 1;
    while ((*table)->buckets_num * (*table)->bucket_max_size < init_size) {
      (*table)->buckets_num *= 2;
    }
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
                        (*table)->buckets_num * sizeof(Bucket<K, V, S>)));
  CUDA_CHECK(cudaMemset((*table)->buckets, 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, S>)));

  initialize_buckets<K, V, S>(table, 0, (*table)->buckets_num);
  CudaCheckError();
}

/* Double the capacity on storage, must be followed by calling the
 * rehash_kernel. */
template <class K, class V, class S>
void double_capacity(Table<K, V, S>** table) {
  realloc<Mutex*>(&((*table)->locks), (*table)->buckets_num * sizeof(Mutex),
                  (*table)->buckets_num * sizeof(Mutex) * 2);
  realloc<int*>(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
                (*table)->buckets_num * sizeof(int) * 2);

  realloc_managed<Bucket<K, V, S>*>(
      &((*table)->buckets), (*table)->buckets_num * sizeof(Bucket<K, V, S>),
      (*table)->buckets_num * sizeof(Bucket<K, V, S>) * 2);

  initialize_buckets<K, V, S>(table, (*table)->buckets_num,
                              (*table)->buckets_num * 2);

  (*table)->capacity *= 2;
  (*table)->buckets_num *= 2;
}

/* free all of the resource of a Table. */
template <class K, class V, class S>
void destroy_table(Table<K, V, S>** table) {
  for (int i = 0; i < (*table)->buckets_num; i++) {
    CUDA_CHECK(cudaFree((*table)->buckets[i].digests_));
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
  CUDA_CHECK(cudaDeviceSynchronize());
  CudaCheckError();
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void defragmentation_for_rehash(
    Bucket<K, V, S>* __restrict bucket, uint32_t remove_pos,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim) {
  uint32_t key_idx;
  size_t global_idx = 0;
  size_t start_idx = 0;
  K find_key;
  K hashed_key;

  uint32_t empty_pos = remove_pos;

  int i = 1;
  while (i < bucket_max_size) {
    key_idx = (remove_pos + i) & (bucket_max_size - 1);
    find_key = (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    if (find_key == static_cast<K>(EMPTY_KEY)) {
      break;
    }
    hashed_key = Murmur3HashDevice(find_key);
    global_idx = hashed_key % (buckets_num * bucket_max_size);
    start_idx = global_idx % bucket_max_size;

    if ((start_idx <= empty_pos && empty_pos < key_idx) ||
        (key_idx < start_idx && start_idx <= empty_pos) ||
        (empty_pos <= key_idx && key_idx < start_idx)) {
      const K key =
          (*(bucket->keys(key_idx))).load(cuda::std::memory_order_relaxed);
      bucket->digests(empty_pos)[0] = get_digest<K>(key);
      (*(bucket->keys(empty_pos))).store(key, cuda::std::memory_order_relaxed);
      const S score =
          (*(bucket->scores(key_idx))).load(cuda::std::memory_order_relaxed);
      (*(bucket->scores(empty_pos)))
          .store(score, cuda::std::memory_order_relaxed);
      for (int j = 0; j < dim; j++) {
        bucket->vectors[empty_pos * dim + j] =
            bucket->vectors[key_idx * dim + j];
      }
      bucket->digests(key_idx)[0] = empty_digest<K>();
      (*(bucket->keys(key_idx)))
          .store(static_cast<K>(EMPTY_KEY), cuda::std::memory_order_relaxed);
      empty_pos = key_idx;
      remove_pos = key_idx;
      i = 1;
    } else {
      i++;
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void refresh_bucket_score(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, S>* bucket,
    const size_t bucket_max_size) {
  S min_val = MAX_SCORE;
  int min_pos = 0;

  for (int i = g.thread_rank(); i < bucket_max_size; i += TILE_SIZE) {
    const K key = (bucket->keys(i))->load(cuda::std::memory_order_relaxed);
    if (key == static_cast<K>(EMPTY_KEY) ||
        key == static_cast<K>(RECLAIM_KEY)) {
      continue;
    }
    const S score = bucket->scores(i)->load(cuda::std::memory_order_relaxed);
    if (score < min_val) {
      min_pos = i;
      min_val = score;
    }
  }
  S global_min_val = cg::reduce(g, min_val, cg::less<S>());
  if (min_val == global_min_val) {
    bucket->min_pos.store(min_pos, cuda::std::memory_order_relaxed);
    bucket->min_score.store(min_val, cuda::std::memory_order_relaxed);
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void move_key_to_new_bucket(
    cg::thread_block_tile<TILE_SIZE> g, int rank, const K& key, const S& score,
    const V* __restrict vector, Bucket<K, V, S>* __restrict new_bucket,
    const size_t new_bkt_idx, const size_t new_start_idx,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim) {
  uint32_t key_pos;
  unsigned empty_vote;
  int local_size;
  int src_lane;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    size_t key_offset =
        (new_start_idx + tile_offset + rank) & (bucket_max_size - 1);
    const K current_key =
        (*(new_bucket->keys(key_offset))).load(cuda::std::memory_order_relaxed);
    empty_vote = g.ballot(current_key == static_cast<K>(EMPTY_KEY));
    if (empty_vote) {
      src_lane = __ffs(empty_vote) - 1;
      key_pos =
          (new_start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
      local_size = buckets_size[new_bkt_idx];
      if (rank == src_lane) {
        new_bucket->digests(key_pos)[0] = get_digest<K>(key);
        new_bucket->keys(key_pos)->store(key, cuda::std::memory_order_relaxed);
        new_bucket->scores(key_pos)->store(score,
                                           cuda::std::memory_order_relaxed);
        atomicAdd(&(buckets_size[new_bkt_idx]), 1);
      }
      local_size = g.shfl(local_size, src_lane);
      if (local_size >= bucket_max_size) {
        refresh_bucket_score<K, V, S, TILE_SIZE>(g, new_bucket,
                                                 bucket_max_size);
      }
      copy_vector<V, TILE_SIZE>(g, vector, new_bucket->vectors + key_pos * dim,
                                dim);
      break;
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void rehash_kernel_for_fast_mode(
    const Table<K, V, S>* __restrict table, size_t N) {
  Bucket<K, V, S>* buckets = table->buckets;
  int* __restrict buckets_size = table->buckets_size;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t buckets_num = table->buckets_num;
  const size_t dim = table->dim;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t global_idx;
  uint32_t start_idx = 0;
  K target_key = 0;
  S target_score = 0;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t / TILE_SIZE;
    Bucket<K, V, S>* bucket = (buckets + bkt_idx);

    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    uint32_t key_idx = 0;
    while (key_idx < bucket_max_size) {
      key_idx = g.shfl(key_idx, 0);
      target_key =
          (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
      target_score =
          bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);
      if (target_key != static_cast<K>(EMPTY_KEY) &&
          target_key != static_cast<K>(RECLAIM_KEY)) {
        const K hashed_key = Murmur3HashDevice(target_key);
        global_idx = hashed_key % (buckets_num * bucket_max_size);
        uint32_t new_bkt_idx = global_idx / bucket_max_size;
        if (new_bkt_idx != bkt_idx) {
          start_idx = global_idx % bucket_max_size;
          move_key_to_new_bucket<K, V, S, TILE_SIZE>(
              g, rank, target_key, target_score,
              (bucket->vectors + key_idx * dim), buckets + new_bkt_idx,
              new_bkt_idx, start_idx, buckets_size, bucket_max_size,
              buckets_num, table->dim);
          if (rank == 0) {
            bucket->digests(key_idx)[0] = empty_digest<K>();
            (bucket->keys(key_idx))
                ->store(static_cast<K>(EMPTY_KEY),
                        cuda::std::memory_order_relaxed);
            atomicSub(&(buckets_size[bkt_idx]), 1);
            defragmentation_for_rehash<K, V, S, TILE_SIZE>(
                bucket, key_idx, bucket_max_size, buckets_num / 2, dim);
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

/* Add a @delta[i] to the the value saved in the address @dst[i].

   `delta`: a delta value which should be add to.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class S>
__global__ void write_with_accum_kernel(const V* __restrict delta,
                                        V** __restrict dst,
                                        const int* __restrict src_offset,
                                        const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;

    if (dst[vec_index] != nullptr) {
      dst[vec_index][dim_index] +=
          delta[src_offset[vec_index] * dim + dim_index];
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
template <class K, class V, class S>
__global__ void read_kernel(const V* const* __restrict src, V* __restrict dst,
                            const bool* mask, const int* __restrict dst_offset,
                            const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;

    /// Copy selected values and fill in default value for all others.
    if (mask[real_dst_offset] && src[vec_index] != nullptr) {
      dst[real_dst_offset * dim + dim_index] = src[vec_index][dim_index];
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
template <class K, class V, class S>
__global__ void read_kernel(const V** __restrict src, V* __restrict dst,
                            const int* __restrict dst_offset, const size_t dim,
                            const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;
    int dim_index = t % dim;
    if (src[vec_index] != nullptr) {
      dst[real_dst_offset * dim + dim_index] = src[vec_index * dim + dim_index];
    }
  }
}

/* Accum kernel with customized scores.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void accum_kernel(
    const Table<K, V, S>* __restrict table, const K* __restrict keys,
    V** __restrict vectors, const S* __restrict scores,
    const bool* __restrict existed, Bucket<K, V, S>* __restrict buckets,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, int* __restrict src_offset,
    bool* __restrict status, size_t N) {
  const size_t dim = table->dim;
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

    if (IS_RESERVED_KEY(insert_key)) continue;

    const K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, S>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

    for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      size_t key_offset =
          (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      K current_key =
          bucket->keys(key_offset)->load(cuda::std::memory_order_relaxed);
      found_or_empty_vote = g.ballot(current_key == static_cast<K>(EMPTY_KEY) ||
                                     insert_key == current_key);
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
            bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
            (bucket->keys(key_pos))
                ->store(insert_key, cuda::std::memory_order_relaxed);
            if (!local_found) {
              buckets_size[bkt_idx]++;
              local_size++;
            }
            *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
            update_score(bucket, key_pos, scores, key_idx);
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_score<K, V, S, TILE_SIZE>(g, bucket, bucket_max_size);
        }
        break;
      }
    }
    if (!found_or_empty_vote) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
        (bucket->keys(key_pos))
            ->store(insert_key, cuda::std::memory_order_relaxed);
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_score(bucket, key_pos, scores, key_idx);
      }
      refresh_bucket_score<K, V, S, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class S>
__global__ void clear_kernel(Table<K, V, S>* __restrict table, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, S>* bucket = &(table->buckets[bkt_idx]);

    bucket->digests(key_idx)[0] = empty_digest<K>();
    (bucket->keys(key_idx))
        ->store(static_cast<K>(EMPTY_KEY), cuda::std::memory_order_relaxed);
    if (key_idx == 0) {
      table->buckets_size[bkt_idx] = 0;
    }
  }
}

/* Remove specified keys. */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void remove_kernel(const Table<K, V, S>* __restrict table,
                              const K* __restrict keys,
                              Bucket<K, V, S>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;
    K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    int key_pos = -1;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    uint32_t tile_offset = 0;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    unsigned found_vote = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_pos = (start_idx + tile_offset + rank) & (bucket_max_size - 1);

      const K current_key =
          (bucket->keys(key_pos))->load(cuda::std::memory_order_relaxed);

      found_vote = g.ballot(find_key == current_key);
      if (found_vote) {
        break;
      }

      if (g.any(current_key == static_cast<K>(EMPTY_KEY))) {
        break;
      }
    }

    if (found_vote) {
      const int src_lane = __ffs(found_vote) - 1;

      if (g.thread_rank() == src_lane) {
        const int key_pos =
            (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        bucket->digests(key_pos)[0] = empty_digest<K>();
        (bucket->keys(key_pos))
            ->store(static_cast<K>(RECLAIM_KEY),
                    cuda::std::memory_order_relaxed);
        (bucket->scores(key_pos))
            ->store(static_cast<S>(EMPTY_SCORE),
                    cuda::std::memory_order_relaxed);
        atomicSub(&buckets_size[bkt_idx], 1);
      }
      break;
    }
  }
}

/* Remove specified keys which match the Predict. */
template <class K, class V, class S,
          template <typename, typename> class PredFunctor,
          uint32_t TILE_SIZE = 1>
__global__ void remove_kernel(const Table<K, V, S>* __restrict table,
                              const K pattern, const S threshold,
                              size_t* __restrict count,
                              Bucket<K, V, S>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  PredFunctor<K, S> pred;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t;
    uint32_t key_pos = 0;

    Bucket<K, V, S>* bucket = buckets + bkt_idx;

    K current_key = 0;
    S current_score = 0;
    uint32_t key_offset = 0;
    while (key_offset < bucket_max_size) {
      current_key =
          bucket->keys(key_offset)->load(cuda::std::memory_order_relaxed);
      current_score =
          bucket->scores(key_offset)->load(cuda::std::memory_order_relaxed);
      if (!IS_RESERVED_KEY(current_key)) {
        if (pred(current_key, current_score, pattern, threshold)) {
          atomicAdd(count, 1);
          key_pos = key_offset;
          bucket->digests(key_pos)[0] = empty_digest<K>();
          (bucket->keys(key_pos))
              ->store(static_cast<K>(RECLAIM_KEY),
                      cuda::std::memory_order_relaxed);
          (bucket->scores(key_pos))
              ->store(static_cast<S>(EMPTY_SCORE),
                      cuda::std::memory_order_relaxed);
          atomicSub(&buckets_size[bkt_idx], 1);
        } else {
          key_offset++;
        }
      } else {
        key_offset++;
      }
    }
  }
}

/* Dump with score. */
template <class K, class V, class S>
inline std::tuple<size_t, size_t> dump_kernel_shared_memory_size(
    const size_t available_shared_memory) {
  const size_t block_size{std::min(
      available_shared_memory / 2 / sizeof(KVM<K, V, S>), UINT64_C(1024))};
  MERLIN_CHECK(
      block_size > 0,
      "[HierarchicalKV] block_size <= 0, the K-V-S size may be too large!");

  return {block_size * sizeof(KVM<K, V, S>), block_size};
}

template <class K, class V, class S>
__global__ void dump_kernel(const Table<K, V, S>* __restrict table, K* d_key,
                            V* __restrict d_val, S* __restrict d_score,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  KVM<K, V, S>* const block_tuples{reinterpret_cast<KVM<K, V, S>*>(s)};

  const size_t bucket_max_size{table->bucket_max_size};
  const size_t dim{table->dim};

  __shared__ size_t block_acc;
  __shared__ size_t global_acc;

  const size_t tid{blockIdx.x * blockDim.x + threadIdx.x};

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    Bucket<K, V, S>* const bucket{
        &table->buckets[(tid + offset) / bucket_max_size]};

    const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
    const K key{(bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed)};

    if (!IS_RESERVED_KEY(key)) {
      size_t local_index{atomicAdd(&block_acc, 1)};
      block_tuples[local_index] = {
          key, &bucket->vectors[key_idx * dim],
          bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed)};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    const KVM<K, V, S>& tuple{block_tuples[threadIdx.x]};

    const size_t j{global_acc + threadIdx.x};
    d_key[j] = tuple.key;
    for (int i{0}; i < dim; ++i) {
      d_val[j * dim + i] = tuple.value[i];
    }
    if (d_score != nullptr) {
      d_score[j] = tuple.score;
    }
  }
}

/* Dump with score. */
template <class K, class V, class S,
          template <typename, typename> class PredFunctor>
__global__ void dump_kernel(const Table<K, V, S>* __restrict table,
                            const K pattern, const S threshold, K* d_key,
                            V* __restrict d_val, S* __restrict d_score,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t dim = table->dim;
  K* smem = (K*)s;
  K* block_result_key = smem;
  V* block_result_val = (V*)&(smem[blockDim.x]);
  S* block_result_score = (S*)&(block_result_val[blockDim.x * dim]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;
  PredFunctor<K, S> pred;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / bucket_max_size;
    int key_idx = (tid + offset) % bucket_max_size;
    Bucket<K, V, S>* bucket = &(table->buckets[bkt_idx]);

    const K key =
        (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    S score = bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);

    if (!IS_RESERVED_KEY(key) && pred(key, score, pattern, threshold)) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = key;
      for (int i = 0; i < dim; i++) {
        atomicExch(&(block_result_val[local_index * dim + i]),
                   bucket->vectors[key_idx * dim + i]);
      }
      if (d_score != nullptr) {
        block_result_score[local_index] = score;
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
    for (int i = 0; i < dim; i++) {
      d_val[(global_acc + threadIdx.x) * dim + i] =
          block_result_val[threadIdx.x * dim + i];
    }
    if (d_score != nullptr) {
      d_score[global_acc + threadIdx.x] = block_result_score[threadIdx.x];
    }
  }
}

}  // namespace merlin
}  // namespace nv
