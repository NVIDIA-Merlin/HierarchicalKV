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

#include <cstdint>
#include "allocator.cuh"
#include "core_kernels/accum_or_assign.cuh"
#include "core_kernels/contains.cuh"
#include "core_kernels/find_or_insert.cuh"
#include "core_kernels/find_ptr_or_insert.cuh"
#include "core_kernels/kernel_utils.cuh"
#include "core_kernels/lookup.cuh"
#include "core_kernels/lookup_ptr.cuh"
#include "core_kernels/update.cuh"
#include "core_kernels/update_score.cuh"
#include "core_kernels/update_values.cuh"
#include "core_kernels/upsert.cuh"
#include "core_kernels/upsert_and_evict.cuh"

namespace nv {
namespace merlin {

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
  }
}

template <class K, class V, class S>
__global__ void allocate_bucket_vectors(Bucket<K, V, S>* __restrict buckets,
                                        const size_t index, V* address) {
  buckets[index].vectors = address;
}

template <class K, class V, class S>
__global__ void allocate_bucket_others(Bucket<K, V, S>* __restrict buckets,
                                       size_t total_size_per_bucket,
                                       size_t num_of_buckets,
                                       const int start_index, uint8_t* address,
                                       const uint32_t reserve_size,
                                       const size_t bucket_max_size) {
  for (size_t step = 0; step < num_of_buckets; step++) {
    size_t index = start_index + step;
    buckets[index].digests_ = address;
    buckets[index].keys_ =
        reinterpret_cast<AtomicKey<K>*>(buckets[index].digests_ + reserve_size);
    buckets[index].scores_ = reinterpret_cast<AtomicScore<S>*>(
        buckets[index].keys_ + bucket_max_size);
    address += total_size_per_bucket;
  }
}

template <class K, class V, class S>
__global__ void get_bucket_others_address(Bucket<K, V, S>* __restrict buckets,
                                          const int index, uint8_t** address) {
  *address = buckets[index].digests_;
}

template <class P>
void realloc(P* ptr, size_t old_size, size_t new_size,
             BaseAllocator* allocator) {
  // Truncate old_size to limit dowstream copy ops.
  old_size = std::min(old_size, new_size);

  // Alloc new buffer and copy at old data.
  char* new_ptr;
  allocator->alloc(MemoryType::Device, (void**)&new_ptr, new_size);
  if (*ptr != nullptr) {
    CUDA_CHECK(cudaMemcpy(new_ptr, *ptr, old_size, cudaMemcpyDefault));
    allocator->free(MemoryType::Device, *ptr);
  }

  // Zero-fill remainder.
  CUDA_CHECK(cudaMemset(new_ptr + old_size, 0, new_size - old_size));

  // Switch to new pointer.
  *ptr = reinterpret_cast<P>(new_ptr);
  return;
}

template <class P>
void realloc_host(P* ptr, size_t old_size, size_t new_size,
                  BaseAllocator* allocator) {
  // Truncate old_size to limit dowstream copy ops.
  old_size = std::min(old_size, new_size);

  // Alloc new buffer and copy at old data.
  char* new_ptr = nullptr;
  allocator->alloc(MemoryType::Host, (void**)&new_ptr, new_size);

  if (*ptr != nullptr) {
    std::memcpy(new_ptr, *ptr, old_size);
    allocator->free(MemoryType::Host, *ptr);
  }

  // Zero-fill remainder.
  std::memset(new_ptr + old_size, 0, new_size - old_size);

  // Switch to new pointer.
  *ptr = reinterpret_cast<P>(new_ptr);
  return;
}

/* Initialize the buckets with index from start to end. */
template <class K, class V, class S>
void initialize_buckets(Table<K, V, S>** table, BaseAllocator* allocator,
                        const size_t start, const size_t end) {
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

  realloc_host<V**>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V*),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V*),
      allocator);

  bool mixed_hbm = false;
  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == (*table)->num_of_memory_slices + num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size = num_of_buckets_in_one_slice *
                             (*table)->bucket_max_size * sizeof(V) *
                             (*table)->dim;
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      if (!(*table)->is_pure_hbm) {
        mixed_hbm = true;
      }
      allocator->alloc(MemoryType::Device, (void**)&((*table)->slices[i]),
                       slice_real_size);
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      (*table)->is_pure_hbm = false;
      allocator->alloc(MemoryType::Pinned, (void**)&((*table)->slices[i]),
                       slice_real_size, cudaHostAllocMapped);
    }
    for (int j = 0; j < num_of_buckets_in_one_slice; j++) {
      if ((*table)->is_pure_hbm || mixed_hbm) {
        size_t index = start + num_of_allocated_buckets + j;
        V* address =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
        allocate_bucket_vectors<K, V, S>
            <<<1, 1>>>((*table)->buckets, index, address);
        CUDA_CHECK(cudaDeviceSynchronize());
      } else {
        V* h_ptr =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
        V* address = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&address, h_ptr, 0));
        size_t index = start + num_of_allocated_buckets + j;
        allocate_bucket_vectors<K, V, S>
            <<<1, 1>>>((*table)->buckets, index, address);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }

  (*table)->num_of_memory_slices += num_of_memory_slices;
  uint32_t bucket_max_size = static_cast<uint32_t>((*table)->bucket_max_size);
  size_t bucket_memory_size =
      bucket_max_size * (sizeof(AtomicKey<K>) + sizeof(AtomicScore<S>));
  // Align to the cache line size.
  constexpr uint32_t CACHE_LINE_SIZE = 128U / sizeof(uint8_t);
  uint32_t reserve_size =
      bucket_max_size < CACHE_LINE_SIZE ? CACHE_LINE_SIZE : bucket_max_size;
  bucket_memory_size += reserve_size * sizeof(uint8_t);

  MERLIN_CHECK(start % (*table)->num_of_buckets_per_alloc == 0,
               "initialize_buckets, start must be times of "
               "num_of_buckets_per_alloc!");
  /* NOTICE: Only the buckets which index is the times of
   * `num_of_buckets_per_alloc` will allocate a real address, that provides the
   * callers a method to avoid memory fragmentation.
   */
  for (int i = start; i < end; i += (*table)->num_of_buckets_per_alloc) {
    uint8_t* address = nullptr;
    size_t num_of_buckets =
        std::min(end - i, (*table)->num_of_buckets_per_alloc);
    allocator->alloc(MemoryType::Device, (void**)&(address),
                     bucket_memory_size * num_of_buckets);
    allocate_bucket_others<K, V, S>
        <<<1, 1>>>((*table)->buckets, bucket_memory_size, num_of_buckets, i,
                   address, reserve_size, bucket_max_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaDeviceSynchronize());
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
void create_table(Table<K, V, S>** table, BaseAllocator* allocator,
                  const size_t dim, const size_t init_size = 134217728,
                  const size_t max_size = std::numeric_limits<size_t>::max(),
                  const size_t max_hbm_for_vectors = 0,
                  const size_t bucket_max_size = 128,
                  const size_t num_of_buckets_per_alloc = 1,
                  const size_t tile_size = 32, const bool primary = true) {
  allocator->alloc(MemoryType::Host, (void**)table, sizeof(Table<K, V, S>));
  std::memset(*table, 0, sizeof(Table<K, V, S>));
  (*table)->dim = dim;
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->max_size = std::max(init_size, max_size);
  (*table)->tile_size = tile_size;
  (*table)->is_pure_hbm = true;
  (*table)->bytes_per_slice = get_slice_size<K, V, S>(table);
  (*table)->num_of_buckets_per_alloc = num_of_buckets_per_alloc;

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

  allocator->alloc(MemoryType::Device, (void**)&((*table)->locks),
                   (*table)->buckets_num * sizeof(Mutex));
  CUDA_CHECK(
      cudaMemset((*table)->locks, 0, (*table)->buckets_num * sizeof(Mutex)));

  allocator->alloc(MemoryType::Device, (void**)&((*table)->buckets_size),
                   (*table)->buckets_num * sizeof(int));
  CUDA_CHECK(cudaMemset((*table)->buckets_size, 0,
                        (*table)->buckets_num * sizeof(int)));

  allocator->alloc(MemoryType::Device, (void**)&((*table)->buckets),
                   (*table)->buckets_num * sizeof(Bucket<K, V, S>));
  CUDA_CHECK(cudaMemset((*table)->buckets, 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, S>)));

  initialize_buckets<K, V, S>(table, allocator, 0, (*table)->buckets_num);
  CudaCheckError();
}

/* Double the capacity on storage, must be followed by calling the
 * rehash_kernel. */
template <class K, class V, class S>
void double_capacity(Table<K, V, S>** table, BaseAllocator* allocator) {
  realloc<Mutex*>(&((*table)->locks), (*table)->buckets_num * sizeof(Mutex),
                  (*table)->buckets_num * sizeof(Mutex) * 2, allocator);
  realloc<int*>(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
                (*table)->buckets_num * sizeof(int) * 2, allocator);

  realloc<Bucket<K, V, S>*>(
      &((*table)->buckets), (*table)->buckets_num * sizeof(Bucket<K, V, S>),
      (*table)->buckets_num * sizeof(Bucket<K, V, S>) * 2, allocator);

  initialize_buckets<K, V, S>(table, allocator, (*table)->buckets_num,
                              (*table)->buckets_num * 2);

  (*table)->capacity *= 2;
  (*table)->buckets_num *= 2;
}

/* free all of the resource of a Table. */
template <class K, class V, class S>
void destroy_table(Table<K, V, S>** table, BaseAllocator* allocator) {
  uint8_t** d_address = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_address, sizeof(uint8_t*)));
  /* NOTICE: Only the buckets which index is the times of
   * `num_of_buckets_per_alloc` will hold a real address, and need to be freed
   */
  for (int i = 0; i < (*table)->buckets_num;
       i += (*table)->num_of_buckets_per_alloc) {
    uint8_t* h_address;
    get_bucket_others_address<K, V, S>
        <<<1, 1>>>((*table)->buckets, i, d_address);
    CUDA_CHECK(cudaMemcpy(&h_address, d_address, sizeof(uint8_t*),
                          cudaMemcpyDeviceToHost));
    allocator->free(MemoryType::Device, h_address);
  }
  CUDA_CHECK(cudaFree(d_address));

  for (int i = 0; i < (*table)->num_of_memory_slices; i++) {
    if (is_on_device((*table)->slices[i])) {
      allocator->free(MemoryType::Device, (*table)->slices[i]);
    } else {
      allocator->free(MemoryType::Pinned, (*table)->slices[i]);
    }
  }
  {
    const size_t block_size = 512;
    const size_t N = (*table)->buckets_num;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    release_locks<Mutex>
        <<<grid_size, block_size>>>((*table)->locks, 0, (*table)->buckets_num);
  }
  allocator->free(MemoryType::Host, (*table)->slices);
  allocator->free(MemoryType::Device, (*table)->buckets_size);
  allocator->free(MemoryType::Device, (*table)->buckets);
  allocator->free(MemoryType::Device, (*table)->locks);
  allocator->free(MemoryType::Host, *table);
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
    start_idx = get_start_position(global_idx, bucket_max_size);

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
__forceinline__ __device__ void move_key_to_new_bucket(
    cg::thread_block_tile<TILE_SIZE> g, int rank, const K& key, const S& score,
    const V* __restrict vector, Bucket<K, V, S>* __restrict new_bucket,
    const size_t new_bkt_idx, const size_t new_start_idx,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim) {
  uint32_t key_pos;
  unsigned empty_vote;
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
      if (rank == src_lane) {
        new_bucket->digests(key_pos)[0] = get_digest<K>(key);
        new_bucket->keys(key_pos)->store(key, cuda::std::memory_order_relaxed);
        new_bucket->scores(key_pos)->store(score,
                                           cuda::std::memory_order_relaxed);
        atomicAdd(&(buckets_size[new_bkt_idx]), 1);
      }
      copy_vector<V, TILE_SIZE>(g, vector, new_bucket->vectors + key_pos * dim,
                                dim);
      break;
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void rehash_kernel_for_fast_mode(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    size_t N) {
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
          start_idx = get_start_position(global_idx, bucket_max_size);
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
__global__ void read_kernel(const V* const* __restrict src, V* __restrict dst,
                            const int* __restrict dst_offset, const size_t dim,
                            const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;
    int dim_index = t % dim;
    if (src[vec_index] != nullptr) {
      dst[real_dst_offset * dim + dim_index] = src[vec_index][dim_index];
    }
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class S>
__global__ void clear_kernel(Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, S>* bucket = &(buckets[bkt_idx]);

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
    if (IS_RESERVED_KEY<K>(find_key)) continue;

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
        bucket->digests(key_pos)[0] = reclaim_digest<K>();
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
      if (!IS_RESERVED_KEY<K>(current_key)) {
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

template <typename K, typename V, typename S, typename PredFunctor,
          uint32_t GroupSize = 32>
__global__ void remove_kernel_v2(const uint64_t search_length,
                                 const uint64_t offset, PredFunctor pred,
                                 Bucket<K, V, S>* buckets,
                                 int* __restrict buckets_size,
                                 const uint64_t bucket_capacity,
                                 const uint64_t dim, uint64_t* remove_counter) {
  cg::thread_block_tile<GroupSize> g =
      cg::tiled_partition<GroupSize>(cg::this_thread_block());

  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (uint64_t i = tid; i < search_length; i += gridDim.x * blockDim.x) {
    uint64_t bkt_idx = (i + offset) / bucket_capacity;
    uint64_t key_idx = (i + offset) % bucket_capacity;

    // May be different for threads within the same group.
    Bucket<K, V, S>* bucket = buckets + bkt_idx;

    const K key = bucket->keys(key_idx)->load(cuda::std::memory_order_relaxed);
    const S score =
        bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);
    const V* value = bucket->vectors + key_idx * dim;

    bool match = pred.template operator()<GroupSize>(key, value, score, g);
    if (IS_RESERVED_KEY<K>(key)) {
      match = false;
    }
    uint32_t vote = g.ballot(match);
    int group_cnt = __popc(vote);
    if (g.thread_rank() == 0) {
      atomicAdd(remove_counter, static_cast<uint64_t>(group_cnt));
      if (bucket_capacity >= GroupSize) {
        atomicSub(&buckets_size[bkt_idx], group_cnt);
      }
    }
    // Only matched threads need to erase.
    if (match) {
      bucket->digests(key_idx)[0] = empty_digest<K>();
      bucket->keys(key_idx)->store(static_cast<K>(RECLAIM_KEY),
                                   cuda::std::memory_order_relaxed);
      bucket->scores(key_idx)->store(static_cast<S>(EMPTY_SCORE),
                                     cuda::std::memory_order_relaxed);
      if (bucket_capacity < GroupSize) {
        atomicSub(&buckets_size[bkt_idx], 1);
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

  return std::make_tuple(block_size * sizeof(KVM<K, V, S>), block_size);
}

template <class K, class V, class S>
__global__ void dump_kernel(const Table<K, V, S>* __restrict table,
                            Bucket<K, V, S>* buckets, K* d_key,
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
    Bucket<K, V, S>* const bucket{&buckets[(tid + offset) / bucket_max_size]};

    const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
    const K key{(bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed)};

    if (!IS_RESERVED_KEY<K>(key)) {
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
                            Bucket<K, V, S>* buckets, const K pattern,
                            const S threshold, K* d_key, V* __restrict d_val,
                            S* __restrict d_score, const size_t offset,
                            const size_t search_length,
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
    Bucket<K, V, S>* bucket = &(buckets[bkt_idx]);

    const K key =
        (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    S score = bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);

    if (!IS_RESERVED_KEY<K>(key) && pred(key, score, pattern, threshold)) {
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

template <class K, class V, class S, class VecV,
          template <typename, typename> class PredFunctor, int TILE_SIZE>
__global__ void dump_kernel_v2(const Table<K, V, S>* __restrict table,
                               Bucket<K, V, S>* buckets, const K pattern,
                               const S threshold, K* d_key, V* __restrict d_val,
                               S* __restrict d_score, const size_t offset,
                               const size_t search_length,
                               size_t* d_dump_counter) {
  const size_t bucket_max_size = table->bucket_max_size;
  int vec_dim = table->dim * sizeof(V) / sizeof(VecV);
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  PredFunctor<K, S> pred;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (size_t i = tid; i < search_length; i += gridDim.x * blockDim.x) {
    size_t bkt_idx = (i + offset) / bucket_max_size;
    size_t key_idx = (i + offset) % bucket_max_size;
    size_t leading_key_idx = key_idx / TILE_SIZE * TILE_SIZE;
    Bucket<K, V, S>* bucket = &(buckets[bkt_idx]);

    const K key =
        (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    S score = bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);

    bool match =
        (!IS_RESERVED_KEY<K>(key)) && pred(key, score, pattern, threshold);
    unsigned int vote = g.ballot(match);
    int tile_cnt = __popc(vote);
    size_t tile_offset = 0;
    if (g.thread_rank() == 0) {
      tile_offset = atomicAdd(d_dump_counter, static_cast<size_t>(tile_cnt));
    }
    tile_offset = g.shfl(tile_offset, 0);
    int bias_g = tile_cnt - __popc(vote >> (key_idx % TILE_SIZE));

    if (match) {
      d_key[tile_offset + bias_g] = key;
      if (d_score) {
        d_score[tile_offset + bias_g] = score;
      }
    }

#pragma unroll
    for (int r = 0; r < TILE_SIZE; r++) {
      unsigned int biased_vote = vote >> r;
      bool cur_match = biased_vote & 1;
      if (cur_match) {
        int bias = tile_cnt - __popc(biased_vote);
        size_t cur_idx = leading_key_idx + r;

        VecV* d_val_vec = reinterpret_cast<VecV*>(d_val);
        VecV* vec = reinterpret_cast<VecV*>(bucket->vectors);
        for (int j = g.thread_rank(); j < vec_dim; j += TILE_SIZE) {
          d_val_vec[(tile_offset + bias) * vec_dim + j] =
              vec[cur_idx * vec_dim + j];
        }
      }
    }
  }
}

template <typename K, typename V, typename S, typename PredFunctor,
          uint32_t GroupSize = 32>
__global__ void dump_kernel(const uint64_t search_length, const uint64_t offset,
                            PredFunctor pred, Bucket<K, V, S>* buckets,
                            const uint64_t bucket_capacity, const uint64_t dim,
                            K* __restrict__ out_keys, V* __restrict__ out_vals,
                            S* __restrict__ out_scores,
                            uint64_t* dump_counter) {
  cg::thread_block_tile<GroupSize> g =
      cg::tiled_partition<GroupSize>(cg::this_thread_block());

  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (uint64_t i = tid; i < search_length; i += gridDim.x * blockDim.x) {
    uint64_t bkt_idx = (i + offset) / bucket_capacity;
    uint64_t key_idx = (i + offset) % bucket_capacity;

    // May be different for threads within the same group.
    Bucket<K, V, S>* bucket = buckets + bkt_idx;

    const K key = bucket->keys(key_idx)->load(cuda::std::memory_order_relaxed);
    const S score =
        bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);
    const V* value = bucket->vectors + key_idx * dim;

    bool match = pred.template operator()<GroupSize>(key, value, score, g);
    uint32_t vote = g.ballot(match);
    int group_cnt = __popc(vote);
    uint64_t group_offset = 0;
    if (g.thread_rank() == 0) {
      group_offset = atomicAdd(dump_counter, static_cast<uint64_t>(group_cnt));
    }
    group_offset = g.shfl(group_offset, 0);
    // Each thread gets the count of previous matches ranks.
    // Using `g.thread_rank()` instead of `key_idx % GroupSize` to handle case:
    // bucket_capacity < GroupSize.
    int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
    // Only matched threads need to output.
    if (match) {
      out_keys[group_offset + previous_cnt] = key;
      if (out_scores) {
        out_scores[group_offset + previous_cnt] = score;
      }
    }

    for (int r = 0; r < GroupSize; r++) {
      uint32_t biased_vote = vote >> r;
      bool cur_match = biased_vote & 1;
      if (cur_match) {
        int bias = group_cnt - __popc(biased_vote);

        /// TODO:timing them
        //----------------------- Solution 1
        // uint64_t cur_bkt_idx = g.shfl(bkt_idx, r);
        // uint64_t cur_key_idx = g.shfl(key_idx, r);
        // auto cur_bucket = buckets + cur_bkt_idx;
        //----------------------- Solution 2
        uint64_t cur_idx = (i / GroupSize) * GroupSize + r + offset;
        uint64_t cur_bkt_idx = cur_idx / bucket_capacity;
        uint64_t cur_key_idx = cur_idx % bucket_capacity;
        Bucket<K, V, S>* cur_bucket = buckets + cur_bkt_idx;

        for (int j = g.thread_rank(); j < dim; j += GroupSize) {
          out_vals[(group_offset + bias) * dim + j] =
              cur_bucket->vectors[cur_key_idx * dim + j];
        }
      }
    }
  }
}

template <class K, class V, class S,
          template <typename, typename> class PredFunctor>
__global__ void size_if_kernel(const Table<K, V, S>* __restrict table,
                               Bucket<K, V, S>* buckets, const K pattern,
                               const S threshold, size_t* d_counter) {
  extern __shared__ unsigned char s[];

  const size_t bucket_max_size{table->bucket_max_size};

  size_t local_acc = 0;
  __shared__ size_t block_acc;
  PredFunctor<K, S> pred;

  const size_t tid{blockIdx.x * blockDim.x + threadIdx.x};

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  for (size_t i = tid; i < table->capacity; i += blockDim.x * gridDim.x) {
    Bucket<K, V, S>* const bucket{&buckets[i / bucket_max_size]};

    const int key_idx{static_cast<int>(i % bucket_max_size)};
    const K key{(bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed)};
    S score = bucket->scores(key_idx)->load(cuda::std::memory_order_relaxed);

    if ((!IS_RESERVED_KEY(key)) && pred(key, score, pattern, threshold)) {
      ++local_acc;
    }
  }
  atomicAdd(&block_acc, local_acc);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(d_counter, block_acc);
  }
}

template <typename K, typename V, typename S, typename ExecutionFunc,
          uint32_t GroupSize = 32>
__global__ void traverse_kernel(const uint64_t search_length,
                                const uint64_t offset, ExecutionFunc f,
                                Bucket<K, V, S>* buckets,
                                const uint64_t bucket_capacity,
                                const uint64_t dim) {
  cg::thread_block_tile<GroupSize> g =
      cg::tiled_partition<GroupSize>(cg::this_thread_block());

  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (uint64_t i = tid; i < search_length; i += gridDim.x * blockDim.x) {
    uint64_t bkt_idx = (i + offset) / bucket_capacity;
    uint64_t key_idx = (i + offset) % bucket_capacity;

    // May be different for threads within the same group.
    Bucket<K, V, S>* bucket = buckets + bkt_idx;

    const K key = bucket->keys(key_idx)->load(cuda::std::memory_order_relaxed);
    S* score = reinterpret_cast<S*>(bucket->scores(key_idx));
    V* value = bucket->vectors + key_idx * dim;

    f.template operator()<GroupSize>(key, value, score, g);
  }
}

template <typename K>
__global__ void unlock_keys_kernel(uint64_t n, K** __restrict__ locked_key_ptrs,
                                   const K* __restrict__ keys,
                                   bool* __restrict__ succeededs) {
  int kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (kv_idx < n) {
    K* locked_key_ptr = locked_key_ptrs[kv_idx];
    bool flag = true;
    if (locked_key_ptr != nullptr) {
      K locked_key = *locked_key_ptr;
      K expected_key = static_cast<K>(LOCKED_KEY);
      K key = keys[kv_idx];
      if (locked_key == expected_key) {
        *locked_key_ptr = key;
      } else {
        flag = false;
      }
    } else {
      flag = false;
    }
    if (succeededs != nullptr) {
      succeededs[kv_idx] = flag;
    }
  }
}

template <typename K, typename V, typename S, typename Tidx, int TILE_SIZE = 8>
__global__ void compact_key_value_score_kernel(
    const bool* masks, size_t n, const Tidx* offsets,
    K* __restrict const src_keys, V* __restrict const src_values,
    S* __restrict const src_scores, K* __restrict dst_keys,
    V* __restrict dst_values, S* __restrict dst_scores, const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);
  K empty_key = (K)EMPTY_KEY;
  Tidx bias;
  if (is_existed) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    int prefix_n = __popc(r_vote) - 1;
    bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);
    dst_keys[bias] = src_keys[tid];
    if (src_scores and dst_scores) dst_scores[bias] = src_scores[tid];
  }

  int group_offset = (tid / TILE_SIZE) * TILE_SIZE;
  for (int i = 0; i < TILE_SIZE; i++) {
    if (group_offset + i >= n) return;
    auto cur_existed = g.shfl(is_existed, i);
    if (cur_existed) {
      auto cur_bias = g.shfl(bias, i);
      for (size_t j = rank; j < dim; j += TILE_SIZE) {
        dst_values[dim * cur_bias + j] =
            src_values[dim * (group_offset + i) + j];
      }
    }
  }
}

template <typename K, typename V, typename S>
__global__ void lock_kernel_with_filter(
    Bucket<K, V, S>* __restrict__ buckets, uint64_t const buckets_num,
    uint32_t bucket_capacity, uint32_t const dim, K const* __restrict__ keys,
    K** __restrict locked_keys_ptr, bool* __restrict succeed, uint64_t n) {
  using BUCKET = Bucket<K, V, S>;
  // Load `STRIDE` digests every time.
  constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  K* bucket_keys_ptr{nullptr};
  uint32_t key_pos = {0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
    } else {
      occupy_result = OccupyResult::ILLEGAL;
      goto WRITE_BACK;
    }
  } else {
    return;
  }

  // One more loop to handle empty keys.
  for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
    VecD_Load digests_vec = *(reinterpret_cast<VecD_Load*>(digests_ptr));
    VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                digests_vec.w};

    for (int i = 0; i < 4; i++) {
      VecD_Comp probe_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      int cmp_result = __vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        // CUDA uses little endian,
        // and the lowest byte in register stores in the lowest address.
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        auto current_key = bucket_keys_ptr[possible_pos];
        if (current_key == key) {
          key_pos = possible_pos;
          occupy_result = OccupyResult::DUPLICATE;
          goto WRITE_BACK;
        }
      } while (true);
      VecD_Comp empty_digests_ = empty_digests<K>();
      cmp_result = __vcmpeq4(probe_digests, empty_digests_);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = bucket_keys_ptr[possible_pos];
        if (current_key == static_cast<K>(EMPTY_KEY)) {
          occupy_result = OccupyResult::OCCUPIED_EMPTY;
          goto WRITE_BACK;
        }
      } while (true);
    }
  }

WRITE_BACK:
  bool found_ = occupy_result == OccupyResult::DUPLICATE;
  if (found_) {
    auto current_key = BUCKET::keys(bucket_keys_ptr, key_pos);
    K expected_key = key;
    // Modifications to the bucket will not before this instruction.
    bool result = current_key->compare_exchange_strong(
        expected_key, static_cast<K>(LOCKED_KEY),
        cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
    if (not result) {
      found_ = false;
    }
  }
  if (found_) {
    locked_keys_ptr[kv_idx] = bucket_keys_ptr + key_pos;
  } else {
    locked_keys_ptr[kv_idx] = nullptr;
  }
  if (succeed) {
    succeed[kv_idx] = found_;
  }
}

}  // namespace merlin
}  // namespace nv
