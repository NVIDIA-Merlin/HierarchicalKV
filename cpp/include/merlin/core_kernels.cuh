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

#include "types.cuh"
#include "utils.cuh"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace nv {
namespace merlin {

template <class M>
__global__ void create_locks(M *__restrict mutex, size_t start, size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    new (mutex + start + tid) M(1);
  }
}

template <class M>
__global__ void release_locks(M *__restrict mutex, size_t start, size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    (mutex + start + tid)->~M();
  }
}

template <typename mutex>
__forceinline__ __device__ void lock(mutex &set_mutex) {
  set_mutex.acquire();
}

template <typename mutex>
__forceinline__ __device__ void unlock(mutex &set_mutex) {
  set_mutex.release();
}

/* 2GB per slice by default.*/
constexpr size_t kDefaultBytesPerSlice = (2ul << 30);

/* Initialize the buckets with index from start to end. */
template <class K, class V, class M, size_t DIM>
void initialize_buckets(Table<K, V, M, DIM> **table, size_t start, size_t end) {
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

  realloc_managed<V **>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V *),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V *));

  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size =
        num_of_buckets_in_one_slice * (*table)->bucket_max_size * sizeof(V);
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      std::cout << "remaining_hbm_for_vectors="
                << (*table)->remaining_hbm_for_vectors << std::endl;
      CUDA_CHECK(cudaMalloc(&((*table)->slices[i]), slice_real_size));
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      (*table)->is_pure_hbm = false;
      CUDA_CHECK(cudaMallocHost(&((*table)->slices[i]), slice_real_size,
                                cudaHostRegisterMapped));
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
void create_table(Table<K, V, M, DIM> **table, size_t init_size = 134217728,
                  size_t max_size = std::numeric_limits<size_t>::max(),
                  size_t max_hbm_for_vectors = 0, size_t bucket_max_size = 128,
                  size_t tile_size = 32, bool primary = true,
                  size_t bytes_per_slice = kDefaultBytesPerSlice) {
  std::cout << "[merlin-kv] requested configuration: \n"
            << "\tinit_size = " << init_size << std::endl
            << "\tmax_size = " << max_size << std::endl
            << "\tmax_hbm_for_vectors = " << max_hbm_for_vectors << std::endl
            << "\tbucket_max_size = " << bucket_max_size << std::endl
            << "\ttile_size = " << tile_size << std::endl
            << "\tprimary = " << primary << std::endl
            << "\tbytes_per_slice = " << bytes_per_slice << std::endl;
  CUDA_CHECK(cudaMallocManaged((void **)table, sizeof(Table<K, V, M, DIM>)));
  CUDA_CHECK(cudaMemset(*table, 0, sizeof(Table<K, V, M, DIM>)));
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->bytes_per_slice = bytes_per_slice;
  (*table)->max_size = max_size;
  (*table)->tile_size = tile_size;

  (*table)->buckets_num = 1;
  while ((*table)->buckets_num * (*table)->bucket_max_size < init_size) {
    (*table)->buckets_num *= 2;
  }
  std::cout << "[merlin-kv] requested capacity=" << init_size
            << ", real capacity="
            << (*table)->buckets_num * (*table)->bucket_max_size << std::endl;
  (*table)->capacity = (*table)->buckets_num * (*table)->bucket_max_size;
  (*table)->max_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->remaining_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->primary = primary;

  CUDA_CHECK(cudaMalloc((void **)&((*table)->locks),
                        (*table)->buckets_num * sizeof(Mutex)));
  CUDA_CHECK(
      cudaMemset((*table)->locks, 0, (*table)->buckets_num * sizeof(Mutex)));

  CUDA_CHECK(cudaMalloc((void **)&((*table)->buckets_size),
                        (*table)->buckets_num * sizeof(int)));
  CUDA_CHECK(cudaMemset((*table)->buckets_size, 0,
                        (*table)->buckets_num * sizeof(int)));

  CUDA_CHECK(
      cudaMallocManaged((void **)&((*table)->buckets),
                        (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>)));
  CUDA_CHECK(cudaMemset((*table)->buckets, 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>)));

  initialize_buckets<K, V, M, DIM>(table, 0, (*table)->buckets_num);
}

/* Double the capacity on storage, must be followed by calling the
 * rehash_kernel. */
template <class K, class V, class M, size_t DIM>
void double_capacity(Table<K, V, M, DIM> **table) {
  realloc<Mutex *>(&((*table)->locks), (*table)->buckets_num * sizeof(Mutex),
                   (*table)->buckets_num * sizeof(Mutex) * 2);
  realloc<int *>(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
                 (*table)->buckets_num * sizeof(int) * 2);

  realloc_managed<Bucket<K, V, M, DIM> *>(
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
void destroy_table(Table<K, V, M, DIM> **table) {
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
}

template <class K, class V, class M, size_t DIM>
__global__ void rehash_kernel(const Table<K, V, M, DIM> *__restrict table,
                              size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;
  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int bkt_idx = t / bucket_max_size;
    int key_idx = t % bucket_max_size;
    lock<Mutex>(table->locks[bkt_idx]);
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    K target_key = bucket->keys[key_idx];
    if (target_key != EMPTY_KEY) {
      K hashed_key = Murmur3HashDevice(target_key);
      int key_bkt_idx = hashed_key % buckets_num;

      if (key_bkt_idx != bkt_idx) {
        Bucket<K, V, M, DIM> *new_bucket = &(table->buckets[key_bkt_idx]);
        atomicExch(&(new_bucket->keys[key_idx]), target_key);
        atomicExch(&(bucket->keys[key_idx]), EMPTY_KEY);

        atomicSub(&(table->buckets_size[bkt_idx]), 1);
        atomicMax(&(table->buckets_size[bkt_idx]), 0);

        atomicExch(&(new_bucket->metas[key_idx].val),
                   bucket->metas[key_idx].val);
        for (int i = 0; i < DIM; i++) {
          new_bucket->vectors[key_idx].value[i] =
              bucket->vectors[key_idx].value[i];
        }
        atomicAdd(&(table->buckets_size[key_bkt_idx]), 1);
        atomicMin(&(table->buckets_size[key_bkt_idx]), bucket_max_size);
      }
    }
    unlock<Mutex>(table->locks[bkt_idx]);
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
__global__ void write_kernel(const V *__restrict src, V **__restrict dst,
                             const int *__restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr) {
      (*(dst[vec_index])).value[dim_index] =
          src[src_offset[vec_index]].value[dim_index];
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
__global__ void write_with_accum_kernel(const V *__restrict delta_or_val,
                                        V **__restrict dst,
                                        const bool *__restrict existed,
                                        const bool *__restrict status,
                                        const int *__restrict src_offset,
                                        size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr &&
        existed[src_offset[vec_index]] == status[src_offset[vec_index]]) {
      if (status[src_offset[vec_index]]) {
        (*(dst[vec_index])).value[dim_index] +=
            delta_or_val[src_offset[vec_index]].value[dim_index];
      } else {
        (*(dst[vec_index])).value[dim_index] =
            delta_or_val[src_offset[vec_index]].value[dim_index];
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
__global__ void write_with_accum_kernel(const V *__restrict delta,
                                        V **__restrict dst,
                                        const int *__restrict src_offset,
                                        size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;

    if (dst[vec_index] != nullptr) {
      (*(dst[vec_index])).value[dim_index] +=
          delta[src_offset[vec_index]].value[dim_index];
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
__global__ void read_kernel(const V *const *__restrict src, V *__restrict dst,
                            const bool *mask, const V *__restrict default_val,
                            const int *__restrict dst_offset, size_t N,
                            bool full_size_default) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;
    int default_index = full_size_default ? dst_offset[vec_index] : 0;

    /// Copy selected values and fill in default value for all others.
    if (mask[dst_offset[vec_index]] && src[vec_index] != nullptr) {
      dst[dst_offset[vec_index]].value[dim_index] =
          src[vec_index]->value[dim_index];
    } else {
      dst[dst_offset[vec_index]].value[dim_index] =
          default_val[default_index].value[dim_index];
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
   `N`: Number of vectors needed to be read.
*/
template <class K, class V, class M, size_t DIM>
__global__ void read_kernel(V **__restrict src, V *__restrict dst,
                            const int *__restrict dst_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / DIM);
    int dim_index = t % DIM;
    if (src[vec_index] != nullptr) {
      dst[dst_offset[vec_index]].value[dim_index] =
          src[vec_index]->value[dim_index];
    }
  }
}

template <class K, class V, class M, size_t DIM>
__forceinline__ __device__ void insert_if_empty(
    const Bucket<K, V, M, DIM> *bucket, const size_t bucket_max_size,
    const K &insert_key, int *key_pos, bool *empty) {
  for (int i = 0; i < bucket_max_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      *empty = true;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__forceinline__ __device__ void insert_if_empty(
    const Bucket<K, V, M, DIM> *bucket, const size_t bucket_max_size,
    const K &insert_key, int *key_pos, bool *empty, const bool found,
    const bool existed) {
  if (found != existed) return;
  for (int i = 0; i < bucket_max_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      *empty = true;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__forceinline__ __device__ void refresh_bucket_meta(
    Bucket<K, V, M, DIM> *bucket, const size_t bucket_max_size) {
  M min_val = MAX_META;
  int min_pos = 0;
  for (int i = 0; i < bucket_max_size; i++) {
    if (bucket->keys[i] == EMPTY_KEY) {
      continue;
    }
    if (bucket->metas[i].val < min_val) {
      min_pos = i;
      min_val = bucket->metas[i].val;
    }
  }
  bucket->min_pos = min_pos;
  bucket->min_meta = min_val;
}
template <class K>
__forceinline__ __device__ constexpr bool key_compare(const K *k1,
                                                      const K *k2) {
  auto __lhs_c = reinterpret_cast<unsigned char const *>(k1);
  auto __rhs_c = reinterpret_cast<unsigned char const *>(k2);

#pragma unroll
  for (int i = 0; i < sizeof(K); i++) {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v != __rhs_v) {
      return false;
    }
  }
  return true;
}

template <class K>
__forceinline__ __device__ constexpr bool key_empty(const K *k) {
  constexpr K empty_key = EMPTY_KEY;
  auto __lhs_c = reinterpret_cast<unsigned char const *>(k);
  auto __rhs_c = reinterpret_cast<unsigned char const *>(&empty_key);

#pragma unroll
  for (int i = 0; i < sizeof(K); i++) {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v != __rhs_v) {
      return false;
    }
  }
  return true;
}

/* Insert or update a Key-Value in the table,
   this operation will not really write the vector data
   into the bucket. Instead, it will only return the address in bucket for each
   key through `vectors`. To actually write the data the addresses in `vectors`
   the caller must call the `write_kernel`.

   `table`: The table to be operated.
   `keys`: Keys for upsert, if the key exists, will return
           the current address, if not found, a new location
           in bucket will return for the key through `vectors`.
   `metas`: The corresponding meta value for each keys.
   `vectors`: The addresses in buckets for each keys where the
              corresponding vector values should be really saved into.
   `N`: Number of vectors needed to be read.
*/
template <class K, class V, class M, size_t DIM, size_t TILE_SIZE = 8>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              const M *__restrict metas,
                              int *__restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_pos = -1;
    int empty_key_pos = -1;
    bool found = false;
    bool empty = false;
    const size_t bucket_max_size = table->bucket_max_size;
    size_t key_idx = t / TILE_SIZE;
    K insert_key = EMPTY_KEY;
    K hashed_key = EMPTY_KEY;
    size_t bkt_idx = 0;

    if (rank == 0) {
      insert_key = keys[key_idx];
      hashed_key = Murmur3HashDevice(insert_key);
      bkt_idx = hashed_key % table->buckets_num;
    }
    insert_key = g.shfl(insert_key, 0);
    bkt_idx = g.shfl(bkt_idx, 0);

    Bucket<K, V, M, DIM> *bucket = table->buckets + bkt_idx;

    if (rank == 0) {
      lock<Mutex>(table->locks[bkt_idx]);
    }
    g.sync();

    size_t tile_offset = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      K current_key = *(bucket->keys + tile_offset + rank);
      auto const found_vote =
          g.ballot(key_compare<K>(&insert_key, &current_key));
      if (found_vote) {
        found = true;
        key_pos = tile_offset + __ffs(found_vote) - 1;
        break;
      } else {
        if (!empty && *(table->buckets_size + bkt_idx) < bucket_max_size) {
          auto const empty_vote = g.ballot(key_empty<K>(&current_key));
          if (empty_vote) {
            empty_key_pos = tile_offset + __ffs(empty_vote) - 1;
            empty = true;
          }
        }
      }
    }
    if (rank == 0) {
      if (metas[key_idx] >= bucket->min_meta || found || empty) {
        if (!found && empty) {
          key_pos = empty_key_pos;
          table->buckets_size[bkt_idx]++;
        }
        if (!found) {
          key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
        }

        bucket->keys[key_pos] = insert_key;
        bucket->metas[key_pos].val = metas[key_idx];

        /// Re-locate the smallest meta.
        if (table->buckets_size[bkt_idx] >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, DIM>(bucket, bucket_max_size);
        }

        /// Record storage offset. This will be used by write_kernel to map
        /// the input to the output data.
        if (vectors[key_idx] == nullptr) {
          vectors[key_idx] = (bucket->vectors + key_pos);
        }
        src_offset[key_idx] = key_idx;
      }
    }

    g.sync();
    if (rank == 0) {
      unlock<Mutex>(table->locks[bkt_idx]);
    }
  }
}

/* Upsert with no user specified meta.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `size_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM, size_t TILE_SIZE = 8>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              int *__restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_pos = -1;
    int empty_key_pos = -1;
    bool found = false;
    bool empty = false;
    const size_t bucket_max_size = table->bucket_max_size;
    size_t key_idx = t / TILE_SIZE;
    K insert_key = EMPTY_KEY;
    K hashed_key = EMPTY_KEY;
    size_t bkt_idx = 0;

    if (rank == 0) {
      insert_key = keys[key_idx];
      hashed_key = Murmur3HashDevice(insert_key);
      bkt_idx = hashed_key % table->buckets_num;
    }
    insert_key = g.shfl(insert_key, 0);
    bkt_idx = g.shfl(bkt_idx, 0);

    Bucket<K, V, M, DIM> *bucket = table->buckets + bkt_idx;

    if (rank == 0) {
      lock<Mutex>(table->locks[bkt_idx]);
    }
    g.sync();

    size_t tile_offset = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      K current_key = *(bucket->keys + tile_offset + rank);
      auto const found_vote =
          g.ballot(key_compare<K>(&insert_key, &current_key));
      if (found_vote) {
        found = true;
        key_pos = tile_offset + __ffs(found_vote) - 1;
        break;
      } else {
        if (!empty && *(table->buckets_size + bkt_idx) < bucket_max_size) {
          auto const empty_vote = g.ballot(key_empty<K>(&current_key));
          if (empty_vote) {
            empty_key_pos = tile_offset + __ffs(empty_vote) - 1;
            empty = true;
          }
        }
      }
    }
    if (rank == 0) {
      if (!found && empty) {
        key_pos = empty_key_pos;
        table->buckets_size[bkt_idx]++;
      }
      if (!found) {
        key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
      }
      M cur_meta = 1 + bucket->cur_meta;
      bucket->keys[key_pos] = insert_key;
      bucket->metas[key_pos].val = cur_meta;
      bucket->cur_meta = cur_meta;

      /// Re-locate the smallest meta.
      if (table->buckets_size[bkt_idx] >= bucket_max_size) {
        refresh_bucket_meta<K, V, M, DIM>(bucket, bucket_max_size);
      }

      /// Record storage offset. This will be used by write_kernel to map
      /// the input to the output data.
      if (vectors[key_idx] == nullptr) {
        vectors[key_idx] = (bucket->vectors + key_pos);
      }
      src_offset[key_idx] = key_idx;
    }

    g.sync();
    if (rank == 0) {
      unlock<Mutex>(table->locks[bkt_idx]);
    }
  }
}

/* Upsert with no user specified meta and return if the keys already exists.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `size_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM, size_t TILE_SIZE = 8>
__global__ void accum_kernel(const Table<K, V, M, DIM> *__restrict table,
                             const K *__restrict keys, V **__restrict vectors,
                             const bool *__restrict existed,
                             int *__restrict src_offset,
                             bool *__restrict status, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_pos = -1;
    int empty_key_pos = -1;
    bool found = false;
    bool empty = false;
    const size_t bucket_max_size = table->bucket_max_size;
    size_t key_idx = t / TILE_SIZE;
    K insert_key = EMPTY_KEY;
    K hashed_key = EMPTY_KEY;
    size_t bkt_idx = 0;

    if (rank == 0) {
      insert_key = keys[key_idx];
      hashed_key = Murmur3HashDevice(insert_key);
      bkt_idx = hashed_key % table->buckets_num;
    }
    insert_key = g.shfl(insert_key, 0);
    bkt_idx = g.shfl(bkt_idx, 0);

    Bucket<K, V, M, DIM> *bucket = table->buckets + bkt_idx;

    if (rank == 0) {
      lock<Mutex>(table->locks[bkt_idx]);
    }
    g.sync();

    size_t tile_offset = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      K current_key = *(bucket->keys + tile_offset + rank);
      auto const found_vote =
          g.ballot(key_compare<K>(&insert_key, &current_key));
      if (found_vote) {
        found = true;
        key_pos = tile_offset + __ffs(found_vote) - 1;
        break;
      } else {
        if (!empty && *(table->buckets_size + bkt_idx) < bucket_max_size) {
          auto const empty_vote = g.ballot(key_empty<K>(&current_key));
          if (empty_vote) {
            empty_key_pos = tile_offset + __ffs(empty_vote) - 1;
            empty = true;
          }
        }
      }
    }
    if (rank == 0) {
      status[key_idx] = found;
    }

    if (rank == 0 && (found == existed[key_idx])) {
      if (!found && empty) {
        key_pos = empty_key_pos;
        table->buckets_size[bkt_idx]++;
      }
      if (!found) {
        key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
      }
      M cur_meta = 1 + bucket->cur_meta;
      bucket->keys[key_pos] = insert_key;
      bucket->metas[key_pos].val = cur_meta;
      bucket->cur_meta = cur_meta;

      /// Re-locate the smallest meta.
      if (table->buckets_size[bkt_idx] >= bucket_max_size) {
        refresh_bucket_meta<K, V, M, DIM>(bucket, bucket_max_size);
      }

      /// Record storage offset. This will be used by write_kernel to map
      /// the input to the output data.
      if (vectors[key_idx] == nullptr) {
        vectors[key_idx] = (bucket->vectors + key_pos);
      }
      src_offset[key_idx] = key_idx;
    }

    g.sync();
    if (rank == 0) {
      unlock<Mutex>(table->locks[bkt_idx]);
    }
  }
}
//
// template <class K, class V, class M, size_t DIM>
//__global__ void accum_kernel_old(const Table<K, V, M, DIM> *__restrict table,
//                             const K *__restrict keys, V **__restrict vectors,
//                             const bool *__restrict existed,
//                             int *__restrict src_offset,
//                             const bool *__restrict status,
//                             const int *__restrict bucket_offset, size_t N) {
//  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
//    int key_pos = -1;
//    int key_idx = t;
//    K hashed_key = Murmur3HashDevice(keys[t]);
//    int bkt_idx = hashed_key % table->buckets_num;
//    const size_t bucket_max_size = table->bucket_max_size;
//    const K insert_key = keys[t];
//    bool found = status[key_idx];
//    bool empty = false;
//
//    lock<Mutex>(table->locks[bkt_idx]);
//    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
//    if (found) {
//      key_pos = bucket_offset[key_idx];
//    } else {
//      insert_if_empty(bucket, bucket_max_size, insert_key, &key_pos, &empty,
//                      found, existed[key_idx]);
//    }
//    if (found == existed[key_idx]) {
//      key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
//      if (!found) {
//        atomicAdd(&(table->buckets_size[bkt_idx]), 1);
//        atomicMin(&(table->buckets_size[bkt_idx]), bucket_max_size);
//      }
//      atomicExch(&(bucket->keys[key_pos]), insert_key);
//      M cur_meta = 1 + atomicAdd(&(bucket->cur_meta), 1);
//      atomicExch(&(bucket->metas[key_pos].val), cur_meta);
//      refresh_bucket_meta<K, V, M, DIM>(bucket, bucket_max_size);
//      atomicCAS((size_t *)&(vectors[t]), (size_t)(nullptr),
//                (size_t)((V *)(bucket->vectors) + key_pos));
//      atomicExch(&(src_offset[key_idx]), key_idx);
//    }
//    unlock<Mutex>(table->locks[bkt_idx]);
//  }
//}

/* Lookup with no meta.*/
template <class K, class V, class M, size_t DIM, unsigned int TILE_SIZE = 8>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              M *__restrict metas, bool *__restrict found,
                              int *__restrict dst_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;
  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int rank = g.thread_rank();

    int key_idx = t / TILE_SIZE;
    int key_pos = -1;
    bool local_found = false;

    K find_key = EMPTY_KEY;
    K hashed_key = EMPTY_KEY;
    int bkt_idx = -1;

    if (rank == 0) {
      find_key = keys[key_idx];
      hashed_key = Murmur3HashDevice(find_key);
      bkt_idx = hashed_key % buckets_num;
    }
    find_key = g.shfl(find_key, 0);
    bkt_idx = g.shfl(bkt_idx, 0);

    Bucket<K, V, M, DIM> *bucket = table->buckets + bkt_idx;

    if (rank == 0) {
      lock<Mutex>(*(table->locks + bkt_idx));
    }
    g.sync();

    size_t tile_offset = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      K current_key = *(bucket->keys + tile_offset + rank);
      auto const found_vote = g.ballot(key_compare<K>(&find_key, &current_key));
      if (found_vote) {
        local_found = true;
        key_pos = tile_offset + __ffs(found_vote) - 1;
        break;
      }
    }

    g.sync();
    if (rank == 0) {
      *(vectors + key_idx) =
          local_found ? (bucket->vectors + key_pos) : nullptr;
      if (metas != nullptr && local_found) {
        *(metas + key_idx) = bucket->metas[key_pos].val;
      }
      if (found != nullptr) {
        *(found + key_idx) = local_found;
      }
      *(dst_offset + key_idx) = key_idx;
      unlock<Mutex>(*(table->locks + bkt_idx));
    }
  }
}

/* Lookup for and return offsets in buckets.*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_for_upsert_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    bool *__restrict found, int *__restrict offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;
  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t / bucket_max_size;
    int key_pos = t % bucket_max_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      found[key_idx] = true;
      atomicExch((int *)&(offset[key_idx]), key_pos);
    }
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class M, size_t DIM>
__global__ void clear_kernel(Table<K, V, M, DIM> *__restrict table, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    /// M_LANGER: Without lock, potential race condition here?
    atomicExch((K *)&(bucket->keys[key_idx]), EMPTY_KEY);
    if (key_idx == 0) {
      atomicExch(&(table->buckets_size[bkt_idx]), 0);
    }
  }
}

/* Remove specified keys. */
template <class K, class V, class M, size_t DIM>
__global__ void remove_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys,
                              size_t *__restrict count, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t / bucket_max_size;
    int key_pos = t % bucket_max_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    lock<Mutex>(table->locks[bkt_idx]);
    /// Prober the current key. Clear it if equal. Then clear metadata to
    /// indicate the field is free.
    K old_key = atomicCAS((K *)&bucket->keys[key_pos], target_key, EMPTY_KEY);
    if (old_key == target_key) {
      atomicExch((K *)&(bucket->metas[key_pos].val), EMPTY_META);
      atomicSub(&(table->buckets_size[bkt_idx]), 1);
      atomicMax(&(table->buckets_size[bkt_idx]), 0);
      atomicAdd(count, 1);
    }
    unlock<Mutex>(table->locks[bkt_idx]);
  }
}

/* Remove specified keys which match the Predict. */
template <class K, class V, class M, size_t DIM>
__global__ void remove_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const Predict<K, M> pred,
                              size_t *__restrict count, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;
  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    lock<Mutex>(table->locks[bkt_idx]);
    if (table->buckets_size[bkt_idx] > 0) {
      K target_key = bucket->keys[key_idx];
      if (target_key != EMPTY_KEY) {
        if (pred(target_key, bucket->metas[key_idx].val)) {
          atomicExch((K *)&(bucket->keys[key_idx]), EMPTY_KEY);
          atomicExch((K *)&(bucket->metas[key_idx].val), EMPTY_META);
          atomicSub(&(table->buckets_size[bkt_idx]), 1);
          atomicMax(&(table->buckets_size[bkt_idx]), 0);
          atomicAdd(count, 1);
        }
      }
    }
    unlock<Mutex>(table->locks[bkt_idx]);
  }
}

/* Dump without meta. */
template <class K, class V, class M, size_t DIM>
__global__ void dump_kernel(const Table<K, V, M, DIM> *__restrict table,
                            K *__restrict d_key, V *__restrict d_val,
                            const size_t offset, const size_t search_length,
                            size_t *__restrict d_dump_counter) {
  extern __shared__ unsigned char s[];
  K *smem = (K *)s;
  K *block_result_key = smem;
  V *block_result_val = (V *)&(smem[blockDim.x]);
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
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_idx] != EMPTY_KEY) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = bucket->keys[key_idx];
      for (int i = 0; i < DIM; i++) {
        atomicExch(&(block_result_val[local_index].value[i]),
                   bucket->vectors[key_idx].value[i]);
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
      d_val[global_acc + threadIdx.x].value[i] =
          block_result_val[threadIdx.x].value[i];
    }
  }
}

/* Dump with meta. */
template <class K, class V, class M, size_t DIM>
__global__ void dump_kernel(const Table<K, V, M, DIM> *__restrict table,
                            K *d_key, V *__restrict d_val, M *__restrict d_meta,
                            const size_t offset, const size_t search_length,
                            size_t *d_dump_counter) {
  extern __shared__ unsigned char s[];
  K *smem = (K *)s;
  K *block_result_key = smem;
  V *block_result_val = (V *)&(smem[blockDim.x]);
  M *block_result_meta = (M *)&(block_result_val[blockDim.x]);
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
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_idx] != EMPTY_KEY) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = bucket->keys[key_idx];
      for (int i = 0; i < DIM; i++) {
        atomicExch(&(block_result_val[local_index].value[i]),
                   bucket->vectors[key_idx].value[i]);
      }
      block_result_meta[local_index] = bucket->metas[key_idx];
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
      d_val[global_acc + threadIdx.x].value[i] =
          block_result_val[threadIdx.x].value[i];
    }
    d_meta[global_acc + threadIdx.x] = block_result_meta[threadIdx.x];
  }
}

static inline size_t GB(size_t n) { return n << 30; }

static inline size_t MB(size_t n) { return n << 20; }

static inline size_t KB(size_t n) { return n << 10; }

}  // namespace merlin
}  // namespace nv
