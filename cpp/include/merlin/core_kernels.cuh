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

#include <cuda_runtime.h>

#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {

/* 2GB per slice by default.*/
constexpr size_t kDefaultBytesPerSlice = (2 * 1024 * 1024 * 1024ul);

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
      buckets_num * (*table)->buckets_size * sizeof(V);
  const size_t num_of_memory_slices =
      1 + (total_size_of_vectors - 1) / (*table)->bytes_per_slice;
  size_t num_of_buckets_in_one_slice =
      (*table)->bytes_per_slice / ((*table)->buckets_size * sizeof(V));
  size_t num_of_allocated_buckets = 0;

  realloc_managed<V **>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V *),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V *));

  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }

    if ((*table)->vector_on_gpu) {
      CUDA_CHECK(cudaMalloc(
          &((*table)->slices[i]),
          num_of_buckets_in_one_slice * (*table)->buckets_size * sizeof(V)));
    } else {
      CUDA_CHECK(cudaMallocHost(
          &((*table)->slices[i]),
          num_of_buckets_in_one_slice * (*table)->buckets_size * sizeof(V),
          cudaHostRegisterMapped));
    }
    for (int j = 0; j < num_of_buckets_in_one_slice; j++) {
      (*table)->buckets[start + num_of_allocated_buckets + j].vectors =
          (*table)->slices[i] + j * (*table)->buckets_size;
    }
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }
  (*table)->num_of_memory_slices += num_of_memory_slices;
  for (int i = start; i < end; i++) {
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].keys),
                          (*table)->buckets_size * sizeof(K)));
    CUDA_CHECK(cudaMemset((*table)->buckets[i].keys, 0xFF,
                          (*table)->buckets_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].metas),
                          (*table)->buckets_size * sizeof(M)));
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].cache),
                          (*table)->cache_size * sizeof(V)));
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
void create_table(Table<K, V, M, DIM> **table, uint64_t init_size = 134217728,
                  uint64_t max_size = std::numeric_limits<uint64_t>::max(),
                  uint64_t cache_size = 0, uint64_t buckets_size = 128,
                  bool vector_on_gpu = false, bool master = true,
                  size_t bytes_per_slice = kDefaultBytesPerSlice) {
  CUDA_CHECK(cudaMallocManaged((void **)table, sizeof(Table<K, V, M, DIM>)));
  CUDA_CHECK(cudaMemset(*table, 0, sizeof(Table<K, V, M, DIM>)));
  (*table)->buckets_size = buckets_size;
  (*table)->bytes_per_slice = bytes_per_slice;
  (*table)->max_size = max_size;

  (*table)->buckets_num = 1;
  while ((*table)->buckets_num * (*table)->buckets_size < init_size) {
    (*table)->buckets_num *= 2;
  }
  std::cout << "[create_table] init requested capacity=" << init_size
            << ", real capacity="
            << (*table)->buckets_num * (*table)->buckets_size << std::endl;
  (*table)->capacity = (*table)->buckets_num * (*table)->buckets_size;
  (*table)->cache_size = 0;
  (*table)->vector_on_gpu = vector_on_gpu;
  (*table)->primary_table = master;

  CUDA_CHECK(cudaMalloc((void **)&((*table)->locks),
                        (*table)->buckets_num * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset((*table)->locks, 0,
                        (*table)->buckets_num * sizeof(unsigned int)));

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
  realloc<unsigned int *>(&((*table)->locks),
                          (*table)->buckets_num * sizeof(unsigned int),
                          (*table)->buckets_num * sizeof(unsigned int) * 2);
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
    CUDA_CHECK(cudaFree((*table)->buckets[i].cache));
  }

  for (int i = 0; i < (*table)->num_of_memory_slices; i++) {
    if ((*table)->vector_on_gpu) {
      CUDA_CHECK(cudaFree((*table)->slices[i]));
    } else {
      CUDA_CHECK(cudaFreeHost((*table)->slices[i]));
    }
  }

  CUDA_CHECK(cudaFree((*table)->slices));
  CUDA_CHECK(cudaFree((*table)->locks));
  CUDA_CHECK(cudaFree((*table)->buckets));
  CUDA_CHECK(cudaFree(*table));
}

template <class K, class V, class M, size_t DIM>
__global__ void rehash_kernel(const Table<K, V, M, DIM> *__restrict table,
                              int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int bkt_idx = tid / buckets_size;
    int key_idx = tid % buckets_size;
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    K target_key = bucket->keys[key_idx];
    if (target_key != EMPTY_KEY) {
      K hashed_key = Murmur3HashDevice(target_key);
      int key_bkt_idx = hashed_key % buckets_num;

      if (key_bkt_idx != bkt_idx) {
        Bucket<K, V, M, DIM> *new_bucket = &(table->buckets[key_bkt_idx]);
        atomicExch(&(new_bucket->keys[key_idx]), target_key);
        atomicExch(&(bucket->keys[key_idx]), EMPTY_KEY);

        atomicExch(&(new_bucket->metas[key_idx].val),
                   bucket->metas[key_idx].val);
        atomicExch(&(bucket->metas[key_idx].val), 0ul);
        for (int i = 0; i < DIM; i++) {
          new_bucket->vectors[key_idx].value[i] =
              bucket->vectors[key_idx].value[i];
        }
      }
    }
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
                             const int *__restrict src_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

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
                                        int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

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
                                        int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

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
                            const int *__restrict dst_offset, int N,
                            bool full_size_default) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
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
                            const int *__restrict dst_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    if (src[vec_index] != nullptr) {
      dst[dst_offset[vec_index]].value[dim_index] =
          src[vec_index]->value[dim_index];
    }
  }
}

template <class K, class V, class M, size_t DIM>
__inline__ __device__ void find_in_bucket(const Bucket<K, V, M, DIM> *bucket,
                                          const uint64_t buckets_size,
                                          const K &insert_key, int *key_pos,
                                          bool *found) {
  for (int i = 0; i < buckets_size; i++) {
    if (bucket->keys[i] == insert_key) {
      *found = true;
      *key_pos = i;
      return;
    }
  }
  for (int i = 0; i < buckets_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__inline__ __device__ void find_in_bucket(const Bucket<K, V, M, DIM> *bucket,
                                          const uint64_t buckets_size,
                                          const K &insert_key, int *key_pos,
                                          bool *found, const bool existed) {
  for (int i = 0; i < buckets_size; i++) {
    if (bucket->keys[i] == insert_key) {
      *found = true;
      *key_pos = i;
      return;
    }
  }
  if (*found != existed) return;
  for (int i = 0; i < buckets_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__inline__ __device__ void insert_if_empty(const Bucket<K, V, M, DIM> *bucket,
                                           const uint64_t buckets_size,
                                           const K &insert_key, int *key_pos,
                                           bool *empty) {
  for (int i = 0; i < buckets_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      *empty = true;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__inline__ __device__ void insert_if_empty(const Bucket<K, V, M, DIM> *bucket,
                                           const uint64_t buckets_size,
                                           const K &insert_key, int *key_pos,
                                           bool *empty, const bool found,
                                           const bool existed) {
  if (found != existed) return;
  for (int i = 0; i < buckets_size; i++) {
    K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
    if (old_key == EMPTY_KEY) {
      *key_pos = i;
      *empty = true;
      break;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__inline__ __device__ void refresh_bucket_meta(Bucket<K, V, M, DIM> *bucket,
                                               const uint64_t buckets_size) {
  M min_val = MAX_META;
  int min_pos = 0;
  for (int i = 0; i < buckets_size; i++) {
    if (bucket->keys[i] == EMPTY_KEY) {
      continue;
    }
    if (bucket->metas[i].val < min_val) {
      min_pos = i;
      min_val = bucket->metas[i].val;
    }
  }
  atomicExch(&(bucket->min_pos), min_pos);
  atomicExch(&(bucket->min_meta), min_val);
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
template <class K, class V, class M, size_t DIM>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys,
                              const M *__restrict metas, V **__restrict vectors,
                              int *__restrict src_offset,
                              const bool *__restrict status,
                              const int *__restrict bucket_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];
    bool found = status[key_idx];
    bool empty = false;

    bool release_lock = false;
    while (!release_lock) {
      /// Spin-wait until we get access.
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        if (found) {
          key_pos = bucket_offset[key_idx];
        } else {
          insert_if_empty(bucket, buckets_size, insert_key, &key_pos, &empty);
        }

        /// Insert if either of the following cases is fulfilled:
        /// 1) We found the key: Then, we override the associated value and
        ///    meta.
        /// 2) The bucket is not yet full: Hence, we can append the key.
        /// 3) Meta of key to be insered is larger than smallest meta in bucket:
        ///    In this case we replace that key.
        if (metas[key_idx] >= bucket->min_meta || found || empty) {
          if (!found) {
            key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          atomicExch(&(bucket->metas[key_pos].val), metas[key_idx]);

          /// Re-locate the smallest meta.
          refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);

          /// Record storage offset. This will be used by write_kernel to map
          /// the input to the output data.
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
          atomicExch(&(src_offset[key_idx]), key_idx);
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

/* Upsert with no user specified meta.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `uint64_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              int *__restrict src_offset,
                              const bool *__restrict status,
                              const int *__restrict bucket_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];
    bool found = status[key_idx];
    bool empty = false;

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        if (found) {
          key_pos = bucket_offset[key_idx];
        } else {
          insert_if_empty(bucket, buckets_size, insert_key, &key_pos, &empty);
        }

        key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
        atomicExch(&(bucket->keys[key_pos]), insert_key);
        M cur_meta = atomicAdd(&(bucket->cur_meta), 1);
        atomicExch(&(bucket->metas[key_pos].val), cur_meta);

        refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);
        atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                  (uint64_t)((V *)(bucket->vectors) + key_pos));
        atomicExch(&(src_offset[key_idx]), key_idx);
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void upsert_allow_duplicate_keys_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    const M *__restrict metas, V **__restrict vectors,
    int *__restrict src_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        find_in_bucket<K, V, M, DIM>(bucket, buckets_size, insert_key, &key_pos,
                                     &found);

        if (metas[key_idx] >= bucket->min_meta || found ||
            bucket->size < buckets_size) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), buckets_size);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          atomicExch(&(bucket->metas[key_pos].val), metas[key_idx]);

          refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
          atomicExch(&(src_offset[key_idx]), key_idx);
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

/* Upsert with no user specified meta.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `uint64_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM>
__global__ void upsert_allow_duplicate_keys_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    V **__restrict vectors, int *__restrict src_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        find_in_bucket<K, V, M, DIM>(bucket, buckets_size, insert_key, &key_pos,
                                     &found);
        if (found || bucket->size < buckets_size) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), buckets_size);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          M cur_meta = atomicAdd(&(bucket->cur_meta), 1);
          atomicExch(&(bucket->metas[key_pos].val), cur_meta);

          refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
          atomicExch(&(src_offset[key_idx]), key_idx);
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

/* Upsert with no user specified meta and return if the keys already exists.
   The meta will be specified by kernel internally according to
   the `bucket->cur_meta` which always increment by 1 when insert happens,
   we assume the cur_meta with `uint64_t` type will never overflow.
*/
template <class K, class V, class M, size_t DIM>
__global__ void accum_kernel(const Table<K, V, M, DIM> *__restrict table,
                             const K *__restrict keys, V **__restrict vectors,
                             const bool *__restrict existed,
                             int *__restrict src_offset,
                             const bool *__restrict status,
                             const int *__restrict bucket_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];
    bool found = status[key_idx];
    bool empty = false;

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        if (found) {
          key_pos = bucket_offset[key_idx];
        } else {
          insert_if_empty(bucket, buckets_size, insert_key, &key_pos, &empty,
                          found, existed[key_idx]);
        }
        if (found == existed[key_idx]) {
          key_pos = (key_pos == -1) ? bucket->min_pos : key_pos;
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          M cur_meta = atomicAdd(&(bucket->cur_meta), 1);
          atomicExch(&(bucket->metas[key_pos].val), cur_meta);

          refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
          atomicExch(&(src_offset[key_idx]), key_idx);
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void accum_allow_duplicate_keys_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    V **__restrict vectors, const bool *__restrict existed,
    bool *__restrict now_exists, int *__restrict src_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    K hashed_key = Murmur3HashDevice(keys[tid]);
    int bkt_idx = hashed_key % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        find_in_bucket<K, V, M, DIM>(bucket, buckets_size, insert_key, &key_pos,
                                     &found, existed[key_idx]);
        now_exists[key_idx] = found;
        if (found == existed[key_idx] &&
            (found || bucket->size < buckets_size)) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), buckets_size);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          M cur_meta = atomicAdd(&(bucket->cur_meta), 1);
          atomicExch(&(bucket->metas[key_pos].val), cur_meta);

          refresh_bucket_meta<K, V, M, DIM>(bucket, buckets_size);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
          atomicExch(&(src_offset[key_idx]), key_idx);
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

/* Lookup keys from the table,
   this operation will not really read the vector and meta data
   from buckets and only return the address in bucket for each key
   through `vectors`. Usually caller should call the `read_kernel`
   to really read the data the addresses saved in `vectors`.

   `table`: The table to be operated.
   `keys`: Keys for search, if the key exists, will return
           the current address, if not found, a `nullptr`
           will be returned for the key through `vectors`.
   `vectors`: The addresses in buckets for each keys where the
              corresponding vector values should be really read from.
   `metas`: The corresponding meta value for each keys.
   `found`: Flag of existence for each keys.
   `N`: The number of key-value needed to be read.
*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              M *__restrict metas, bool *__restrict found,
                              int *__restrict dst_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      metas[key_idx] = bucket->metas[key_pos].val;
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
      atomicExch(&(dst_offset[key_idx]), key_idx);
    }
  }
}

/* Lookup with no meta.*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              bool *__restrict found,
                              int *__restrict dst_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (key_pos == 0) {
      atomicExch(&(dst_offset[key_idx]), key_idx);
    }

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

/* Lookup with no meta.*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              int *__restrict dst_offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (key_pos == 0) {
      atomicExch(&(dst_offset[key_idx]), key_idx);
    }

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
    }
  }
}

/* Lookup for and return offsets in buckets.*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_for_upsert_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    bool *__restrict found, int *__restrict offset, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
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

/* Get the number of stored Key-value in the table. */
template <class K, class V, class M, size_t DIM>
__global__ void size_kernel(Table<K, V, M, DIM> *table, size_t *size, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    int key_idx = tid % buckets_size;
    int bkt_idx = tid / buckets_size;

    /// Just count non empty bucket cells.
    if (table->buckets[bkt_idx].keys[key_idx] != EMPTY_KEY) {
      atomicAdd((unsigned long long int *)(size), 1);
    }
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class M, size_t DIM>
__global__ void clear_kernel(Table<K, V, M, DIM> *__restrict table, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    int key_idx = tid % buckets_size;
    int bkt_idx = tid / buckets_size;
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    atomicExch((K *)&(bucket->keys[key_idx]), EMPTY_KEY);

    /// M_LANGER: Without lock, potential race condition here?
    atomicExch((K *)&(bucket->metas[key_idx].val), EMPTY_META);
    if (key_idx == 0) {
      atomicExch(&(bucket->size), 0);
    }
  }
}

/* Remove specified keys. */
template <class K, class V, class M, size_t DIM>
__global__ void remove_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    K hashed_key = Murmur3HashDevice(keys[key_idx]);
    int bkt_idx = hashed_key % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    /// Prober the current key. Clear it if equal. Then clear metadata to
    /// indicate the field is free.
    K old_key = atomicCAS((K *)&bucket->keys[key_pos], target_key, EMPTY_KEY);
    if (old_key == target_key) {
      /// M_LANGER: Without lock, potential race condition here?
      atomicExch((K *)&(bucket->metas[key_pos].val), EMPTY_META);
      atomicDec((unsigned int *)&(bucket->size), buckets_size);
    }
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
  const uint64_t buckets_size = table->buckets_size;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / buckets_size;
    int key_idx = (tid + offset) % buckets_size;
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
  const uint64_t buckets_size = table->buckets_size;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / buckets_size;
    int key_idx = (tid + offset) % buckets_size;
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

}  // namespace merlin
}  // namespace nv
