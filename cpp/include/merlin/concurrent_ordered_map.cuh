/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include "util.cuh"

namespace nv {
namespace merlin {

template <class M>
struct Meta {
  M val;
};

constexpr uint64_t EMPTY_KEY = std::numeric_limits<uint64_t>::max();
constexpr uint64_t MAX_META = std::numeric_limits<uint64_t>::max();

template <class K, class V, class M, size_t DIM>
struct Bucket {
  K *keys;         // HBM
  Meta<M> *metas;  // HBM
  V *cache;        // HBM(optional)
  V *vectors;      // Pinned memory or HBM

  /* For upsert_kernel without user specified metas
     recording the current meta, the cur_meta will
     increment by 1 when a new inserting happens. */
  M cur_meta;

  /* min_meta and min_pos is for or upsert_kernel
     with user specified meta. They record the minimum
     meta and its pos in the bucket. */
  M min_meta;
  int min_pos;

  /* The number of saved key-value in this buckets */
  int size;
};

template <class K, class V, class M, size_t DIM>
struct Table {
  Bucket<K, V, M, DIM> *buckets;
  unsigned int *locks;            // Write lock for each bucket.
  uint64_t capacity = 134217728;  // Initial capacity.
  uint64_t buckets_num;
  uint64_t buckets_size = 128;
  uint64_t cache_size = 0;
  bool vector_on_gpu = false;
};

/* Initial a Table struct.

   K: The key type
   V: The value type which should be static array type and C++ class
      with customized construct is not supported.
   M: The meta type, the meta will be used to store the timestamp
      or occurrence frequency or any thing for eviction.
   DIM: Vector dimension.
*/
template <class K, class V, class M, size_t DIM>
void create_table(Table<K, V, M, DIM> **table, uint64_t capacity = 134217728,
                  uint64_t cache_size = 0, uint64_t buckets_size = 128,
                  bool vector_on_gpu = false) {
  cudaMallocManaged((void **)table, sizeof(Table<K, V, M, DIM>));
  (*table)->capacity = capacity;
  (*table)->buckets_size = buckets_size;
  (*table)->buckets_num = 1 + (capacity - 1) / (*table)->buckets_size;
  (*table)->cache_size = 0;
  (*table)->vector_on_gpu = vector_on_gpu;
  cudaMallocManaged((void **)&((*table)->buckets),
                    (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>));
  cudaMemset((*table)->buckets, 0,
             (*table)->buckets_num * sizeof(Bucket<K, V, M, DIM>));

  cudaMalloc((void **)&((*table)->locks), (*table)->buckets_num * sizeof(int));
  cudaMemset((*table)->locks, 0, (*table)->buckets_num * sizeof(unsigned int));

  for (int i = 0; i < (*table)->buckets_num; i++) {
    cudaMalloc(&((*table)->buckets[i].keys),
               (*table)->buckets_size * sizeof(K));
    cudaMemset((*table)->buckets[i].keys, 0xFF,
               (*table)->buckets_size * sizeof(K));
    cudaMalloc(&((*table)->buckets[i].metas),
               (*table)->buckets_size * sizeof(M));
    cudaMalloc(&((*table)->buckets[i].cache), (*table)->cache_size * sizeof(V));
    if ((*table)->vector_on_gpu) {
      cudaMalloc(&((*table)->buckets[i].vectors),
                 (*table)->buckets_size * sizeof(V));
    } else {
      cudaMallocHost(&((*table)->buckets[i].vectors),
                     (*table)->buckets_size * sizeof(V),
                     cudaHostRegisterMapped);
    }
  }
}

/* free all of the resource of a Table. */
template <class K, class V, class M, size_t DIM>
void destroy_table(Table<K, V, M, DIM> **table) {
  for (int i = 0; i < (*table)->buckets_num; i++) {
    cudaFree((*table)->buckets[i].keys);
    cudaFree((*table)->buckets[i].metas);
    cudaFree((*table)->buckets[i].cache);
    if ((*table)->vector_on_gpu) {
      cudaFree((*table)->buckets[i].vectors);
    } else {
      cudaFreeHost((*table)->buckets[i].vectors);
    }
  }
  cudaFree((*table)->locks);
  cudaFree((*table)->buckets);
  cudaFree(*table);
}

/* Write the N data from src to each address in *dst,
   usually called by upsert kernel.

   `src`: A continue memory pointer with Vector
          which can be HBM.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class M, size_t DIM>
__global__ void write_kernel(const V *__restrict src, V **__restrict dst,
                             int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    if (dst[vec_index] != nullptr) {
      (*(dst[vec_index])).value[dim_index] = src[vec_index].value[dim_index];
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
__global__ void read_kernel(V **__restrict src, V *__restrict dst,
                            const bool *mask, const V *__restrict default_val,
                            int N, bool full_size_default) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    int default_index = full_size_default ? vec_index : 0;
    if (mask[vec_index]) {
      dst[vec_index].value[dim_index] = (*(src[vec_index])).value[dim_index];
    } else {
      dst[vec_index].value[dim_index] =
          default_val[default_index].value[dim_index];
    }
  }
}

/* Insert or update a Key-Value in the table,
   this operation will not really write the vector data
   into buckets and only return the address in bucket for each key
   through `vectors`. Usually caller should call the `write_kernel`
   to really write the data the addresses in `vectors`.

   `table`: The table to be operated.
   `keys`: Keys for upsert, if the key exists, will return
           the current address, if not found, a new location
           in bucket will return for the key through `vectors`.
   `metas`: The corresponding meta value for each keys.
   `vectors`: The addresses in buckets for each keys where the
              corresponding vector values should be really saved into.
   `N`: The number of key-value needed to be read.
*/
template <class K, class V, class M, size_t DIM>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys,
                              const M *__restrict metas, V **__restrict vectors,
                              int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        for (int i = 0; i < buckets_size; i++) {
          if (bucket->keys[i] == insert_key) {
            found = true;
            key_pos = i;
            break;
          }
        }
        for (int i = 0; i < buckets_size; i++) {
          if (found) break;
          K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
          if (old_key == EMPTY_KEY) {
            key_pos = i;
            break;
          }
        }
        if (metas[key_idx] >= bucket->min_meta || found ||
            bucket->size < buckets_size) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), buckets_size);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          atomicExch(&(bucket->metas[key_pos].val), metas[key_idx]);

          M tmp_min_val = MAX_META;
          int tmp_min_pos = 0;
          for (int i = 0; i < buckets_size; i++) {
            if (bucket->keys[i] == EMPTY_KEY) {
              continue;
            }
            if (bucket->metas[i].val < tmp_min_val) {
              tmp_min_pos = i;
              tmp_min_val = bucket->metas[i].val;
            }
          }
          atomicExch(&(bucket->min_pos), tmp_min_pos);
          atomicExch(&(bucket->min_meta), tmp_min_val);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
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
                              int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % table->buckets_num;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
        for (int i = 0; i < buckets_size; i++) {
          if (bucket->keys[i] == insert_key) {
            found = true;
            key_pos = i;
            break;
          }
        }
        for (int i = 0; i < buckets_size; i++) {
          if (found) break;
          K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
          if (old_key == EMPTY_KEY) {
            key_pos = i;
            break;
          }
        }
        if (found || bucket->size < buckets_size) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), buckets_size);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          M cur_meta = atomicAdd(&(bucket->cur_meta), 1);
          atomicExch(&(bucket->metas[key_pos].val), cur_meta);

          M tmp_min_val = MAX_META;
          int tmp_min_pos = 0;
          for (int i = 0; i < buckets_size; i++) {
            if (bucket->keys[i] == EMPTY_KEY) {
              continue;
            }
            if (bucket->metas[i].val < tmp_min_val) {
              tmp_min_pos = i;
              tmp_min_val = bucket->metas[i].val;
            }
          }
          atomicExch(&(bucket->min_pos), tmp_min_pos);
          atomicExch(&(bucket->min_meta), tmp_min_val);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((V *)(bucket->vectors) + key_pos));
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
                              int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    int bkt_idx = keys[key_idx] % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      metas[key_idx] = bucket->metas[key_pos].val;
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

/* Lookup with no meta.*/
template <class K, class V, class M, size_t DIM>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              bool *__restrict found, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_num = table->buckets_num;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    int bkt_idx = keys[key_idx] % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
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
    if (table->buckets[bkt_idx].keys[key_idx] != EMPTY_KEY) {
      atomicAdd((unsigned long long int *)(size), 1);
    }
  }
}

/* Clear all key-value in the table. */
template <class K, class V, class M, size_t DIM>
__global__ void clear_kernel(Table<K, V, M, DIM> *table, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    int key_idx = tid % buckets_size;
    int bkt_idx = tid / buckets_size;
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    atomicExch((K *)&(bucket->keys[key_idx]), EMPTY_KEY);
    atomicExch((K *)&(bucket->metas[key_idx].val), MAX_META);
    if (key_idx == 0) atomicExch(&(bucket->size), 0);
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
    int bkt_idx = keys[key_idx] % buckets_num;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    K old_key = atomicCAS((K *)&bucket->keys[key_pos], target_key, EMPTY_KEY);
    if (old_key == target_key) {
      atomicExch((K *)&(bucket->metas[key_pos].val), MAX_META);
      atomicDec((unsigned int *)&(bucket->size), buckets_size);
    }
  }
}

/* Dump without meta. */
template <class K, class V, class M, size_t DIM>
__global__ void dump_kernel(const Table<K, V, M, DIM> *table, K *d_key,
                            V *d_val, const size_t offset,
                            const size_t search_length,
                            size_t *d_dump_counter) {
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
        block_result_val[local_index].value[i] =
            bucket->vectors[key_idx].value[i];
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
__global__ void dump_kernel(const Table<K, V, M, DIM> *table, K *d_key,
                            V *d_val, M *d_meta, const size_t offset,
                            const size_t search_length,
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
        block_result_val[local_index].value[i] =
            bucket->vectors[key_idx].value[i];
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