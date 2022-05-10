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
  K *keys;         // Device memory
  Meta<M> *metas;  // Device memory
  V *cache;        // Device memory
  V *vectors;      // Pinned host memory
  M min_meta;
  int min_pos;
  int size;
};

template <class K, class V, class M, size_t DIM>
struct Table {
  Bucket<K, V, M, DIM> *buckets;
  unsigned int *locks;
  uint64_t init_size;
  uint64_t table_size;
  uint64_t buckets_size;
  uint64_t cache_size;
};

template <class K, class V, class M, size_t DIM>
void create_table(Table<K, V, M, DIM> **table, uint64_t init_size = 134217728,
                  uint64_t buckets_size = 128) {
  cudaMallocManaged((void **)table, sizeof(Table<K, V, M, DIM>));
  (*table)->init_size = init_size;
  (*table)->buckets_size = buckets_size;
  (*table)->table_size = 1 + (init_size - 1) / (*table)->buckets_size;
  (*table)->cache_size = 0;
  cudaMallocManaged((void **)&((*table)->buckets),
                    (*table)->table_size * sizeof(Bucket<K, V, M, DIM>));

  cudaMalloc((void **)&((*table)->locks), (*table)->table_size * sizeof(int));
  cudaMemset((*table)->locks, 0, (*table)->table_size * sizeof(unsigned int));

  for (int i = 0; i < (*table)->table_size; i++) {
    cudaMalloc(&((*table)->buckets[i].keys),
               (*table)->buckets_size * sizeof(K));
    cudaMemset((*table)->buckets[i].keys, 0xFF,
               (*table)->buckets_size * sizeof(K));
    cudaMalloc(&((*table)->buckets[i].metas),
               (*table)->buckets_size * sizeof(M));
    cudaMalloc(&((*table)->buckets[i].cache), (*table)->cache_size * sizeof(V));
    cudaMallocHost(&((*table)->buckets[i].vectors),
                   (*table)->buckets_size * sizeof(V), cudaHostRegisterMapped);
  }
}

template <class K, class V, class M, size_t DIM>
void destroy_table(Table<K, V, M, DIM> **table) {
  for (int i = 0; i < (*table)->table_size; i++) {
    cudaFree((*table)->buckets[i].keys);
    cudaFree((*table)->buckets[i].metas);
    cudaFree((*table)->buckets[i].cache);
    cudaFreeHost((*table)->buckets[i].vectors);
  }
  cudaFree((*table)->locks);
  cudaFree((*table)->buckets);
  cudaFree(*table);
}

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

template <class K, class V, class M, size_t DIM>
__global__ void read_kernel(V **__restrict src, V *__restrict dst,
                            const bool *found, const V *__restrict d_def_val,
                            int N, bool full_size_default) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    int default_index = full_size_default ? vec_index : 0;
    if (found[vec_index]) {
      dst[vec_index].value[dim_index] = (*(src[vec_index])).value[dim_index];
    } else {
      dst[vec_index].value[dim_index] =
          d_def_val[default_index].value[dim_index];
    }
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void upsert_with_meta_kernel(
    const Table<K, V, M, DIM> *__restrict table, const K *__restrict keys,
    const M *__restrict metas, V **__restrict vectors, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = 0;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % table->table_size;
    const uint64_t buckets_size = table->buckets_size;
    const K insert_key = keys[tid];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        for (int i = 0; i < buckets_size; i++) {
          if (bucket->keys[i] == insert_key) {
            found = true;
            key_pos = i;
            break;
          }
        }
        if (metas[key_idx] < bucket->min_meta && !found &&
            bucket->size >= buckets_size) {
          vectors[tid] = nullptr;
        } else {
          if (!found) {
            bucket->size += 1;
            key_pos = (bucket->size <= buckets_size) ? bucket->size + 1
                                                     : bucket->min_pos;
            if (bucket->size > buckets_size) {
              bucket->size = buckets_size;
            }
          }
          bucket->keys[key_pos] = insert_key;
          bucket->metas[key_pos].val = metas[key_idx];

          M tmp_min_val = MAX_META;
          int tmp_min_pos = 0;
          for (int i = 0; i < buckets_size; i++) {
            if (bucket->keys[i] == EMPTY_KEY) {
              break;
            }
            if (bucket->metas[i].val < tmp_min_val) {
              tmp_min_pos = i;
              tmp_min_val = bucket->metas[i].val;
            }
          }
          bucket->min_pos = tmp_min_pos;
          bucket->min_meta = tmp_min_val;
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }

    vectors[tid] = (V *)((V *)(bucket->vectors) + key_pos);
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void upsert_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = 0;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % table->table_size;
    const K insert_key = keys[tid];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);
    while (key_pos < buckets_size) {
      const K old_key =
          atomicCAS((K *)&bucket->keys[key_pos], EMPTY_KEY, insert_key);
      if (EMPTY_KEY == old_key || insert_key == old_key) {
        break;
      }
      key_pos++;
    }
    key_pos = key_pos % buckets_size;
    vectors[tid] = (V *)((V *)(bucket->vectors) + key_pos);
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void lookup_kernel(const Table<K, V, M, DIM> *__restrict table,
                              const K *__restrict keys, V **__restrict vectors,
                              bool *__restrict found, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_size = table->buckets_size;
  if (tid < N) {
    int key_idx = tid / buckets_size;
    int key_pos = tid % buckets_size;
    int bkt_idx = keys[key_idx] % buckets_size;
    K target_key = keys[key_idx];
    Bucket<K, V, M, DIM> *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (V *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

template <class K, class V, class M, size_t DIM>
__global__ void size_kernel(Table<K, V, M, DIM> *table, uint64_t *size, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const uint64_t buckets_size = table->buckets_size;

  if (tid < N) {
    for (int i = 0; i < buckets_size; i++) {
      if (table->buckets[tid].keys[i] != EMPTY_KEY) {
        atomicAdd((unsigned long long int *)&(size[tid]), 1);
      }
    }
  }
}

}  // namespace merlin
}  // namespace nv