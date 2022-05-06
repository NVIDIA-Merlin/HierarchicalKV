/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
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

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

namespace nv {
namespace merlin {

// constexpr M MAX_META = (M)(0xFFFFFFFFFFFFFFFF);

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus) {                                          \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
      exit(1);                                                                \
    }                                                                         \
  }
#endif

inline uint64_t Murmur3Hash(const uint64_t &key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

inline uint64_t Murmur3Hash(const uint64_t &key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <typename K, typename V, size_t DIM,
          uint64_t INIT_SIZE = 4 * 32 * 1024 * 1024,
          uint64_t BUCKETS_SIZE = 128,
          uint64_t TABLE_SIZE = INIT_SIZE / BUCKETS_SIZE, typename M = uint64_t,
          K EMPTY_KEY = std::numeric_limits<K>::max(),
          M MAX_META = std::numeric_limits<M>::max()>
class concurrent_ordered_map : public managed {
 private:
  struct Bucket {
    K *keys;          // Device memory
    Meta *metas;      // Device memory
    Vector *cache;    // Device memory
    Vector *vectors;  // Pinned host memory
    M min_meta;
    int min_pos;
    int size;
  };

  struct __align__(16) Vector {
    V values[DIM];
  };

  struct Table {
    Bucket *buckets;
    unsigned int *locks;
  };

 private:
  void create_table() {
    cudaMallocManaged((void **)table_, sizeof(Table));
    cudaMallocManaged((void **)&((*table_)->buckets),
                      table_size * sizeof(Bucket));

    cudaMalloc((void **)&((*table_)->locks), TABLE_SIZE * sizeof(int));
    cudaMemset((*table_)->locks, 0, TABLE_SIZE * sizeof(unsigned int));

    for (int i = 0; i < TABLE_SIZE; i++) {
      cudaMalloc(&((*table_)->buckets[i].keys), BUCKETS_SIZE * sizeof(K));
      cudaMemset((*table_)->buckets[i].keys, 0xFF, BUCKETS_SIZE * sizeof(K));
      cudaMalloc(&((*table_)->buckets[i].metas), BUCKETS_SIZE * sizeof(M));
      cudaMalloc(&((*table_)->buckets[i].cache), CACHE_SIZE * sizeof(Vector));
      cudaMallocHost(&((*table_)->buckets[i].vectors),
                     BUCKETS_SIZE * sizeof(Vector), cudaHostRegisterMapped);
    }
  }

  void destroy_table() {
    for (int i = 0; i < TABLE_SIZE; i++) {
      cudaFree((*table_)->buckets[i].keys);
      cudaFree((*table_)->buckets[i].metas);
      cudaFree((*table_)->buckets[i].cache);
      cudaFreeHost((*table_)->buckets[i].vectors);
    }
    cudaFree((*table_)->locks);
    cudaFree((*table_)->buckets);
    cudaFree(*table_);
  }

 public:
  concurrent_ordered_map(const concurrent_ordered_map &) = delete;
  concurrent_ordered_map &operator=(const concurrent_ordered_map &) = delete;

  explicit concurrent_unordered_map() { create_table(); }
  ~concurrent_unordered_map() { destroy_table(); }

  __global__ void write_kernel(const Vector *__restrict src,
                               Vector **__restrict dst, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
      int vec_index = int(tid / DIM);
      int dim_index = tid % DIM;

      if (dst[vec_index] != nullptr) {
        (*(dst[vec_index])).values[dim_index] =
            src[vec_index].values[dim_index];
      }
    }
  }

  __global__ void read_kernel(Vector **__restrict src, Vector *__restrict dst,
                              int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
      int vec_index = int(tid / DIM);
      int dim_index = tid % DIM;
      if (src[vec_index] != nullptr) {
        dst[vec_index].values[dim_index] =
            (*(src[vec_index])).values[dim_index];
      }
    }
  }

  __global__ void upsert_kernel(const K *__restrict keys,
                                const M *__restrict metas,
                                Vector **__restrict vectors, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int key_pos = 0;
    bool found = false;

    if (tid < N) {
      int key_idx = tid;
      int bkt_idx = keys[tid] % TABLE_SIZE;
      const K insert_key = keys[tid];
      Bucket *bucket = &(table_->buckets[bkt_idx]);

      bool release_lock = false;
      while (!release_lock) {
        if (atomicExch(&(table_->locks[bkt_idx]), 1u) == 0u) {
          for (int i = 0; i < BUCKETS_SIZE; i++) {
            if (bucket->keys[i] == insert_key) {
              found = true;
              key_pos = i;
              break;
            }
          }
          if (metas[key_idx] < bucket->min_meta && !found &&
              bucket->size >= BUCKETS_SIZE) {
            vectors[tid] = nullptr;
          } else {
            if (!found) {
              bucket->size += 1;
              key_pos = (bucket->size <= BUCKETS_SIZE) ? bucket->size + 1
                                                       : bucket->min_pos;
              if (bucket->size > BUCKETS_SIZE) {
                bucket->size = BUCKETS_SIZE;
              }
            }
            bucket->keys[key_pos] = insert_key;
            bucket->metas[key_pos].val = metas[key_idx];

            M tmp_min_val = MAX_META;
            int tmp_min_pos = 0;
            for (int i = 0; i < BUCKETS_SIZE; i++) {
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
          atomicExch(&(table_->locks[bkt_idx]), 0u);
        }
      }

      vectors[tid] = (Vector *)((Vector *)(bucket->vectors) + key_pos);
    }
  }

  __global__ void lookup_kernel(const K *__restrict keys,
                                Vector **__restrict vectors,
                                bool *__restrict found, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
      int key_idx = tid / BUCKETS_SIZE;
      int key_pos = tid % BUCKETS_SIZE;
      int bkt_idx = keys[key_idx] % TABLE_SIZE;
      K target_key = keys[key_idx];
      Bucket *bucket = &(table_->buckets[bkt_idx]);

      if (bucket->keys[key_pos] == target_key) {
        vectors[key_idx] = (Vector *)&(bucket->vectors[key_pos]);
        found[key_idx] = true;
      }
    }
  }

  __global__ void size_kernel(size_t *size, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
      for (int i = 0; i < BUCKETS_SIZE; i++) {
        if (table_->buckets[tid].keys[i] != EMPTY_KEY) {
          atomicAdd((unsigned long long int *)&(size[tid]), 1);
        }
      }
    }
  }

 private:
  Table **table_;
}

template <typename KeyType, typename ValType, typename BaseValType,
          KeyType empty_key, size_t DIM>
class HashTable {
 public:
  HashTable(uint64_t init_size) {
    dim_ = DIM;
    init_size_ = init_size;
    buckets_size_ = 128;

    table_ = new Table<KeyType, BaseValType, dim_>();
  }
  ~HashTable() { delete table_; }
  HashTable(const HashTable &) = delete;
  HashTable &operator=(const HashTable &) = delete;

  void upsert(const KeyType *d_keys, const BaseValType *d_vals, size_t len,
              cudaStream_t stream) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    upsert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, (const ValType *)d_vals, len);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void get(const KeyType *d_keys, BaseValType *d_vals, bool *d_status,
           size_t len, BaseValType *d_def_val, cudaStream_t stream,
           bool full_size_default) const {
    if (len == 0) {
      return;
    }
    CUDA_CHECK(cudaMemset((void *)d_vals, 0, sizeof(ValType) * len));
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        table_, d_keys, (ValType *)d_vals, d_status, len, (ValType *)d_def_val,
        full_size_default);
  }

  void accum(const KeyType *d_keys, const BaseValType *d_vals_or_deltas,
             const bool *d_exists, size_t len, cudaStream_t stream) {}

  size_t get_size(cudaStream_t stream) const {
    /* size variable on Host and device, total capacity of the hashtable */
    size_t table_size;
    size_t *d_table_size;
    const size_t hash_capacity = table_->size();

    /* grid_size and allocating/initializing variable on dev, lauching kernel*/
    const int grid_size = (hash_capacity - 1) / BLOCK_SIZE_ + 1;
    CUDA_CHECK(cudaMallocManaged((void **)&d_table_size, sizeof(size_t)));
    CUDA_CHECK(cudaMemset(d_table_size, 0, sizeof(size_t)));
    size_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, hash_capacity,
                                                       d_table_size, empty_key);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&table_size, d_table_size, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    /* Copy result back and do clean up*/
    CUDA_CHECK(cudaFree(d_table_size));
    return table_size;
  }

  size_t get_capacity() const { return (table_->size()); }
  void clear(cudaStream_t stream) { table_->clear_async(stream); }

  void remove(const KeyType *d_keys, size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    delete_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, len);
  }

 private:
  uint64_t dim_;
  uint64_t init_size_;

  Table *table_;
}

}  // namespace merlin
}  // namespace nv