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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"

#define MERLIN_EXPECT_TRUE(cond, msg)                                    \
  if (!cond) {                                                           \
    fprintf(stderr, "[ERROR] %s at %s : %d\n", msg, __FILE__, __LINE__); \
    exit(-1);                                                            \
  }

namespace test_util {

__global__ void all_true(const bool* conds, size_t n, int* nfalse) {
  const size_t stripe =
      (n + gridDim.x - 1) /
      gridDim.x;  // number of elements assigned to each block.
  size_t start = blockIdx.x * stripe + threadIdx.x;
  size_t end = min(start + stripe, n);

  __shared__ int local_nfalse;
  if (threadIdx.x == 0) {
    local_nfalse = 0;
  }
  __syncthreads();

  for (size_t i = start; i < end; i += blockDim.x) {
    if (!conds[i]) {
      atomicAdd(&local_nfalse, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(nfalse, local_nfalse);
  }
}

template <typename T>
__global__ void all_equal(T* a, T* b, size_t n, int* ndiff) {
  const size_t stripe =
      (n + gridDim.x - 1) /
      gridDim.x;  // number of elements assigned to each block.
  size_t start = blockIdx.x * stripe + threadIdx.x;
  size_t end = min(start + stripe, n);

  __shared__ int local_ndiff;
  if (threadIdx.x == 0) {
    local_ndiff = 0;
  }
  __syncthreads();

  for (size_t i = start; i < end; i += blockDim.x) {
    if (a[i] != b[i]) {
      atomicAdd(&local_ndiff, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(ndiff, local_ndiff);
  }
}

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <class K, class M>
void create_random_keys(K* h_keys, M* h_metas, int KEY_NUM) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <typename T>
void getBufferOnDevice(T** ptr, size_t size, cudaStream_t stream) {
  MERLIN_EXPECT_TRUE((*ptr == nullptr), "Pointer is already assigned.");
  CUDA_CHECK(cudaMallocAsync(ptr, size, stream));
  CUDA_CHECK(cudaMemsetAsync(*ptr, 0, size, stream));
}

void freeBufferOnDevice(void* ptr, cudaStream_t stream) {
  CUDA_CHECK(cudaFreeAsync(ptr, stream));
  ptr = nullptr;
}

bool allTrueGpu(const bool* conds, size_t n, cudaStream_t stream) {
  int nfalse = 0;
  int* d_nfalse = nullptr;
  getBufferOnDevice(&d_nfalse, sizeof(int), stream);
  int block_size = 128;
  int grid_size = (n + block_size - 1) / block_size;
  all_true<<<grid_size, block_size, 0, stream>>>(conds, n, d_nfalse);
  CUDA_CHECK(cudaMemcpyAsync(&nfalse, d_nfalse, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  freeBufferOnDevice(d_nfalse, stream);
  cudaStreamSynchronize(stream);
  return nfalse == 0;
}

template <typename T>
bool allEqualGpu(T* a, T* b, size_t n, cudaStream_t stream) {
  int ndiff = 0;
  int* d_ndiff = nullptr;
  getBufferOnDevice(&d_ndiff, sizeof(int), stream);
  int block_size = 128;
  int grid_size = (n + block_size - 1) / block_size;
  all_equal<<<grid_size, block_size, 0, stream>>>(a, b, n, d_ndiff);
  CUDA_CHECK(cudaMemcpyAsync(&ndiff, d_ndiff, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  freeBufferOnDevice(d_ndiff, stream);
  cudaStreamSynchronize(stream);
  return ndiff == 0;
}

#define TableType nv::merlin::HashTable<K, V, M>
template <typename K, typename V, typename M>
bool tables_equal(TableType* a, TableType* b, cudaStream_t stream) {
  size_t size = a->size(stream);
  if (size != b->size(stream)) {
    return false;
  }

  if (a->dim() != b->dim()) {
    return false;
  }

  size_t* d_size = nullptr;
  K* d_keys = nullptr;
  V* d_vectors = nullptr;
  M* d_metas = nullptr;
  bool* d_founds_in_b = nullptr;
  V* d_vectors_in_b = nullptr;
  M* d_metas_in_b = nullptr;

  getBufferOnDevice(&d_size, sizeof(size_t), stream);
  getBufferOnDevice(&d_keys, sizeof(K) * size, stream);
  getBufferOnDevice(&d_vectors, sizeof(V) * size * a->dim(), stream);
  getBufferOnDevice(&d_metas, sizeof(M) * size, stream);
  getBufferOnDevice(&d_founds_in_b, sizeof(bool) * size, stream);
  getBufferOnDevice(&d_vectors_in_b, sizeof(V) * size * a->dim(), stream);
  getBufferOnDevice(&d_metas_in_b, sizeof(M) * size, stream);

  a->export_batch(a->capacity(), 0, d_size, d_keys, d_vectors, d_metas, stream);
  b->find(size, d_keys, d_vectors_in_b, d_founds_in_b, d_metas_in_b, stream);
  if (!allTrueGpu(d_founds_in_b, size, stream)) {
    CUDA_FREE_POINTERS(stream, d_size, d_keys, d_vectors, d_metas,
                       d_founds_in_b, d_vectors_in_b, d_metas_in_b);
    return false;
  }
  if (!allEqualGpu(d_vectors, d_vectors_in_b, size * a->dim(), stream)) {
    CUDA_FREE_POINTERS(stream, d_size, d_keys, d_vectors, d_metas,
                       d_founds_in_b, d_vectors_in_b, d_metas_in_b);
    return false;
  }
  return true;
}

template <typename T, std::size_t N>
std::array<T, N> range(const T start) {
  std::array<T, N> result;
  size_t i = 0;
  while (i < N) {
    result[i] = start + i;
    i++;
  }
  return result;
}
}  // namespace test_util
