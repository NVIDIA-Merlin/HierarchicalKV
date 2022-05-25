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
#include <curand.h>
#include <curand_kernel.h>

#include "curand_philox4x32_x.h"
#include "types.cuh"
#include "util.cuh"

namespace nv {
namespace merlin {
namespace initializers {

inline void cuda_rand_check_(curandStatus_t val, const char *file, int line) {
  if (val != CURAND_STATUS_SUCCESS) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CURAND error " + std::to_string(val));
  }
}

#define CURAND_CHECK(val) \
  { cuda_rand_check_((val), __FILE__, __LINE__); }

template <class K, class V, class T, class M, size_t DIM>
void zeros(Table<K, V, M, DIM> **table) {
  for (int i = 0; i < (*table)->buckets_num; i++) {
    cudaMemset((*table)->buckets[i].vectors, 0,
               (*table)->buckets_size * sizeof(V));
  }
}

template <class K, class V, class M, class T, size_t DIM>
void random_normal(Table<K, V, M, DIM> **table, T mean = 0.0, T stddev = 0.05,
                   unsigned long long seed = 2022ULL) {
  curandGenerator_t generator;
  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
  for (int i = 0; i < (*table)->buckets_num; i++) {
    CURAND_CHECK(curandGenerateNormal(generator, (*table)->buckets[i].vectors,
                                      (*table)->buckets_size * sizeof(V), mean,
                                      stddev));
  }
}

template <class T>
__global__ void adjust_max_min(T *d_data, T minval, T maxval, size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    d_data[tid] =
        d_data[tid] * (maxval - minval) + (0.5 * (maxval + minval) - 0.5);
  }
}

template <class K, class V, class M, class T, size_t DIM>
void random_uniform(Table<K, V, M, DIM> **table, T minval = 0.0, T maxval = 1.0,
                    unsigned long long seed = 2022ULL) {
  cudaStream_t stream;
  curandGenerator_t generator;

  CUDA_CHECK(cudaStreamCreate(&stream));
  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));

  for (int i = 0; i < (*table)->buckets_num; i++) {
    int N = (*table)->buckets_size * sizeof(V);
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    CURAND_CHECK(
        curandGenerateUniform(generator, (*table)->buckets[i].vectors, N));
    adjust_max_min<T><<<grid_size, block_size, 0, stream>>>(
        (*table)->buckets[i].vectors, minval, maxval, N);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

__global__ void init_states(curandStatePhilox4_32_10_t *states,
                            unsigned long long seed, size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    curand_init(seed, tid, 0, &states[tid]);
  }
}

template <class T>
__global__ void make_truncated_normal(T *d_data,
                                      curandStatePhilox4_32_10_t *states,
                                      size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    constexpr T truncated_val = T(2.0);
    while (fabsf(d_data[tid]) > truncated_val) {
      d_data[tid] = curand_normal(&states[tid]);
    }
  }
}

template <class K, class V, class M, class T, size_t DIM>
void truncated_normal(Table<K, V, M, DIM> **table, T minval = 0.0,
                      T maxval = 1.0, unsigned long long seed = 2022ULL) {
  cudaStream_t stream;
  curandGenerator_t generator;

  CUDA_CHECK(cudaStreamCreate(&stream));
  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
  for (int i = 0; i < (*table)->buckets_num; i++) {
    int N = (*table)->buckets_size * sizeof(V);
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    curandStatePhilox4_32_10_t *d_states;
    cudaMalloc(&d_states, N);

    init_states<<<grid_size, block_size, 0, stream>>>(d_states, seed, N);

    make_truncated_normal<T><<<grid_size, block_size, 0, stream>>>(
        (*table)->buckets[i].vectors, d_states, N);

    adjust_max_min<T><<<grid_size, block_size, 0, stream>>>(
        (*table)->buckets[i].vectors, minval, maxval, N);

    cudaFree(d_states);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace initializers
}  // namespace merlin
}  // namespace nv