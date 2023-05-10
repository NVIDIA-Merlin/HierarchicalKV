/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <chrono>
#include <cmath>
#include <cstdint>
#include "merlin/utils.cuh"

namespace benchmark {
enum class TimeUnit {
  Second = 0,
  MilliSecond = 3,
  MicroSecond = 6,
  NanoSecond = 9,
};

template <typename Rep>
struct Timer {
  explicit Timer(TimeUnit tu = TimeUnit::Second) : tu_(tu) {}
  void start() { startRecord = std::chrono::steady_clock::now(); }
  void end() { endRecord = std::chrono::steady_clock::now(); }
  Rep getResult() {
    auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endRecord - startRecord);
    auto pow_ =
        static_cast<int32_t>(tu_) - static_cast<int32_t>(TimeUnit::NanoSecond);
    auto factor = static_cast<Rep>(std::pow(10, pow_));
    return static_cast<Rep>(duration_.count()) * factor;
  }

 private:
  TimeUnit tu_;
  std::chrono::time_point<std::chrono::steady_clock> startRecord{};
  std::chrono::time_point<std::chrono::steady_clock> endRecord{};
};
template <class V>
__global__ void read_from_ptr_kernel(const V* const* __restrict src,
                                     V* __restrict dst, const size_t dim,
                                     size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    dst[vec_index * dim + dim_index] = src[vec_index][dim_index];
  }
}

template <class V>
void read_from_ptr(const V* const* __restrict src, V* __restrict dst,
                   const size_t dim, size_t n, cudaStream_t stream) {
  const size_t block_size = 1024;
  const size_t N = n * dim;
  const size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(N, block_size);

  read_from_ptr_kernel<V>
      <<<grid_size, block_size, 0, stream>>>(src, dst, dim, N);
}

template <class V>
__global__ void array2ptr_kernel(V** ptr, V* __restrict array, const size_t dim,
                                 size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t);
    ptr[vec_index] = array + vec_index * dim;
  }
}

template <class V>
void array2ptr(V** ptr, V* __restrict array, const size_t dim, size_t n,
               cudaStream_t stream) {
  const size_t block_size = 1024;
  const size_t N = n;
  const size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(N, block_size);

  array2ptr_kernel<V><<<grid_size, block_size, 0, stream>>>(ptr, array, dim, N);
}
}  // namespace benchmark
