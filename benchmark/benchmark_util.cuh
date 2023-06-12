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

enum class API_Select {
  find = 0,
  insert_or_assign = 1,
  find_or_insert = 2,
  assign = 3,
  insert_and_evict = 4,
  find_ptr = 5,
  find_or_insert_ptr = 6,
  export_batch = 7,
  export_batch_if = 8,
};

enum class Hit_Mode {
  random = 0,
  last_insert = 1,
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

// RAII Timer using CUDA Event
template <typename Rep>
struct KernelTimer {
  explicit KernelTimer(TimeUnit tu = TimeUnit::Second) : tu_(tu) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&end_));
  }
  ~KernelTimer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(end_));
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  void end() {
    CUDA_CHECK(cudaEventRecord(end_));
    CUDA_CHECK(cudaEventSynchronize(end_));
    CUDA_CHECK(cudaEventElapsedTime(&time, start_, end_));
  }
  Rep getResult() {
    auto pow_ =
        static_cast<int32_t>(tu_) - static_cast<int32_t>(TimeUnit::MilliSecond);
    auto factor = static_cast<Rep>(std::pow(10, pow_));
    return static_cast<Rep>(time * factor);
  }

 private:
  TimeUnit tu_;
  float time{-1.0f};
  cudaEvent_t start_;
  cudaEvent_t end_;
};

inline uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <class K, class S>
void create_continuous_keys(K* h_keys, S* h_scores, const int key_num_per_op,
                            const K start = 0) {
  for (K i = 0; i < key_num_per_op; i++) {
    h_keys[i] = start + static_cast<K>(i);
    if (h_scores != nullptr) h_scores[i] = getTimestamp();
  }
}

template <class K, class S>
void create_random_keys(K* h_keys, S* h_scores, const int key_num_per_op) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < key_num_per_op) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) h_scores[i] = getTimestamp();
    i++;
  }
}

template <typename K, typename S>
void create_keys_for_hitrate(K* h_keys, S* h_scores, const int key_num_per_op,
                             const float hitrate = 0.6f,
                             const Hit_Mode hit_mode = Hit_Mode::last_insert,
                             const K end = 0, const bool reset = false) {
  int divide = static_cast<int>(key_num_per_op * hitrate);
  if (Hit_Mode::random == hit_mode) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    K existed_max = end == 0 ? 1 : (end - 1);
    std::uniform_int_distribution<K> distr(0, existed_max);

    if (existed_max < divide) {
      std::cout << "# Can not generate enough keys for hit!";
      exit(-1);
    }
    std::unordered_set<K> numbers;
    while (numbers.size() < divide) {
      numbers.insert(distr(eng));
    }
    int i = 0;
    for (auto existed_value : numbers) {
      h_keys[i] = existed_value;
      if (h_scores != nullptr) h_scores[i] = getTimestamp();
      i++;
    }
  } else {
    // else keep its original value, but update scores
    for (int i = 0; i < divide; i++) {
      if (h_scores != nullptr) h_scores[i] = getTimestamp();
    }
  }

  static K new_value = std::numeric_limits<K>::max();
  if (reset) {
    new_value = std::numeric_limits<K>::max();
  }
  for (int i = divide; i < key_num_per_op; i++) {
    h_keys[i] = new_value--;
    if (h_scores != nullptr) h_scores[i] = getTimestamp();
  }
}

template <typename S>
void refresh_scores(S* h_scores, const int key_num_per_op) {
  for (int i = 0; i < key_num_per_op; i++) {
    h_scores[i] = getTimestamp();
  }
}

template <class K, class V>
void init_value_using_key(K* h_keys, V* h_vectors, const int key_num_per_op,
                          int dim) {
  for (size_t i = 0; i < key_num_per_op; i++) {
    for (size_t j = 0; j < dim; j++) {
      h_vectors[i * dim + j] = static_cast<V>(h_keys[i] * 0.00001);
    }
  }
}

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

template <class S>
__global__ void host_nano_kernel(S* d_clk) {
  S mclk;
  asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(mclk));
  *d_clk = mclk;
}

template <class S>
S host_nano(cudaStream_t stream = 0) {
  S h_clk = 0;
  S* d_clk;

  CUDA_CHECK(cudaMalloc((void**)&(d_clk), sizeof(S)));
  host_nano_kernel<S><<<1, 1, 0, stream>>>(d_clk);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpy(&h_clk, d_clk, sizeof(S), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_clk));
  return h_clk;
}

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __device__ bool operator()(const K& key, S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score > threshold;
  }
};

}  // namespace benchmark
