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

#include <exception>
#include <string>

#include "cuda_runtime_api.h"

#define CUDA_CHECK(val) \
  { nv::merlin::cuda_check_((val), __FILE__, __LINE__); }

__inline__ __device__ uint64_t atomicCAS(uint64_t* address, uint64_t compare,
                                         uint64_t val) {
  return (uint64_t)atomicCAS((unsigned long long*)address,
                             (unsigned long long)compare,
                             (unsigned long long)val);
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare,
                                        int64_t val) {
  return (int64_t)atomicCAS((unsigned long long*)address,
                            (unsigned long long)compare,
                            (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicExch(uint64_t* address, uint64_t val) {
  return (uint64_t)atomicExch((unsigned long long*)address,
                              (unsigned long long)val);
}

__inline__ __device__ int64_t atomicExch(int64_t* address, int64_t val) {
  return (int64_t)atomicExch((unsigned long long*)address,
                             (unsigned long long)val);
}

__inline__ __device__ int64_t atomicAdd(int64_t* address, const int64_t val) {
  return (int64_t)atomicAdd((unsigned long long*)address,
                            (const unsigned long long)val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t* address,
                                         const uint64_t val) {
  return (uint64_t)atomicAdd((unsigned long long*)address,
                             (const unsigned long long)val);
}

namespace nv {
namespace merlin {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
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

__inline__ __device__ uint64_t Murmur3HashDevice(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

class CudaDeviceRestorer {
 public:
  CudaDeviceRestorer() { CUDA_CHECK(cudaGetDevice(&dev_)); }
  ~CudaDeviceRestorer() { CUDA_CHECK(cudaSetDevice(dev_)); }

 private:
  int dev_;
};

inline int get_dev(const void* ptr) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  int dev = -1;

#if CUDART_VERSION >= 10000
  if (attr.type == cudaMemoryTypeDevice)
#else
  if (attr.memoryType == cudaMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}

inline void switch_to_dev(const void* ptr) {
  int dev = get_dev(ptr);
  if (dev >= 0) {
    CUDA_CHECK(cudaSetDevice(dev));
  }
}

}  // namespace merlin
}  // namespace nv
