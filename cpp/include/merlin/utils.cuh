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

#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

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

__inline__ __device__ signed char atomicExch(signed char* address,
                                             signed char val) {
  signed char old = *address;
  *address = val;
  return old;
}

// TODO(jamesrong): this API will not confirm atomic, just for compiling
// successfully with framework in the TensorFlow ecosystem.
#ifdef GOOGLE_CUDA
__inline__ __device__ Eigen::half atomicExch(Eigen::half* address,
                                             Eigen::half val) {
  Eigen::half old = *address;
  *address = val;
  return old;
}
#endif

__inline__ __device__ int64_t atomicAdd(int64_t* address, const int64_t val) {
  return (int64_t)atomicAdd((unsigned long long*)address, val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t* address,
                                         const uint64_t val) {
  return (uint64_t)atomicAdd((unsigned long long*)address, val);
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

#define CUDA_CHECK(val) \
  { nv::merlin::cuda_check_((val), __FILE__, __LINE__); }

inline void __cudaCheckError(const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}
#define CudaCheckError() nv::merlin::__cudaCheckError(__FILE__, __LINE__)

inline void merlin_check_(bool cond, const std::string& msg, const char* file,
                          int line) {
  if (!cond) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": Merlin-KV error " + msg);
  }
}

#define MERLIN_CHECK(cond, msg) \
  { nv::merlin::merlin_check_((cond), (msg), __FILE__, __LINE__); }

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__inline__ __device__ uint64_t Murmur3HashDevice(uint64_t const& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__inline__ __device__ int64_t Murmur3HashDevice(int64_t const& key) {
  uint64_t k = uint64_t(key);
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return int64_t(k);
}

__inline__ __device__ uint32_t Murmur3HashDevice(uint32_t const& key) {
  uint32_t k = key;
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;

  return k;
}

__inline__ __device__ int32_t Murmur3HashDevice(int32_t const& key) {
  uint32_t k = uint32_t(key);
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;

  return int32_t(k);
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

template <typename TOUT, typename TIN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<__half, float> {
  static __forceinline__ __device__ __half convert(float val) {
    return __float2half(val);
  }
};

template <>
struct TypeConvertFunc<float, __half> {
  static __forceinline__ __device__ float convert(__half val) {
    return __half2float(val);
  }
};

template <>
struct TypeConvertFunc<float, float> {
  static __forceinline__ __device__ float convert(float val) { return val; }
};

template <>
struct TypeConvertFunc<float, long long> {
  static __forceinline__ __device__ float convert(long long val) {
    return static_cast<float>(val);
  }
};

template <>
struct TypeConvertFunc<float, unsigned int> {
  static __forceinline__ __device__ float convert(unsigned int val) {
    return static_cast<float>(val);
  }
};

template <>
struct TypeConvertFunc<int, long long> {
  static __forceinline__ __device__ int convert(long long val) {
    return static_cast<int>(val);
  }
};

template <>
struct TypeConvertFunc<int, unsigned int> {
  static __forceinline__ __device__ int convert(unsigned int val) {
    return static_cast<int>(val);
  }
};

template <class P>
void realloc(P* ptr, size_t old_size, size_t new_size) {
  void* new_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&new_ptr, new_size));
  CUDA_CHECK(cudaMemset(new_ptr, 0, new_size));
  if (*ptr != nullptr && old_size != 0) {
    CUDA_CHECK(cudaMemcpy(new_ptr, *ptr, old_size, cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(*ptr));
  }
  *ptr = (P)new_ptr;
  return;
}

template <class P>
void realloc_managed(P* ptr, size_t old_size, size_t new_size) {
  void* new_ptr = nullptr;

  CUDA_CHECK(cudaMallocManaged(&new_ptr, new_size));
  CUDA_CHECK(cudaMemset(new_ptr, 0, new_size));
  if (*ptr != nullptr && old_size != 0) {
    CUDA_CHECK(cudaMemcpy(new_ptr, *ptr, old_size, cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(*ptr));
  }
  *ptr = (P)new_ptr;
  return;
}

}  // namespace merlin
}  // namespace nv
