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

#include "debug.hpp"
#include "utils.cuh"

namespace nv {
namespace merlin {

enum MemoryType { Device, Managed, Pinned };

class DefaultAllocator {
 public:
  DefaultAllocator(const DefaultAllocator&) = delete;
  DefaultAllocator(DefaultAllocator&&) = delete;

  DefaultAllocator& operator=(const DefaultAllocator&) = delete;
  DefaultAllocator& operator=(DefaultAllocator&&) = delete;

  DefaultAllocator(){};

  void alloc(const MemoryType type, void** ptr, size_t size,
             unsigned int flags = cudaHostAllocDefault) {
    switch (type) {
      case MemoryType::Device:
        CUDA_CHECK(cudaMalloc(ptr, size));
        break;
      case MemoryType::Managed:
        CUDA_CHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
        break;
      case MemoryType::Pinned:
        CUDA_CHECK(cudaMallocHost(ptr, size, flags));
        break;
    }
    return;
  }

  void alloc_async(const MemoryType type, void** ptr, size_t size,
                   cudaStream_t stream) {
    switch (type) {
      case MemoryType::Device:
        CUDA_CHECK(cudaMallocAsync(ptr, size, stream));
        break;
      case MemoryType::Managed:
      case MemoryType::Pinned:
        MERLIN_CHECK(false,
                     "[DefaultAllocator] alloc_async is only support for "
                     "MemoryType::Device!");
        break;
    }
    return;
  }

  void free(const MemoryType type, void* ptr) {
    if (ptr == nullptr) {
      return;
    }

    if (type == MemoryType::Pinned) {
      CUDA_CHECK(cudaFreeHost(ptr));
    } else {
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  void free_async(const MemoryType type, void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
      return;
    }

    if (type == MemoryType::Device) {
      CUDA_CHECK(cudaFreeAsync(ptr, stream));
    } else {
      MERLIN_CHECK(false,
                   "[DefaultAllocator] free_async is only support for "
                   "MemoryType::Device!");
    }
  }
};

}  // namespace merlin
}  // namespace nv
