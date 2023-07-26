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

#include <stdlib.h>
#include "debug.hpp"
#include "utils.cuh"

namespace nv {
namespace merlin {

enum MemoryType {
  Device,   // HBM
  Pinned,   // Pinned Host Memory
  Host,     // Host Memory
  Managed,  // Pageable Host Memory(Not required)
};

/* This abstract class defines the allocator APIs required by HKV.
   Any of the customized allocators should inherit from it.
 */
class BaseAllocator {
 public:
  BaseAllocator(const BaseAllocator&) = delete;
  BaseAllocator(BaseAllocator&&) = delete;

  BaseAllocator& operator=(const BaseAllocator&) = delete;
  BaseAllocator& operator=(BaseAllocator&&) = delete;

  BaseAllocator() = default;
  virtual ~BaseAllocator() = default;

  virtual void alloc(const MemoryType type, void** ptr, size_t size,
                     unsigned int pinned_flags = cudaHostAllocDefault) = 0;

  virtual void alloc_async(const MemoryType type, void** ptr, size_t size,
                           cudaStream_t stream) = 0;

  virtual void free(const MemoryType type, void* ptr) = 0;

  virtual void free_async(const MemoryType type, void* ptr,
                          cudaStream_t stream) = 0;
};

class DefaultAllocator : public virtual BaseAllocator {
 public:
  DefaultAllocator(){};
  ~DefaultAllocator() override{};

  void alloc(const MemoryType type, void** ptr, size_t size,
             unsigned int pinned_flags = cudaHostAllocDefault) override {
    switch (type) {
      case MemoryType::Device:
        CUDA_CHECK(cudaMalloc(ptr, size));
        break;
      case MemoryType::Pinned:
        CUDA_CHECK(cudaMallocHost(ptr, size, pinned_flags));
        break;
      case MemoryType::Host:
        *ptr = std::malloc(size);
        break;
    }
    return;
  }

  void alloc_async(const MemoryType type, void** ptr, size_t size,
                   cudaStream_t stream) override {
    if (type == MemoryType::Device) {
      CUDA_CHECK(cudaMallocAsync(ptr, size, stream));
    } else {
      MERLIN_CHECK(false,
                   "[DefaultAllocator] alloc_async is only support for "
                   "MemoryType::Device!");
    }
    return;
  }

  void free(const MemoryType type, void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    switch (type) {
      case MemoryType::Pinned:
        CUDA_CHECK(cudaFreeHost(ptr));
        break;
      case MemoryType::Device:
        CUDA_CHECK(cudaFree(ptr));
        break;
      case MemoryType::Host:
        std::free(ptr);
        break;
    }
    return;
  }

  void free_async(const MemoryType type, void* ptr,
                  cudaStream_t stream) override {
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
