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

#include "utils.cuh"

namespace nv {
namespace merlin {

template <class T>
class FlexPinnedBuffer {
 public:
  FlexPinnedBuffer(size_t size = 1) : ptr_(nullptr) {
    if (!ptr_) {
      size_ = size;
      CUDA_CHECK(cudaMallocHost(&ptr_, sizeof(T) * size_));
    }
  }
  ~FlexPinnedBuffer() {
    if (!ptr_) CUDA_CHECK(cudaFreeHost(ptr_));
  }

  __inline__ T* alloc_or_reuse(size_t size = 0) {
    if (size > size_) {
      CUDA_CHECK(cudaFreeHost(ptr_));
      size_ = size;
      CUDA_CHECK(cudaMallocHost(&ptr_, sizeof(T) * size_));
    }
    return ptr_;
  }

 private:
  T* ptr_;
  size_t size_;
};

}  // namespace merlin
}  // namespace nv