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

#include "merlin/initializers.cuh"
#include "merlin_hashtable.cuh"

namespace nv {
namespace merlin {
namespace initializers {

template <class T>
class RandomNormal final : public Initializer<T> {
 public:
  RandomNormal(T mean = 0.0, T stddev = 0.5, unsigned long long seed = 2022ULL)
      : mean_(mean), stddev_(stddev), seed_(seed) {}
  ~RandomNormal() {}

  void initialize(T* data, size_t len, cudaStream_t stream) override {
    random_normal<T>(data, len, stream, mean_, stddev_, seed_);
    CudaCheckError();
  }

 private:
  T mean_;
  T stddev_;
  unsigned long long seed_;
};

}  // namespace initializers
}  // namespace merlin
}  // namespace nv