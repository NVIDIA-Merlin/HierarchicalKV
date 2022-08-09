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

#include "merlin/embedding_kernels.cuh"
#include "merlin/optimizers.cuh"
#include "merlin_hashtable.cuh"

namespace nv {
namespace merlin {
namespace optimizers {

template <class K, class V, class M, class T, size_t DIM>
class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void update(const K* d_keys, const V* d_grad, int len,
                      cudaStream_t stream) {}
};

template <class K, class V, class M, class T, size_t DIM>
class Adam : public Optimizer<K, V, M, T, DIM> {
 public:
  using Table = nv::merlin::Table<K, V, M, DIM>;

 public:
  Adam(Table* weights, float alpha, float beta1, float beta2, float epsilon,
       float scaler)
      : w_(weights),
        alpha_(alpha),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        scaler_(scaler) {
    create_slot(weights, &m_);
    create_slot(weights, &v_);
  }
  ~Adam() {}

  void update(const K* d_keys, const V* d_grad, int len,
              cudaStream_t stream) override {
    if (len == 0) return;
    V* d_w;
    V* d_m;
    V* d_v;

    // TODO(jamesrong): change to flexible buffer.
    CUDA_CHECK(cudaMallocAsync((void**)&d_w, len * sizeof(V), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_m, len * sizeof(V), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_v, len * sizeof(V), stream));

    w_->get(d_keys, d_w, len, stream);
    m_->get(d_keys, d_m, len, stream);
    v_->get(d_keys, d_v, len, stream);

    const int N = len * DIM;
    const int grid_size = (N - 1) / block_size_ + 1;
    adam_update_kernel<T><<<grid_size, block_size_, 0, stream>>>(
        N, d_w, d_m, d_v, d_grad, alpha_, beta1_, beta2_, epsilon_, scaler_);

    w_->insert_or_assign(d_keys, d_w, len, stream);
    m_->insert_or_assign(d_keys, d_m, len, stream);
    v_->insert_or_assign(d_keys, d_v, len, stream);

    CUDA_CHECK(cudaFreeAsync(d_w, stream))
    CUDA_CHECK(cudaFreeAsync(d_m, stream));
    CUDA_CHECK(cudaFreeAsync(d_v, stream));
    CudaCheckError();
  }

 private:
  Table* w_;
  Table* m_;
  Table* v_;
  float alpha_;
  float beta1_;
  float beta2_;
  float epsilon_;
  float scaler_;
  int block_size_ = 1024;
};

}  // namespace optimizers
}  // namespace merlin
}  // namespace nv