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

#include "types.cuh"
#include "util.cuh"

namespace nv {
namespace merlin {
namespace embedding {

/* template <typename T>
__global__ __launch_bounds__(1024) void ApplyAdamKernel(
    int32 data_dim, T* var, T* m, T* v, const T* const beta1_power_,
    const T* const beta2_power_, const T* const lr_, const T* const beta1_,
    const T* const beta2_, const T* const epsilon_, const T* grad,
    bool use_nesterov) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  const T mul_factor =
      (*lr_) * Eigen::numext::sqrt(static_cast<T>(1.0) - (*beta2_power_)) /
      (static_cast<T>(1.0) - (*beta1_power_));
  const T epsilon = (*epsilon_);
  const T beta1 = (*beta1_);
  const T one_minus_beta1 = static_cast<T>(1.0) - (beta1);
  const T one_minus_beta2 = static_cast<T>(1.0) - (*beta2_);
  const int32 stripe = gridDim.x * blockDim.x;

  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < data_dim;
       i += stripe) {
    auto m_i = m[i];
    auto g_i = grad[i];
    auto v_i = v[i];

    // Avoid += and -= due to std::complex<T> issues on device for MSVC.
    m_i = m_i + one_minus_beta1 * (g_i - m_i);
    v_i = v_i + one_minus_beta2 * (g_i * g_i - v_i);
    if (use_nesterov) {
      var[i] = var[i] - mul_factor * (m_i * beta1 + one_minus_beta1 * g_i) /
                            (epsilon + Eigen::numext::sqrt(v_i));
    } else {
      var[i] = var[i] - mul_factor * m_i / (epsilon + Eigen::numext::sqrt(v_i));
    }

    m[i] = m_i;
    v[i] = v_i;
  }
} */

}  // namespace embedding
}  // namespace merlin
}  // namespace nv