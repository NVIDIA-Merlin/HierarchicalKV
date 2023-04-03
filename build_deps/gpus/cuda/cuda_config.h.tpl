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

#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_

#define CUDA_VERSION "%{cuda_version}"
#define CUDART_VERSION "%{cudart_version}"
#define CUPTI_VERSION "%{cupti_version}"
#define CUBLAS_VERSION "%{cublas_version}"
#define CUSOLVER_VERSION "%{cusolver_version}"
#define CURAND_VERSION "%{curand_version}"
#define CUFFT_VERSION "%{cufft_version}"
#define CUSPARSE_VERSION "%{cusparse_version}"
#define CUDNN_VERSION "%{cudnn_version}"

#define CUDA_TOOLKIT_PATH "%{cuda_toolkit_path}"

#define CUDA_COMPUTE_CAPABILITIES %{cuda_compute_capabilities}

#endif  // CUDA_CUDA_CONFIG_H_
