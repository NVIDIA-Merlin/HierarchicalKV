/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/scan.h"
#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {

template <typename K>
__global__ void keys_not_empty(const K* keys, bool* masks, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    masks[tid] = keys[tid] != EMPTY_KEY;
  }
}

template <typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_cell_count(const bool* masks, Tidx* offsets, size_t n,
                               size_t* n_existed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  int g_ones = __popc((int)vote);
  if (rank == 0 && tid < n) {
    offsets[tid / TILE_SIZE] = static_cast<Tidx>(g_ones);
    atomicAdd(static_cast<uint64_t*>(n_existed), static_cast<uint64_t>(g_ones));
  }
}

template <typename K, typename V, typename M, typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_select_kvm_kernel(const bool* masks, size_t n,
                                      const Tidx* offsets, K* __restrict keys,
                                      V* __restrict values, M* __restrict metas,
                                      const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);
  K empty_key = (K)EMPTY_KEY;

  if (tid < n) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    if (masks[tid]) {
      int prefix_n = __popc(r_vote) - 1;
      Tidx bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);

      if (bias == tid) return;

      K target_key = 0;
      while (target_key != empty_key) {
        target_key = atomicCAS(keys + bias, empty_key, keys[tid]);
      }
      if (metas) metas[bias] = metas[tid];
      for (size_t j = 0; j < dim; j++) {
        values[dim * bias + j] = values[dim * tid + j];
      }
      atomicExch(keys + tid, empty_key);
    }
  }
}

template <typename K, typename V, typename M, typename Tidx, int TILE_SIZE = 8>
void gpu_boolean_mask(size_t grid_size, size_t block_size, const bool* masks,
                      size_t n, size_t* n_evicted, Tidx* offsets,
                      K* __restrict keys, V* __restrict values,
                      M* __restrict metas, size_t dim, cudaStream_t stream) {
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  gpu_cell_count<Tidx, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, offsets, n, n_evicted);
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<Tidx> d_src(offsets);
  thrust::device_ptr<Tidx> d_dest(offsets);
  thrust::exclusive_scan(policy, d_src, d_src + n_offsets, d_dest);
  gpu_select_kvm_kernel<K, V, M, Tidx, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, n, offsets, keys, values,
                                             metas, dim);
}

}  // namespace merlin
}  // namespace nv
