/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <stdio.h>
#include <array>
#include <map>
#include <unordered_map>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;

void test_lock_and_unlock() {
  TableOptions opt;

  // table setting
  const size_t U = 4 * 1024 * 1024UL;
  const size_t M = 65536UL;
  opt.max_capacity = U;
  opt.init_capacity = U;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.num_of_buckets_per_alloc = 8;

  using Table =
      nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kCustomized>;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  bool *d_found, *d_lock_results;
  i64** lock_keys_ptr;
  CUDA_CHECK(cudaMalloc(&d_found, M * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_lock_results, M * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&lock_keys_ptr, M * sizeof(i64*)));

  // step1
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  // step2
  test_util::KVMSBuffer<i64, f32, u64> buffer;
  buffer.Reserve(M, dim, stream);

  i64 start = 0;
  for (int i = 0; i < U / M; i++) {
    buffer.ToRange(start, 1, stream);
    start += M;
    buffer.Setscore((u64)i, stream);
    table->insert_or_assign(M, buffer.keys_ptr(), buffer.values_ptr(),
                            buffer.scores_ptr(), stream);

    CUDA_CHECK(cudaMemsetAsync(d_found, 0, M * sizeof(bool), stream));
    CUDA_CHECK(cudaMemsetAsync(d_lock_results, 0, M * sizeof(bool), stream));
    table->contains(M, buffer.keys_ptr(), d_found, stream);
    table->lock_keys(M, buffer.keys_ptr(), lock_keys_ptr, d_lock_results,
                     stream);
    bool result = test_util::allEqualGpu(d_found, d_lock_results, M, stream);
    ASSERT_EQ(result, true);
    result = test_util::allTrueGpu(d_found, M, stream);
    ASSERT_EQ(result, true);

    CUDA_CHECK(cudaMemsetAsync(d_found, 0, M * sizeof(bool), stream));
    CUDA_CHECK(cudaMemsetAsync(d_lock_results, 0, M * sizeof(bool), stream));
    table->contains(M, buffer.keys_ptr(), d_found, stream);
    result = test_util::allEqualGpu(d_found, d_lock_results, M, stream);
    ASSERT_EQ(result, true);

    CUDA_CHECK(cudaMemsetAsync(d_found, 0, M * sizeof(bool), stream));
    table->unlock_keys(M, lock_keys_ptr, buffer.keys_ptr(), d_lock_results,
                       stream);
    table->contains(M, buffer.keys_ptr(), d_found, stream);
    result = test_util::allEqualGpu(d_found, d_lock_results, M, stream);
    ASSERT_EQ(result, true);
    result = test_util::allTrueGpu(d_found, M, stream);
    ASSERT_EQ(result, true);
  }

  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_lock_results));
  CUDA_CHECK(cudaFree(lock_keys_ptr));
}

TEST(LockAndUnlockTest, test_lock_and_unlock) { test_lock_and_unlock(); }