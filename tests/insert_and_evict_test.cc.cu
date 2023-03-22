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

#include <gtest/gtest.h>
#include <stdio.h>
#include <array>
#include <map>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"

constexpr size_t dim = 4;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using Table = nv::merlin::HashTable<i64, f32, u64>;
using TableOptions = nv::merlin::HashTableOptions;

/*
 * There are several steps to check whether if
 * the insert_and_evict API is safe to use:
 *
 *   step1: Create a table with max_capacity U
 *   step2: Insert M keys into table while M < U. And
 *     the table size became m <= M. M - m keys was
 *     evicted.
 *   step3: Insert N keys into table while m + N > U, with
 *     no same key with M keys. And p keys gets evicted.
 *     If now the table size is v. Then total number of
 *     keys T = v + p + M - m, must equal to VT = M + N,
 *     while the keys, values, and metas match.
 *   step4: export table and check all values.
 */
void test_insert_and_evict() {
  TableOptions opt;

  // table setting
  const size_t init_capacity = 1024;

  // numeric setting
  const size_t U = 2llu << 18;
  const size_t M = (U >> 1);
  const size_t N = (U >> 1) + 17;  // Add a prime to test the non-aligned case.

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  opt.dim = dim;

  std::map<i64, test_util::ValueArray<f32, dim>> summarized_kvs;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // step1
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  // step2
  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(M, dim, stream);
  evict_buffer.ToZeros(stream);

  test_util::KVMSBuffer<i64, f32, u64> buffer;
  buffer.Reserve(M, dim, stream);
  buffer.ToRange(0, 1, stream);
  buffer.SetMeta((u64)1, stream);
  size_t n_evicted = table->insert_and_evict(
      M, buffer.keys_ptr(), buffer.values_ptr(), buffer.metas_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.metas_ptr(), stream);
  size_t table_size_m = table->size(stream);
  buffer.SyncData(/*h2d=*/false, stream);
  evict_buffer.SyncData(/*h2d=*/false, stream);
  ASSERT_EQ(n_evicted + table_size_m, M);
  for (size_t i = 0; i < n_evicted; i++) {
    test_util::ValueArray<f32, dim>* vec =
        reinterpret_cast<test_util::ValueArray<f32, dim>*>(
            evict_buffer.values_ptr(false) + i * dim);
    summarized_kvs.emplace(evict_buffer.keys_ptr(false)[i], *vec);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //  step3
  evict_buffer.Reserve(N, dim, stream);
  buffer.Reserve(N, dim, stream);
  buffer.ToRange(M, 1, stream);
  buffer.SetMeta((u64)2, stream);
  n_evicted = table->insert_and_evict(
      N, buffer.keys_ptr(), buffer.values_ptr(), buffer.metas_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.metas_ptr(), stream);
  size_t table_size_n = table->size(stream);
  buffer.SyncData(/*h2d=*/false, stream);
  evict_buffer.SyncData(/*h2d=*/false, stream);
  ASSERT_EQ(table_size_m + N, table_size_n + n_evicted);
  for (size_t i = 0; i < n_evicted; i++) {
    test_util::ValueArray<f32, dim>* vec =
        reinterpret_cast<test_util::ValueArray<f32, dim>*>(
            evict_buffer.values_ptr(false) + i * dim);
    summarized_kvs.emplace(evict_buffer.keys_ptr(false)[i], *vec);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // step4
  buffer.Reserve(table_size_n, dim, stream);
  size_t n_exported =
      table->export_batch(table->capacity(), 0, buffer.keys_ptr(),
                          buffer.values_ptr(), buffer.metas_ptr(), stream);
  ASSERT_EQ(table_size_n, n_exported);
  buffer.SyncData(/*h2d=*/false, stream);
  for (size_t i = 0; i < n_exported; i++) {
    test_util::ValueArray<f32, dim>* vec =
        reinterpret_cast<test_util::ValueArray<f32, dim>*>(
            buffer.values_ptr(false) + i * dim);
    summarized_kvs.emplace(buffer.keys_ptr(false)[i], *vec);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  buffer.Free(stream);
  evict_buffer.Free(stream);

  size_t k = 0;
  for (auto it = summarized_kvs.begin(); it != summarized_kvs.end(); it++) {
    i64 key = it->first;
    test_util::ValueArray<f32, dim>& value = it->second;
    ASSERT_EQ(key, (i64)k);
    for (size_t j = 0; j < dim; j++) {
      ASSERT_EQ(value[j], (f32)k);
    }
    ++k;
  }
  ASSERT_EQ(summarized_kvs.size(), M + N);
  summarized_kvs.clear();
}

TEST(MerlinHashTableTest, test_insert_and_evict) { test_insert_and_evict(); }
