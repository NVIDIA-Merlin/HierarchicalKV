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

constexpr size_t dim = 64;
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
void test_insert_and_evict_basic() {
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

template <typename K, typename V, typename M>
void CheckInsertAndEvict(Table* table, K* keys, V* values, M* metas,
                         K* evicted_keys, V* evicted_values, M* evicted_metas,
                         size_t len, cudaStream_t stream) {
  std::map<i64, test_util::ValueArray<f32, dim>> map_before_insert;
  std::map<i64, test_util::ValueArray<f32, dim>> map_after_insert;
  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  M* h_tmp_metas = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  M* d_tmp_metas = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_metas, cap * sizeof(M), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_metas, 0, cap * sizeof(M), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_metas = (M*)malloc(cap * sizeof(M));

  size_t table_size_verify0 = table->export_batch(
      table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_metas, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                             table_size_before * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                             table_size_before * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas, d_tmp_metas,
                             table_size_before * sizeof(M),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_before, keys,
                             len * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_before * dim, values,
                             len * dim * sizeof(V), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas + table_size_before, metas,
                             len * sizeof(M), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < cap; i++) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    map_before_insert[h_tmp_keys[i]] = *vec;
  }

  auto start = std::chrono::steady_clock::now();
  size_t filtered_len =
      table->insert_and_evict(len, keys, values, nullptr, evicted_keys,
                              evicted_values, evicted_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  float dur = diff.count();

  size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = table->export_batch(
      table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_metas, stream);

  ASSERT_EQ(table_size_verify1, table_size_after);

  size_t new_cap = table_size_after + filtered_len;
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                             table_size_after * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                             table_size_after * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas, d_tmp_metas,
                             table_size_after * sizeof(M),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_after, evicted_keys,
                             filtered_len * sizeof(K), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_after * dim,
                             evicted_values, filtered_len * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas + table_size_after, evicted_metas,
                             filtered_len * sizeof(M), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int64_t new_cap_i64 = (int64_t)new_cap;
  for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    map_after_insert[h_tmp_keys[i]] = *vec;
  }

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  for (auto& it : map_before_insert) {
    if (map_after_insert.find(it.first) == map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<V, dim>& vec0 = it.second;
    test_util::ValueArray<V, dim>& vec1 = map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; j++) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);

  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_metas, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_metas);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_insert_and_evict_advanced() {
  const size_t U = 524288;
  const size_t init_capacity = 1024;
  const size_t B = 524288 + 13;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kLru;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(B, dim, stream);
  evict_buffer.ToZeros(stream);

  test_util::KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.Reserve(B, dim, stream);

  size_t offset = 0;
  u64 meta = 0;
  for (int i = 0; i < 20; i++) {
    test_util::create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.metas_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvict<i64, f32, u64>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.metas_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.metas_ptr(), B, stream);

    offset += B;
    meta += 1;
  }
}

TEST(MerlinHashTableTest, test_insert_and_evict_basic) {
  test_insert_and_evict_basic();
}
TEST(MerlinHashTableTest, test_insert_and_evict_advanced) {
  test_insert_and_evict_advanced();
}
