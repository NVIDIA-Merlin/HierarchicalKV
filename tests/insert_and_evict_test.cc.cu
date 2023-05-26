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
 *     while the keys, values, and scores match.
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
  buffer.Setscore((u64)1, stream);
  size_t n_evicted = table->insert_and_evict(
      M, buffer.keys_ptr(), buffer.values_ptr(), buffer.scores_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.scores_ptr(), stream);
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
  buffer.Setscore((u64)2, stream);
  n_evicted = table->insert_and_evict(
      N, buffer.keys_ptr(), buffer.values_ptr(), buffer.scores_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.scores_ptr(), stream);
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
                          buffer.values_ptr(), buffer.scores_ptr(), stream);
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

template <typename K, typename V, typename S>
void CheckInsertAndEvict(Table* table, K* keys, V* values, S* scores,
                         K* evicted_keys, V* evicted_values, S* evicted_scores,
                         size_t len, cudaStream_t stream) {
  std::map<i64, test_util::ValueArray<f32, dim>> map_before_insert;
  std::map<i64, test_util::ValueArray<f32, dim>> map_after_insert;
  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));

  size_t table_size_verify0 = table->export_batch(
      table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                             table_size_before * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                             table_size_before * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                             table_size_before * sizeof(S),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_before, keys,
                             len * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_before * dim, values,
                             len * dim * sizeof(V), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores + table_size_before, scores,
                             len * sizeof(S), cudaMemcpyDeviceToHost, stream));
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
                              evicted_values, evicted_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  float dur = diff.count();

  size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = table->export_batch(
      table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);

  ASSERT_EQ(table_size_verify1, table_size_after);

  size_t new_cap = table_size_after + filtered_len;
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                             table_size_after * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                             table_size_after * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                             table_size_after * sizeof(S),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_after, evicted_keys,
                             filtered_len * sizeof(K), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_after * dim,
                             evicted_values, filtered_len * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores + table_size_after, evicted_scores,
                             filtered_len * sizeof(S), cudaMemcpyDeviceToHost,
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
  std::cout << "Check insert_and_evict behavior got "
            << "key_miss_cnt: " << key_miss_cnt
            << ", value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);

  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_scores, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_scores);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_insert_and_evict_advanced() {
  const size_t U = 524288;
  const size_t init_capacity = U;
  const size_t B = 524288 + 13;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
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
  u64 score = 0;
  for (int i = 0; i < 8; i++) {
    test_util::create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvict<i64, f32, u64>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, stream);

    offset += B;
    score += 1;
  }
}

void test_insert_and_evict_with_export_batch() {
  size_t max_capacity = 4096;
  size_t init_capacity = 2048;
  size_t offset = 0;
  size_t uplimit = 1048576;
  size_t len = 4096 + 13;

  TableOptions opt;
  opt.max_capacity = max_capacity;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = uplimit * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kLru;
  opt.dim = dim;

  using Vec_t = test_util::ValueArray<f32, dim>;
  std::map<i64, Vec_t> ref_map;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> buffer;
  buffer.Reserve(len, dim, stream);
  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(len, dim, stream);

  size_t total_len = 0;
  buffer.ToRange(offset, /*skip=1*/ 1, stream);
  size_t n_evicted = table->insert_and_evict(
      len, buffer.keys_ptr(), buffer.values_ptr(), nullptr,
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(), nullptr, stream);
  printf("Insert %zu keys and evict %zu\n", len, n_evicted);
  offset += len;
  total_len += len;
  evict_buffer.SyncData(/*h2d=*/false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < n_evicted; i++) {
    Vec_t* vec =
        reinterpret_cast<Vec_t*>(evict_buffer.values_ptr(false) + i * dim);
    ref_map[evict_buffer.keys_ptr(false)[i]] = *vec;
  }

  offset = 0;
  for (; offset < table->capacity(); offset += len) {
    size_t search_len = len;
    if (offset + search_len > table->capacity()) {
      search_len = table->capacity() - offset;
    }
    size_t n_exported =
        table->export_batch(search_len, offset, buffer.keys_ptr(),
                            buffer.values_ptr(), /*scores=*/nullptr, stream);
    buffer.SyncData(/*h2d=*/false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < n_exported; i++) {
      Vec_t* vec = reinterpret_cast<Vec_t*>(buffer.values_ptr(false) + i * dim);
      for (size_t j = 0; j < dim; j++) {
        ASSERT_EQ(buffer.keys_ptr(false)[i], vec->operator[](j));
      }
      ref_map[buffer.keys_ptr(false)[i]] = *vec;
    }
  }

  for (auto& it : ref_map) {
    for (size_t j = 0; j < dim; j++) {
      ASSERT_EQ(static_cast<f32>(it.first), it.second.data[j]);
    }
  }
}

template <typename K, typename V, typename S>
void BatchCheckInsertAndEvict(Table* table, K* keys, V* values, S* scores,
                              K* evicted_keys, V* evicted_values,
                              S* evicted_scores, size_t len,
                              std::atomic<int>* step, size_t total_step,
                              cudaStream_t stream, bool if_check = true) {
  std::map<i64, test_util::ValueArray<f32, dim>> map_before_insert;
  std::map<i64, test_util::ValueArray<f32, dim>> map_after_insert;

  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;

  while (step->load() < total_step) {
    size_t table_size_before = table->size(stream);
    size_t cap = table_size_before + len;
    size_t key_miss_cnt = 0;
    size_t value_diff_cnt = 0;
    size_t table_size_after = 0;
    size_t table_size_verify1 = 0;

    int s = step->load();

    if (if_check) {
      CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
      CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
      CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      h_tmp_keys = (K*)malloc(cap * sizeof(K));
      h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
      h_tmp_scores = (S*)malloc(cap * sizeof(S));

      CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
      CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));

      size_t table_size_verify0 = table->export_batch(
          table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);
      ASSERT_EQ(table_size_before, table_size_verify0);

      CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                                 table_size_before * sizeof(K),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                                 table_size_before * dim * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                                 table_size_before * sizeof(S),
                                 cudaMemcpyDeviceToHost, stream));

      CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_before, keys + len * s,
                                 len * sizeof(K), cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_before * dim,
                                 values + len * s * dim, len * dim * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores + table_size_before,
                                 scores + len * s, len * sizeof(S),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (size_t i = 0; i < cap; i++) {
        test_util::ValueArray<V, dim>* vec =
            reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                             i * dim);
        map_before_insert[h_tmp_keys[i]] = *vec;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto start = std::chrono::steady_clock::now();
    size_t filtered_len = table->insert_and_evict(
        len, keys + len * s, values + len * s * dim, nullptr, evicted_keys,
        evicted_values, evicted_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    float dur = diff.count();

    if (if_check) {
      table_size_after = table->size(stream);
      table_size_verify1 = table->export_batch(
          table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);

      ASSERT_EQ(table_size_verify1, table_size_after);

      size_t new_cap = table_size_after + filtered_len;
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                                 table_size_after * sizeof(K),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                                 table_size_after * dim * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                                 table_size_after * sizeof(S),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_after, evicted_keys,
                                 filtered_len * sizeof(K),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_after * dim,
                                 evicted_values, filtered_len * dim * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores + table_size_after,
                                 evicted_scores, filtered_len * sizeof(S),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      int64_t new_cap_i64 = (int64_t)new_cap;
      for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
        test_util::ValueArray<V, dim>* vec =
            reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                             i * dim);
        map_after_insert[h_tmp_keys[i]] = *vec;
      }

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
      CUDA_CHECK(cudaFreeAsync(d_tmp_scores, stream));
      free(h_tmp_keys);
      free(h_tmp_values);
      free(h_tmp_scores);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::cout << "Check insert behavior got step: " << step->load()
              << ",\tduration: " << dur
              << ",\twhile value_diff_cnt: " << value_diff_cnt
              << ", while table_size_before: " << table_size_before
              << ", while table_size_after: " << table_size_after
              << ", while len: " << len << std::endl;

    step->fetch_add(1);
  }
}

template <typename K, typename V, typename S>
void BatchCheckFind(Table* table, K* keys, V* values, S* scores, size_t len,
                    std::atomic<int>* step, size_t total_step,
                    size_t find_interval, cudaStream_t stream,
                    bool if_check = true) {
  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_founds = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_founds = nullptr;

  int find_step = 0;
  size_t cap = len * find_interval;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_founds, cap * sizeof(bool), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_founds = (bool*)malloc(cap * sizeof(bool));

  while (step->load() < total_step) {
    while (find_step >= (step->load() / find_interval)) continue;

    size_t found_num = 0;
    size_t value_diff_cnt = 0;

    CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
    CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
    CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
    CUDA_CHECK(cudaMemsetAsync(d_tmp_founds, 0, cap * sizeof(bool), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_tmp_keys, keys + cap * find_step,
                               cap * sizeof(K), cudaMemcpyDeviceToDevice,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto start = std::chrono::steady_clock::now();
    table->find(cap, d_tmp_keys, d_tmp_values, d_tmp_founds, d_tmp_scores,
                stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    float dur = diff.count();

    if (if_check) {
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys, cap * sizeof(K),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                                 cap * dim * sizeof(V), cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores, cap * sizeof(S),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, cap * sizeof(bool),
                                 cudaMemcpyDeviceToHost, stream));

      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (int i = 0; i < cap; i++) {
        if (h_tmp_founds[i]) {
          for (int j = 0; j < dim; j++) {
            if (h_tmp_values[i * dim + j] !=
                static_cast<float>(h_tmp_keys[i] * 0.00001)) {
              value_diff_cnt++;
            };
          }
          found_num++;
        }
      }
      ASSERT_EQ(value_diff_cnt, 0);
    }
    std::cout << std::endl
              << "\nCheck find behavior got step: " << find_step
              << ",\tduration: " << dur
              << ",\twhile value_diff_cnt: " << value_diff_cnt
              << ", while cap: " << cap << std::endl
              << std::endl;
    ASSERT_EQ(value_diff_cnt, 0);
    find_step++;
  }
  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_scores, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_founds, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_scores);
  free(h_tmp_founds);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_insert_and_evict_run_with_batch_find() {
  const size_t U = 16 * 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 256 * 1024;
  constexpr size_t batch_num = 256;
  constexpr size_t find_interval = 8;

  const bool if_check = false;

  std::thread insert_and_evict_thread;
  std::thread find_thread;
  std::atomic<int> step{0};

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kLru;
  opt.dim = dim;

  cudaStream_t insert_stream;
  cudaStream_t find_stream;
  CUDA_CHECK(cudaStreamCreate(&insert_stream));
  CUDA_CHECK(cudaStreamCreate(&find_stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> global_buffer;
  global_buffer.Reserve(B * batch_num, dim, insert_stream);

  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(B, dim, insert_stream);
  evict_buffer.ToZeros(insert_stream);

  for (int i = 0; i < batch_num; i++) {
    test_util::create_random_keys<i64, u64, f32, dim>(
        global_buffer.keys_ptr(false) + B * i,
        global_buffer.scores_ptr(false) + B * i,
        global_buffer.values_ptr(false) + B * i * dim, (int)B);
  }
  global_buffer.SyncData(true, insert_stream);
  CUDA_CHECK(cudaStreamSynchronize(insert_stream));

  auto insert_and_evict_func = [&table, &global_buffer, &evict_buffer, &B,
                                &step, &batch_num, &insert_stream]() {
    BatchCheckInsertAndEvict<i64, f32, u64>(
        table.get(), global_buffer.keys_ptr(), global_buffer.values_ptr(),
        global_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, &step,
        batch_num, insert_stream, if_check);
  };

  auto find_func = [&table, &global_buffer, &B, &step, &batch_num,
                    &find_interval, &find_stream]() {
    BatchCheckFind<i64, f32, u64>(
        table.get(), global_buffer.keys_ptr(), global_buffer.values_ptr(),
        global_buffer.scores_ptr(), B, &step, batch_num, find_interval,
        find_stream, if_check);
  };

  find_thread = std::thread(find_func);
  insert_and_evict_thread = std::thread(insert_and_evict_func);
  find_thread.join();
  insert_and_evict_thread.join();
  CUDA_CHECK(cudaStreamDestroy(insert_stream));
  CUDA_CHECK(cudaStreamDestroy(find_stream));
}

TEST(InsertAndEvictTest, test_insert_and_evict_basic) {
  test_insert_and_evict_basic();
}

TEST(InsertAndEvictTest, test_insert_and_evict_advanced) {
  test_insert_and_evict_advanced();
}

TEST(InsertAndEvictTest, test_insert_and_evict_with_export_batch) {
  test_insert_and_evict_with_export_batch();
}

TEST(InsertAndEvictTest, test_insert_and_evict_run_with_batch_find) {
  test_insert_and_evict_run_with_batch_find();
}
