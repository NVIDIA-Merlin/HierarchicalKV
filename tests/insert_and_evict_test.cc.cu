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
  using Table =
      nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kCustomized>;
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

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvict(Table* table, K* keys, V* values, S* scores,
                         K* evicted_keys, V* evicted_values, S* evicted_scores,
                         size_t len, cudaStream_t stream, TableOptions& opt) {
  std::map<i64, test_util::ValueArray<f32, dim>> map_before_insert;
  std::map<i64, test_util::ValueArray<f32, dim>> map_after_insert;
  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_founds = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_founds = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_founds, cap * sizeof(bool), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_founds, 0, cap * sizeof(bool), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_founds = (bool*)malloc(cap * sizeof(bool));

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
  size_t filtered_len = table->insert_and_evict(
      len, keys, values,
      Table::evict_strategy == EvictStrategy::kLru ? nullptr : scores,
      evicted_keys, evicted_values, evicted_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  {
    table->find(len, keys, values, d_tmp_founds, scores, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t found_counter = 0;
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) found_counter++;
    }
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;

    CUDA_CHECK(cudaMemset(d_tmp_founds, 0, len * sizeof(bool)));
    table->contains(len, keys, d_tmp_founds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int contains_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) contains_counter++;
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

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
  CUDA_CHECK(cudaFreeAsync(d_tmp_founds, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_scores);
  free(h_tmp_founds);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_insert_and_evict_advanced_on_lru() {
  const size_t U = 524288;
  const size_t init_capacity = U;
  const size_t B = 524288 + 13;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kLru>;
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
  for (int i = 0; i < 16; i++) {
    test_util::create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvict<i64, f32, u64, Table>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, stream, opt);

    offset += B;
    score += 1;
  }
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnLfu(Table* table,
                              test_util::KVMSBuffer<K, V, S>* data_buffer,
                              test_util::KVMSBuffer<K, V, S>* evict_buffer,
                              size_t len, cudaStream_t stream,
                              TableOptions& opt, unsigned int global_epoch) {
  std::map<K, test_util::ValueArray<V, dim>> values_map_before_insert;
  std::map<K, test_util::ValueArray<V, dim>> values_map_after_insert;

  std::unordered_map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;

  std::map<K, S> scores_map_current_batch;
  std::map<K, S> scores_map_current_evict;

  K* keys = data_buffer->keys_ptr();
  V* values = data_buffer->values_ptr();
  S* scores = data_buffer->scores_ptr();

  K* evicted_keys = evict_buffer->keys_ptr();
  V* evicted_values = evict_buffer->values_ptr();
  S* evicted_scores = evict_buffer->scores_ptr();

  for (size_t i = 0; i < len; i++) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_founds = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_founds = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_founds, cap * sizeof(bool), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_founds, 0, cap * sizeof(bool), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_founds = (bool*)malloc(cap * sizeof(bool));

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
    values_map_before_insert[h_tmp_keys[i]] = *vec;
  }

  for (size_t i = 0; i < table_size_before; i++) {
    scores_map_before_insert[h_tmp_keys[i]] = h_tmp_scores[i];
  }

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  size_t filtered_len = table->insert_and_evict(
      len, keys, values,
      (Table::evict_strategy == EvictStrategy::kLru ||
       Table::evict_strategy == EvictStrategy::kEpochLru)
          ? nullptr
          : scores,
      evicted_keys, evicted_values, evicted_scores, stream);
  evict_buffer->SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  {
    table->find(len, keys, values, d_tmp_founds, scores, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t found_counter = 0;
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) found_counter++;
    }

    CUDA_CHECK(cudaMemset(d_tmp_founds, 0, len * sizeof(bool)));
    table->contains(len, keys, d_tmp_founds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int contains_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) contains_counter++;
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  for (size_t i = 0; i < filtered_len; i++) {
    scores_map_current_evict[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
  }

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

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt = 0;

  for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    values_map_after_insert[h_tmp_keys[i]] = *vec;
    scores_map_after_insert[h_tmp_keys[i]] = h_tmp_scores[i];
  }

  for (auto it : scores_map_current_batch) {
    const K key = it.first;
    const K score = it.second;
    S current_score = scores_map_after_insert[key];
    S score_before_insert = 0;
    if (scores_map_before_insert.find(key) != scores_map_before_insert.end() &&
        scores_map_current_evict.find(key) == scores_map_current_evict.end()) {
      score_before_insert = scores_map_before_insert[key];
    } else {
      continue;
    }
    bool valid = (current_score == score + score_before_insert);
    if (!valid) {
      score_error_cnt++;
    }
  }

  ASSERT_EQ(values_map_before_insert.size(), values_map_after_insert.size());

  for (auto& it : values_map_before_insert) {
    if (values_map_after_insert.find(it.first) ==
        values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<V, dim>& vec0 = it.second;
    test_util::ValueArray<V, dim>& vec1 = values_map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; j++) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
      }
    }
  }
  std::cout << "Check insert_and_evict behavior got "
            << "key_miss_cnt: " << key_miss_cnt
            << ", value_diff_cnt: " << value_diff_cnt
            << ", score_error_cnt: " << score_error_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(score_error_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);

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

void test_insert_and_evict_advanced_on_lfu() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 256 * 1024;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kLfu>;
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
  for (unsigned int global_epoch = 1; global_epoch <= 32; global_epoch++) {
    test_util::create_random_keys_advanced<i64, u64, f32>(
        dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16, 100);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvictOnLfu<i64, f32, u64, Table>(
        table.get(), &data_buffer, &evict_buffer, B, stream, opt, global_epoch);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    offset += B;
  }
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnEpochLru(Table* table,
                                   test_util::KVMSBuffer<K, V, S>* data_buffer,
                                   test_util::KVMSBuffer<K, V, S>* evict_buffer,
                                   size_t len, cudaStream_t stream,
                                   TableOptions& opt,
                                   unsigned int global_epoch) {
  std::map<K, test_util::ValueArray<V, dim>> values_map_before_insert;
  std::map<K, test_util::ValueArray<V, dim>> values_map_after_insert;

  std::map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;

  std::map<K, S> scores_map_current_batch;

  K* keys = data_buffer->keys_ptr();
  V* values = data_buffer->values_ptr();
  S* scores = data_buffer->scores_ptr();

  K* evicted_keys = evict_buffer->keys_ptr();
  V* evicted_values = evict_buffer->values_ptr();
  S* evicted_scores = evict_buffer->scores_ptr();

  for (size_t i = 0; i < len; i++) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_founds = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_founds = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_founds, cap * sizeof(bool), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_founds, 0, cap * sizeof(bool), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_founds = (bool*)malloc(cap * sizeof(bool));

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
    values_map_before_insert[h_tmp_keys[i]] = *vec;
  }

  for (size_t i = 0; i < table_size_before; i++) {
    scores_map_before_insert[h_tmp_keys[i]] = h_tmp_scores[i];
  }

  S nano_before_insert = test_util::host_nano<S>();

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  size_t filtered_len = table->insert_and_evict(
      len, keys, values,
      (Table::evict_strategy == EvictStrategy::kLru ||
       Table::evict_strategy == EvictStrategy::kEpochLru)
          ? nullptr
          : scores,
      evicted_keys, evicted_values, evicted_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  S nano_after_insert = test_util::host_nano<S>();

  {
    table->find(len, keys, values, d_tmp_founds, scores, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t found_counter = 0;
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) found_counter++;
    }
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    CUDA_CHECK(cudaMemset(d_tmp_founds, 0, len * sizeof(bool)));
    table->contains(len, keys, d_tmp_founds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int contains_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) contains_counter++;
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

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

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    values_map_after_insert[h_tmp_keys[i]] = *vec;
    scores_map_after_insert[h_tmp_keys[i]] = h_tmp_scores[i];
    if (i >= (new_cap_i64 - filtered_len)) {
      bool valid = ((h_tmp_scores[i] >> 32) < (global_epoch - 2));
      if (!valid) {
        score_error_cnt1++;
      }
    }
  }

  for (auto& it : scores_map_current_batch) {
    S score = scores_map_after_insert[it.first];
    bool valid =
        ((score >> 32) == global_epoch) &&
        ((score & 0xFFFFFFFF) >= (0xFFFFFFFF & (nano_before_insert >> 20))) &&
        ((score & 0xFFFFFFFF) <= (0xFFFFFFFF & (nano_after_insert >> 20)));
    if (!valid) {
      score_error_cnt2++;
    }
  }
  for (auto& it : values_map_before_insert) {
    if (values_map_after_insert.find(it.first) ==
        values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<V, dim>& vec0 = it.second;
    test_util::ValueArray<V, dim>& vec1 = values_map_after_insert.at(it.first);
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
            << ", score_error_cnt1: " << score_error_cnt1
            << ", score_error_cnt2: " << score_error_cnt2
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);

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

void test_insert_and_evict_advanced_on_epochlru() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 128 * 1024;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kEpochLru>;
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
  for (unsigned int global_epoch = 1; global_epoch <= 64; global_epoch++) {
    test_util::create_random_keys_advanced<i64, u64, f32>(
        dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvictOnEpochLru<i64, f32, u64, Table>(
        table.get(), &data_buffer, &evict_buffer, B, stream, opt, global_epoch);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    offset += B;
    score += 1;
  }
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnEpochLfu(
    Table* table, test_util::KVMSBuffer<K, V, S>* data_buffer,
    test_util::KVMSBuffer<K, V, S>* evict_buffer,
    test_util::KVMSBuffer<K, V, S>* pre_data_buffer, size_t len,
    cudaStream_t stream, TableOptions& opt, unsigned int global_epoch) {
  std::map<K, test_util::ValueArray<V, dim>> values_map_before_insert;
  std::map<K, test_util::ValueArray<V, dim>> values_map_after_insert;

  std::unordered_map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;

  std::map<K, S> scores_map_current_batch;
  std::map<K, S> scores_map_current_evict;

  K* keys = data_buffer->keys_ptr();
  V* values = data_buffer->values_ptr();
  S* scores = data_buffer->scores_ptr();

  K* evicted_keys = evict_buffer->keys_ptr();
  V* evicted_values = evict_buffer->values_ptr();
  S* evicted_scores = evict_buffer->scores_ptr();

  for (size_t i = 0; i < len; i++) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_founds = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_founds = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_founds, cap * sizeof(bool), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_founds, 0, cap * sizeof(bool), stream));
  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_founds = (bool*)malloc(cap * sizeof(bool));

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
    values_map_before_insert[h_tmp_keys[i]] = *vec;
  }

  for (size_t i = 0; i < table_size_before; i++) {
    scores_map_before_insert[h_tmp_keys[i]] = h_tmp_scores[i];
  }

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  size_t filtered_len = table->insert_and_evict(
      len, keys, values,
      (Table::evict_strategy == EvictStrategy::kLru ||
       Table::evict_strategy == EvictStrategy::kEpochLru)
          ? nullptr
          : scores,
      evicted_keys, evicted_values, evicted_scores, stream);
  evict_buffer->SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  {
    table->find(len, pre_data_buffer->keys_ptr(), values, d_tmp_founds,
                pre_data_buffer->scores_ptr(), stream);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    pre_data_buffer->SyncData(false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t found_counter = 0;
    size_t old_epoch_counter = 0;
    size_t new_epoch_counter = 0;
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) found_counter++;
      S score = pre_data_buffer->scores_ptr(false)[i];
      S cur_epoch = score >> 32;
      if (global_epoch == cur_epoch) new_epoch_counter++;
      if (global_epoch - 1 == cur_epoch) old_epoch_counter++;
    }
    ASSERT_EQ(len, new_epoch_counter + old_epoch_counter);
    std::cout << "old_epoch_counter:" << old_epoch_counter
              << ", new_epoch_counter:" << new_epoch_counter << std::endl
              << ", pre_data filtered_len:" << filtered_len
              << ", pre_data miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    CUDA_CHECK(cudaMemset(d_tmp_founds, 0, len * sizeof(bool)));
    table->contains(len, pre_data_buffer->keys_ptr(), d_tmp_founds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int contains_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) contains_counter++;
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  {
    table->find(len, keys, values, d_tmp_founds, scores, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    data_buffer->SyncData(false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t found_counter = 0;
    size_t new_epoch_counter = 0;
    for (int i = 0; i < len; i++) {
      S score = data_buffer->scores_ptr(false)[i];
      S cur_epoch = score >> 32;
      if (h_tmp_founds[i]) found_counter++;
      if (global_epoch == cur_epoch) new_epoch_counter++;
    }
    ASSERT_EQ(len, new_epoch_counter);
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    CUDA_CHECK(cudaMemset(d_tmp_founds, 0, len * sizeof(bool)));
    table->contains(len, keys, d_tmp_founds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int contains_counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, len * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < len; i++) {
      if (h_tmp_founds[i]) contains_counter++;
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  {
    std::unordered_set<K> unique_keys;
    for (int i = 0; i < len; i++) {
      unique_keys.insert(data_buffer->keys_ptr(false)[i]);
      unique_keys.insert(pre_data_buffer->keys_ptr(false)[i]);
    }
    float repeat_rate = (len * 2.0 - unique_keys.size()) / (len * 1.0);
    std::cout << "repeat_rate:" << repeat_rate << std::endl;
  }

  for (size_t i = 0; i < filtered_len; i++) {
    scores_map_current_evict[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
  }

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

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    values_map_after_insert[h_tmp_keys[i]] = *vec;
    scores_map_after_insert[h_tmp_keys[i]] = h_tmp_scores[i];
    if (i >= (new_cap_i64 - filtered_len)) {
      bool valid = ((h_tmp_scores[i] >> 32) < (global_epoch - 2));
      if (!valid) {
        score_error_cnt1++;
      }
    }
  }

  for (auto it : scores_map_current_batch) {
    const K key = it.first;
    const K score = it.second;
    S current_score = scores_map_after_insert[key];
    S score_before_insert = 0;
    if (scores_map_before_insert.find(key) != scores_map_before_insert.end() &&
        scores_map_current_evict.find(key) == scores_map_current_evict.end()) {
      score_before_insert = scores_map_before_insert[key];
    }
    bool valid = ((current_score >> 32) == global_epoch) &&
                 ((current_score & 0xFFFFFFFF) ==
                  ((0xFFFFFFFF & score_before_insert) + (0xFFFFFFFF & score)));

    if (!valid) {
      score_error_cnt2++;
    }
  }
  for (auto& it : values_map_before_insert) {
    if (values_map_after_insert.find(it.first) ==
        values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<V, dim>& vec0 = it.second;
    test_util::ValueArray<V, dim>& vec1 = values_map_after_insert.at(it.first);
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
            << ", score_error_cnt1: " << score_error_cnt1
            << ", score_error_cnt2: " << score_error_cnt2
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);

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

void test_insert_and_evict_advanced_on_epochlfu() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 128 * 1024;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kEpochLfu>;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(B, dim, stream);
  evict_buffer.ToZeros(stream);

  test_util::KVMSBuffer<i64, f32, u64> data_buffer;
  test_util::KVMSBuffer<i64, f32, u64> pre_data_buffer;
  data_buffer.Reserve(B, dim, stream);
  pre_data_buffer.Reserve(B, dim, stream);

  size_t offset = 0;
  int freq_range = 100;
  float repeat_rate = 0.9;
  for (unsigned int global_epoch = 1; global_epoch <= 64; global_epoch++) {
    if (global_epoch <= 1) {
      test_util::create_random_keys_advanced<i64, u64, f32>(
          dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
          data_buffer.values_ptr(false), (int)B, B * 16, freq_range);
    } else {
      test_util::create_random_keys_advanced<i64, u64, f32>(
          dim, data_buffer.keys_ptr(false), pre_data_buffer.keys_ptr(false),
          data_buffer.scores_ptr(false), data_buffer.values_ptr(false), (int)B,
          B * 16, freq_range, repeat_rate);
    }
    data_buffer.SyncData(true, stream);
    if (global_epoch <= 1) {
      pre_data_buffer.CopyFrom(data_buffer, stream);
    }

    CheckInsertAndEvictOnEpochLfu<i64, f32, u64, Table>(
        table.get(), &data_buffer, &evict_buffer, &pre_data_buffer, B, stream,
        opt, global_epoch);

    pre_data_buffer.CopyFrom(data_buffer, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    offset += B;
  }
}

void test_insert_and_evict_advanced_on_customized() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 100000;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  using Table =
      nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kCustomized>;
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
  for (int i = 0; i < 32; i++) {
    test_util::create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, (int)B * 16);
    data_buffer.SyncData(true, stream);

    CheckInsertAndEvict<i64, f32, u64, Table>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, stream, opt);

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
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kLru>;
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
  size_t search_len = (table->capacity() >> 2);
  for (; offset < table->capacity(); offset += search_len) {
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

template <typename K, typename V, typename S, typename Table>
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

template <typename K, typename V, typename S, typename Table>
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

      CUDA_CHECK(cudaMemset(d_tmp_founds, 0, cap * sizeof(bool)));
      table->contains(cap, keys, d_tmp_founds, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      int contains_num = 0;
      CUDA_CHECK(cudaMemcpyAsync(h_tmp_founds, d_tmp_founds, cap * sizeof(bool),
                                 cudaMemcpyDeviceToHost, stream));
      for (int i = 0; i < cap; i++) {
        if (h_tmp_founds[i]) contains_num++;
      }
      ASSERT_EQ(contains_num, found_num);
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
  using Table = nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kLru>;
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
    BatchCheckInsertAndEvict<i64, f32, u64, Table>(
        table.get(), global_buffer.keys_ptr(), global_buffer.values_ptr(),
        global_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, &step,
        batch_num, insert_stream, if_check);
  };

  auto find_func = [&table, &global_buffer, &B, &step, &batch_num,
                    &find_interval, &find_stream]() {
    BatchCheckFind<i64, f32, u64, Table>(
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

TEST(InsertAndEvictTest, test_insert_and_evict_advanced_on_lru) {
  test_insert_and_evict_advanced_on_lru();
}

TEST(InsertAndEvictTest, test_insert_and_evict_advanced_on_lfu) {
  test_insert_and_evict_advanced_on_lfu();
}

TEST(InsertAndEvictTest, test_insert_and_evict_advanced_on_epochlru) {
  test_insert_and_evict_advanced_on_epochlru();
}

TEST(InsertAndEvictTest, test_insert_and_evict_advanced_on_epochlfu) {
  test_insert_and_evict_advanced_on_epochlfu();
}

TEST(InsertAndEvictTest, test_insert_and_evict_advanced_on_customized) {
  test_insert_and_evict_advanced_on_customized();
}

TEST(InsertAndEvictTest, test_insert_and_evict_with_export_batch) {
  test_insert_and_evict_with_export_batch();
}

TEST(InsertAndEvictTest, test_insert_and_evict_run_with_batch_find) {
  test_insert_and_evict_run_with_batch_find();
}
