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

/*
 * test APIs: find_or_insert and assign,
 * move insert operation from `insert_or_assign` to `find`.
 */

#include <gtest/gtest.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <thread>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;

void test_evict_strategy_lru_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  for (int i = 0; i < options.dim; i++) {
    h_vectors_test[2 * options.dim + i] = h_vectors_base[72 * options.dim + i];
    h_vectors_test[3 * options.dim + i] = h_vectors_base[73 * options.dim + i];
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      S start_ts = test_util::host_nano<S>(stream);
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp, nullptr,
                            stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts = test_util::host_nano<S>(stream);

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, BASE_KEY_NUM> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      ASSERT_GE(h_scores_temp_sorted[0], start_ts);
      ASSERT_LE(h_scores_temp_sorted[TEST_KEY_NUM - 1], end_ts);
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp, nullptr,
                            stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      S start_ts = test_util::host_nano<S>(stream);
      table->assign(TEST_KEY_NUM, d_keys_temp, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts = test_util::host_nano<S>(stream);

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, TEST_KEY_NUM> h_scores_temp_sorted;
      int ctr = 0;
      for (int i = 0; i < TEMP_KEY_NUM; i++) {
        if (h_keys_test.end() !=
            std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i])) {
          ASSERT_GT(h_scores_temp[i], BUCKET_MAX_SIZE);
          h_scores_temp_sorted[ctr++] = h_scores_temp[i];
        } else {
          ASSERT_LE(h_scores_temp[i], start_ts);
        }
      }
      std::sort(h_scores_temp_sorted.begin(),
                h_scores_temp_sorted.begin() + ctr);

      ASSERT_GE(h_scores_temp_sorted[0], start_ts);
      ASSERT_LE(h_scores_temp_sorted[ctr - 1], end_ts);
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_lfu_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLfu>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  int freq_range = 1000;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;

  for (int i = 0; i < options.dim; i++) {
    h_vectors_test[2 * options.dim + i] = h_vectors_base[72 * options.dim + i];
    h_vectors_test[3 * options.dim + i] = h_vectors_base[73 * options.dim + i];
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  S global_epoch = 1;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        ASSERT_EQ(h_scores_temp[i], h_keys_temp[i] % freq_range);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      global_epoch++;
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));

      table->assign(TEST_KEY_NUM, d_keys_temp, d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        bool in_base =
            h_keys_base.end() !=
            std::find(h_keys_base.begin(), h_keys_base.end(), h_keys_temp[i]);
        bool in_test =
            h_keys_test.end() !=
            std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]);
        if (in_base && in_test) {
          ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % freq_range) * 2);
        } else {
          ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % freq_range));
        }
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_epochlru_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr int RSHIFT_ON_NANO = 20;

  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLru>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  for (int i = 0; i < options.dim; i++) {
    h_vectors_test[2 * options.dim + i] = h_vectors_base[72 * options.dim + i];
    h_vectors_test[3 * options.dim + i] = h_vectors_base[73 * options.dim + i];
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  S global_epoch = 1;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      S start_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp, nullptr,
                            stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, BASE_KEY_NUM> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      ASSERT_GE(h_scores_temp_sorted[0], (global_epoch << 32 | start_ts));
      ASSERT_LE(h_scores_temp_sorted[BASE_KEY_NUM - 1],
                (global_epoch << 32 | end_ts));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      global_epoch++;
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      S start_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->assign(TEST_KEY_NUM, d_keys_temp, nullptr, stream);
      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp, nullptr,
                            stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, TEST_KEY_NUM> h_scores_temp_sorted;
      int ctr = 0;
      for (int i = 0; i < TEMP_KEY_NUM; i++) {
        if (h_keys_test.end() !=
            std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i])) {
          ASSERT_GE(h_scores_temp[i], (global_epoch << 32 | start_ts));
          h_scores_temp_sorted[ctr++] = h_scores_temp[i];
        } else {
          ASSERT_LE(h_scores_temp[i], (global_epoch << 32 | start_ts));
        }
      }
      std::sort(h_scores_temp_sorted.begin(),
                h_scores_temp_sorted.begin() + ctr);

      ASSERT_GE(h_scores_temp_sorted[0], (global_epoch << 32 | start_ts));
      ASSERT_LE(h_scores_temp_sorted[ctr - 1], (global_epoch << 32 | end_ts));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_epochlfu_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLfu>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  int freq_range = 1000;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  // Simulate overflow of low 32bits.
  h_scores_base[71] = static_cast<S>(std::numeric_limits<uint32_t>::max() -
                                     static_cast<uint32_t>(1));

  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_scores_test[1] = h_scores_base[71];
  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;

  for (int i = 0; i < options.dim; i++) {
    h_vectors_test[1 * options.dim + i] = h_vectors_base[71 * options.dim + i];
    h_vectors_test[2 * options.dim + i] = h_vectors_base[72 * options.dim + i];
    h_vectors_test[3 * options.dim + i] = h_vectors_base[73 * options.dim + i];
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  S global_epoch = 1;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->set_global_epoch(global_epoch);
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        if (h_keys_temp[i] == h_keys_base[71]) {
          S expected_score = test_util::make_expected_score_for_epochlfu<S>(
              global_epoch, h_scores_base[71]);
          ASSERT_EQ(h_scores_temp[i], expected_score);
        } else {
          S expected_score = test_util::make_expected_score_for_epochlfu<S>(
              global_epoch, (h_keys_temp[i] % freq_range));
          ASSERT_EQ(h_scores_temp[i], expected_score);
        }
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      global_epoch++;
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->set_global_epoch(global_epoch);
      table->assign(TEST_KEY_NUM, d_keys_temp, d_scores_temp, stream);
      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      ASSERT_TRUE(h_keys_temp.end() != std::find(h_keys_temp.begin(),
                                                 h_keys_temp.end(),
                                                 h_keys_base[71]));

      for (int i = 0; i < dump_counter; i++) {
        bool in_base =
            h_keys_base.end() !=
            std::find(h_keys_base.begin(), h_keys_base.end(), h_keys_temp[i]);
        bool in_test =
            h_keys_test.end() !=
            std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]);

        if (in_base && in_test) {
          if (h_keys_temp[i] == h_keys_base[71]) {
            S expected_score = test_util::make_expected_score_for_epochlfu<S>(
                global_epoch, h_scores_base[71] * 2);
            ASSERT_EQ(h_scores_temp[i], expected_score);
          } else {
            S expected_score = test_util::make_expected_score_for_epochlfu<S>(
                global_epoch, (h_keys_temp[i] % freq_range) * 2);
            ASSERT_EQ(h_scores_temp[i], expected_score);
          }
        } else {
          if (h_keys_temp[i] == h_keys_base[71]) {
            S expected_score = test_util::make_expected_score_for_epochlfu<S>(
                global_epoch - static_cast<S>(in_base), h_scores_base[71]);
            ASSERT_EQ(h_scores_temp[i], expected_score);
          } else {
            S expected_score = test_util::make_expected_score_for_epochlfu<S>(
                global_epoch - static_cast<S>(in_base),
                (h_keys_temp[i] % freq_range));

            ASSERT_EQ(h_scores_temp[i], expected_score);
          }
        }
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 128;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  const S base_score_start = 1000;
  for (int i = 0; i < BASE_KEY_NUM; i++) {
    h_scores_base[i] = base_score_start + i;
  }

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);
  const S test_score_start = base_score_start + BASE_KEY_NUM;
  for (int i = 0; i < TEST_KEY_NUM; i++) {
    h_scores_test[i] = test_score_start + i;
  }
  for (int i = 64; i < TEST_KEY_NUM; i++) {
    h_keys_test[i] = h_keys_base[i];
    for (int j = 0; j < options.dim; j++) {
      h_vectors_test[i * options.dim + j] = h_vectors_base[i * options.dim + j];
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, BASE_KEY_NUM> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      ASSERT_TRUE((h_scores_temp_sorted ==
                   test_util::range<S, TEMP_KEY_NUM>(base_score_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->assign(TEST_KEY_NUM, d_keys_temp, d_scores_temp, stream);
      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, TEST_KEY_NUM> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      ASSERT_TRUE((h_scores_temp_sorted ==
                   test_util::range<S, TEST_KEY_NUM>(test_score_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 256;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<S, BASE_KEY_NUM> h_scores_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<S, TEST_KEY_NUM> h_scores_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<S, TEMP_KEY_NUM> h_scores_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  const S base_score_start = 1000;
  for (int i = 0; i < BASE_KEY_NUM; i++) {
    h_scores_base[i] = base_score_start + i;
  }

  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];

  // replace four new keys to lower scores, would not be inserted.
  h_scores_test[0] = 20;
  h_scores_test[1] = 78;
  h_scores_test[2] = 97;
  h_scores_test[3] = 98;

  // replace three exist keys to new scores, just refresh the score for them.
  h_scores_test[4] = 99;
  h_scores_test[5] = 1010;
  h_scores_test[6] = 1020;
  h_scores_test[7] = 1035;

  for (int i = 4; i < TEST_KEY_NUM; i++) {
    for (int j = 0; j < options.dim; j++) {
      h_vectors_test[i * options.dim + j] =
          static_cast<V>(h_keys_test[i] * 0.00001);
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base.data(),
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->find_or_insert(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<S, BASE_KEY_NUM> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      ASSERT_TRUE((h_scores_temp_sorted ==
                   test_util::range<S, TEMP_KEY_NUM>(base_score_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->assign(TEST_KEY_NUM, d_keys_temp, d_scores_temp, stream);
      table->find_or_insert(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                            d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if (i < 4) {
          ASSERT_EQ(h_keys_temp.end(),
                    std::find(h_keys_temp.begin(), h_keys_temp.end(),
                              h_keys_test[i]));
        } else {
          ASSERT_NE(h_keys_temp.end(),
                    std::find(h_keys_temp.begin(), h_keys_temp.end(),
                              h_keys_test[i]));
        }
      }
      for (int i = 0; i < TEMP_KEY_NUM; i++) {
        if (h_keys_temp[i] == h_keys_test[4])
          ASSERT_EQ(h_scores_temp[i], h_scores_test[4]);
        if (h_keys_temp[i] == h_keys_test[5])
          ASSERT_EQ(h_scores_temp[i], h_scores_test[5]);
        if (h_keys_temp[i] == h_keys_test[6])
          ASSERT_EQ(h_scores_temp[i], h_scores_test[6]);
        if (h_keys_temp[i] == h_keys_test[7])
          ASSERT_EQ(h_scores_temp[i], h_scores_test[7]);

        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

template <typename K, typename V, typename S, typename Table, size_t dim = 64>
void CheckAssignOnEpochLfu(Table* table,
                           test_util::KVMSBuffer<K, V, S>* data_buffer,
                           test_util::KVMSBuffer<K, V, S>* evict_buffer,
                           test_util::KVMSBuffer<K, V, S>* pre_data_buffer,
                           size_t len, cudaStream_t stream, TableOptions& opt,
                           unsigned int global_epoch) {
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

  table->set_global_epoch(global_epoch);
  table->assign(len, keys, scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  {
    size_t table_size_verify1 = table->export_batch(
        table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);

    CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                               table_size_before * sizeof(K),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                               table_size_before * dim * sizeof(V),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                               table_size_before * sizeof(S),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(table_size_verify1, table_size_before);

    size_t score_error_cnt = 0;

    for (int64_t i = table_size_before - 1; i >= 0; i--) {
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
      if (scores_map_before_insert.find(key) !=
          scores_map_before_insert.end()) {
        score_before_insert = scores_map_before_insert[key];
        bool valid =
            ((current_score >> 32) == global_epoch) &&
            ((current_score & 0xFFFFFFFF) ==
             ((0xFFFFFFFF & score_before_insert) + (0xFFFFFFFF & score)));

        if (!valid) {
          score_error_cnt++;
        }
      }
    }
    std::cout << "Check assign behavior got "
              << ", score_error_cnt: " << score_error_cnt
              << ", while len: " << len << std::endl;
    ASSERT_EQ(score_error_cnt, 0);
  }

  for (int64_t i = 0; i < table_size_before; i++) {
    values_map_before_insert[h_tmp_keys[i]] =
        values_map_after_insert[h_tmp_keys[i]];
    scores_map_before_insert[h_tmp_keys[i]] =
        scores_map_after_insert[h_tmp_keys[i]];
  }
  values_map_after_insert.clear();
  scores_map_after_insert.clear();

  auto start = std::chrono::steady_clock::now();
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

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (int64_t i = new_cap - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    values_map_after_insert[h_tmp_keys[i]] = *vec;
    scores_map_after_insert[h_tmp_keys[i]] = h_tmp_scores[i];
    if (i >= (new_cap - filtered_len)) {
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
    if (values_map_after_insert.find(key) != values_map_after_insert.end() &&
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
            << ", while len: " << len << std::endl;

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

void test_assign_advanced_on_epochlfu(size_t max_hbm_for_vectors) {
  const size_t U = 1024 * 1024;
  const size_t B = 100000;
  constexpr size_t dim = 16;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = U;
  opt.max_hbm_for_vectors = U * dim * sizeof(V);
  opt.max_bucket_size = 128;
  opt.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLfu>;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<K, V, S> evict_buffer;
  evict_buffer.Reserve(B, dim, stream);
  evict_buffer.ToZeros(stream);

  test_util::KVMSBuffer<K, V, S> data_buffer;
  test_util::KVMSBuffer<K, V, S> pre_data_buffer;
  data_buffer.Reserve(B, dim, stream);
  pre_data_buffer.Reserve(B, dim, stream);

  size_t offset = 0;
  int freq_range = 100;
  float repeat_rate = 0.9;
  for (unsigned int global_epoch = 1; global_epoch <= 20; global_epoch++) {
    repeat_rate = global_epoch <= 1 ? 0.0 : 0.1;
    if (global_epoch <= 1) {
      test_util::create_random_keys_advanced<K, S, V>(
          dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
          data_buffer.values_ptr(false), (int)B, B * 32, freq_range);
    } else {
      test_util::create_random_keys_advanced<K, S, V>(
          dim, data_buffer.keys_ptr(false), pre_data_buffer.keys_ptr(false),
          data_buffer.scores_ptr(false), data_buffer.values_ptr(false), (int)B,
          B * 32, freq_range, repeat_rate);
    }
    data_buffer.SyncData(true, stream);
    if (global_epoch <= 1) {
      pre_data_buffer.CopyFrom(data_buffer, stream);
    }

    CheckAssignOnEpochLfu<K, V, S, Table, dim>(table.get(), &data_buffer,
                                               &evict_buffer, &pre_data_buffer,
                                               B, stream, opt, global_epoch);

    pre_data_buffer.CopyFrom(data_buffer, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    offset += B;
  }
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  float expected_correct_rate = 0.964;
  const int rounds = 12;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  K* h_keys_base = test_util::HostBuffer<K>(BATCH_SIZE).ptr();
  S* h_scores_base = test_util::HostBuffer<S>(BATCH_SIZE).ptr();
  V* h_vectors_base = test_util::HostBuffer<V>(BATCH_SIZE * options.dim).ptr();

  K* h_keys_temp = test_util::HostBuffer<K>(MAX_CAPACITY).ptr();
  S* h_scores_temp = test_util::HostBuffer<S>(MAX_CAPACITY).ptr();
  V* h_vectors_temp =
      test_util::HostBuffer<V>(MAX_CAPACITY * options.dim).ptr();

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, MAX_CAPACITY * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t global_start_key = 100000;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    size_t start_key = global_start_key;

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    for (int r = 0; r < rounds; r++) {
      size_t expected_min_key = global_start_key + INIT_CAPACITY * r;
      size_t expected_max_key = global_start_key + INIT_CAPACITY * (r + 1) - 1;
      size_t expected_table_size =
          (r == 0) ? size_t(expected_correct_rate * INIT_CAPACITY)
                   : INIT_CAPACITY;

      for (int s = 0; s < STEPS; s++) {
        test_util::create_continuous_keys<K, S, V, DIM>(
            h_keys_base, h_scores_base, h_vectors_base, BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;

        CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base, BATCH_SIZE * sizeof(K),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base,
                              BATCH_SIZE * sizeof(S), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base,
                              BATCH_SIZE * sizeof(V) * options.dim,
                              cudaMemcpyHostToDevice));
        table->assign(BATCH_SIZE, d_keys_temp, d_scores_temp, stream);
        table->find_or_insert(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                              d_scores_temp, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());

      size_t dump_counter = table->export_batch(
          MAX_CAPACITY, 0, d_keys_temp, d_vectors_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaMemcpy(h_keys_temp, d_keys_temp, MAX_CAPACITY * sizeof(K),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_scores_temp, d_scores_temp,
                            MAX_CAPACITY * sizeof(S), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp, d_vectors_temp,
                            MAX_CAPACITY * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));

      size_t bigger_score_counter = 0;
      K max_key = 0;
      size_t values_error_counter = 0;
      for (int i = 0; i < dump_counter; i++) {
        ASSERT_EQ(h_keys_temp[i], h_scores_temp[i]);
        max_key = std::max(max_key, h_keys_temp[i]);
        if (h_scores_temp[i] >= expected_min_key) bigger_score_counter++;
        for (int j = 0; j < options.dim; j++) {
          if (h_vectors_temp[i * options.dim + j] !=
              static_cast<float>(h_keys_temp[i] * 0.00001)) {
            values_error_counter++;
          }
        }
      }

      ASSERT_EQ(values_error_counter, 0);
      float correct_rate = (bigger_score_counter * 1.0) / MAX_CAPACITY;
      std::cout << std::setprecision(3) << "[Round " << r << "]"
                << "correct_rate=" << correct_rate << std::endl;
      ASSERT_GE(max_key, expected_max_key);
      ASSERT_GE(correct_rate, expected_correct_rate);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

template <typename K, typename V, typename S, typename Table, size_t dim = 64>
void CheckFindOrInsertValues(Table* table, K* keys, V* values, S* scores,
                             size_t len, cudaStream_t stream) {
  std::map<K, test_util::ValueArray<V, dim>> map_before_insert;
  std::map<K, test_util::ValueArray<V, dim>> map_after_insert;
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

  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < table_size_verify0; i++) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    map_before_insert[h_tmp_keys[i]] = *vec;
  }

  auto start = std::chrono::steady_clock::now();
  table->find_or_insert(len, keys, values, nullptr, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  float dur = diff.count();

  size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = table->export_batch(
      table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_scores, stream);

  ASSERT_EQ(table_size_verify1, table_size_after);

  size_t new_cap = table_size_after;
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys,
                             table_size_after * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values,
                             table_size_after * dim * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_scores, d_tmp_scores,
                             table_size_after * sizeof(S),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int64_t new_cap_K = (int64_t)new_cap;
  for (int64_t i = new_cap_K - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    map_after_insert[h_tmp_keys[i]] = *vec;
  }

  size_t value_diff_cnt = 0;
  for (auto& it : map_after_insert) {
    test_util::ValueArray<V, dim>& vec = map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; j++) {
      if (vec[j] != static_cast<float>(it.first * 0.00001)) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  ASSERT_EQ(value_diff_cnt, 0);
  std::cout << "Check find_or_insert behavior got "
            << "value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;

  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_scores, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_scores);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_find_or_insert_values_check(size_t max_hbm_for_vectors) {
  const size_t U = 524288;
  const size_t init_capacity = 1024;
  const size_t B = 524288 + 13;
  constexpr size_t dim = 64;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
  opt.dim = 64;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<K, V, S> data_buffer;
  data_buffer.Reserve(B, dim, stream);

  size_t offset = 0;
  S score = 0;
  for (int i = 0; i < 20; i++) {
    test_util::create_random_keys<K, S, V, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), (int)B, B * 16);
    data_buffer.SyncData(true, stream);

    CheckFindOrInsertValues<K, V, S, Table, dim>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), B, stream);

    offset += B;
    score += 1;
  }
}

TEST(AssignScoreTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic(16);
  test_evict_strategy_lru_basic(0, 34);
}
TEST(AssignScoreTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic(16);
  test_evict_strategy_lfu_basic(0, 2);
}
TEST(AssignScoreTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_epochlru_basic(16, 51);
  test_evict_strategy_epochlru_basic(0);
}
TEST(AssignScoreTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_epochlfu_basic(16, 4);
  test_evict_strategy_epochlfu_basic(0);
}
TEST(AssignScoreTest, test_evict_strategy_customized_basic) {
  test_evict_strategy_customized_basic(16);
  test_evict_strategy_customized_basic(0, 11);
}
TEST(AssignScoreTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced(16, 33);
  test_evict_strategy_customized_advanced(0);
}
TEST(AssignScoreTest, test_assign_advanced_on_epochlfu) {
  test_assign_advanced_on_epochlfu(16);
}
TEST(AssignScoreTest, test_evict_strategy_customized_correct_rate) {
  // TODO(rhdong): after blossom CI issue is resolved, the skip logic.
  const bool skip_hmem_check = (nullptr != std::getenv("IS_BLOSSOM_CI"));
  test_evict_strategy_customized_correct_rate(16, 44);
  if (!skip_hmem_check) {
    test_evict_strategy_customized_correct_rate(0);
  } else {
    std::cout << "The HMEM check is skipped in blossom CI!" << std::endl;
  }
}