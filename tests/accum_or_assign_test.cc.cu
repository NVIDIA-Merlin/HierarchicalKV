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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;

template <class K, class S>
struct EraseIfPredFunctor {
  __forceinline__ __device__ bool operator()(const K& key, S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return (((key & 0x1u) == 0x1u) && (score > threshold));
  }
};

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __device__ bool operator()(const K& key, S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score > threshold;
  }
};

void test_basic_when_full(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t INIT_CAPACITY = 1 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_accum_or_assigns;
  bool* h_found;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

  test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, nullptr,
                                              KEY_NUM);

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  bool* d_accum_or_assigns;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_def_val, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_vectors, 1, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                           d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t total_size_after_insert = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    table->erase(KEY_NUM, d_keys, stream);
    size_t total_size_after_erase = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_erase, 0);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                           d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t total_size_after_reinsert = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_insert, total_size_after_reinsert);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_accum_or_assigns));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_erase_if_pred(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr float true_ratio = 0.5;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;
  bool* h_accum_or_assigns;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  bool* d_found;
  bool* d_accum_or_assigns;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    test_util::create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys, h_scores, h_vectors, KEY_NUM, INIT_CAPACITY);

    test_util::create_random_bools<K>(h_accum_or_assigns, KEY_NUM, true_ratio);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                          KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                           d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t expected_size = 0;
    for (size_t i = 0; i < KEY_NUM; i++) {
      if (!h_accum_or_assigns[i]) expected_size++;
    }
    ASSERT_EQ(total_size, expected_size);

    K pattern = 100;
    S threshold = 0;
    size_t erase_num = table->template erase_if<EraseIfPredFunctor>(
        pattern, threshold, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ((erase_num + total_size), expected_size);

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_scores, 0, KEY_NUM * sizeof(S)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_EQ(h_scores[i], h_keys[i]);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<float>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, (expected_size - erase_num));

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_accum_or_assigns));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_rehash(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 4 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  constexpr float true_ratio = 0.5;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;
  bool* h_accum_or_assigns;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  bool* d_found;
  bool* d_accum_or_assigns;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    test_util::create_random_bools<K>(h_accum_or_assigns, KEY_NUM, true_ratio);
    test_util::create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys, h_scores, h_vectors, KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                          KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                           d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t expected_size = 0;
    for (size_t i = 0; i < KEY_NUM; i++) {
      if (!h_accum_or_assigns[i]) expected_size++;
    }
    ASSERT_EQ(total_size, expected_size);

    dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                       d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(dump_counter, expected_size);

    table->reserve(MAX_CAPACITY, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(table->capacity(), MAX_CAPACITY);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, expected_size);

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_scores, 0, KEY_NUM * sizeof(S)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_EQ(h_scores[i], h_keys[i]);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<float>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, expected_size);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_accum_or_assigns));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_rehash_on_big_batch(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t INIT_CAPACITY = 1024;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024;
  constexpr uint64_t INIT_KEY_NUM = 1024;
  constexpr uint64_t KEY_NUM = 2048;

  std::unordered_map<K, float> expected_values;
  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;
  bool* h_accum_or_assigns;
  bool* h_accum_or_assigns_init;
  float true_ratio = 0.6f;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.max_load_factor = 0.6;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns_init, KEY_NUM * sizeof(bool)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  bool* d_accum_or_assigns;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  uint64_t expected_size = 0;
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, h_vectors,
                                              KEY_NUM);

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_accum_or_assigns, 0, KEY_NUM * sizeof(bool)));

  total_size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(total_size, 0);

  test_util::create_random_bools<K>(h_accum_or_assigns, INIT_KEY_NUM,
                                    true_ratio);
  CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                        INIT_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));

  table->accum_or_assign(INIT_KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                         d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_GE(table->capacity(), INIT_CAPACITY * 2);

  expected_size = 0;
  for (int i = 0; i < INIT_KEY_NUM; i++)
    expected_size += (h_accum_or_assigns[i] ? 0 : 1);

  total_size = table->size(stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  ASSERT_EQ(total_size, expected_size);
  ASSERT_EQ(table->capacity(), (INIT_CAPACITY * 2));

  CUDA_CHECK(cudaMemcpy(h_accum_or_assigns_init, h_accum_or_assigns,
                        KEY_NUM * sizeof(bool), cudaMemcpyHostToHost));
  test_util::create_random_bools<K>(h_accum_or_assigns, KEY_NUM, true_ratio);
  CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                        KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));

  table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                         d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  expected_size = 0;
  for (int i = 0; i < KEY_NUM; i++) {
    if (i < INIT_KEY_NUM) {
      if (h_accum_or_assigns_init[i]) {
        if (h_accum_or_assigns[i]) {
        } else {
          expected_size++;
          expected_values[h_keys[i]] = static_cast<float>(h_keys[i] * 0.00001);
        }
      } else {
        expected_size++;
        if (h_accum_or_assigns[i]) {
          expected_values[h_keys[i]] = static_cast<float>(h_keys[i] * 0.00002);
        } else {
          expected_values[h_keys[i]] = static_cast<float>(h_keys[i] * 0.00001);
        }
      }
    }
    if (i >= INIT_KEY_NUM && (!h_accum_or_assigns[i])) {
      expected_size++;
      expected_values[h_keys[i]] = static_cast<float>(h_keys[i] * 0.00001);
    }
  }

  total_size = table->size(stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  ASSERT_EQ(total_size, expected_size);
  ASSERT_GE(table->capacity(), KEY_NUM * 2);

  dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                     d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(dump_counter, expected_size);

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_scores, 0, KEY_NUM * sizeof(S)));
  table->find(KEY_NUM, d_keys, d_vectors, d_found, d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int found_num = 0;

  CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMemset(h_scores, 0, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_scores, d_scores, KEY_NUM * sizeof(S),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) {
      found_num++;
      ASSERT_EQ(h_scores[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j], expected_values[h_keys[i]]);
      }
    }
  }
  ASSERT_EQ(found_num, expected_size);

  table->clear(stream);
  total_size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(total_size, 0);
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns_init));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_accum_or_assigns));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}
//
// void test_dynamic_rehash_on_multi_threads(size_t max_hbm_for_vectors) {
//  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
//  constexpr uint64_t INIT_CAPACITY = 4 * 1024;
//  constexpr uint64_t MAX_CAPACITY = 16 * 1024 * INIT_CAPACITY;
//  constexpr uint64_t KEY_NUM = 256;
//  constexpr uint64_t THREAD_N = 8;
//
//  std::vector<std::thread> threads;
//
//  TableOptions options;
//
//  options.init_capacity = INIT_CAPACITY;
//  options.max_capacity = MAX_CAPACITY;
//  options.dim = DIM;
//  options.max_load_factor = 0.50f;
//  options.max_bucket_size = BUCKET_MAX_SIZE;
//  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
//  using Table = nv::merlin::HashTable<K, V, S,
//  EvictStrategy::kLru>;
//
//  std::shared_ptr<Table> table = std::make_shared<Table>();
//  table->init(options);
//
//  auto worker_function = [&table, KEY_NUM, options](int task_n) {
//    K* h_keys;
//    V* h_vectors;
//    bool* h_found;
//
//    size_t current_capacity = table->capacity();
//
//    CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
//    CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
//    CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));
//
//    K* d_keys;
//    V* d_vectors;
//    bool* d_found;
//
//    CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
//    CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
//    CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
//
//    cudaStream_t stream;
//    CUDA_CHECK(cudaStreamCreate(&stream));
//
//    while (table->capacity() < MAX_CAPACITY) {
//      test_util::create_random_keys<K, S, V, DIM>(h_keys, nullptr, h_vectors,
//                                                  KEY_NUM);
//      CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
//                            cudaMemcpyHostToDevice));
//      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
//                            KEY_NUM * sizeof(V) * options.dim,
//                            cudaMemcpyHostToDevice));
//      CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));
//
//      table->accum_or_assign(KEY_NUM, d_keys, d_vectors, nullptr, stream);
//      CUDA_CHECK(cudaStreamSynchronize(stream));
//
//      CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
//      table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
//
//      CUDA_CHECK(cudaStreamSynchronize(stream));
//      int found_num = 0;
//
//      CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
//      CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
//      CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
//                            cudaMemcpyDeviceToHost));
//      CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
//                            cudaMemcpyDeviceToHost));
//
//      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
//                            KEY_NUM * sizeof(V) * options.dim,
//                            cudaMemcpyDeviceToHost));
//      for (int i = 0; i < KEY_NUM; i++) {
//        if (h_found[i]) {
//          found_num++;
//          for (int j = 0; j < options.dim; j++) {
//            ASSERT_EQ(h_vectors[i * options.dim + j],
//                      static_cast<float>(h_keys[i] * 0.00001));
//          }
//        }
//      }
//      ASSERT_EQ(found_num, KEY_NUM);
//      if (task_n == 0 && current_capacity != table->capacity()) {
//        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
//                     "changed from "
//                  << current_capacity << " to " << table->capacity()
//                  << std::endl;
//        current_capacity = table->capacity();
//      }
//      CUDA_CHECK(cudaStreamSynchronize(stream));
//    }
//    CUDA_CHECK(cudaStreamDestroy(stream));
//
//    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
//                          KEY_NUM * sizeof(V) * options.dim,
//                          cudaMemcpyDeviceToHost));
//
//    CUDA_CHECK(cudaFreeHost(h_keys));
//    CUDA_CHECK(cudaFreeHost(h_found));
//    CUDA_CHECK(cudaFreeHost(h_vectors));
//
//    CUDA_CHECK(cudaFree(d_keys));
//    CUDA_CHECK(cudaFree(d_vectors));
//    CUDA_CHECK(cudaFree(d_found));
//    CUDA_CHECK(cudaDeviceSynchronize());
//
//    CudaCheckError();
//  };
//
//  for (int i = 0; i < THREAD_N; ++i)
//    threads.emplace_back(std::thread(worker_function, i));
//
//  for (auto& th : threads) {
//    th.join();
//  }
//  ASSERT_EQ(table->capacity(), MAX_CAPACITY);
//}
//
void test_export_batch_if(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr float true_ratio = 0.6;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_accum_or_assigns;
  size_t h_dump_counter = 0;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  bool* d_found;
  bool* d_accum_or_assigns;
  size_t* d_dump_counter;
  int found_num = 0;
  bool* h_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));

  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  K pattern = 100;
  S threshold = test_util::host_nano<S>(stream);

  for (int i = 0; i < TEST_TIMES; i++) {
    test_util::create_random_bools<K>(h_accum_or_assigns, KEY_NUM, true_ratio);
    test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, h_vectors,
                                                KEY_NUM);

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                          KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                           nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t expected_size = 0;
    for (size_t i = 0; i < KEY_NUM; i++) {
      if (!h_accum_or_assigns[i]) expected_size++;
    }

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, expected_size);

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    found_num = 0;
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<float>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, expected_size);

    table->template export_batch_if<ExportIfPredFunctor>(
        pattern, threshold, table->capacity(), 0, d_dump_counter, d_keys,
        d_vectors, d_scores, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&h_dump_counter, d_dump_counter, sizeof(size_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, h_dump_counter * sizeof(S),
                          cudaMemcpyDeviceToHost));

    size_t expected_export_count = 0;
    for (int i = 0; i < h_dump_counter; i++) {
      if (h_scores[i] > threshold) expected_export_count++;
    }
    ASSERT_EQ(expected_export_count, h_dump_counter);

    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_dump_counter; i++) {
      ASSERT_GT(h_scores[i], threshold);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<float>(h_keys[i] * 0.00001));
      }
    }

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));
  CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_accum_or_assigns));
  CUDA_CHECK(cudaFree(d_dump_counter));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_lru_basic(size_t max_hbm_for_vectors) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr float true_ratio = 0.5;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));
      S start_ts = test_util::host_nano<S>(stream);
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts = test_util::host_nano<S>(stream);

      size_t total_size = table->size(stream);
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        ASSERT_GE(h_scores_temp[i], start_ts);
        ASSERT_LE(h_scores_temp[i], end_ts);
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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
      S start_ts = test_util::host_nano<S>(stream);
      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts = test_util::host_nano<S>(stream);

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if ((h_keys_base.end() == std::find(h_keys_base.begin(),
                                            h_keys_base.end(),
                                            h_keys_test[i])) &&
            !h_accum_or_assigns_test[i])
          expected_size++;
      }
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        bool is_accum = (h_keys_temp[i] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]));
        if (is_accum) {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00002));
          }
        } else {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00001));
          }
        }
        if (is_accum || (is_new_insert && (h_keys_temp[i] != h_keys_test[3]))) {
          ASSERT_GE(h_scores_temp[i], start_ts);
          ASSERT_LE(h_scores_temp[i], end_ts);
        } else {
          ASSERT_LE(h_scores_temp[i], start_ts);
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_lfu_basic(size_t max_hbm_for_vectors, int key_start) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 1024;
  constexpr float true_ratio = 0.5;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLfu>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  int freq_range = 1000;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, true_ratio);

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, true_ratio);

  for (int i = 0; i < TEST_TIMES; i++) {
    test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
        BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
        freq_range);

    test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
        TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFD, freq_range);

    h_accum_or_assigns_base[72] = false;
    h_accum_or_assigns_base[73] = false;

    h_keys_test[2] = h_keys_base[72];
    h_keys_test[3] = h_keys_base[73];

    h_accum_or_assigns_test[2] = true;
    h_accum_or_assigns_test[3] = false;

    h_scores_test[2] = h_keys_base[72] % freq_range;
    h_scores_test[3] = h_keys_base[73] % freq_range;

    for (int i = 0; i < options.dim; i++) {
      h_vectors_test[2 * options.dim + i] =
          h_vectors_base[72 * options.dim + i];
      h_vectors_test[3 * options.dim + i] =
          h_vectors_base[73 * options.dim + i];
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t total_size = 0;
    size_t dump_counter = 0;
    S global_epoch = 1;
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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if ((h_keys_base.end() == std::find(h_keys_base.begin(),
                                            h_keys_base.end(),
                                            h_keys_test[i])) &&
            !h_accum_or_assigns_test[i])
          expected_size++;
      }
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        bool is_accum = (h_keys_temp[i] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]));

        if (is_accum) {
          ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % freq_range) * 2);
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00002));
          }
        } else {
          ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % freq_range));
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00001));
          }
        }
      }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_epochlru_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
  constexpr int RSHIFT_ON_NANO = 20;

  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr float true_ratio = 0.5;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLru>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));
      S start_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;

      size_t total_size = table->size(stream);
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        ASSERT_GE(h_scores_temp[i] & 0xFFFFFFFF, start_ts);
        ASSERT_LE(h_scores_temp[i] & 0xFFFFFFFF, end_ts);
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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
      S start_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      S end_ts =
          (test_util::host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if ((h_keys_base.end() == std::find(h_keys_base.begin(),
                                            h_keys_base.end(),
                                            h_keys_test[i])) &&
            !h_accum_or_assigns_test[i])
          expected_size++;
      }
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        bool is_accum = (h_keys_temp[i] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]));
        if (is_accum) {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00002));
          }
        } else {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00001));
          }
        }
        if (is_accum || (is_new_insert && (h_keys_temp[i] != h_keys_test[3]))) {
          ASSERT_GE(h_scores_temp[i] & 0xffffffff, start_ts);
          ASSERT_LE(h_scores_temp[i] & 0xffffffff, end_ts);
          ASSERT_EQ(h_scores_temp[i] >> 32 & 0xffffffff, global_epoch);
        } else {
          ASSERT_LE(h_scores_temp[i] & 0xffffffff, start_ts);
          ASSERT_EQ(h_scores_temp[i] >> 32 & 0xffffffff, global_epoch - 1);
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_epochlfu_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 1024;
  constexpr float true_ratio = 0.5;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLfu>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  int freq_range = 1000;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, true_ratio);

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, true_ratio);

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);

  test_util::create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  h_accum_or_assigns_base[71] = false;
  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;

  // Simulate overflow of low 32bits.
  h_scores_base[71] = static_cast<S>(std::numeric_limits<uint32_t>::max() -
                                     static_cast<uint32_t>(1));

  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_accum_or_assigns_test[1] = true;
  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if ((h_keys_base.end() == std::find(h_keys_base.begin(),
                                            h_keys_base.end(),
                                            h_keys_test[i])) &&
            !h_accum_or_assigns_test[i])
          expected_size++;
      }
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

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

        bool is_accum = (h_keys_temp[i] == h_keys_test[1] ||
                         h_keys_temp[i] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]));

        if (is_accum) {
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
                global_epoch, h_scores_base[71] * 2);
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
                    static_cast<float>(h_keys_temp[i] *
                                       (is_accum ? 0.00002 : 0.00001)))
              << ",i=" << i << ",is_accum=" << is_accum;
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_basic(size_t max_hbm_for_vectors,
                                          int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 128;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr float true_ratio = 0.3;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  std::vector<uint8_t> h_found_temp(TEMP_KEY_NUM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;
  bool* d_found_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, true_ratio);
  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, true_ratio);

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
    //    h_scores_test[i] = h_scores_base[i];
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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        S expected_score = 0ul;
        bool is_accum = false;
        for (int j = 0; j < BASE_KEY_NUM; j++) {
          if (h_keys_base[j] == h_keys_temp[i]) {
            expected_score = h_scores_base[j];
            is_accum = h_accum_or_assigns_base[j];
          }
        }
        ASSERT_FALSE(is_accum);
        ASSERT_EQ(expected_score, h_scores_temp[i]);
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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
      table->find(TEST_KEY_NUM, d_keys_temp, d_vectors_temp, d_found_temp,
                  nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemcpy(h_found_temp.data(), d_found_temp,
                            TEST_KEY_NUM * sizeof(bool),
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t expected_size = 0;
      for (int i = 0; i < BASE_KEY_NUM; i++) {
        if (!h_accum_or_assigns_base[i]) expected_size++;
      }

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if ((h_keys_base.end() == std::find(h_keys_base.begin(),
                                            h_keys_base.end(),
                                            h_keys_test[i])) &&
            !h_accum_or_assigns_test[i])
          expected_size++;
      }
      expected_size = std::min(expected_size, BUCKET_MAX_SIZE);

      // Some keys in base could be evicted in one operation that allows the
      // same key with `assign` flag in the test can be inserted.
      ASSERT_GE(total_size, expected_size);
      ASSERT_LE(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, total_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            TEMP_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < dump_counter; i++) {
        S expected_score = 888ul;
        S base_score = 888ul;
        S test_score = 888ul;
        bool is_accum_test = false;
        bool is_found_on_base = false;
        for (int j = 0; j < BASE_KEY_NUM; j++) {
          if (h_keys_base[j] == h_keys_temp[i]) {
            is_found_on_base = !h_accum_or_assigns_base[j];
            base_score = h_scores_base[j];
            break;
          }
        }
        for (int j = 0; j < TEST_KEY_NUM; j++) {
          if (h_keys_test[j] == h_keys_temp[i]) {
            is_accum_test = h_accum_or_assigns_test[j];
            test_score = h_scores_test[j];
            break;
          }
        }
        if (is_found_on_base && is_accum_test) expected_score = test_score;
        if (is_found_on_base && !is_accum_test) expected_score = base_score;
        if (!is_found_on_base && is_accum_test) assert(false);
        if (!is_found_on_base && !is_accum_test) expected_score = test_score;

        // Some keys in base could be evicted in one operation that allows the
        // same key with `assign` flag in the test can be inserted.
        ASSERT_EQ(expected_score, h_scores_temp[i])
            << " " << is_found_on_base << " " << is_accum_test << " "
            << base_score << " " << test_score;
        if (is_found_on_base && is_accum_test) {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00002));
          }
        } else {
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                      static_cast<float>(h_keys_temp[i] * 0.00001));
          }
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));
  CUDA_CHECK(cudaFree(d_found_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors,
                                             int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 256;
  constexpr float base_true_ratio = 0.0f;
  constexpr float test_true_ratio = 0.5f;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, TEMP_KEY_NUM * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, TEMP_KEY_NUM * sizeof(bool)));

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                                    BASE_KEY_NUM, base_true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  const S base_score_start = 1000;
  for (int i = 0; i < BASE_KEY_NUM; i++) {
    h_scores_base[i] = base_score_start + i;
  }

  test_util::create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                                    TEST_KEY_NUM, test_true_ratio);
  test_util::create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];

  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;
  h_accum_or_assigns_base[74] = false;
  h_accum_or_assigns_base[75] = false;
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

  h_accum_or_assigns_test[0] = false;
  h_accum_or_assigns_test[1] = false;
  h_accum_or_assigns_test[2] = false;
  h_accum_or_assigns_test[3] = false;

  h_accum_or_assigns_test[4] = true;
  h_accum_or_assigns_test[5] = true;
  h_accum_or_assigns_test[6] = true;
  h_accum_or_assigns_test[7] = false;

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
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base.data(),
                     BASE_KEY_NUM * sizeof(uint8_t), cudaMemcpyHostToDevice));
      table->accum_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t expected_size = 0;
      for (const auto accum : h_accum_or_assigns_base) {
        if (!accum) expected_size++;
      }

      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp.data(), d_scores_temp,
                            BASE_KEY_NUM * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::vector<S> h_scores_temp_sorted(h_scores_temp);
      std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());

      for (int i = 0; i < dump_counter; i++) {
        S expected_score = 0ul;
        for (int j = 0; j < BASE_KEY_NUM; j++) {
          if (h_keys_temp[i] == h_keys_base[j]) {
            expected_score = h_scores_base[j];
            break;
          }
        }
        ASSERT_EQ(h_scores_temp[i], expected_score);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      std::unordered_map<K, bool> base_found_map;
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_test.data(),
                            TEST_KEY_NUM * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(
          cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_test.data(),
                     TEST_KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));

      table->accum_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                             d_accum_or_assigns_temp, d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t expected_size = 0;
      for (const auto accum : h_accum_or_assigns_base) {
        if (!accum) expected_size++;
      }
      expected_size = std::max(expected_size, BUCKET_MAX_SIZE);
      ASSERT_EQ(total_size, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_scores_temp, stream);
      ASSERT_EQ(dump_counter, expected_size);

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
          ASSERT_NE(h_scores_temp[i], h_scores_test[7]);
        bool is_accum =
            (h_keys_temp[i] != h_keys_test[7]) &&
            (h_keys_test.end() != std::find(h_keys_test.begin() + 4,
                                            h_keys_test.end(), h_keys_temp[i]));
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] *
                                       (is_accum ? 0.00002 : 0.00001)));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors,
                                                 int key_start = 0) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  float expected_correct_rate = 0.964;
  const int rounds = 3;
  constexpr float true_ratio = 0.0;

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
  bool* h_accum_or_assigns_base = test_util::HostBuffer<bool>(BATCH_SIZE).ptr();

  K* h_keys_temp = test_util::HostBuffer<K>(MAX_CAPACITY).ptr();
  S* h_scores_temp = test_util::HostBuffer<S>(MAX_CAPACITY).ptr();
  V* h_vectors_temp =
      test_util::HostBuffer<V>(MAX_CAPACITY * options.dim).ptr();

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;
  bool* d_accum_or_assigns_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, MAX_CAPACITY * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_temp, MAX_CAPACITY * sizeof(bool)));

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
        test_util::create_random_bools<K>(h_accum_or_assigns_base, BATCH_SIZE,
                                          true_ratio);
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
        CUDA_CHECK(cudaMemcpy(d_accum_or_assigns_temp, h_accum_or_assigns_base,
                              BATCH_SIZE * sizeof(bool),
                              cudaMemcpyHostToDevice));
        table->accum_or_assign(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                               d_accum_or_assigns_temp, d_scores_temp, stream);
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
  CUDA_CHECK(cudaFree(d_accum_or_assigns_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_dynamic_rehash_on_multi_threads(size_t max_hbm_for_vectors,
                                          int key_start = 0) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = 4 * 1024 - BUCKET_MAX_SIZE - 1;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 256;
  constexpr uint64_t THREAD_N = 8;

  std::vector<std::thread> threads;

  TableOptions options;

  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_load_factor = 0.50f;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;

  std::shared_ptr<Table> table = std::make_shared<Table>();
  table->init(options);
  ASSERT_EQ(table->bucket_count(), 32);

  auto worker_function = [&table, KEY_NUM, options](int task_n) {
    constexpr float true_ratio = 0.5;
    K* h_keys;
    V* h_vectors;
    bool* h_found;
    bool* h_accum_or_assigns;
    K* h_keys_temp;
    V* h_vectors_temp;
    bool* h_found_temp;

    size_t current_capacity = table->capacity();

    CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMallocHost(&h_keys_temp, KEY_NUM * sizeof(K)));
    CUDA_CHECK(
        cudaMallocHost(&h_vectors_temp, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMallocHost(&h_found_temp, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMallocHost(&h_accum_or_assigns, KEY_NUM * sizeof(bool)));

    K* d_keys;
    V* d_vectors;
    bool* d_found;
    bool* d_accum_or_assigns;

    CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_accum_or_assigns, KEY_NUM * sizeof(bool)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    while (table->capacity() * 2 < MAX_CAPACITY) {
      test_util::create_random_bools<K>(h_accum_or_assigns, KEY_NUM,
                                        true_ratio);

      test_util::create_random_keys<K, S, V, DIM>(h_keys, nullptr, h_vectors,
                                                  KEY_NUM);
      CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_accum_or_assigns, h_accum_or_assigns,
                            KEY_NUM * sizeof(bool), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

      table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemcpy(h_keys_temp, d_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp, d_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_found_temp, d_found, KEY_NUM * sizeof(bool),
                            cudaMemcpyDeviceToHost));

      table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_accum_or_assigns,
                             nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
      table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      int found_num = 0;
      size_t expected_size = 0;
      for (int i = 0; i < KEY_NUM; i++) {
        if (h_found_temp[i] || !h_accum_or_assigns[i]) expected_size++;
      }

      CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
      CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
      CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < KEY_NUM; i++) {
        if (h_found[i]) {
          found_num++;
          for (int j = 0; j < options.dim; j++) {
            if (h_found_temp[i] && h_accum_or_assigns[i]) {
              ASSERT_EQ(h_vectors[i * options.dim + j],
                        h_vectors_temp[i * options.dim + j] +
                            static_cast<float>(h_keys[i] * 0.00001));
            } else {
              ASSERT_EQ(h_vectors[i * options.dim + j],
                        static_cast<float>(h_keys[i] * 0.00001));
            }
          }
        }
      }
      ASSERT_EQ(found_num, expected_size);
      if (task_n == 0 && current_capacity != table->capacity()) {
        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
                     "changed from "
                  << current_capacity << " to " << table->capacity()
                  << std::endl;
        current_capacity = table->capacity();
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_found));
    CUDA_CHECK(cudaFreeHost(h_vectors));
    CUDA_CHECK(cudaFreeHost(h_keys_temp));
    CUDA_CHECK(cudaFreeHost(h_found_temp));
    CUDA_CHECK(cudaFreeHost(h_vectors_temp));
    CUDA_CHECK(cudaFreeHost(h_accum_or_assigns));

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_accum_or_assigns));
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaCheckError();
  };

  for (int i = 0; i < THREAD_N; ++i)
    threads.emplace_back(std::thread(worker_function, i));

  for (auto& th : threads) {
    th.join();
  }
  ASSERT_GE(table->capacity() * 2, MAX_CAPACITY);
}

template <typename K, typename V, typename S, typename Table, size_t dim = 64>
void CheckAccumOrAssignValues(Table* table,
                              test_util::KVMSBuffer<K, V, S>& data_buffer,
                              size_t len, cudaStream_t stream) {
  std::map<K, test_util::ValueArray<V, dim>> map_before_insert;
  std::map<K, test_util::ValueArray<V, dim>> map_after_insert;
  std::unordered_set<K> map_current_batch;

  K* keys = data_buffer.keys_ptr();
  V* values = data_buffer.values_ptr();
  S* scores = data_buffer.scores_ptr();

  for (int i = 0; i < len; i++) {
    map_current_batch.insert(data_buffer.keys_ptr(false)[i]);
  }

  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  S* h_tmp_scores = nullptr;
  bool* h_tmp_accum_or_assigns = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  S* d_tmp_scores = nullptr;
  bool* d_tmp_accum_or_assigns = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  h_tmp_keys = (K*)malloc(cap * sizeof(K));
  h_tmp_values = (V*)malloc(cap * dim * sizeof(V));
  h_tmp_scores = (S*)malloc(cap * sizeof(S));
  h_tmp_accum_or_assigns = (bool*)malloc(cap * sizeof(bool));

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_scores, cap * sizeof(S), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_scores, 0, cap * sizeof(S), stream));
  CUDA_CHECK(
      cudaMallocAsync(&d_tmp_accum_or_assigns, cap * sizeof(bool), stream));
  CUDA_CHECK(
      cudaMemsetAsync(d_tmp_accum_or_assigns, 0, cap * sizeof(bool), stream));

  table->find(len, keys, d_tmp_values, d_tmp_accum_or_assigns, nullptr, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_accum_or_assigns, d_tmp_accum_or_assigns,
                             len * sizeof(bool), cudaMemcpyDeviceToHost,
                             stream));

  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
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

  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < table_size_verify0; i++) {
    test_util::ValueArray<V, dim>* vec =
        reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values +
                                                         i * dim);
    map_before_insert[h_tmp_keys[i]] = *vec;
  }

  auto start = std::chrono::steady_clock::now();
  table->accum_or_assign(len, keys, values, d_tmp_accum_or_assigns, nullptr,
                         stream);
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
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_accum_or_assigns, d_tmp_accum_or_assigns,
                             table_size_after * sizeof(bool),
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
    bool existed =
        (map_before_insert.end() != map_before_insert.find(it.first));
    bool required =
        (map_current_batch.end() != map_current_batch.find(it.first));
    V expected_value = 0;

    if (existed) {
      if (required) {
        expected_value = (map_before_insert.at(it.first)[0] +
                          static_cast<V>(it.first * 0.00001));
      } else {
        expected_value = map_before_insert.at(it.first)[0];
      }
    } else {
      if (required) {
        expected_value = static_cast<V>(it.first * 0.00001);
      }
    }
    for (size_t j = 0; j < dim; j++) {
      if (vec[j] != expected_value) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  ASSERT_EQ(value_diff_cnt, 0);
  std::cout << "Check accum_or_assign behavior got "
            << "value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;

  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_scores, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_accum_or_assigns, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_scores);
  free(h_tmp_accum_or_assigns);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void test_accum_or_assign_values_check(size_t max_hbm_for_vectors) {
  const size_t U = 524288;
  const size_t init_capacity = 1024;
  const size_t B = 524288 + 13;
  constexpr size_t dim = 64;

  TableOptions opt;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
  opt.dim = dim;

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

    CheckAccumOrAssignValues<K, V, S, Table, dim>(table.get(), data_buffer, B,
                                                  stream);

    offset += B;
    score += 1;
  }
}

TEST(AccumOrAssignTest, test_export_batch_if) {
  test_export_batch_if(16, 22);
  test_export_batch_if(0, 0);
}
TEST(AccumOrAssignTest, test_basic_when_full) {
  test_basic_when_full(16, 2);
  test_basic_when_full(0, 0);
}
TEST(AccumOrAssignTest, test_erase_if_pred) {
  test_erase_if_pred(16, 0);
  test_erase_if_pred(0, 5);
}
TEST(AccumOrAssignTest, test_rehash) {
  test_rehash(16, 7);
  test_rehash(0, 0);
}
TEST(AccumOrAssignTest, test_rehash_on_big_batch) {
  test_rehash_on_big_batch(16, 9);
  test_rehash_on_big_batch(0, 0);
}
TEST(AccumOrAssignTest, test_dynamic_rehash_on_multi_threads) {
  test_dynamic_rehash_on_multi_threads(16, 56);
  test_dynamic_rehash_on_multi_threads(0);
}
TEST(AccumOrAssignTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic(16);
  test_evict_strategy_lru_basic(0);
}
TEST(AccumOrAssignTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic(16, 3);
  // TODO(rhdong): Add back when diff error issue fixed in hybrid mode.
  // test_evict_strategy_lfu_basic(0);
}

TEST(AccumOrAssignTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_epochlru_basic(16, 33);
  test_evict_strategy_epochlru_basic(0);
}

TEST(AccumOrAssignTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_epochlfu_basic(16);
  test_evict_strategy_epochlfu_basic(0, 44);
}
TEST(AccumOrAssignTest, test_evict_strategy_customized_basic) {
  test_evict_strategy_customized_basic(16);
  test_evict_strategy_customized_basic(0, 23);
}
TEST(AccumOrAssignTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced(16, 16);
  test_evict_strategy_customized_advanced(0);
}
TEST(AccumOrAssignTest, test_evict_strategy_customized_correct_rate) {
  // TODO(rhdong): after blossom CI issue is resolved, the skip logic.
  const bool skip_hmem_check = (nullptr != std::getenv("IS_BLOSSOM_CI"));
  test_evict_strategy_customized_correct_rate(16, 61);
  if (!skip_hmem_check) {
    test_evict_strategy_customized_correct_rate(0);
  } else {
    std::cout << "The HMEM check is skipped in blossom CI!" << std::endl;
  }
}

TEST(AccumOrAssignTest, test_accum_or_assign_values_check) {
  test_accum_or_assign_values_check(16);
  // TODO(rhdong): Add back when diff error issue fixed in hybrid mode.
  // test_insert_or_assign_values_check(0);
}