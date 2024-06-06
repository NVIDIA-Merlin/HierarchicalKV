/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;

void test_find(size_t max_hbm_for_vectors, size_t max_bucket_size,
               double load_factor, bool pipeline_lookup, int key_start = 0) {
  MERLIN_CHECK(load_factor >= 0.0 && load_factor <= 1.0,
               "Invalid `load_factor`");

  constexpr uint64_t INIT_CAPACITY = 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  K* h_missed_keys;
  int* h_missed_indices;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::MB(max_hbm_for_vectors);
  if (pipeline_lookup) {
    options.max_bucket_size = 128;
  } else {
    options.max_bucket_size = 256;
  }
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_missed_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_missed_indices, KEY_NUM * sizeof(int)));

  K* d_keys;
  S* d_scores;
  V* d_vectors;
  K* d_missed_keys;
  int* d_missed_indices;
  int* d_missed_size;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_missed_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_missed_indices, KEY_NUM * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_missed_size, sizeof(int)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int missed_size;
  for (int i = 0; i < TEST_TIMES; ++i) {
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, h_vectors,
                                                KEY_NUM);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));

    Table table;
    table.init(options);
    size_t size = table.size(stream);
    ASSERT_EQ(size, 0);

    size_t insert_num = (double)KEY_NUM * load_factor;
    table.insert_or_assign(insert_num, d_keys, d_vectors, d_scores, stream);
    table.find(KEY_NUM, d_keys, d_vectors, d_missed_keys, d_missed_indices,
               d_missed_size, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&missed_size, d_missed_size, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_missed_keys, d_missed_keys, missed_size * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_missed_indices, d_missed_indices,
                          missed_size * sizeof(int), cudaMemcpyDeviceToHost));

    if (insert_num == 0) {
      ASSERT_EQ(missed_size, KEY_NUM);
    } else {
      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));

      ASSERT_TRUE(missed_size > 0 && missed_size < KEY_NUM);
      std::vector<bool> founds(KEY_NUM, true);
      // Check missed
      for (int j = 0; j < missed_size; ++j) {
        int idx = h_missed_indices[i];
        ASSERT_TRUE(idx >= 0 && idx < KEY_NUM);
        ASSERT_EQ(h_keys[idx], h_missed_keys[i]);
        founds[idx] = false;
      }
      // Check hitted
      for (uint64_t j = 0; j < KEY_NUM; ++j) {
        if (founds[j]) {
          for (int k = 0; k < options.dim; ++k) {
            ASSERT_EQ(h_vectors[j * options.dim + k],
                      static_cast<float>(h_keys[j] * 0.00001));
          }
        }
      }
    }
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_vectors));
  CUDA_CHECK(cudaFreeHost(h_missed_keys));
  CUDA_CHECK(cudaFreeHost(h_missed_indices));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_missed_keys));
  CUDA_CHECK(cudaFree(d_missed_indices));
  CUDA_CHECK(cudaFree(d_missed_size));

  CudaCheckError();
}

TEST(FindTest, test_find_when_empty) {
  // pure HMEM
  test_find(0, 128, 0.0, true, 12);
  test_find(0, 256, 0.0, false);
  // hybrid
  test_find(32, 128, 0.0, true, 58);
  test_find(32, 256, 0.0, false);
  // pure HBM
  test_find(1024, 128, 0.0, true);
  test_find(1024, 256, 0.0, false, 12);
}

TEST(FindTest, test_find_when_full) {
  // pure HMEM
  test_find(0, 128, 1.0, true);
  test_find(0, 256, 1.0, false);
  // hybrid
  test_find(32, 128, 1.0, true);
  test_find(32, 256, 1.0, false, 60);
  // pure HBM
  test_find(1024, 128, 1.0, true);
  test_find(1024, 256, 1.0, false);
}

TEST(FindTest, test_find_load_factor) {
  // pure HMEM
  test_find(0, 128, 0.2, true, 45);
  test_find(0, 256, 0.2, false, 12);
  // hybrid
  test_find(32, 128, 0.2, true, 27);
  test_find(32, 256, 0.2, false, 53);
  // pure HBM
  test_find(1024, 128, 0.2, true, 9);
  test_find(1024, 256, 0.2, false, 38);

  // pure HMEM
  test_find(0, 128, 0.5, true, 21);
  test_find(0, 256, 0.5, false, 46);
  // hybrid
  test_find(32, 128, 0.5, true, 31);
  test_find(32, 256, 0.5, false, 59);
  // pure HBM
  test_find(1024, 128, 0.5, true, 4);
  test_find(1024, 256, 0.5, false, 22);

  // pure HMEM
  test_find(0, 128, 0.75, true, 11);
  test_find(0, 256, 0.75, false, 34);
  // hybrid
  test_find(32, 128, 0.75, true, 18);
  test_find(32, 256, 0.75, false, 47);
  // pure HBM
  test_find(1024, 128, 0.75, true, 7);
  test_find(1024, 256, 0.75, false, 29);
}
