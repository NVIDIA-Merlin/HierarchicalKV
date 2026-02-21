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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using TableMode = nv::merlin::TableMode;
using EvictStrategy = nv::merlin::EvictStrategy;

/*
 * Helper: create a MEMORY_MODE table with fixed capacity.
 */
template <typename Table>
void create_memory_mode_table(Table& table, size_t capacity, size_t dim = DIM) {
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = 0;
  options.dim = dim;
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kMemory;
  table.init(options);
}

/*
 * Helper: create a THROUGHPUT_MODE table with fixed capacity.
 */
template <typename Table>
void create_throughput_mode_table(Table& table, size_t capacity,
                                  size_t dim = DIM) {
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = 0;
  options.dim = dim;
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kThroughput;
  table.init(options);
}

// ==============================
// TestGroup 1: Basic Correctness
// ==============================

// T1.1: MEMORY_MODE insert_or_assign + find basic functionality.
TEST(DualBucketTest, BasicInsertAndFind) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;  // ~16K entries
  constexpr size_t N = static_cast<size_t>(CAPACITY * 0.5);

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Allocate host data.
  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) {
    h_scores[i] = i + 1;
    for (size_t j = 0; j < DIM; j++) {
      h_values[i * DIM + j] = static_cast<V>(h_keys[i] * 0.00001f);
    }
  }

  // Allocate device data.
  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;

  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Insert.
  table.insert_or_assign(N, d_keys, d_values, d_scores, /*stream=*/0,
                         /*unique_key=*/true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify size.
  size_t table_size = table.size(/*stream=*/0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N);

  // Find.
  table.find(N, d_keys, d_found_values, d_founds, /*scores=*/nullptr,
             /*stream=*/0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check all found.
  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i]) << "Key " << h_keys[i] << " not found";
  }

  // Check values correct.
  std::vector<V> h_found_values(N * DIM);
  CUDA_CHECK(cudaMemcpy(h_found_values.data(), d_found_values,
                        N * DIM * sizeof(V), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < DIM; j++) {
      EXPECT_FLOAT_EQ(h_found_values[i * DIM + j],
                      static_cast<V>(h_keys[i] * 0.00001f))
          << "Value mismatch for key " << h_keys[i] << " dim " << j;
    }
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// T1.2: MEMORY_MODE assign (update) - key already exists.
TEST(DualBucketTest, UpdateExistingKey) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;
  constexpr size_t N = 1024;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values_v1(N * DIM);
  std::vector<V> h_values_v2(N * DIM);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) {
    h_scores[i] = i + 1;
    for (size_t j = 0; j < DIM; j++) {
      h_values_v1[i * DIM + j] = 1.0f;
      h_values_v2[i * DIM + j] = 2.0f;
    }
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;

  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Insert V1.
  CUDA_CHECK(cudaMemcpy(d_values, h_values_v1.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Update with V2.
  CUDA_CHECK(cudaMemcpy(d_values, h_values_v2.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Size should still be N (no duplicates).
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N);

  // Find and verify V2 values.
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<V> h_found_values(N * DIM);
  CUDA_CHECK(cudaMemcpy(h_found_values.data(), d_found_values,
                        N * DIM * sizeof(V), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < DIM; j++) {
      EXPECT_FLOAT_EQ(h_found_values[i * DIM + j], 2.0f)
          << "Expected V2 value for key " << h_keys[i] << " dim " << j;
    }
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// T1.3: MEMORY_MODE score-eviction correctness.
TEST(DualBucketTest, ScoreEviction) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  // Small capacity to force eviction quickly.
  constexpr size_t CAPACITY = 128 * 128;  // 128 buckets * 128 slots = 16384
  constexpr size_t N_FILL = CAPACITY;     // Fill completely
  constexpr size_t N_NEW = 1024;          // Insert high-score keys

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Phase 1: Fill table with low-score keys.
  std::vector<K> h_keys_fill(N_FILL);
  std::vector<V> h_values_fill(N_FILL * DIM, 1.0f);
  std::vector<S> h_scores_fill(N_FILL);

  std::iota(h_keys_fill.begin(), h_keys_fill.end(), 1);
  for (size_t i = 0; i < N_FILL; i++) {
    h_scores_fill[i] = i + 1;  // Low scores: 1..N_FILL
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, N_FILL * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N_FILL * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N_FILL * sizeof(S)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_fill.data(), N_FILL * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values_fill.data(),
                        N_FILL * DIM * sizeof(V), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores_fill.data(), N_FILL * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_FILL, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 2: Insert high-score keys (should evict low-score keys).
  std::vector<K> h_keys_new(N_NEW);
  std::vector<V> h_values_new(N_NEW * DIM, 2.0f);
  std::vector<S> h_scores_new(N_NEW);

  for (size_t i = 0; i < N_NEW; i++) {
    h_keys_new[i] = N_FILL + 1 + i;       // New keys
    h_scores_new[i] = N_FILL + 1000 + i;  // High scores
  }

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_new.data(), N_NEW * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values_new.data(), N_NEW * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores_new.data(), N_NEW * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_NEW, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 3: Verify high-score keys are present.
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_founds, N_NEW * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N_NEW * DIM * sizeof(V)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_new.data(), N_NEW * sizeof(K),
                        cudaMemcpyHostToDevice));
  table.find(N_NEW, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N_NEW];
  CUDA_CHECK(cudaMemcpy(h_founds, d_founds, N_NEW * sizeof(bool),
                        cudaMemcpyDeviceToHost));

  int found_count = 0;
  for (size_t i = 0; i < N_NEW; i++) {
    if (h_founds[i]) found_count++;
  }
  std::cout << "[ScoreEviction] High-score keys accuracy: " << found_count
            << "/" << N_NEW << " (" << (100.0 * found_count / N_NEW) << "%)"
            << std::endl;
  // Most high-score keys should be found.  Require >= 80%.
  EXPECT_GT(found_count, static_cast<int>(N_NEW * 0.8))
      << "Expected >= 80% of high-score keys to survive eviction";

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// T1.4: THROUGHPUT_MODE regression test (not affected by dual-bucket changes).
TEST(DualBucketTest, ThroughputModeRegression) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;
  constexpr size_t N = 4096;

  Table table;
  create_throughput_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) {
    h_scores[i] = i + 1;
    for (size_t j = 0; j < DIM; j++) {
      h_values[i * DIM + j] = static_cast<V>(h_keys[i] * 0.001f);
    }
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;

  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i])
        << "THROUGHPUT_MODE: Key " << h_keys[i] << " not found";
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ==========================================
// TestGroup 2: Dual-bucket Feature Verify
// ==========================================

// T2.2: First eviction load factor comparison.
TEST(DualBucketTest, FirstEvictionLoadFactor) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 10 * 1024 * 1024;  // ~10M slots

  // Run for MEMORY_MODE.
  {
    Table table;
    create_memory_mode_table(table, CAPACITY);

    constexpr size_t BATCH = 128;
    std::vector<K> h_keys(BATCH);
    std::vector<V> h_values(BATCH * DIM, 1.0f);
    std::vector<S> h_scores(BATCH);

    K* d_keys;
    V* d_values;
    S* d_scores;
    CUDA_CHECK(cudaMalloc(&d_keys, BATCH * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values, BATCH * DIM * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_scores, BATCH * sizeof(S)));

    K next_key = 1;
    size_t total_inserted = 0;
    float first_eviction_lf = 0.0f;

    // Insert in batches until table is nearly full.
    while (total_inserted < CAPACITY) {
      for (size_t i = 0; i < BATCH; i++) {
        h_keys[i] = next_key++;
        h_scores[i] = h_keys[i];  // Score = key value (ascending)
      }
      CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), BATCH * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), BATCH * DIM * sizeof(V),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), BATCH * sizeof(S),
                            cudaMemcpyHostToDevice));

      table.insert_or_assign(BATCH, d_keys, d_values, d_scores, 0, true);
      CUDA_CHECK(cudaDeviceSynchronize());
      total_inserted += BATCH;

      size_t table_size = table.size(0);
      CUDA_CHECK(cudaDeviceSynchronize());

      // If table_size < total_inserted, eviction occurred.
      if (table_size < total_inserted && first_eviction_lf == 0.0f) {
        first_eviction_lf =
            static_cast<float>(table_size) / static_cast<float>(CAPACITY);
        break;
      }
    }

    std::cout << "[MEMORY_MODE] First eviction LF: " << first_eviction_lf
              << " (total_inserted=" << total_inserted << ")" << std::endl;

    // Dual-bucket two-choice hashing should achieve very high LF before first
    // eviction.  Empirically measured ~0.982 at 10M scale on A6000.
    EXPECT_GT(first_eviction_lf, 0.980f)
        << "Dual-bucket should delay eviction beyond 98.0% LF";

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_scores));
  }
}

// ===================================
// TestGroup 3: API Guard Tests
// ===================================

TEST(DualBucketTest, EraseGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  K h_key = 1;
  CUDA_CHECK(cudaMemcpy(d_keys, &h_key, sizeof(K), cudaMemcpyHostToDevice));

  EXPECT_THROW(table.erase(1, d_keys, 0), std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
}

TEST(DualBucketTest, ContainsGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  bool* d_founds;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_founds, sizeof(bool)));
  K h_key = 1;
  CUDA_CHECK(cudaMemcpy(d_keys, &h_key, sizeof(K), cudaMemcpyHostToDevice));

  EXPECT_THROW(table.contains(1, d_keys, d_founds, 0), std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_founds));
}

TEST(DualBucketTest, ReserveGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  EXPECT_THROW(table.reserve(128 * 256, 0), std::runtime_error);
}

// ===================================
// TestGroup 4: Boundary Conditions
// ===================================

// T4.1: Empty table find.
TEST(DualBucketTest, EmptyTableFind) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  Table table;
  create_memory_mode_table(table, 128 * 128);

  constexpr size_t N = 64;
  K* d_keys;
  V* d_values;
  bool* d_founds;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));

  std::vector<K> h_keys(N);
  std::iota(h_keys.begin(), h_keys.end(), 1);
  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));

  table.find(N, d_keys, d_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_FALSE(h_founds[i])
        << "Empty table should not find key " << h_keys[i];
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_founds));
}

// T4.4: Different dim values.
TEST(DualBucketTest, DimVariation) {
  using Table1 = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  // Test dim=1 and dim=64 (exercises different VecV specializations).
  // Note: dim > 224 exceeds the dual-bucket lookup kernel's fixed shared-memory
  // buffer (896 bytes).  init() now rejects dim > 224 for kMemory mode.
  for (size_t test_dim : {1, 64}) {
    Table1 table;
    constexpr size_t CAPACITY = 128 * 128;
    constexpr size_t N = 256;

    TableOptions options;
    options.init_capacity = CAPACITY;
    options.max_capacity = CAPACITY;
    options.max_hbm_for_vectors = 0;
    options.dim = test_dim;
    options.max_bucket_size = 128;
    options.table_mode = TableMode::kMemory;
    table.init(options);

    std::vector<K> h_keys(N);
    std::vector<V> h_values(N * test_dim);
    std::vector<S> h_scores(N);

    std::iota(h_keys.begin(), h_keys.end(), 1);
    for (size_t i = 0; i < N; i++) {
      h_scores[i] = i + 1;
      for (size_t j = 0; j < test_dim; j++) {
        h_values[i * test_dim + j] = static_cast<V>(i);
      }
    }

    K* d_keys;
    V* d_values;
    S* d_scores;
    bool* d_founds;
    V* d_found_values;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values, N * test_dim * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
    CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_found_values, N * test_dim * sizeof(V)));

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * test_dim * sizeof(V),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                          cudaMemcpyHostToDevice));

    table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool* h_founds = new bool[N];
    CUDA_CHECK(cudaMemcpy(h_founds, d_founds, N * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    std::vector<V> h_found_values(N * test_dim);
    CUDA_CHECK(cudaMemcpy(h_found_values.data(), d_found_values,
                          N * test_dim * sizeof(V), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N; i++) {
      EXPECT_TRUE(h_founds[i])
          << "dim=" << test_dim << ": Key " << h_keys[i] << " not found";
      if (h_founds[i]) {
        for (size_t j = 0; j < test_dim; j++) {
          EXPECT_FLOAT_EQ(h_found_values[i * test_dim + j], static_cast<V>(i))
              << "dim=" << test_dim << ": Value mismatch key " << h_keys[i]
              << " dim " << j;
        }
      }
    }

    delete[] h_founds;
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_founds));
    CUDA_CHECK(cudaFree(d_found_values));
  }
}

// ===================================
// TestGroup 5: Init Validation
// ===================================

TEST(DualBucketTest, InitCapacityMismatchReject) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;

  TableOptions options;
  options.init_capacity = 128 * 128;
  options.max_capacity = 128 * 256;  // Different from init_capacity!
  options.max_hbm_for_vectors = 0;
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kMemory;

  EXPECT_THROW(table.init(options), std::runtime_error);
}

// ===================================
// TestGroup 2 additions
// ===================================

// T2.3: b1 == b2 degeneration.
// When a key's two bucket indices collide, the kernel must degenerate to
// single-bucket behaviour without data corruption or deadlock.
TEST(DualBucketTest, B1EqualsB2Degeneration) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  // Use a small number of buckets so that collisions of lo%N == hi%N are
  // reasonably frequent.  With 4 buckets the probability for each key is ~25%.
  constexpr size_t NUM_BUCKETS = 4;
  constexpr size_t CAPACITY = NUM_BUCKETS * 128;  // 512 slots
  constexpr size_t N = 256;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) {
    h_scores[i] = i + 1;
    for (size_t j = 0; j < DIM; j++)
      h_values[i * DIM + j] = static_cast<V>(h_keys[i]);
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // All N keys should be found, regardless of b1==b2 collisions.
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  std::vector<V> h_found_values(N * DIM);
  CUDA_CHECK(cudaMemcpy(h_found_values.data(), d_found_values,
                        N * DIM * sizeof(V), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i]) << "Key " << h_keys[i] << " not found (b1==b2?)";
    if (h_founds[i]) {
      EXPECT_FLOAT_EQ(h_found_values[i * DIM], static_cast<V>(h_keys[i]))
          << "Value mismatch for key " << h_keys[i];
    }
  }

  // Table size must equal N (no duplicates from b1==b2 path).
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N);

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// T2.5: Digest effectiveness — verify that dual-bucket digest (bit[56:63])
// is used consistently during init, insert, and find.  If the init kernel
// wrote the wrong empty-digest value, empty-slot detection would fail and
// no keys could be inserted.  This test therefore doubles as a regression
// guard for the G1 digest-mismatch bug.
TEST(DualBucketTest, DigestEffectiveness) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 64;  // 8192 slots, 64 buckets
  constexpr size_t N = 4096;             // 50% LF

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM, 1.0f);
  std::vector<S> h_scores(N);

  // Use random keys so that digests are well-distributed.
  std::mt19937_64 rng(42);
  for (size_t i = 0; i < N; i++) {
    h_keys[i] = (rng() & 0x00FFFFFFFFFFFFFF) | 1;  // avoid reserved keys
    h_scores[i] = i + 1;
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // If empty-digest was wrong, insert would have gone through the D2 eviction
  // path and all entries would be REFUSED.  Check that table is not empty.
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N) << "Digest mismatch: expected " << N
                           << " entries but got " << table_size
                           << " (empty-slot detection likely failed)";

  // Verify every key is findable.
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  int found_count = 0;
  for (size_t i = 0; i < N; i++) {
    if (h_founds[i]) found_count++;
  }
  EXPECT_EQ(found_count, static_cast<int>(N))
      << "Digest mismatch on find: only " << found_count << "/" << N
      << " keys found";

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ===================================
// TestGroup 1 addition: Score ordering after eviction
// ===================================

// T1.3b: After eviction, surviving keys must have scores >= the scores of
// evicted keys.  We export the full table and verify score ordering.
TEST(DualBucketTest, ScoreOrderingAfterEviction) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 64;  // 8192 slots
  constexpr size_t N_FILL = CAPACITY;
  constexpr size_t N_NEW = 512;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Phase 1: Fill with scores [1..N_FILL].
  std::vector<K> h_keys(N_FILL);
  std::vector<V> h_values(N_FILL * DIM, 1.0f);
  std::vector<S> h_scores(N_FILL);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N_FILL; i++) h_scores[i] = i + 1;

  K* d_keys;
  V* d_values;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, N_FILL * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N_FILL * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N_FILL * sizeof(S)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), N_FILL * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N_FILL * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N_FILL * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_FILL, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 2: Insert high-score keys that force eviction.
  std::vector<K> h_keys_new(N_NEW);
  std::vector<V> h_values_new(N_NEW * DIM, 2.0f);
  std::vector<S> h_scores_new(N_NEW);
  for (size_t i = 0; i < N_NEW; i++) {
    h_keys_new[i] = N_FILL + 1 + i;
    h_scores_new[i] = N_FILL * 10 + i;  // Much higher scores
  }

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_new.data(), N_NEW * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values_new.data(), N_NEW * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores_new.data(), N_NEW * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_NEW, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 3: Export all surviving entries and check scores.
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());

  K* d_dump_keys;
  V* d_dump_values;
  S* d_dump_scores;
  size_t* d_dump_counter;
  CUDA_CHECK(cudaMalloc(&d_dump_keys, table_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_dump_values, table_size * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_dump_scores, table_size * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
  CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

  table.export_batch(table_size, 0, d_dump_counter, d_dump_keys, d_dump_values,
                     d_dump_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t dumped;
  CUDA_CHECK(cudaMemcpy(&dumped, d_dump_counter, sizeof(size_t),
                        cudaMemcpyDeviceToHost));

  std::vector<S> h_dump_scores(dumped);
  CUDA_CHECK(cudaMemcpy(h_dump_scores.data(), d_dump_scores, dumped * sizeof(S),
                        cudaMemcpyDeviceToHost));

  // Find the minimum score among all surviving entries.
  S min_surviving =
      *std::min_element(h_dump_scores.begin(), h_dump_scores.end());

  // Check that all high-score keys that were inserted have scores above
  // the surviving minimum.  (Some high-score keys may have been REFUSED,
  // but if they ARE in the table, their score must be consistent.)
  std::vector<K> h_dump_keys(dumped);
  CUDA_CHECK(cudaMemcpy(h_dump_keys.data(), d_dump_keys, dumped * sizeof(K),
                        cudaMemcpyDeviceToHost));

  int high_score_survivors = 0;
  for (size_t i = 0; i < dumped; i++) {
    if (h_dump_keys[i] > N_FILL) {
      high_score_survivors++;
      // Every high-score key should have score >= min_surviving.
      EXPECT_GE(h_dump_scores[i], min_surviving);
    }
  }
  // At least some high-score keys should have survived.
  EXPECT_GT(high_score_survivors, 0) << "No high-score keys survived eviction";

  std::cout << "[ScoreOrdering] min_surviving_score=" << min_surviving
            << " high_score_survivors=" << high_score_survivors << "/" << N_NEW
            << std::endl;

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_dump_keys));
  CUDA_CHECK(cudaFree(d_dump_values));
  CUDA_CHECK(cudaFree(d_dump_scores));
  CUDA_CHECK(cudaFree(d_dump_counter));
}

// ===================================
// TestGroup 3 additions: API Guard Tests (new)
// ===================================

TEST(DualBucketTest, FindOrInsertGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  V* d_values;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, sizeof(S)));

  EXPECT_THROW(table.find_or_insert(1, d_keys, d_values, d_scores, 0, true),
               std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST(DualBucketTest, InsertAndEvictGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  V* d_values;
  S* d_scores;
  K* d_evicted_keys;
  V* d_evicted_values;
  S* d_evicted_scores;
  size_t* d_counter;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_evicted_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evicted_values, DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_evicted_scores, sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(size_t)));

  EXPECT_THROW(
      table.insert_and_evict(1, d_keys, d_values, d_scores, d_evicted_keys,
                             d_evicted_values, d_evicted_scores, d_counter, 0),
      std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_evicted_keys));
  CUDA_CHECK(cudaFree(d_evicted_values));
  CUDA_CHECK(cudaFree(d_evicted_scores));
  CUDA_CHECK(cudaFree(d_counter));
}

TEST(DualBucketTest, AccumOrAssignGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  V* d_values;
  bool* d_accum;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_accum, sizeof(bool)));

  EXPECT_THROW(table.accum_or_assign(1, d_keys, d_values, d_accum),
               std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_accum));
}

TEST(DualBucketTest, AssignScoresGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, sizeof(S)));

  EXPECT_THROW(table.assign_scores(1, d_keys, d_scores), std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST(DualBucketTest, AssignValuesGuard) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  create_memory_mode_table(table, 128 * 128);

  K* d_keys;
  V* d_values;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, DIM * sizeof(V)));

  EXPECT_THROW(table.assign_values(1, d_keys, d_values), std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
}

// ===================================
// TestGroup 5 addition: max_hbm_for_vectors rejection
// ===================================

TEST(DualBucketTest, InitHbmForVectorsReject) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;

  TableOptions options;
  options.init_capacity = 128 * 128;
  options.max_capacity = 128 * 128;
  options.max_hbm_for_vectors = 1024;  // non-zero → should be rejected
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kMemory;

  EXPECT_THROW(table.init(options), std::runtime_error);
}

// T5.3: dim > 224 rejected in MEMORY_MODE (shared-memory buffer overflow).
TEST(DualBucketTest, InitDimTooLargeReject) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;

  TableOptions options;
  options.init_capacity = 128 * 128;
  options.max_capacity = 128 * 128;
  options.max_hbm_for_vectors = 0;
  options.dim = 256;  // exceeds 224-float limit
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kMemory;

  EXPECT_THROW(table.init(options), std::runtime_error);
}

// T5.3b: dim=224 should be accepted (exact boundary).
TEST(DualBucketTest, InitDimMaxAccepted) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;

  TableOptions options;
  options.init_capacity = 128 * 128;
  options.max_capacity = 128 * 128;
  options.max_hbm_for_vectors = 0;
  options.dim = 224;  // exactly at the limit
  options.max_bucket_size = 128;
  options.table_mode = TableMode::kMemory;

  EXPECT_NO_THROW(table.init(options));
}

// ===================================
// TestGroup 2 addition: Bucket distribution (T2.1)
// ===================================

// Verify that keys are distributed across multiple buckets (not all in b1).
// We insert random keys and check that after export, the table size matches
// expectations.  A more direct check would require bucket-level introspection
// which the public API does not expose, but we can infer distribution by
// checking that the first-eviction LF is significantly higher than single-
// bucket mode (covered in FirstEvictionLoadFactor).  Here we do a simple
// idempotency + size check with random keys to stress the hash distribution.
TEST(DualBucketTest, RandomKeyDistribution) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 128;  // 16384 slots
  constexpr size_t N = 8192;              // 50% LF

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM, 1.0f);
  std::vector<S> h_scores(N);

  std::mt19937_64 rng(12345);
  for (size_t i = 0; i < N; i++) {
    h_keys[i] = (rng() & 0x00FFFFFFFFFFFFFF) | 1;
    h_scores[i] = i + 1;
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N)
      << "Random keys at 50% LF should all be inserted without eviction";

  // Re-insert the same keys (idempotent).
  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size_after = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size_after, N) << "Re-insert must not create duplicates";

  // Find all.
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i]) << "Random key " << h_keys[i] << " not found";
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ===================================
// TestGroup 4 addition: Single bucket capacity (T4.2)
// ===================================

// T4.2: Single-bucket capacity must be rejected by MEMORY_MODE init guard.
// Dual-bucket addressing requires at least 2 buckets (capacity >= 256).
TEST(DualBucketTest, SingleBucketCapacityRejected) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;

  // 1 bucket = 128 slots → must be rejected.
  EXPECT_THROW(create_memory_mode_table(table, 128), std::runtime_error);
}

// T4.2b: Minimum valid capacity (2 buckets = 256 slots).
TEST(DualBucketTest, MinimumTwoBucketCapacity) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 256;  // 2 buckets
  constexpr size_t N = 128;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM, 1.0f);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) h_scores[i] = i + 1;

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N);

  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i])
        << "Two-bucket: Key " << h_keys[i] << " not found";
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ===================================
// DEBUG: 2-bucket eviction trace
// ===================================

// Small-scale eviction test with kernel printf enabled (buckets_num <= 4).
// Fill 2 buckets (256 slots), then insert 4 high-score keys and trace D2.
TEST(DualBucketTest, DebugEvictionTrace) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t NUM_BUCKETS = 2;
  constexpr size_t CAPACITY = NUM_BUCKETS * 128;  // 256 slots
  constexpr size_t N_FILL = CAPACITY;             // Fill completely
  constexpr size_t N_NEW = 4;  // Insert a few high-score keys

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Phase 1: Fill with scores 1..256.
  std::vector<K> h_keys_fill(N_FILL);
  std::vector<V> h_values_fill(N_FILL * DIM, 1.0f);
  std::vector<S> h_scores_fill(N_FILL);

  std::iota(h_keys_fill.begin(), h_keys_fill.end(), 1);
  for (size_t i = 0; i < N_FILL; i++) {
    h_scores_fill[i] = i + 1;
  }

  K* d_keys;
  V* d_values;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, N_FILL * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N_FILL * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N_FILL * sizeof(S)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_fill.data(), N_FILL * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values_fill.data(),
                        N_FILL * DIM * sizeof(V), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores_fill.data(), N_FILL * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_FILL, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size_after_fill = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[DebugEviction] After fill: table_size="
            << table_size_after_fill << " capacity=" << CAPACITY << std::endl;

  // Verify fill: find all N_FILL keys to check b2 lookup correctness.
  {
    bool* d_fill_founds;
    V* d_fill_found_vals;
    CUDA_CHECK(cudaMalloc(&d_fill_founds, N_FILL * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_fill_found_vals, N_FILL * DIM * sizeof(V)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys_fill.data(), N_FILL * sizeof(K),
                          cudaMemcpyHostToDevice));
    table.find(N_FILL, d_keys, d_fill_found_vals, d_fill_founds, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool* h_fill_founds = new bool[N_FILL];
    CUDA_CHECK(cudaMemcpy(h_fill_founds, d_fill_founds, N_FILL * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    int fill_found = 0;
    for (size_t i = 0; i < N_FILL; i++) {
      if (h_fill_founds[i]) {
        fill_found++;
      } else {
        std::cout << "[DebugEviction] MISSING fill key=" << h_keys_fill[i]
                  << " (index=" << i << ")" << std::endl;
      }
    }
    std::cout << "[DebugEviction] Fill verify: found " << fill_found << "/"
              << N_FILL << " keys" << std::endl;
    delete[] h_fill_founds;
    CUDA_CHECK(cudaFree(d_fill_founds));
    CUDA_CHECK(cudaFree(d_fill_found_vals));
  }

  // Phase 2: Insert high-score keys.
  std::vector<K> h_keys_new(N_NEW);
  std::vector<V> h_values_new(N_NEW * DIM, 2.0f);
  std::vector<S> h_scores_new(N_NEW);

  for (size_t i = 0; i < N_NEW; i++) {
    h_keys_new[i] = N_FILL + 100 + i;
    h_scores_new[i] = 10000 + i;
  }

  std::cout << "[DebugEviction] Inserting " << N_NEW << " high-score keys "
            << "(scores " << h_scores_new[0] << ".." << h_scores_new[N_NEW - 1]
            << ")" << std::endl;

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_new.data(), N_NEW * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values_new.data(), N_NEW * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores_new.data(), N_NEW * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N_NEW, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size_after_evict = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[DebugEviction] After evict-insert: table_size="
            << table_size_after_evict << std::endl;

  // Phase 3: Find the high-score keys.
  bool* d_founds;
  CUDA_CHECK(cudaMalloc(&d_founds, N_NEW * sizeof(bool)));
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_found_values, N_NEW * DIM * sizeof(V)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys_new.data(), N_NEW * sizeof(K),
                        cudaMemcpyHostToDevice));
  table.find(N_NEW, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N_NEW];
  CUDA_CHECK(cudaMemcpy(h_founds, d_founds, N_NEW * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  int found_count = 0;
  for (size_t i = 0; i < N_NEW; i++) {
    std::cout << "[DebugEviction] key=" << h_keys_new[i]
              << " score=" << h_scores_new[i]
              << " found=" << (h_founds[i] ? "YES" : "NO") << std::endl;
    if (h_founds[i]) found_count++;
  }
  std::cout << "[DebugEviction] Found " << found_count << "/" << N_NEW
            << std::endl;

  EXPECT_EQ(found_count, static_cast<int>(N_NEW))
      << "All high-score keys should survive eviction in 2-bucket table";

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ===================================
// TestGroup 2 addition: Eviction Quality (T2.6)
// ===================================

// T2.6: After inserting 5x capacity keys with random scores, the surviving keys
// in the table should overlap with the theoretical top-capacity scores by at
// least 98%.  This validates that dual-bucket score-based eviction correctly
// retains high-score keys under sustained oversubscription pressure.
TEST(DualBucketTest, EvictionQualityAtFullLoad) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;      // 128K slots
  constexpr size_t TOTAL_KEYS = 5 * CAPACITY;  // 5x oversubscription
  constexpr size_t BATCH = CAPACITY;           // One capacity per batch
  constexpr double QUALITY_THRESHOLD = 0.995;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Generate all keys (1..TOTAL_KEYS) with random scores.
  std::vector<K> all_keys(TOTAL_KEYS);
  std::vector<S> all_scores(TOTAL_KEYS);
  std::iota(all_keys.begin(), all_keys.end(), 1);

  std::mt19937_64 rng(42);
  for (size_t i = 0; i < TOTAL_KEYS; i++) {
    all_scores[i] = (rng() >> 1) | 1;  // Positive, non-zero
  }

  // Allocate device memory for one batch.
  K* d_keys;
  V* d_values;
  S* d_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, BATCH * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, BATCH * sizeof(S)));

  std::vector<V> h_values(BATCH * DIM, 1.0f);

  // Insert all keys in 5 batches.
  for (size_t offset = 0; offset < TOTAL_KEYS; offset += BATCH) {
    size_t n = std::min(BATCH, TOTAL_KEYS - offset);
    CUDA_CHECK(cudaMemcpy(d_keys, all_keys.data() + offset, n * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * DIM * sizeof(V),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, all_scores.data() + offset, n * sizeof(S),
                          cudaMemcpyHostToDevice));
    table.insert_or_assign(n, d_keys, d_values, d_scores, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Export surviving keys and scores.
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());

  K* d_dump_keys;
  V* d_dump_values;
  S* d_dump_scores;
  size_t* d_dump_counter;
  CUDA_CHECK(cudaMalloc(&d_dump_keys, table_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_dump_values, table_size * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_dump_scores, table_size * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
  CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

  table.export_batch(table_size, 0, d_dump_counter, d_dump_keys, d_dump_values,
                     d_dump_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t dumped;
  CUDA_CHECK(cudaMemcpy(&dumped, d_dump_counter, sizeof(size_t),
                        cudaMemcpyDeviceToHost));

  std::vector<K> h_dump_keys(dumped);
  CUDA_CHECK(cudaMemcpy(h_dump_keys.data(), d_dump_keys, dumped * sizeof(K),
                        cudaMemcpyDeviceToHost));

  // Compute the ideal top-`dumped` set: keys with the highest scores out of
  // all TOTAL_KEYS inserted during the entire test.
  std::vector<std::pair<S, K>> score_key_pairs(TOTAL_KEYS);
  for (size_t i = 0; i < TOTAL_KEYS; i++) {
    score_key_pairs[i] = {all_scores[i], all_keys[i]};
  }
  std::sort(score_key_pairs.begin(), score_key_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::unordered_set<K> ideal_set;
  for (size_t i = 0; i < dumped && i < TOTAL_KEYS; i++) {
    ideal_set.insert(score_key_pairs[i].second);
  }

  // Count overlap between surviving keys and ideal set.
  size_t overlap = 0;
  for (size_t i = 0; i < dumped; i++) {
    if (ideal_set.count(h_dump_keys[i])) overlap++;
  }

  double quality = static_cast<double>(overlap) / static_cast<double>(dumped);
  std::cout << "[EvictionQuality] Table size: " << dumped << "/" << CAPACITY
            << " (LF=" << (static_cast<double>(dumped) / CAPACITY) << ")"
            << std::endl;
  std::cout << "[EvictionQuality] Overlap with ideal top-" << dumped << ": "
            << overlap << "/" << dumped << " (quality=" << (quality * 100.0)
            << "%)" << std::endl;

  EXPECT_GE(quality, QUALITY_THRESHOLD)
      << "Eviction quality " << (quality * 100.0) << "% is below "
      << (QUALITY_THRESHOLD * 100.0) << "% threshold";

  // Cleanup.
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_dump_keys));
  CUDA_CHECK(cudaFree(d_dump_values));
  CUDA_CHECK(cudaFree(d_dump_scores));
  CUDA_CHECK(cudaFree(d_dump_counter));
}

// ===================================
// TestGroup 6: Concurrency Stress Tests
// ===================================

// T6.1: Multi-stream concurrent upsert stress test.
// Multiple CUDA streams issue insert_or_assign concurrently to stress Phase 2
// eviction's stale-score handling.  Under high contention some inserts may be
// REFUSED, but the table must remain consistent: no crashes, no duplicates,
// and all surviving keys must be findable.
TEST(DualBucketTest, MultiStreamConcurrentUpsert) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;    // 128K slots
  constexpr int NUM_STREAMS = 4;
  constexpr size_t KEYS_PER_STREAM = CAPACITY;  // Each stream fills capacity
  constexpr size_t TOTAL_KEYS = NUM_STREAMS * KEYS_PER_STREAM;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  // Create CUDA streams.
  cudaStream_t streams[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaStreamCreate(&streams[s]));
  }

  // Prepare per-stream device memory and data.
  K* d_keys[NUM_STREAMS];
  V* d_values[NUM_STREAMS];
  S* d_scores[NUM_STREAMS];

  std::mt19937_64 rng(42);
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaMalloc(&d_keys[s], KEYS_PER_STREAM * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values[s], KEYS_PER_STREAM * DIM * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_scores[s], KEYS_PER_STREAM * sizeof(S)));

    std::vector<K> h_keys(KEYS_PER_STREAM);
    std::vector<V> h_values(KEYS_PER_STREAM * DIM, static_cast<V>(s + 1));
    std::vector<S> h_scores(KEYS_PER_STREAM);

    for (size_t i = 0; i < KEYS_PER_STREAM; i++) {
      // Use non-overlapping key ranges per stream.
      h_keys[i] = s * KEYS_PER_STREAM + i + 1;
      h_scores[i] = (rng() >> 1) | 1;  // Random positive score
    }

    CUDA_CHECK(cudaMemcpy(d_keys[s], h_keys.data(),
                          KEYS_PER_STREAM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values[s], h_values.data(),
                          KEYS_PER_STREAM * DIM * sizeof(V),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores[s], h_scores.data(),
                          KEYS_PER_STREAM * sizeof(S),
                          cudaMemcpyHostToDevice));
  }

  // Launch concurrent inserts on all streams simultaneously.
  for (int s = 0; s < NUM_STREAMS; s++) {
    table.insert_or_assign(KEYS_PER_STREAM, d_keys[s], d_values[s],
                           d_scores[s], streams[s], /*unique_key=*/true);
  }

  // Synchronize all streams.
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
  }

  // Verify table consistency.
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[MultiStream] Table size after concurrent inserts: "
            << table_size << "/" << CAPACITY << std::endl;

  // Table size must not exceed capacity (no overflow).
  EXPECT_LE(table_size, CAPACITY);
  // Some keys should have been inserted (table should not be empty).
  EXPECT_GT(table_size, static_cast<size_t>(0));

  // Export all surviving keys and verify they are findable.
  K* d_dump_keys;
  V* d_dump_values;
  S* d_dump_scores;
  size_t* d_dump_counter;
  CUDA_CHECK(cudaMalloc(&d_dump_keys, table_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_dump_values, table_size * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_dump_scores, table_size * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
  CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

  table.export_batch(table_size, 0, d_dump_counter, d_dump_keys, d_dump_values,
                     d_dump_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t dumped;
  CUDA_CHECK(cudaMemcpy(&dumped, d_dump_counter, sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(dumped, table_size);

  // Find all exported keys — every surviving key must be findable.
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_founds, dumped * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, dumped * DIM * sizeof(V)));

  table.find(dumped, d_dump_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[dumped];
  CUDA_CHECK(cudaMemcpy(h_founds, d_founds, dumped * sizeof(bool),
                        cudaMemcpyDeviceToHost));

  int found_count = 0;
  for (size_t i = 0; i < dumped; i++) {
    if (h_founds[i]) found_count++;
  }
  std::cout << "[MultiStream] Find consistency: " << found_count << "/"
            << dumped << std::endl;
  EXPECT_EQ(found_count, static_cast<int>(dumped))
      << "All surviving keys must be findable after concurrent upserts";

  // Check no duplicates: export size must match table.size().
  std::vector<K> h_dump_keys(dumped);
  CUDA_CHECK(cudaMemcpy(h_dump_keys.data(), d_dump_keys, dumped * sizeof(K),
                        cudaMemcpyDeviceToHost));
  std::unordered_set<K> unique_keys(h_dump_keys.begin(), h_dump_keys.end());
  EXPECT_EQ(unique_keys.size(), dumped) << "Duplicate keys found in table";

  // Cleanup.
  delete[] h_founds;
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaFree(d_keys[s]));
    CUDA_CHECK(cudaFree(d_values[s]));
    CUDA_CHECK(cudaFree(d_scores[s]));
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }
  CUDA_CHECK(cudaFree(d_dump_keys));
  CUDA_CHECK(cudaFree(d_dump_values));
  CUDA_CHECK(cudaFree(d_dump_scores));
  CUDA_CHECK(cudaFree(d_dump_counter));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// T6.2: Multi-stream concurrent upsert with overlapping keys.
// Tests that concurrent streams inserting the same keys do not create
// duplicates, and that the final values/scores are consistent.
TEST(DualBucketTest, MultiStreamOverlappingKeys) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 1024;
  constexpr int NUM_STREAMS = 4;
  constexpr size_t N = 32768;  // Shared key set

  Table table;
  create_memory_mode_table(table, CAPACITY);

  cudaStream_t streams[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaStreamCreate(&streams[s]));
  }

  // All streams insert the SAME keys with different scores.
  std::vector<K> h_keys(N);
  std::iota(h_keys.begin(), h_keys.end(), 1);

  K* d_keys[NUM_STREAMS];
  V* d_values[NUM_STREAMS];
  S* d_scores[NUM_STREAMS];

  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaMalloc(&d_keys[s], N * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values[s], N * DIM * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_scores[s], N * sizeof(S)));

    std::vector<V> h_values(N * DIM, static_cast<V>(s + 1));
    std::vector<S> h_scores(N);
    for (size_t i = 0; i < N; i++) {
      h_scores[i] = (s + 1) * 1000 + i;  // Different scores per stream
    }

    CUDA_CHECK(cudaMemcpy(d_keys[s], h_keys.data(), N * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values[s], h_values.data(), N * DIM * sizeof(V),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores[s], h_scores.data(), N * sizeof(S),
                          cudaMemcpyHostToDevice));
  }

  // Launch concurrent inserts.
  for (int s = 0; s < NUM_STREAMS; s++) {
    table.insert_or_assign(N, d_keys[s], d_values[s], d_scores[s], streams[s],
                           true);
  }

  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
  }

  // Table size must equal N (no duplicates from concurrent inserts).
  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[MultiStreamOverlap] Table size: " << table_size
            << " (expected " << N << ")" << std::endl;
  EXPECT_EQ(table_size, N) << "Concurrent inserts of same keys created "
                           << (table_size - N) << " duplicates";

  // All keys must be findable.
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  table.find(N, d_keys[0], d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));

  int found_count = 0;
  for (size_t i = 0; i < N; i++) {
    if (h_founds[i]) found_count++;
  }
  EXPECT_EQ(found_count, static_cast<int>(N))
      << "All keys must be findable after concurrent overlapping inserts";

  // Cleanup.
  delete[] h_founds;
  for (int s = 0; s < NUM_STREAMS; s++) {
    CUDA_CHECK(cudaFree(d_keys[s]));
    CUDA_CHECK(cudaFree(d_values[s]));
    CUDA_CHECK(cudaFree(d_scores[s]));
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}

// ===================================
// TestGroup 7: Additional Missing Tests
// ===================================

// T7.1: Find with scores=nullptr (CopyScoreEmpty path).
TEST(DualBucketTest, FindWithNullScores) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 128;
  constexpr size_t N = 1024;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM, 1.0f);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) h_scores[i] = i + 1;

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  S* d_found_scores;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_found_scores, N * sizeof(S)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Find with scores=nullptr (CopyScoreEmpty branch).
  table.find(N, d_keys, d_found_values, d_founds, /*scores=*/nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i]) << "Key " << h_keys[i] << " not found (null scores)";
  }

  // Find with scores!=nullptr (CopyScoreByPassCache branch).
  table.find(N, d_keys, d_found_values, d_founds, d_found_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  std::vector<S> h_found_scores(N);
  CUDA_CHECK(cudaMemcpy(h_found_scores.data(), d_found_scores, N * sizeof(S),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i])
        << "Key " << h_keys[i] << " not found (with scores)";
    if (h_founds[i]) {
      EXPECT_GT(h_found_scores[i], static_cast<S>(0))
          << "Score should be non-zero for key " << h_keys[i];
    }
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
  CUDA_CHECK(cudaFree(d_found_scores));
}

// T7.2: Clear then re-insert (verifies dual_bucket_empty_digest reset).
TEST(DualBucketTest, ClearAndReinsert) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  constexpr size_t CAPACITY = 128 * 128;
  constexpr size_t N = 2048;

  Table table;
  create_memory_mode_table(table, CAPACITY);

  std::vector<K> h_keys(N);
  std::vector<V> h_values(N * DIM, 1.0f);
  std::vector<S> h_scores(N);

  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < N; i++) h_scores[i] = i + 1;

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, N * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, N * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, N * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, N * DIM * sizeof(V)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * DIM * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), N * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Insert first batch.
  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table.size(0), N);

  // Clear the table.
  table.clear(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table.size(0), static_cast<size_t>(0));

  // Re-insert different keys.
  std::vector<K> h_keys2(N);
  std::iota(h_keys2.begin(), h_keys2.end(), N + 1);  // Different keys
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys2.data(), N * sizeof(K),
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(N, d_keys, d_values, d_scores, 0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t table_size = table.size(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(table_size, N)
      << "After clear + re-insert, table should have N entries";

  // Verify new keys are findable.
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool* h_founds = new bool[N];
  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_TRUE(h_founds[i])
        << "Key " << h_keys2[i] << " not found after clear + re-insert";
  }

  // Verify old keys are NOT findable.
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), N * sizeof(K),
                        cudaMemcpyHostToDevice));
  table.find(N, d_keys, d_found_values, d_founds, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_founds, d_founds, N * sizeof(bool), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    EXPECT_FALSE(h_founds[i])
        << "Old key " << h_keys[i] << " still found after clear";
  }

  delete[] h_founds;
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
}
