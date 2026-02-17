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

#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using TableMode = nv::merlin::TableMode;
using EvictStrategy = nv::merlin::EvictStrategy;

template <typename Table>
double benchmark_insert(Table& table, size_t n, K* d_keys, V* d_values,
                        S* d_scores, cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto start = std::chrono::high_resolution_clock::now();
  table.insert_or_assign(n, d_keys, d_values, d_scores, stream, true);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count() /
              1000.0;
  return static_cast<double>(n) / ms / 1000.0;  // Mops/s
}

template <typename Table>
double benchmark_find(Table& table, size_t n, K* d_keys, V* d_values,
                      bool* d_founds, cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto start = std::chrono::high_resolution_clock::now();
  table.find(n, d_keys, d_values, d_founds, nullptr, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count() /
              1000.0;
  return static_cast<double>(n) / ms / 1000.0;  // Mops/s
}

void run_benchmark(size_t capacity, size_t dim, TableMode mode,
                   const char* mode_name) {
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  Table table;
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = 0;
  options.dim = dim;
  options.max_bucket_size = 128;
  options.table_mode = mode;
  table.init(options);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Generate keys.
  size_t max_n = capacity;
  std::vector<K> h_keys(max_n);
  std::vector<V> h_values(max_n * dim, 1.0f);
  std::vector<S> h_scores(max_n);
  std::iota(h_keys.begin(), h_keys.end(), 1);
  for (size_t i = 0; i < max_n; i++) h_scores[i] = i + 1;

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;
  V* d_found_values;
  CUDA_CHECK(cudaMalloc(&d_keys, max_n * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_values, max_n * dim * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_scores, max_n * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_founds, max_n * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_found_values, max_n * dim * sizeof(V)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), max_n * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), max_n * dim * sizeof(V),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), max_n * sizeof(S),
                        cudaMemcpyHostToDevice));

  printf("--- %s (capacity=%zuK, dim=%zu) ---\n", mode_name, capacity / 1024,
         dim);
  printf("  %-12s  %-18s  %-18s\n", "Load Factor", "Insert (Mops/s)",
         "Find (Mops/s)");

  float load_factors[] = {0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 1.00f};
  size_t prev_n = 0;

  for (float lf : load_factors) {
    size_t target_n = static_cast<size_t>(capacity * lf);
    if (target_n > max_n) break;
    size_t batch_n = target_n - prev_n;
    if (batch_n == 0) continue;

    // Insert to reach target load factor.
    double insert_mops =
        benchmark_insert(table, batch_n, d_keys + prev_n,
                         d_values + prev_n * dim, d_scores + prev_n, stream);

    // Find all inserted keys.
    double find_mops = benchmark_find(table, target_n, d_keys, d_found_values,
                                      d_founds, stream);

    printf("  %-12.2f  %-18.1f  %-18.1f\n", lf, insert_mops, find_mops);
    prev_n = target_n;
  }

  // Memory efficiency: first eviction LF.
  // (Already covered in test, report here too.)
  size_t table_size = table.size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("  Final size: %zu / %zu (LF=%.4f)\n", table_size, capacity,
         static_cast<float>(table_size) / capacity);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_found_values));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
  printf("=== Dual-Bucket Benchmark Results ===\n\n");

  // Default: 1M capacity, dim=64.
  size_t capacity = 128 * 1024 * 8;  // ~1M
  size_t dim = 64;

  if (argc > 1) capacity = static_cast<size_t>(atol(argv[1]));
  if (argc > 2) dim = static_cast<size_t>(atol(argv[2]));

  run_benchmark(capacity, dim, TableMode::kThroughput, "THROUGHPUT_MODE");
  printf("\n");
  run_benchmark(capacity, dim, TableMode::kMemory, "MEMORY_MODE");
  printf("\n");

  printf("=== Benchmark Complete ===\n");
  return 0;
}
