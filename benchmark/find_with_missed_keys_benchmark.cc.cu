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

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

void print_tile() {
  std::cout << std::endl
            << "|    \u03BB "
            << "| capacity "
            << "| max_hbm_for_vectors "
            << "| max_bucket_size "
            << "| dim "
            << "| missed_ratio "
            << "| througput(BillionKV/secs) ";
  std::cout << "|\n";

  //<< "| load_factor "
  std::cout << "|------"
            //<< "| capacity "
            << "|----------"
            //<< "| max_hbm_for_vectors "
            << "|---------------------"
            //<< "| max_bucket_size "
            << "|-----------------"
            //<< "| dim "
            << "|-----"
            //<< "| missed_ratio "
            << "|--------------"
            //<< "| througput(BillionKV/secs) "
            << "|---------------------------";
  std::cout << "|\n";
}

template <typename T>
void print_w(const T& t, size_t width) {
  std::cout << "|" << std::setw(width) << t;
}

void print_result(double load_factor, size_t capacity,
                  size_t max_hbm_for_vectors, size_t max_bucket_size,
                  size_t dim, double missed_ratio, float througput) {
  print_w(load_factor, 6);
  print_w(capacity, 10);
  print_w(max_hbm_for_vectors, 21);
  print_w(max_bucket_size, 17);
  print_w(dim, 5);
  print_w(missed_ratio, 14);
  print_w(througput, 27);
  std::cout << "|\n";
}

void test_find(size_t capacity, size_t dim, size_t max_hbm_for_vectors,
               double load_factor, size_t max_bucket_size,
               double missed_ratio) {
  MERLIN_CHECK(load_factor >= 0.0 && load_factor <= 1.0,
               "Invalid `load_factor`");
  K* h_keys;
  S* h_scores;
  V* h_vectors;

  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.dim = dim;

  options.max_hbm_for_vectors = nv::merlin::MB(max_hbm_for_vectors);
  options.max_bucket_size = max_bucket_size;

  size_t key_num = capacity;
  CUDA_CHECK(cudaMallocHost(&h_keys, key_num * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, key_num * options.dim * sizeof(V)));

  K* d_keys;
  S* d_scores;
  V* d_vectors;
  K* d_missed_keys;
  int* d_missed_indices;
  int* d_missed_size;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, key_num * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_missed_keys, key_num * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_missed_indices, key_num * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_missed_size, sizeof(int)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // insert key-value
  size_t insert_num = (double)key_num * load_factor;
  benchmark::create_continuous_keys<K, S>(h_keys, h_scores, insert_num,
                                          0 /*start*/);
  benchmark::init_value_using_key<K, V>(h_keys, h_vectors, insert_num,
                                        options.dim);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, insert_num * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, insert_num * sizeof(S),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                        insert_num * sizeof(V) * options.dim,
                        cudaMemcpyHostToDevice));
  Table table;
  table.init(options);
  table.insert_or_assign(insert_num, d_keys, d_vectors, d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // find key-value
  size_t find_num = (double)insert_num * (1.0 - missed_ratio);
  benchmark::create_continuous_keys<K, S>(h_keys, nullptr, find_num,
                                          0 /*start*/);
  benchmark::create_continuous_keys<K, S>(
      h_keys + find_num, nullptr, insert_num - find_num, insert_num /*start*/);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, insert_num * sizeof(K),
                        cudaMemcpyHostToDevice));

  auto timer = benchmark::Timer<double>();
  timer.start();
  table.find(insert_num, d_keys, d_vectors, d_missed_keys, d_missed_indices,
             d_missed_size, d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.end();

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_vectors));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_missed_keys));
  CUDA_CHECK(cudaFree(d_missed_indices));
  CUDA_CHECK(cudaFree(d_missed_size));

  CudaCheckError();
  float througput = insert_num / timer.getResult() / (1024 * 1024 * 1024.0f);
  print_result(load_factor, capacity, max_hbm_for_vectors, max_bucket_size, dim,
               missed_ratio, througput);
}

void test_main(double load_factor, double missed_ratio) {
  constexpr size_t CAPACITY = 100000000UL;
  print_tile();
  // pure HBM
  test_find(CAPACITY, 8, 8 * 1024UL, load_factor, 256, missed_ratio);
  test_find(CAPACITY, 8, 8 * 1024UL, load_factor, 128, missed_ratio);
  // hybrid
  test_find(CAPACITY, 8, 1 * 1024UL, load_factor, 256, missed_ratio);
  test_find(CAPACITY, 8, 1 * 1024UL, load_factor, 128, missed_ratio);
  // pure HMEM
  test_find(CAPACITY, 8, 0, load_factor, 256, missed_ratio);
  test_find(CAPACITY, 8, 0, load_factor, 128, missed_ratio);
}

int main() {
  test_main(0.2, 0);
  test_main(0.2, 0.5);
  test_main(0.2, 1.0);
  test_main(0.5, 0);
  test_main(0.5, 0.5);
  test_main(0.5, 1.0);
  test_main(1.0, 0);
  test_main(1.0, 0.5);
  test_main(1.0, 1.0);
  return 0;
}
