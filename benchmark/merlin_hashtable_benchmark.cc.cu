/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

using namespace nv::merlin;

inline uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <class K, class M>
void create_random_keys(K* h_keys, M* h_metas, const int key_num_per_op) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < key_num_per_op) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

std::string rep(int n) { return std::string(n, ' '); }

template <class K, class M>
void create_continuous_keys(K* h_keys, M* h_metas, const int key_num_per_op,
                            const K start = 0) {
  for (K i = 0; i < key_num_per_op; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <typename K, typename M>
void set_partial_key(K* h_keys, M* h_metas, const int key_num_per_op,
                     const float hitrate = 0.6f) {
  static K unique_value = std::numeric_limits<K>::max();
  int start = key_num_per_op * (1.0 - hitrate);
  for (int i = start; i < key_num_per_op; i++) {
    h_keys[i] = unique_value--;
    h_metas[i] = getTimestamp();
  }
}

template <typename M>
void refresh_metas(M* h_metas, const int key_num_per_op) {
  for (int i = 0; i < key_num_per_op; i++) {
    h_metas[i] = getTimestamp();
  }
}

void test_main(const size_t dim,
               const size_t init_capacity = 64 * 1024 * 1024UL,
               const size_t key_num_per_op = 1 * 1024 * 1024UL,
               const size_t hbm4values = 16, const float load_factor = 1.0f,
               const float hitrate = 0.6f, const bool io_by_cpu = false) {
  using K = uint64_t;
  using M = uint64_t;
  using V = float;
  using Table = nv::merlin::HashTable<K, float, M>;
  using TableOptions = nv::merlin::HashTableOptions;

  size_t free, total;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  if (free / (1 << 30) < hbm4values) {
    return;
  }

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(hbm4values);
  options.io_by_cpu = io_by_cpu;
  options.evict_strategy = EvictStrategy::kCustomized;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

  CUDA_CHECK(
      cudaMemset(h_vectors, 0, key_num_per_op * sizeof(V) * options.dim));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  K* d_keys_out;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, key_num_per_op * sizeof(K)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemset(d_def_val, 2, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  K start = 0UL;
  float cur_load_factor = table->load_factor();
  auto diff_insert_or_assign = benchmark::Timer<double>();
  auto diff_find = benchmark::Timer<double>();
  auto diff_find_or_insert = benchmark::Timer<double>();
  auto diff_assign = benchmark::Timer<double>();

  while (cur_load_factor < load_factor) {
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_per_op, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                          cudaMemcpyHostToDevice));

    diff_insert_or_assign.start();
    table->insert_or_assign(key_num_per_op, d_keys,
                            reinterpret_cast<float*>(d_vectors), d_metas,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    diff_insert_or_assign.end();

    diff_find.start();
    table->find(key_num_per_op, d_keys, reinterpret_cast<float*>(d_vectors),
                d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    diff_find.end();

    refresh_metas<M>(h_metas, key_num_per_op);
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                          cudaMemcpyHostToDevice));
    diff_assign.start();
    table->assign(key_num_per_op, d_keys, reinterpret_cast<float*>(d_def_val),
                  d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    diff_assign.end();

    set_partial_key<K, M>(h_keys, h_metas, key_num_per_op, hitrate);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                          cudaMemcpyHostToDevice));
    diff_find_or_insert.start();
    table->find_or_insert(key_num_per_op, d_keys,
                          reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    diff_find_or_insert.end();

    cur_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (start == 0) {
      table->erase(key_num_per_op, d_keys, stream);  // warmup for erase kernel.
    }
    start += (key_num_per_op * (1.0 - hitrate));
  }

  table->erase(key_num_per_op, d_keys, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int32_t capacity = static_cast<int32_t>(init_capacity / (1024 * 1024));
  size_t hmem4values =
      init_capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024);
  hmem4values = hmem4values < hbm4values ? 0 : (hmem4values - hbm4values);
  float insert_or_assign_tput = key_num_per_op /
                                diff_insert_or_assign.getResult() /
                                (1024 * 1024 * 1024.0);
  float find_tput =
      key_num_per_op / diff_find.getResult() / (1024 * 1024 * 1024.0f);
  float find_or_insert_tput = key_num_per_op / diff_find_or_insert.getResult() /
                              (1024 * 1024 * 1024.0f);
  float assign_tput =
      key_num_per_op / diff_assign.getResult() / (1024 * 1024 * 1024.0f);

  cout << "|" << rep(1) << setw(3) << setfill(' ') << options.dim << " "
       << "|" << rep(15) << setw(5) << setfill(' ') << capacity << " "
       << "|" << rep(9) << setw(3) << setfill(' ') << hbm4values << " /"
       << setw(3) << setfill(' ') << hmem4values << " "
       << "|" << rep(8) << fixed << setprecision(2) << load_factor << " "
       << "|" << rep(12) << fixed << setprecision(3) << insert_or_assign_tput
       << " "
       << "|" << rep(2) << fixed << setprecision(3) << find_tput << " "
       << "|" << rep(10) << fixed << setprecision(3) << find_or_insert_tput
       << " "
       << "|" << rep(2) << fixed << setprecision(3) << assign_tput << " |"
       << endl;

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));

  CudaCheckError();

  return;
}

void print_title() {
  cout << endl
       << "| dim "
       << "| Capacity<br>(M-KVs) "
       << "| HBM/HMEM<br>(GB) "
       << "| load_factor "
       << "| insert_or_assign "
       << "|   find "
       << "| find_or_insert "
       << "| assign |" << endl;
  cout << "|----:"
       //<< "| Capacity<br>(M-KVs) "
       << "|:-------------------:"
       //<< "| HBM/HMEM<br>(GB) "
       << "|:----------------:"
       //<< "| load_factor "
       << "|------------:"
       //<< "| insert_or_assign "
       << "|:----------------:"
       //<< "|  find "
       << "|-------:"
       //<< "| find_or_insert "
       << "|:--------------:"
       //<< "| assign "
       << "|-------:|" << endl;
}

int main() {
  size_t key_num_per_op = 1 * 1024 * 1024UL;
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  cout << endl
       << "## Benchmark" << endl
       << endl
       << "* GPU: 1 x " << props.name << ": " << props.major << "."
       << props.minor << endl
       << "* Key Type = uint64_t" << endl
       << "* Value Type = float32 * {dim}" << endl
       << "* Key-Values per OP = " << key_num_per_op << endl
       << "* ***Throughput Unit: Billion-KV/second***" << endl
       << endl
       << "### On pure HBM mode: " << endl;
  print_title();
  try {
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 0.50f);
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 0.75f);
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 1.00f);

    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 0.50f);
    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 0.75f);
    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 1.00f);
    cout << endl;

    cout << "### On HBM+HMEM hybrid mode: " << endl;
    print_title();
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 0.50f);
    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 0.75f);
    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 1.00f);

    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 0.50f);
    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 0.75f);
    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 1.00f);
    cout << endl;

    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
