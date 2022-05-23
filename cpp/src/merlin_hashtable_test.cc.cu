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
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "merlin_hashtable.cuh"

using std::begin;
using std::cerr;
using std::copy;
using std::cout;
using std::endl;
using std::generate;
using std::max;
using std::min;

using namespace std::chrono;
using namespace nv::merlin;

uint64_t getTimestamp() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

template <class K, class M>
void create_random_keys(K *h_keys, M *h_metas, int KEY_NUM) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = Murmur3HashHost(num);
    h_metas[i] = getTimestamp() + i;
    i++;
  }
}

template <class V>
struct ValueArrayBase {};

template <class V, size_t DIM>
struct ValueArray : public ValueArrayBase<V> {
  V value[DIM];
};

template <class T>
using ValueType = ValueArrayBase<T>;

int test_main() {
  constexpr uint64_t INIT_SIZE = 32 * 1024 * 1024;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr uint64_t DIM = 64;

  using K = uint64_t;
  using M = uint64_t;
  using Vector = ValueArray<float, DIM>;
  using Table = HashTable<K, Vector, ValueArrayBase<float>, M, DIM>;

  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  bool *h_found;

  Table *table_ = new Table(INIT_SIZE);

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, KEY_NUM * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_found, KEY_NUM * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector));

  create_random_keys<K, M>(h_keys, h_metas, KEY_NUM);

  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector *d_def_val;
  Vector **d_vectors_ptr;
  bool *d_found;
  size_t *d_dump_counter;
  size_t h_dump_counter = 0;

  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_def_val, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_found, KEY_NUM * sizeof(bool));            // 4MB
  cudaMallocManaged((void **)&d_dump_counter, sizeof(size_t));

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));
  cudaMemset(d_def_val, 2, KEY_NUM * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
  cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    total_size = table_->get_size(stream);

    cout << "before upsert: total_size = " << total_size << endl;
    auto start_upsert = std::chrono::steady_clock::now();
    table_->upsert(d_keys, (ValueArrayBase<float> *)d_vectors, d_metas, KEY_NUM,
                   stream);
    auto end_upsert = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_upsert = end_upsert - start_upsert;

    cudaMemset(d_vectors, 2, KEY_NUM * sizeof(Vector));
    table_->upsert(d_keys, (ValueArrayBase<float> *)d_vectors, d_metas, KEY_NUM,
                   stream);

    total_size = table_->get_size(stream);
    cout << "after upsert: total_size = " << total_size << endl;

    auto start_lookup = std::chrono::steady_clock::now();
    table_->get(d_keys, (ValueArrayBase<float> *)d_vectors, d_found, KEY_NUM,
                d_def_val, stream, true);
    auto end_lookup = std::chrono::steady_clock::now();

    table_->dump(d_keys, (ValueArrayBase<float> *)d_vectors, 0,
                 table_->get_capacity(), d_dump_counter, stream);

    std::chrono::duration<double> diff_lookup = end_lookup - start_lookup;
    printf("[timing] upsert=%.2fms\n", diff_upsert.count() * 1000);
    printf("[timing] lookup=%.2fms\n ", diff_lookup.count() * 1000);
  }
  cudaStreamDestroy(stream);

  cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool), cudaMemcpyDeviceToHost);

  cudaMemcpy(&h_dump_counter, d_dump_counter, sizeof(size_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
             cudaMemcpyDeviceToHost);

  size_t max_bucket_len = 0;
  size_t min_bucket_len = KEY_NUM;
  int found_num = 0;

  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) found_num++;
  }

  cout << "Capacity = " << table_->get_capacity()
       << ", total_size = " << total_size
       << ", h_dump_counter = " << h_dump_counter
       << ", max_bucket_len = " << max_bucket_len
       << ", min_bucket_len = " << min_bucket_len
       << ", found_num = " << found_num << endl;

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_found);

  cudaFree(d_dump_counter);
  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_def_val);
  cudaFree(d_vectors_ptr);
  cudaFree(d_found);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}

int main() {
  test_main();
  return 0;
}
