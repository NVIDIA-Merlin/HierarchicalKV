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
#include <cooperative_groups.h>
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

template <typename T>
void create_random_keys(T *h_keys, M *h_metas, int KEY_NUM) {
  std::unordered_set<T> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<T> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const T num : numbers) {
    h_keys[i] = Murmur3Hash(num);
    h_metas[i] = getTimestamp();
    i++;
  }
}

int main() {
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024;
  constexpr uint64_t TEST_TIMES = 1;

  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  size_t *h_size;
  bool *h_found;

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, KEY_NUM * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_size, TABLE_SIZE * sizeof(size_t));  // 8MB
  cudaMallocHost(&h_found, KEY_NUM * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector));

  create_random_keys<K>(h_keys, h_metas, KEY_NUM);

  Table *d_table;
  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector **d_vectors_ptr;
  size_t *d_size;
  bool *d_found;

  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_size, TABLE_SIZE * sizeof(size_t));        // 8MB
  cudaMalloc(&d_found, KEY_NUM * sizeof(bool));            // 4MB

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
  cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint64_t NUM_THREADS = 1024;
  uint64_t N = KEY_NUM;
  uint64_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  //   create_table(&d_table);
  HashTable table_ * = new HashTable(KEY_NUM);
  for (int i = 0; i < TEST_TIMES; i++) {
    // upsert test
    N = KEY_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    auto start_upsert = std::chrono::steady_clock::now();
    upsert<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, d_metas, d_vectors_ptr,
                                        N);

    cudaDeviceSynchronize();
    auto end_upsert = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_upsert = end_upsert - start_upsert;

    N = KEY_NUM * DIM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    auto start_write = std::chrono::steady_clock::now();
    write<<<NUM_BLOCKS, NUM_THREADS>>>(d_vectors, d_vectors_ptr, N);
    cudaDeviceSynchronize();
    auto end_write = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_write = end_write - start_write;

    // size test
    N = TABLE_SIZE;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    size<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    cudaDeviceSynchronize();

    // lookup test
    N = BUCKETS_SIZE * KEY_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
    cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));
    cudaDeviceSynchronize();

    auto start_lookup = std::chrono::steady_clock::now();
    lookup<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, d_vectors_ptr, d_found,
                                        N);
    cudaDeviceSynchronize();
    auto end_lookup = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_lookup = end_lookup - start_lookup;

    N = KEY_NUM * DIM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // TODO: remove
    cudaMemset(d_vectors, 0, KEY_NUM * sizeof(Vector));

    auto start_read = std::chrono::steady_clock::now();
    read<<<NUM_BLOCKS, NUM_THREADS>>>(d_vectors_ptr, d_vectors, N);
    cudaDeviceSynchronize();
    auto end_read = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_read = end_read - start_read;
    printf("[timing] upsert=%.2fms, write=%.2fms\n", diff_upsert.count() * 1000,
           diff_write.count() * 1000);
    printf("[timing] lookup=%.2fms, read = % .2fms\n ",
           diff_lookup.count() * 1000, diff_read.count() * 1000);
  }
  destroy_table(&d_table);
  cudaStreamDestroy(stream);
  cudaMemcpy(h_size, d_size, TABLE_SIZE * sizeof(size_t),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
             cudaMemcpyDeviceToHost);

  int total_size = 0;
  size_t max_bucket_len = 0;
  size_t min_bucket_len = KEY_NUM;
  int found_num = 0;
  std::unordered_map<int, int> size2length;

  //   for (int i = 0; i < DIM; i++) {
  //     V tmp = h_vectors[1].values[i];
  //     int *tmp_int = reinterpret_cast<int *>((V *)(&tmp));
  //     cout << "vector = " << *tmp_int;
  //   }
  cout << endl;

  for (int i = 0; i < TABLE_SIZE; i++) {
    total_size += h_size[i];
    if (size2length.find(h_size[i]) != size2length.end()) {
      size2length[h_size[i]] += 1;
    } else {
      size2length[h_size[i]] = 1;
    }
    max_bucket_len = max(max_bucket_len, h_size[i]);
    min_bucket_len = min(min_bucket_len, h_size[i]);
  }

  //   for(auto n: size2length){
  //     cout << n.first << "    " << n.second << endl;
  //   }

  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) found_num++;
  }
  cout << "Capacity = " << INIT_SIZE << ", total_size = " << total_size
       << ", max_bucket_len = " << max_bucket_len
       << ", min_bucket_len = " << min_bucket_len
       << ", found_num = " << found_num << endl;

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_size);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_vectors_ptr);
  cudaFree(d_size);
  cudaFree(d_found);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
