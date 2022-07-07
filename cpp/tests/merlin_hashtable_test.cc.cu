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

#include "merlin/initializers.cuh"
#include "merlin/optimizers.cuh"
#include "merlin_hashtable.cuh"

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
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
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

template <class K, class M>
void create_continuous_keys(K *h_keys, M *h_metas, int KEY_NUM, K start = 0) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

constexpr uint64_t INIT_SIZE = 64 * 1024 * 1024UL;
constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
constexpr uint64_t TEST_TIMES = 1;
constexpr uint64_t DIM = 64;
constexpr float target_load_factor = 0.50;

template <class K, class M>
__forceinline__ __device__ bool erase_if_pred(const K &key, const M &meta) {
  return ((key % 2) == 1);
}

using K = uint64_t;
using M = uint64_t;
using Vector = ValueArray<float, DIM>;
using Table = nv::merlin::HashTable<K, float, M, DIM>;

/* A demo of Pred for erase_if */
template <class K, class M>
__device__ Table::Pred pred = erase_if_pred<K, M>;

int test_main() {
  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  bool *h_found;

  std::unique_ptr<Table> table_ = std::make_unique<Table>(INIT_SIZE);

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
  size_t dump_counter = 0;

  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_def_val, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_found, KEY_NUM * sizeof(bool));            // 4MB

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));
  cudaMemset(d_def_val, 2, KEY_NUM * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
  cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  K start = 0UL;
  float cur_load_factor = table_->load_factor();

  while (cur_load_factor < target_load_factor) {
    create_continuous_keys<K, M>(h_keys, h_metas, KEY_NUM, start);
    cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

    auto start_insert_or_assign = std::chrono::steady_clock::now();
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, KEY_NUM, false, stream);
    auto end_insert_or_assign = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_insert_or_assign =
        end_insert_or_assign - start_insert_or_assign;

    auto start_find = std::chrono::steady_clock::now();
    table_->find(d_keys, reinterpret_cast<float *>(d_vectors), d_found, KEY_NUM,
                 reinterpret_cast<float *>(d_def_val), true, stream);
    auto end_find = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_find = end_find - start_find;

    cur_load_factor = table_->load_factor();

    printf(
        "[prepare] insert_or_assign=%.2fms, find=%.2fms, "
        "cur_load_factor=%f\n",
        diff_insert_or_assign.count() * 1000, diff_find.count() * 1000,
        cur_load_factor);
    start += KEY_NUM;
  }
  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    total_size = table_->size(stream);

    std::cout << "before insert_or_assign: total_size = " << total_size
              << std::endl;
    auto start_insert_or_assign = std::chrono::steady_clock::now();
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, KEY_NUM, false, stream);
    auto end_insert_or_assign = std::chrono::steady_clock::now();

    total_size = table_->size(stream);
    std::cout << "after 1st insert_or_assign: total_size = " << total_size
              << std::endl;

    auto start_reserve = std::chrono::steady_clock::now();
    table_->reserve(table_->capacity() * 2, stream);
    auto end_reserve = std::chrono::steady_clock::now();

    total_size = table_->size(stream);
    std::cout << "after reserve: total_size = " << total_size << std::endl;

    cudaMemset(d_vectors, 2, KEY_NUM * sizeof(Vector));
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, KEY_NUM, stream);

    total_size = table_->size(stream);
    std::cout << "after 2nd insert_or_assign: total_size = " << total_size
              << std::endl;

    auto start_find = std::chrono::steady_clock::now();
    table_->find(d_keys, reinterpret_cast<float *>(d_vectors), d_found, KEY_NUM,
                 reinterpret_cast<float *>(d_def_val), true, stream);
    auto end_find = std::chrono::steady_clock::now();

    auto start_accum = std::chrono::steady_clock::now();
    table_->accum(d_keys, reinterpret_cast<float *>(d_vectors), d_found,
                  KEY_NUM, false, stream);
    auto end_accum = std::chrono::steady_clock::now();

    auto start_size = std::chrono::steady_clock::now();
    total_size = table_->size(stream);
    auto end_size = std::chrono::steady_clock::now();
    std::cout << "after accum: total_size = " << total_size << std::endl;

    auto start_erase_if = std::chrono::steady_clock::now();
    size_t erase_num = table_->erase_if(pred<K, M>, stream);
    auto end_erase_if = std::chrono::steady_clock::now();
    total_size = table_->size(stream);
    std::cout << "after erase_if: total_size = " << total_size
              << ", erase_num = " << erase_num << std::endl;

    auto start_clear = std::chrono::steady_clock::now();
    table_->clear(stream);
    auto end_clear = std::chrono::steady_clock::now();
    total_size = table_->size(stream);
    std::cout << "after clear: total_size = " << total_size << std::endl;

    table_->clear(stream);
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, KEY_NUM, false, stream);

    dump_counter = table_->dump(d_keys, reinterpret_cast<float *>(d_vectors), 0,
                                table_->capacity(), stream);

    std::chrono::duration<double> diff_insert_or_assign =
        end_insert_or_assign - start_insert_or_assign;
    std::chrono::duration<double> diff_size = end_size - start_size;
    std::chrono::duration<double> diff_find = end_find - start_find;
    std::chrono::duration<double> diff_accum = end_accum - start_accum;
    std::chrono::duration<double> diff_reserve = end_reserve - start_reserve;
    std::chrono::duration<double> diff_erase_if = end_erase_if - start_erase_if;
    std::chrono::duration<double> diff_clear = end_clear - start_clear;
    printf("[timing] insert_or_assign=%.2fms\n",
           diff_insert_or_assign.count() * 1000);
    printf("[timing] size=%.2fms\n", diff_size.count() * 1000);
    printf("[timing] reserve=%.2fms\n", diff_reserve.count() * 1000);
    printf("[timing] find=%.2fms\n", diff_find.count() * 1000);
    printf("[timing] accum=%.2fms\n", diff_accum.count() * 1000);
    printf("[timing] erase_if=%.2fms\n", diff_erase_if.count() * 1000);
    printf("[timing] clear=%.2fms\n", diff_clear.count() * 1000);
  }
  cudaStreamDestroy(stream);

  cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
             cudaMemcpyDeviceToHost);

  int found_num = 0;

  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) found_num++;
  }

  std::cout << "Capacity = " << table_->capacity()
            << ", total_size = " << total_size
            << ", dump_counter = " << dump_counter
            << ", found_num = " << found_num << std::endl;

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_def_val);
  cudaFree(d_vectors_ptr);
  cudaFree(d_found);

  std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

  return 0;
}

int main() {
  test_main();
  return 0;
}
