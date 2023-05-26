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
#include <stdio.h>
#include <array>
#include <map>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using Table = nv::merlin::HashTable<i64, f32, u64>;
using TableOptions = nv::merlin::HashTableOptions;

/*
 * test_dynamic_max_capcity_table creates a table in small
 * capacity and insert random kv pairs until its load_factor
 * became 1.0. Then expand the max_capacity. Keep inserting until
 * the load factor growth to 1.0 again.
 */
void test_dynamic_max_capcity_table() {
  size_t len = 10000llu;
  size_t max_capacity = 1 << 14;
  size_t init_capacity = 1 << 12;
  size_t offset = 0;
  size_t uplimit = 1 << 20;
  float load_factor_threshold = 0.98f;

  TableOptions opt;
  opt.max_capacity = max_capacity;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = uplimit * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kLru;
  opt.dim = dim;

  using Vec_t = test_util::ValueArray<f32, dim>;
  std::map<i64, Vec_t> ref_map;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> buffer;
  buffer.Reserve(len, dim, stream);
  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(len, dim, stream);

  size_t total_len = 0;
  while (true) {
    buffer.ToRange(offset, /*skip=1*/ 1, stream);
    size_t n_evicted = table->insert_and_evict(
        len, buffer.keys_ptr(), buffer.values_ptr(), nullptr,
        evict_buffer.keys_ptr(), evict_buffer.values_ptr(), nullptr, stream);
    printf("Insert %zu keys and evict %zu\n", len, n_evicted);
    offset += len;
    total_len += len;
    evict_buffer.SyncData(/*h2d=*/false, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < n_evicted; i++) {
      Vec_t* vec =
          reinterpret_cast<Vec_t*>(evict_buffer.values_ptr(false) + i * dim);
      ref_map[evict_buffer.keys_ptr(false)[i]] = *vec;
    }

    if (table->load_factor(stream) >= load_factor_threshold) {
      ASSERT_GE(table->size(stream),
                static_cast<size_t>((static_cast<float>(max_capacity) *
                                     load_factor_threshold)));
      max_capacity *= 2;
      if (max_capacity > uplimit) {
        break;
      }
      // What we need.
      printf("----> check change max_capacity from %zu to %zu\n",
             table->capacity(), max_capacity);
      table->set_max_capacity(max_capacity);
      table->reserve(max_capacity, stream);
      ASSERT_EQ(max_capacity, table->capacity());
      ASSERT_LE(table->load_factor(stream), 0.5f);
    }

    if (total_len > uplimit * 2) {
      throw std::runtime_error("Traverse too much keys but not finish test.");
    }
  };

  offset = 0;
  for (; offset < table->capacity(); offset += len) {
    size_t search_len = len;
    if (offset + search_len > table->capacity()) {
      search_len = table->capacity() - offset;
    }
    size_t n_exported =
        table->export_batch(search_len, offset, buffer.keys_ptr(),
                            buffer.values_ptr(), /*scores=*/nullptr, stream);
    buffer.SyncData(/*h2d=*/false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < n_exported; i++) {
      Vec_t* vec = reinterpret_cast<Vec_t*>(buffer.values_ptr(false) + i * dim);
      for (size_t j = 0; j < dim; j++) {
        ASSERT_EQ(buffer.keys_ptr(false)[i], vec->operator[](j));
      }
      ref_map[buffer.keys_ptr(false)[i]] = *vec;
    }
  }

  printf("---> uplimit: %zu\n", uplimit);
  printf("---> table size: %zu\n", table->size(stream));
  printf("---> table cap: %zu\n", table->capacity());
  printf("---> cpu table size: %zu\n", ref_map.size());
  for (auto& it : ref_map) {
    for (size_t j = 0; j < dim; j++) {
      ASSERT_EQ(static_cast<f32>(it.first), it.second.data[j]);
    }
  }
  ASSERT_EQ(table->capacity() * 2, max_capacity);
  ASSERT_GE(static_cast<float>(ref_map.size()),
            static_cast<float>(table->capacity()) * load_factor_threshold);
}

TEST(MerlinHashTableTest, test_dynamic_max_capcity_table) {
  test_dynamic_max_capcity_table();
}
