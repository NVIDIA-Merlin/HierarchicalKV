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
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"

constexpr uint64_t DIM = 4;
using K = int64_t;
using M = uint64_t;
using V = float;
using Table = nv::merlin::HashTable<K, V, M>;
using TableOptions = nv::merlin::HashTableOptions;

constexpr M kMetaV1 = 100llu;
constexpr M kMetaV2 = 10000llu;

void test_insert_and_evict() {
  TableOptions opt;
  size_t bucket_num = 1;
  size_t bucket_len = 128;
  size_t keynum = bucket_num * bucket_len;
  opt.max_capacity = keynum;
  opt.init_capacity = keynum;
  opt.max_hbm_for_vectors = keynum * DIM * sizeof(V);
  opt.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  opt.dim = DIM;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  K* d_keys = nullptr;
  V* d_vectors = nullptr;
  M* d_metas = nullptr;
  test_util::getBufferOnDevice(&d_keys, keynum * sizeof(K), stream);
  test_util::getBufferOnDevice(&d_vectors, keynum * sizeof(V) * DIM, stream);
  test_util::getBufferOnDevice(&d_metas, keynum * sizeof(M), stream);

  K* h_keys = nullptr;
  V* h_vectors = nullptr;
  M* h_metas = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_keys, keynum * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, keynum * sizeof(V) * DIM));
  CUDA_CHECK(cudaMallocHost(&h_metas, keynum * sizeof(M)));
  memset(h_keys, 0, keynum * sizeof(K));
  memset(h_vectors, 0, keynum * sizeof(V) * DIM);
  memset(h_metas, 0, keynum * sizeof(M));
  for (size_t i = 0; i < keynum; i++) {
    h_keys[i] = static_cast<K>(i);
    for (size_t j = 0; j < DIM; j++) {
      h_vectors[i * DIM + j] = static_cast<V>(i);
    }
    h_metas[i] = kMetaV1;
  }

  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, keynum * sizeof(K),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, keynum * DIM * sizeof(V),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, keynum * sizeof(M),
                             cudaMemcpyHostToDevice, stream));

  table->insert_or_assign(keynum, d_keys, d_vectors, d_metas, stream);
  size_t s0 = table->size(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < keynum; i++) {
    h_keys[i] = static_cast<K>(i + keynum);
    for (size_t j = 0; j < DIM; j++) {
      h_vectors[i * DIM + j] = static_cast<V>(i + keynum);
    }
    h_metas[i] = kMetaV2;
  }

  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, keynum * sizeof(K),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, keynum * DIM * sizeof(V),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, keynum * sizeof(M),
                             cudaMemcpyHostToDevice, stream));

  K* d_ev_keys = nullptr;
  V* d_ev_vectors = nullptr;
  M* d_ev_metas = nullptr;
  test_util::getBufferOnDevice(&d_ev_keys, keynum * sizeof(K), stream);
  test_util::getBufferOnDevice(&d_ev_vectors, keynum * sizeof(V) * DIM, stream);
  test_util::getBufferOnDevice(&d_ev_metas, keynum * sizeof(M), stream);

  size_t n_evicted =
      table->insert_and_evict(keynum, d_keys, d_vectors, d_metas, d_ev_keys,
                              d_ev_vectors, d_ev_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t s1 = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  ASSERT_TRUE(s1 == n_evicted);

  size_t h_counter = table->export_batch(table->capacity(), 0, d_keys,
                                         d_vectors, d_metas, stream);
  ASSERT_TRUE(h_counter == s1);
  CUDA_CHECK(cudaMemcpyAsync(h_keys, d_ev_keys, h_counter * sizeof(K),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_metas, d_ev_metas, h_counter * sizeof(M),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_vectors, d_ev_vectors,
                             h_counter * DIM * sizeof(V),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < h_counter; i++) {
    ASSERT_TRUE(h_keys[i] < static_cast<K>(keynum));
    ASSERT_TRUE(h_metas[i] == kMetaV1);
    for (size_t j = 0; j < DIM; j++) {
      ASSERT_TRUE(h_vectors[i * DIM + j] == static_cast<V>(h_keys[i]));
    }
  }
}

TEST(MerlinHashTableTest, test_insert_and_evict) { test_insert_and_evict(); }
