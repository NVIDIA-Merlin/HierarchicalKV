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

#include <stdio.h>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.h"
#include "test_util.cuh"

constexpr uint64_t DIM = 64;
using K = int64_t;
using M = uint64_t;
using V = float;
using Table = nv::merlin::HashTable<K, V, M, DIM>;
using TableOptions = nv::merlin::HashTableOptions;

void test_save_to_file(std::string& prefix) {
  size_t keynum = 1 * 1024 * 1024;
  size_t capacity = 2 * 1024 * 1024;
  size_t buffer_size = 1024 * 1024;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  K* h_keys = nullptr;
  V* h_vectors = nullptr;
  M* h_metas = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_keys, keynum * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, keynum * sizeof(V) * DIM));
  CUDA_CHECK(cudaMallocHost(&h_metas, keynum * sizeof(M)));
  memset(h_keys, 0, keynum * sizeof(K));
  memset(h_vectors, 0, keynum * sizeof(V) * DIM);
  memset(h_metas, 0, keynum * sizeof(M));
  test_util::create_random_keys<K, M>(h_keys, h_metas, keynum);
  printf("Pass create random keys.\n");

  K* d_keys = nullptr;
  V* d_vectors = nullptr;
  M* d_metas = nullptr;
  test_util::getBufferOnDevice(&d_keys, keynum * sizeof(K), stream);
  test_util::getBufferOnDevice(&d_vectors, keynum * sizeof(V) * DIM, stream);
  test_util::getBufferOnDevice(&d_metas, keynum * sizeof(M), stream);
  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, keynum * sizeof(K),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, keynum * sizeof(V) * DIM,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, keynum * sizeof(M),
                             cudaMemcpyHostToDevice, stream));
  printf("Create buffers.\n");

  TableOptions options0;
  TableOptions options1;
  options0.init_capacity = capacity;
  options0.max_capacity = capacity;
  options0.evict_strategy = nv::merlin::EvictStrategy::kLru;

  options1 = options0;
  options1.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  std::unique_ptr<Table> table_0 = std::make_unique<Table>();
  std::unique_ptr<Table> table_1 = std::make_unique<Table>();
  table_0->init(options0);
  table_1->init(options1);
  printf("Init tables.\n");

  table_0->insert_or_assign(keynum, d_keys, d_vectors, /*metas=*/nullptr,
                            stream);
  printf("Fill table_0.\n");
  nv::merlin::LocalKVFile<K, V, M, DIM> file;
  std::string keyfile = prefix + ".keys";
  std::string valuefile = prefix + ".values";
  file.open(keyfile, valuefile, "wb");
  table_0->save(&file, buffer_size, stream);
  file.close();
  printf("table_0 saves.\n");
  file.open(keyfile, valuefile, "rb");
  table_1->load(&file, buffer_size, stream);
  file.close();
  printf("table_1 loads.\n");
  MERLIN_EXPECT_TRUE(
      test_util::tables_equal(table_0.get(), table_1.get(), stream),
      "Tables not equal");
  printf("table_0 and table_1 are equal.\n");
  CUDA_FREE_POINTERS(stream, d_keys, d_vectors, d_metas, h_keys, h_vectors,
                     h_metas);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main() {
  std::string prefix = "checkpoint";
  test_save_to_file(prefix);
}
