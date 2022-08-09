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

#include "common.cuh"
#include "merlin/initializers.cuh"
#include "merlin/optimizers.cuh"
#include "merlin_hashtable.cuh"

template <class K, class M, class V, size_t DIM>
void create_keys_in_one_buckets(K* h_keys, M* h_metas, V* h_vectors,
                                int KEY_NUM, int capacity,
                                int bucket_max_size = 128, int bucket_idx = 0) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  K candidate;
  K hashed_key;
  size_t global_idx;
  size_t bkt_idx;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    candidate = distr(eng);
    hashed_key = common::Murmur3HashHost(candidate);
    global_idx = hashed_key & (capacity - 1);
    bkt_idx = global_idx / bucket_max_size;
    if (bkt_idx == bucket_idx) {
      numbers.insert(candidate);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = num;
    for (size_t j = 0; j < DIM; j++) {
      *(h_vectors + i * DIM + j) = static_cast<float>((i * DIM + j) * 0.00001);
    }
    i++;
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

constexpr uint64_t DIM = 2;

using K = uint64_t;
using M = uint64_t;
using Vector = ValueArray<float, DIM>;
using Table = nv::merlin::HashTable<K, float, M, DIM>;
using TableOptions = nv::merlin::HashTableOptions;

template <class K, class M>
__forceinline__ __device__ bool erase_if_pred(const K& key, const M& meta,
                                              const K& pattern,
                                              const M& threshold) {
  return ((key & 0x7f > pattern) && (meta > threshold));
}

/* A demo of Pred for erase_if */
template <class K, class M>
__device__ Table::Pred pred = erase_if_pred<K, M>;

int test_basic() {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  M* h_metas;
  Vector* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_hbm_for_vectors = nv::merlin::GB(16);

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));

  common::create_random_keys<K, M>(h_keys, h_metas, KEY_NUM);

  K* d_keys;
  M* d_metas = nullptr;
  Vector* d_vectors;
  Vector* d_def_val;
  Vector** d_vectors_ptr;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_def_val, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector*)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMemset(d_def_val, 2, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector*)));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert(total_size == 0);

    std::cout << "before insert_or_assign: total_size = " << total_size
              << std::endl;
    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after 1st insert_or_assign: total_size = " << total_size
              << std::endl;
    assert(total_size == KEY_NUM);

    CUDA_CHECK(cudaMemset(d_vectors, 2, KEY_NUM * sizeof(Vector)));
    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after 2nd insert_or_assign: total_size = " << total_size
              << std::endl;
    assert(total_size == KEY_NUM);

    table->find(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_found,
                nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
    }
    std::cout << "after find, found_num = " << found_num << std::endl;
    assert(found_num == KEY_NUM);

    table->accum_or_assign(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors),
                           d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after accum: total_size = " << total_size << std::endl;
    assert(total_size == KEY_NUM);

    K pattern = 120;
    M threshold = 0;
    size_t erase_num = table->erase_if(pred<K, M>, pattern, threshold, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after erase_if: total_size = " << total_size
              << ", erase_num = " << erase_num << std::endl;
    assert((erase_num + total_size) == KEY_NUM);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after clear: total_size = " << total_size << std::endl;
    assert(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);

    // dump_counter = table->export_batch(table->capacity(), 0, d_keys,
    size_t* d_dump_counter = nullptr;
    common::getBufferOnDevice(&d_dump_counter, sizeof(size_t), stream);
    table->export_batch(table->capacity(), 0, d_dump_counter, d_keys,
                        reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaMemcpyAsync(&dump_counter, d_dump_counter, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "after export_batch: dump_counter = " << dump_counter
              << std::endl;
    assert(dump_counter == KEY_NUM);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas))
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();

  return 0;
}

int test_erase_if_pred() {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 64;
  constexpr uint64_t BUCKET_MAX_SIZE = 128;

  K* h_keys;
  M* h_metas;
  Vector* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.max_hbm_for_vectors = nv::merlin::GB(16);

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  Vector* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    create_keys_in_one_buckets<K, M, float, DIM>(
        h_keys, h_metas, reinterpret_cast<float*>(h_vectors), KEY_NUM,
        INIT_CAPACITY);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert(total_size == 0);

    table->insert_or_assign(
        KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert(total_size == BUCKET_MAX_SIZE);

    K pattern = 100;
    M threshold = 0;
    size_t erase_num = table->erase_if(pred<K, M>, pattern, threshold, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert((erase_num + total_size) == BUCKET_MAX_SIZE);

    table->find(KEY_NUM, d_keys, reinterpret_cast<float*>(d_vectors), d_found,
                d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector)));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        assert(h_metas[i] == h_keys[i]);
        for (int j = 0; j < DIM; j++) {
          assert(h_vectors[i].value[j] ==
                 static_cast<float>((i * DIM + j) * 0.00001));
        }
      }
    }
    assert(found_num == (BUCKET_MAX_SIZE - erase_num));

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert(total_size == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas))
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();

  return 0;
}

int main() {
  try {
    test_basic();
    test_erase_if_pred();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "All test cases passed!" << std::endl;
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << e.what() << std::endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
