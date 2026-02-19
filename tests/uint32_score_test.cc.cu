#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 8;
constexpr uint64_t CAPACITY = 1024;
constexpr uint64_t KEY_NUM = 256;

using K = uint64_t;
using V = float;
using S = uint32_t;
using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

namespace {

TableOptions default_options() {
  TableOptions options;
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.max_hbm_for_vectors = nv::merlin::GB(1);
  return options;
}

void fill_sequential(test_util::KVMSBuffer<K, V, S>& buffer) {
  for (size_t i = 0; i < buffer.len(); ++i) {
    K key = static_cast<K>(i + 1);
    buffer.keys.h_data[i] = key;
    buffer.scores.h_data[i] = static_cast<S>(key);
    for (size_t j = 0; j < buffer.dim(); ++j) {
      buffer.values.h_data[i * buffer.dim() + j] =
          static_cast<V>(key * 0.00001f);
    }
  }
}

}  // namespace

TEST(Uint32ScoreTest, FindOrInsertAndFind) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(default_options());

  test_util::KVMSBuffer<K, V, S> input;
  input.Reserve(KEY_NUM, DIM, stream);
  fill_sequential(input);
  input.SyncData(true, stream);

  table->find_or_insert(KEY_NUM, input.keys_ptr(), input.values_ptr(),
                        input.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(size, KEY_NUM);

  test_util::KVMSBuffer<K, V, S> output;
  output.Reserve(KEY_NUM, DIM, stream);
  output.ToZeros(stream);

  table->find(KEY_NUM, input.keys_ptr(), output.values_ptr(),
              output.status_ptr(), output.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  output.SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_TRUE(output.status.h_data[i]);
    ASSERT_EQ(output.scores.h_data[i], input.scores.h_data[i]);
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(output.values.h_data[i * DIM + j],
                input.values.h_data[i * DIM + j]);
    }
  }

  constexpr size_t MISSING_NUM = 16;
  test_util::KVMSBuffer<K, V, S> missing;
  missing.Reserve(MISSING_NUM, DIM, stream);
  missing.ToZeros(stream);
  for (size_t i = 0; i < MISSING_NUM; ++i) {
    missing.keys.h_data[i] = static_cast<K>(KEY_NUM + 1000 + i);
  }
  missing.SyncData(true, stream);

  table->find(MISSING_NUM, missing.keys_ptr(), missing.values_ptr(),
              missing.status_ptr(), missing.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  missing.SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < MISSING_NUM; ++i) {
    ASSERT_FALSE(missing.status.h_data[i]);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CudaCheckError();
}

TEST(Uint32ScoreTest, AssignScoresAndExport) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(default_options());

  test_util::KVMSBuffer<K, V, S> input;
  input.Reserve(KEY_NUM, DIM, stream);
  fill_sequential(input);
  input.SyncData(true, stream);

  table->find_or_insert(KEY_NUM, input.keys_ptr(), input.values_ptr(),
                        input.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < KEY_NUM; ++i) {
    input.scores.h_data[i] = static_cast<S>(1000 + i);
  }
  input.scores.h_data[0] = static_cast<S>(0);
  input.scores.h_data[1] = std::numeric_limits<S>::max();
  input.scores.h_data[2] = static_cast<S>(1);
  input.scores.h_data[3] = std::numeric_limits<S>::max() - 1;
  input.scores.SyncData(true, stream);

  table->assign_scores(KEY_NUM, input.keys_ptr(), input.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  test_util::KVMSBuffer<K, V, S> verify;
  verify.Reserve(KEY_NUM, DIM, stream);
  verify.ToZeros(stream);

  table->find(KEY_NUM, input.keys_ptr(), verify.values_ptr(),
              verify.status_ptr(), verify.scores_ptr(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  verify.SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_TRUE(verify.status.h_data[i]);
    ASSERT_EQ(verify.scores.h_data[i], input.scores.h_data[i]);
  }

  const size_t capacity = table->capacity();
  test_util::KVMSBuffer<K, V, S> exported;
  exported.Reserve(capacity, DIM, stream);
  exported.ToZeros(stream);

  size_t dumped =
      table->export_batch(capacity, 0, exported.keys_ptr(),
                          exported.values_ptr(), exported.scores_ptr(), stream);
  ASSERT_EQ(dumped, KEY_NUM);

  exported.SyncData(false, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::unordered_map<K, S> expected_scores;
  expected_scores.reserve(KEY_NUM);
  for (size_t i = 0; i < KEY_NUM; ++i) {
    expected_scores.emplace(static_cast<K>(i + 1), input.scores.h_data[i]);
  }

  for (size_t i = 0; i < dumped; ++i) {
    K key = exported.keys.h_data[i];
    auto it = expected_scores.find(key);
    ASSERT_NE(it, expected_scores.end());
    ASSERT_EQ(exported.scores.h_data[i], it->second);
    expected_scores.erase(it);
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(exported.values.h_data[i * DIM + j],
                static_cast<V>(key * 0.00001f));
    }
  }
  ASSERT_TRUE(expected_scores.empty());

  CUDA_CHECK(cudaStreamDestroy(stream));
  CudaCheckError();
}

TEST(Uint32ScoreTest, EvictCustomizedCorrectRateFull) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr float EXPECTED_CORRECT_RATE = 0.964f;
  const int rounds = 6;

  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.reserved_key_start_bit = 17;
  options.num_of_buckets_per_alloc = 128;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(16);

  K* h_keys_base = test_util::HostBuffer<K>(BATCH_SIZE).ptr();
  S* h_scores_base = test_util::HostBuffer<S>(BATCH_SIZE).ptr();
  V* h_vectors_base = test_util::HostBuffer<V>(BATCH_SIZE * options.dim).ptr();

  K* h_keys_temp = test_util::HostBuffer<K>(MAX_CAPACITY).ptr();
  S* h_scores_temp = test_util::HostBuffer<S>(MAX_CAPACITY).ptr();
  V* h_vectors_temp =
      test_util::HostBuffer<V>(MAX_CAPACITY * options.dim).ptr();

  K* d_keys_temp = nullptr;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp = nullptr;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, MAX_CAPACITY * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  size_t total_size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(total_size, 0);

  size_t global_start_key = 100000;
  size_t start_key = global_start_key;

  for (int r = 0; r < rounds; ++r) {
    const K expected_min_key =
        static_cast<K>(global_start_key + INIT_CAPACITY * r);
    const K expected_max_key =
        static_cast<K>(global_start_key + INIT_CAPACITY * (r + 1) - 1);
    const size_t expected_table_size =
        (r == 0) ? static_cast<size_t>(EXPECTED_CORRECT_RATE * INIT_CAPACITY)
                 : INIT_CAPACITY;

    for (int s = 0; s < STEPS; ++s) {
      test_util::create_continuous_keys<K, S, V, DIM>(
          h_keys_base, h_scores_base, h_vectors_base, BATCH_SIZE, start_key);
      start_key += BATCH_SIZE;

      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base, BATCH_SIZE * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base,
                            BATCH_SIZE * sizeof(S), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base,
                            BATCH_SIZE * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                              d_scores_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_GE(total_size, expected_table_size);
    ASSERT_EQ(MAX_CAPACITY, table->capacity());

    size_t dump_counter = table->export_batch(
        MAX_CAPACITY, 0, d_keys_temp, d_vectors_temp, d_scores_temp, stream);

    CUDA_CHECK(cudaMemcpy(h_keys_temp, d_keys_temp, MAX_CAPACITY * sizeof(K),
                          cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(h_scores_temp, d_scores_temp,
                          MAX_CAPACITY * sizeof(S), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(h_vectors_temp, d_vectors_temp,
                          MAX_CAPACITY * sizeof(V) * options.dim,
                          cudaMemcpyDefault));

    ASSERT_EQ(total_size, dump_counter);
    size_t bigger_score_counter = 0;
    K max_key = 0;

    for (size_t i = 0; i < dump_counter; ++i) {
      ASSERT_EQ(h_scores_temp[i], static_cast<S>(h_keys_temp[i]));
      max_key = std::max(max_key, h_keys_temp[i]);
      if (h_scores_temp[i] >= static_cast<S>(expected_min_key)) {
        bigger_score_counter++;
      }
      for (size_t j = 0; j < options.dim; ++j) {
        const V expected = static_cast<V>(h_keys_temp[i] * 0.00001);
        ASSERT_EQ(h_vectors_temp[i * options.dim + j], expected);
      }
    }

    float correct_rate =
        (bigger_score_counter * 1.0f) / static_cast<float>(MAX_CAPACITY);
    std::cout << "[Round " << r << "] "
              << "correct_rate=" << correct_rate << std::endl;
    ASSERT_GE(max_key, expected_max_key);
    ASSERT_GE(correct_rate, EXPECTED_CORRECT_RATE);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}
