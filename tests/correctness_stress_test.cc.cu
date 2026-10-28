/*
 * Cache correctness stress tests.
 *
 * These tests document and verify the legal API contract most likely to be
 * affected by contention: duplicate-key handling, dual-bucket unique-key
 * requirements, repeated-batch updates, and score-based admission behavior.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

namespace {

constexpr size_t kDim = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using TableMode = nv::merlin::TableMode;
using EvictStrategy = nv::merlin::EvictStrategy;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

void create_table(Table& table, TableMode mode, size_t capacity,
                  size_t dim = kDim, bool pure_hbm = false,
                  bool io_by_cpu = false) {
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = pure_hbm ? capacity * dim * sizeof(V) : 0;
  options.dim = dim;
  options.max_bucket_size = 128;
  options.table_mode = mode;
  options.io_by_cpu = io_by_cpu;
  options.api_lock = true;
  table.init(options);
}

void fill_sentinel_values(std::vector<V>& values, size_t n, size_t dim,
                          V sentinel) {
  values.assign(n * dim, sentinel);
}

template <typename T>
T* copy_to_device(const std::vector<T>& host) {
  T* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}

size_t count_found(const std::vector<bool>& found) {
  return std::count(found.begin(), found.end(), true);
}

std::vector<bool> find_keys(Table& table, const std::vector<K>& keys,
                            std::vector<V>* values_out = nullptr,
                            std::vector<S>* scores_out = nullptr) {
  K* d_keys = copy_to_device(keys);
  V* d_values = nullptr;
  bool* d_found = nullptr;
  S* d_scores = nullptr;
  const size_t dim = table.dim();
  CUDA_CHECK(cudaMalloc(&d_values, keys.size() * dim * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_found, keys.size() * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_found, 0, keys.size() * sizeof(bool)));
  if (scores_out != nullptr)
    CUDA_CHECK(cudaMalloc(&d_scores, keys.size() * sizeof(S)));

  table.find(keys.size(), d_keys, d_values, d_found, d_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint8_t> found_bytes(keys.size());
  CUDA_CHECK(cudaMemcpy(found_bytes.data(), d_found, keys.size() * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  std::vector<bool> found(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    found[i] = found_bytes[i] != 0;
  }

  if (values_out != nullptr) {
    values_out->resize(keys.size() * dim);
    CUDA_CHECK(cudaMemcpy(values_out->data(), d_values,
                          keys.size() * dim * sizeof(V),
                          cudaMemcpyDeviceToHost));
  }
  if (scores_out != nullptr) {
    scores_out->resize(keys.size());
    CUDA_CHECK(cudaMemcpy(scores_out->data(), d_scores, keys.size() * sizeof(S),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_scores));
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_found));
  return found;
}

void insert_or_assign(Table& table, const std::vector<K>& keys,
                      const std::vector<V>& values,
                      const std::vector<S>& scores, bool unique_key,
                      bool ignore_evict_strategy = false) {
  K* d_keys = copy_to_device(keys);
  V* d_values = copy_to_device(values);
  S* d_scores = copy_to_device(scores);

  table.insert_or_assign(keys.size(), d_keys, d_values, d_scores, 0, unique_key,
                         ignore_evict_strategy);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
}

bool is_whole_sentinel_vector(const std::vector<V>& values, size_t row,
                              const std::vector<V>& legal_sentinels) {
  const V first = values[row * kDim];
  if (std::find(legal_sentinels.begin(), legal_sentinels.end(), first) ==
      legal_sentinels.end()) {
    return false;
  }
  for (size_t d = 1; d < kDim; d++) {
    if (values[row * kDim + d] != first) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(CorrectnessStress, DuplicateSameKeyBatchThroughputNoTornValue) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 1024;
  constexpr K kHotKey = 42;

  Table table;
  create_table(table, TableMode::kThroughput, kCapacity);

  std::vector<K> keys(kBatch, kHotKey);
  std::vector<V> values(kBatch * kDim);
  std::vector<S> scores(kBatch);
  std::vector<V> legal_sentinels(kBatch);

  for (size_t i = 0; i < kBatch; i++) {
    const V sentinel = static_cast<V>(1000 + i);
    legal_sentinels[i] = sentinel;
    scores[i] = i + 1;
    for (size_t d = 0; d < kDim; d++) {
      values[i * kDim + d] = sentinel;
    }
  }

  insert_or_assign(table, keys, values, scores, false);

  EXPECT_EQ(table.size(0), static_cast<size_t>(1));

  std::vector<V> found_values;
  std::vector<S> found_scores;
  auto found =
      find_keys(table, std::vector<K>{kHotKey}, &found_values, &found_scores);
  ASSERT_EQ(count_found(found), static_cast<size_t>(1));
  EXPECT_TRUE(is_whole_sentinel_vector(found_values, 0, legal_sentinels))
      << "duplicate same-key update produced a torn or unknown value";
  ASSERT_GE(found_scores[0], S{1});
  ASSERT_LE(found_scores[0], S{kBatch});
  EXPECT_EQ(found_values[0], static_cast<V>(999 + found_scores[0]));
}

TEST(CorrectnessStress, DualBucketRejectsNonUniqueBatchContract) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 16;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> keys(kBatch, 7);
  std::vector<V> values(kBatch * kDim, 1.0f);
  std::vector<S> scores(kBatch, 1);

  K* d_keys = copy_to_device(keys);
  V* d_values = copy_to_device(values);
  S* d_scores = copy_to_device(scores);

  EXPECT_THROW(
      table.insert_or_assign(kBatch, d_keys, d_values, d_scores, 0, false),
      std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST(CorrectnessStress, DualBucketRepeatedBatchUpdateNoDuplicates) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 2048;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> keys(kBatch);
  std::iota(keys.begin(), keys.end(), 1);
  std::vector<S> scores(kBatch);
  std::iota(scores.begin(), scores.end(), 1);

  std::vector<V> values_v1;
  std::vector<V> values_v2;
  fill_sentinel_values(values_v1, kBatch, kDim, 1.0f);
  fill_sentinel_values(values_v2, kBatch, kDim, 2.0f);

  insert_or_assign(table, keys, values_v1, scores, true);
  for (auto& score : scores) score += kBatch;
  insert_or_assign(table, keys, values_v2, scores, true);

  EXPECT_EQ(table.size(0), kBatch);

  std::vector<V> found_values;
  std::vector<S> found_scores;
  auto found = find_keys(table, keys, &found_values, &found_scores);
  ASSERT_EQ(count_found(found), kBatch);
  for (size_t i = 0; i < kBatch; i++) {
    EXPECT_EQ(found_scores[i], scores[i]);
    for (size_t d = 0; d < kDim; d++) {
      EXPECT_FLOAT_EQ(found_values[i * kDim + d], 2.0f)
          << "key " << keys[i] << " dim " << d;
    }
  }
}

namespace {

struct StressPath {
  TableMode mode;
  bool pure_hbm;
  bool unique;
  bool io_by_cpu;
  const char* name;
};

class CorrectnessPaths : public testing::TestWithParam<StressPath> {};

bool legal_role_generation(const std::string& roles, size_t row, S score,
                           V value, bool completed) {
  if (score == 0 || (score - 1) % 1000 != row ||
      value != static_cast<V>(1000 + score))
    return false;
  const size_t generation = (score - 1) / 1000;
  if (generation == 0)
    return !completed || roles.find_first_of("UI") == std::string::npos;
  return generation <= roles.size() && roles[generation - 1] != 'R';
}

TEST(CorrectnessStress, RoleGenerationOracleRejectsNoOpWrites) {
  for (const std::string roles : {"RR", "UU", "II", "RU", "RI", "UI", "RUI"}) {
    SCOPED_TRACE(roles);
    for (size_t row = 0; row < 33; ++row) {
      EXPECT_TRUE(
          legal_role_generation(roles, row, row + 1, 1001 + row, false));
      EXPECT_EQ(legal_role_generation(roles, row, row + 1, 1001 + row, true),
                roles.find_first_of("UI") == std::string::npos);
      EXPECT_FALSE(
          legal_role_generation(roles, row, row + 1, 1002 + row, true));
      for (size_t worker = 0; worker < roles.size(); ++worker) {
        const S score = (worker + 1) * 1000 + row + 1;
        EXPECT_EQ(legal_role_generation(roles, row, score, 1000 + score, true),
                  roles[worker] != 'R');
        EXPECT_FALSE(
            legal_role_generation(roles, row, score + 1, 1001 + score, true));
      }
    }
  }
}

std::vector<K> collision_keys(size_t count, TableMode mode, K start = 1) {
  constexpr size_t capacity = 512;
  std::vector<K> keys;
  for (K candidate = start; keys.size() < count; ++candidate) {
    const auto hash = nv::merlin::Murmur3HashHost(candidate);
    const size_t first = static_cast<uint32_t>(hash) % 4;
    size_t second = static_cast<uint32_t>(hash >> 32) % 4;
    if (second == first) second = (second + 1) % 4;
    if ((mode == TableMode::kThroughput &&
         (hash & (capacity - 1)) / 128 == 0) ||
        (mode == TableMode::kMemory && first == 0 && second == 1))
      keys.push_back(candidate);
  }
  return keys;
}

using GenerationMap = std::unordered_map<K, std::pair<S, V>>;

void check_residents(Table& table, const GenerationMap& expected,
                     size_t expected_size) {
  const size_t capacity = table.capacity();
  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&device_keys, capacity * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&device_values, capacity * kDim * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&device_scores, capacity * sizeof(S)));
  const size_t exported = table.export_batch(capacity, 0, device_keys,
                                             device_values, device_scores, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<K> keys(exported);
  std::vector<V> values(exported * kDim);
  std::vector<S> scores(exported);
  CUDA_CHECK(cudaMemcpy(keys.data(), device_keys, exported * sizeof(K),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(values.data(), device_values, values.size() * sizeof(V),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scores.data(), device_scores, exported * sizeof(S),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_keys));
  CUDA_CHECK(cudaFree(device_values));
  CUDA_CHECK(cudaFree(device_scores));
  ASSERT_EQ(exported, expected_size);
  EXPECT_EQ(table.size(0), expected_size);
  EXPECT_EQ(std::unordered_set<K>(keys.begin(), keys.end()).size(), exported);
  if (keys.empty()) return;
  std::vector<V> found_values;
  std::vector<S> found_scores;
  const auto found = find_keys(table, keys, &found_values, &found_scores);
  ASSERT_EQ(count_found(found), exported);
  for (size_t row = 0; row < exported; ++row) {
    const auto generation = expected.find(keys[row]);
    ASSERT_NE(generation, expected.end()) << keys[row];
    EXPECT_EQ(scores[row], generation->second.first) << keys[row];
    EXPECT_EQ(found_scores[row], scores[row]);
    for (size_t dim = 0; dim < kDim; ++dim) {
      EXPECT_EQ(values[row * kDim + dim], generation->second.second)
          << keys[row];
      EXPECT_EQ(found_values[row * kDim + dim], values[row * kDim + dim]);
    }
  }
}

TEST_P(CorrectnessPaths, SameBucketOrPairGenerationCompletion) {
  const auto path = GetParam();
  for (size_t count : {2, 31, 32, 33}) {
    SCOPED_TRACE(count);
    Table table;
    create_table(table, path.mode, 512, kDim, path.pure_hbm, path.io_by_cpu);
    const auto distinct = collision_keys(count, path.mode);
    std::vector<K> keys;
    std::vector<V> values;
    std::vector<S> scores;
    const size_t repeats = path.unique ? 1 : 4;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
      for (K key : distinct) {
        keys.push_back(key);
        scores.push_back(keys.size());
        values.insert(values.end(), kDim, static_cast<V>(1000 + keys.size()));
      }
    }
    insert_or_assign(table, keys, values, scores, path.unique);
    std::vector<V> found_values;
    std::vector<S> found_scores;
    const auto found = find_keys(table, distinct, &found_values, &found_scores);
    ASSERT_EQ(count_found(found), count);
    EXPECT_EQ(table.size(0), count);
    for (size_t row = 0; row < count; ++row) {
      ASSERT_GE(found_scores[row], S{1});
      ASSERT_LE(found_scores[row], scores.size());
      const size_t generation = found_scores[row] - 1;
      EXPECT_EQ(keys[generation], distinct[row]);
      for (size_t dim = 0; dim < kDim; ++dim)
        EXPECT_EQ(found_values[row * kDim + dim],
                  values[generation * kDim + dim]);
    }
    GenerationMap expected;
    values.clear();
    for (size_t row = 0; row < count; ++row) {
      const V value = static_cast<V>(2000 + row);
      values.insert(values.end(), kDim, value);
      expected.emplace(distinct[row], std::make_pair(S{42}, value));
    }
    insert_or_assign(table, distinct, values, std::vector<S>(count, 42),
                     path.unique);
    check_residents(table, expected, count);
  }
}

TEST_P(CorrectnessPaths, SaturatedCandidatesLowEqualScoreAndContinuousIngress) {
  const auto path = GetParam();
  const size_t slots = path.mode == TableMode::kMemory ? 256 : 128;
  const auto keys = collision_keys(slots + 33 * 6, path.mode);
  Table table;
  create_table(table, path.mode, 512, kDim, path.pure_hbm, path.io_by_cpu);
  GenerationMap expected;
  auto insert = [&](size_t offset, size_t count, S score) {
    std::vector<K> batch(keys.begin() + offset, keys.begin() + offset + count);
    std::vector<V> values;
    for (size_t row = offset; row < offset + count; ++row) {
      const V value = static_cast<V>(1000 + row);
      expected.emplace(keys[row], std::make_pair(score, value));
      values.insert(values.end(), kDim, value);
    }
    insert_or_assign(table, batch, values, std::vector<S>(count, score),
                     path.unique);
    return batch;
  };
  insert(0, slots, 10);
  check_residents(table, expected, slots);
  auto low = insert(slots, 33, 1);
  EXPECT_EQ(count_found(find_keys(table, low)), size_t{0});
  check_residents(table, expected, slots);
  insert(slots + 33, 33, 10);
  check_residents(table, expected, slots);
  for (size_t round = 0; round < 4; ++round) {
    SCOPED_TRACE(round);
    const auto ingress = insert(slots + 33 * (round + 2), 33, 20 + round);
    EXPECT_EQ(count_found(find_keys(table, ingress)), ingress.size());
    check_residents(table, expected, slots);
  }
}

TEST_P(CorrectnessPaths, SupportedOverlappingRolesCompleteWithGenerationPairs) {
  const auto path = GetParam();
  const std::vector<std::string> combinations =
      path.mode == TableMode::kMemory
          ? std::vector<std::string>{"RR", "II", "RI"}
          : std::vector<std::string>{"RR", "UU", "II", "RU", "RI", "UI", "RUI"};
  constexpr size_t count = 33;
  const auto keys = collision_keys(count, path.mode);
  for (const auto& roles : combinations) {
    SCOPED_TRACE(roles);
    Table table;
    create_table(table, path.mode, 512, kDim, path.pure_hbm, path.io_by_cpu);
    std::vector<S> initial_scores(count);
    std::vector<V> initial_values;
    for (size_t row = 0; row < count; ++row) {
      initial_scores[row] = row + 1;
      initial_values.insert(initial_values.end(), kDim,
                            static_cast<V>(1001 + row));
    }
    insert_or_assign(table, keys, initial_values, initial_scores, path.unique);
    struct Buffers {
      cudaStream_t stream;
      K* keys;
      V* values;
      S* scores;
      V* found_values;
      S* found_scores;
      bool* found;
    };
    std::vector<Buffers> buffers(roles.size());
    for (size_t worker = 0; worker < roles.size(); ++worker) {
      auto& buffer = buffers[worker];
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&buffer.stream, cudaStreamNonBlocking));
      buffer.keys = copy_to_device(keys);
      std::vector<S> scores(count);
      std::vector<V> values;
      for (size_t row = 0; row < count; ++row) {
        scores[row] = (worker + 1) * 1000 + row + 1;
        values.insert(values.end(), kDim, static_cast<V>(1000 + scores[row]));
      }
      buffer.scores = copy_to_device(scores);
      buffer.values = copy_to_device(values);
      CUDA_CHECK(cudaMalloc(&buffer.found_values, count * kDim * sizeof(V)));
      CUDA_CHECK(cudaMalloc(&buffer.found_scores, count * sizeof(S)));
      CUDA_CHECK(cudaMalloc(&buffer.found, count * sizeof(bool)));
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<size_t> errors{0};
    std::atomic<size_t> completed{0};
    std::vector<std::exception_ptr> exceptions(roles.size());
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < roles.size(); ++worker) {
      threads.emplace_back([&, worker] {
        ready.fetch_add(1);
        while (!start.load()) std::this_thread::yield();
        try {
          CUDA_CHECK(cudaSetDevice(device));
          auto& buffer = buffers[worker];
          for (size_t round = 0; round < 4; ++round) {
            if (roles[worker] == 'R') {
              CUDA_CHECK(cudaMemsetAsync(buffer.found, 0, count * sizeof(bool),
                                         buffer.stream));
              table.find(count, buffer.keys, buffer.found_values, buffer.found,
                         buffer.found_scores, buffer.stream);
              std::vector<uint8_t> found(count);
              std::vector<S> scores(count);
              std::vector<V> values(count * kDim);
              CUDA_CHECK(cudaMemcpyAsync(
                  found.data(), buffer.found, count * sizeof(bool),
                  cudaMemcpyDeviceToHost, buffer.stream));
              CUDA_CHECK(cudaMemcpyAsync(
                  scores.data(), buffer.found_scores, count * sizeof(S),
                  cudaMemcpyDeviceToHost, buffer.stream));
              CUDA_CHECK(cudaMemcpyAsync(
                  values.data(), buffer.found_values, values.size() * sizeof(V),
                  cudaMemcpyDeviceToHost, buffer.stream));
              CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
              for (size_t row = 0; row < count; ++row) {
                bool valid = found[row] &&
                             legal_role_generation(roles, row, scores[row],
                                                   values[row * kDim], false);
                for (size_t dim = 1; dim < kDim; ++dim)
                  valid =
                      valid && values[row * kDim + dim] == values[row * kDim];
                errors.fetch_add(!valid);
              }
            } else if (roles[worker] == 'U') {
              table.assign(count, buffer.keys, buffer.values, buffer.scores,
                           buffer.stream, path.unique);
            } else {
              table.insert_or_assign(count, buffer.keys, buffer.values,
                                     buffer.scores, buffer.stream, path.unique);
            }
            CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
            completed.fetch_add(1);
          }
        } catch (...) {
          exceptions[worker] = std::current_exception();
        }
      });
    }
    while (ready.load() != roles.size()) std::this_thread::yield();
    start.store(true);
    for (auto& thread : threads) thread.join();
    for (auto& buffer : buffers) {
      CUDA_CHECK(cudaFree(buffer.keys));
      CUDA_CHECK(cudaFree(buffer.values));
      CUDA_CHECK(cudaFree(buffer.scores));
      CUDA_CHECK(cudaFree(buffer.found_values));
      CUDA_CHECK(cudaFree(buffer.found_scores));
      CUDA_CHECK(cudaFree(buffer.found));
      CUDA_CHECK(cudaStreamDestroy(buffer.stream));
    }
    for (const auto& error : exceptions) {
      if (error) std::rethrow_exception(error);
    }
    EXPECT_EQ(completed.load(), roles.size() * 4);
    EXPECT_EQ(errors.load(), size_t{0});
    std::vector<V> values;
    std::vector<S> scores;
    const auto found = find_keys(table, keys, &values, &scores);
    ASSERT_EQ(count_found(found), count);
    EXPECT_EQ(table.size(0), count);
    for (size_t row = 0; row < count; ++row) {
      EXPECT_TRUE(legal_role_generation(roles, row, scores[row],
                                        values[row * kDim], true))
          << keys[row];
      for (size_t dim = 1; dim < kDim; ++dim)
        EXPECT_EQ(values[row * kDim + dim], values[row * kDim]);
    }
  }
}

TEST(CorrectnessStress, FullSingleBucketPipelineGenerationCompletion) {
  Table table;
  create_table(table, TableMode::kThroughput, 512, kDim, true);
  std::vector<K> keys;
  for (size_t bucket = 0; bucket < 4; ++bucket) {
    size_t selected = 0;
    for (K candidate = 1; selected < 128; ++candidate) {
      if ((nv::merlin::Murmur3HashHost(candidate) & 511) / 128 != bucket)
        continue;
      keys.push_back(candidate);
      ++selected;
    }
  }
  GenerationMap expected;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t row = 0; row < keys.size(); ++row) {
    scores.push_back(row + 1);
    values.insert(values.end(), kDim, static_cast<V>(1001 + row));
    expected.emplace(keys[row], std::make_pair(scores.back(), values.back()));
  }
  insert_or_assign(table, keys, values, scores, true);
  check_residents(table, expected, 512);
  for (size_t count : {2, 31, 32, 33}) {
    SCOPED_TRACE(count);
    const std::vector<K> updates(keys.begin(), keys.begin() + count);
    for (size_t round = 0; round < 8; ++round) {
      const S score = 1000 + count * 8 + round;
      const V value = static_cast<V>(1000 + score);
      insert_or_assign(table, updates, std::vector<V>(count * kDim, value),
                       std::vector<S>(count, score), true);
      for (K key : updates) expected[key] = std::make_pair(score, value);
      check_residents(table, expected, 512);
    }
  }
}

TEST(CorrectnessStress, UniqueLargeValueFallbackGenerationPairs) {
  constexpr size_t dim = 160;
  const auto keys = collision_keys(33, TableMode::kThroughput);
  for (bool pure_hbm : {false, true}) {
    SCOPED_TRACE(pure_hbm);
    Table table;
    create_table(table, TableMode::kThroughput, 512, dim, pure_hbm);
    for (size_t generation = 1; generation <= 2; ++generation) {
      std::vector<V> values;
      std::vector<S> scores;
      for (size_t row = 0; row < keys.size(); ++row) {
        scores.push_back(generation * 1000 + row);
        values.insert(values.end(), dim, static_cast<V>(1000 + scores.back()));
      }
      insert_or_assign(table, keys, values, scores, true);
      std::vector<V> found_values;
      std::vector<S> found_scores;
      const auto found = find_keys(table, keys, &found_values, &found_scores);
      EXPECT_EQ(count_found(found), keys.size());
      EXPECT_EQ(table.size(0), keys.size());
      EXPECT_EQ(found_scores, scores);
      EXPECT_EQ(found_values, values);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Supported, CorrectnessPaths,
    testing::Values(StressPath{TableMode::kThroughput, true, true, false,
                               "SingleHbmUnique"},
                    StressPath{TableMode::kThroughput, true, false, false,
                               "SingleHbmNonUnique"},
                    StressPath{TableMode::kThroughput, false, true, false,
                               "SinglePipelineUnique"},
                    StressPath{TableMode::kThroughput, false, false, false,
                               "SinglePipelineNonUnique"},
                    StressPath{TableMode::kThroughput, false, true, true,
                               "SingleCpuIoUnique"},
                    StressPath{TableMode::kThroughput, false, false, true,
                               "SingleCpuIoNonUnique"},
                    StressPath{TableMode::kMemory, false, true, false,
                               "DualPureHbmUnique"}),
    [](const testing::TestParamInfo<StressPath>& info) {
      return info.param.name;
    });

template <int strategy>
void check_compatible_value_score_completion_case(const size_t dimension,
                                                  const size_t populated,
                                                  const size_t count) {
  constexpr size_t capacity = 128;
  constexpr K missing_key = 1000000;
  SCOPED_TRACE(dimension);
  SCOPED_TRACE(populated);
  SCOPED_TRACE(count);
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = capacity * dimension * sizeof(V);
  options.dim = dimension;
  options.max_bucket_size = capacity;
  options.table_mode = TableMode::kThroughput;
  options.api_lock = true;
  nv::merlin::HashTable<K, V, S, strategy> table;
  table.init(options);
  std::vector<K> keys(populated);
  std::iota(keys.begin(), keys.end(), 1);
  std::vector<V> expected_values(populated * dimension, -1);
  std::vector<S> expected_scores(populated, 100);
  K* device_keys = copy_to_device(keys);
  V* device_values = copy_to_device(expected_values);
  S* device_scores = copy_to_device(expected_scores);
  table.insert_or_assign(populated, device_keys, device_values, device_scores,
                         0, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<K> update_keys(keys.begin(), keys.begin() + count);
  if (count > 1) update_keys.back() = missing_key;
  K* device_update_keys = copy_to_device(update_keys);
  V* device_update_values = nullptr;
  S* device_update_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&device_update_values, count * dimension * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&device_update_scores, count * sizeof(S)));
  std::vector<K> query_keys = keys;
  query_keys.push_back(missing_key);
  K* device_query_keys = copy_to_device(query_keys);
  V* device_query_values = nullptr;
  S* device_query_scores = nullptr;
  bool* device_found = nullptr;
  CUDA_CHECK(cudaMalloc(&device_query_values,
                        query_keys.size() * dimension * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&device_query_scores, query_keys.size() * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&device_found, query_keys.size() * sizeof(bool)));
  cudaStream_t value_stream;
  cudaStream_t score_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&value_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&score_stream, cudaStreamNonBlocking));
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  for (size_t round = 0; round < 8; ++round) {
    SCOPED_TRACE(round);
    std::vector<V> update_values(count * dimension);
    std::vector<S> update_scores(count);
    for (size_t row = 0; row < count; ++row) {
      update_scores[row] = strategy == EvictStrategy::kLfu
                               ? round + row + 1
                               : 1000 + round * count + row;
      for (size_t component = 0; component < dimension; ++component) {
        const V value =
            10000 + round * count * dimension + row * dimension + component;
        update_values[row * dimension + component] = value;
        if (update_keys[row] != missing_key) {
          expected_values[row * dimension + component] = value;
        }
      }
      if (update_keys[row] != missing_key) {
        if (strategy == EvictStrategy::kLfu) {
          expected_scores[row] += update_scores[row];
        } else {
          expected_scores[row] = update_scores[row];
        }
      }
    }
    CUDA_CHECK(cudaMemcpy(device_update_values, update_values.data(),
                          update_values.size() * sizeof(V),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_update_scores, update_scores.data(),
                          update_scores.size() * sizeof(S),
                          cudaMemcpyHostToDevice));
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::exception_ptr> failures(2);
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < 2; ++worker) {
      workers.emplace_back([&, worker] {
        ready.fetch_add(1);
        while (!start.load()) std::this_thread::yield();
        try {
          CUDA_CHECK(cudaSetDevice(device));
          if (worker == 0) {
            table.assign_values(count, device_update_keys, device_update_values,
                                value_stream, true);
          } else {
            table.assign_scores(count, device_update_keys, device_update_scores,
                                score_stream, true);
          }
        } catch (...) {
          failures[worker] = std::current_exception();
        }
      });
    }
    while (ready.load() != workers.size()) std::this_thread::yield();
    start.store(true);
    for (auto& worker : workers) worker.join();
    for (auto failure : failures) {
      if (failure) std::rethrow_exception(failure);
    }
    CUDA_CHECK(cudaStreamSynchronize(value_stream));
    CUDA_CHECK(cudaStreamSynchronize(score_stream));
    CUDA_CHECK(
        cudaMemset(device_found, 0xff, query_keys.size() * sizeof(bool)));
    table.find(query_keys.size(), device_query_keys, device_query_values,
               device_found, device_query_scores, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> found(query_keys.size());
    std::vector<V> actual_values(query_keys.size() * dimension);
    std::vector<S> actual_scores(query_keys.size());
    CUDA_CHECK(cudaMemcpy(found.data(), device_found, found.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_values.data(), device_query_values,
                          actual_values.size() * sizeof(V),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_scores.data(), device_query_scores,
                          actual_scores.size() * sizeof(S),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(table.size(0), populated);
    EXPECT_EQ(found.back(), 0);
    for (size_t row = 0; row < populated; ++row) {
      EXPECT_EQ(found[row], 1) << row;
      EXPECT_EQ(actual_scores[row], expected_scores[row]) << row;
      for (size_t component = 0; component < dimension; ++component) {
        EXPECT_EQ(actual_values[row * dimension + component],
                  expected_values[row * dimension + component])
            << row << ':' << component;
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(value_stream));
  CUDA_CHECK(cudaStreamDestroy(score_stream));
  CUDA_CHECK(cudaFree(device_keys));
  CUDA_CHECK(cudaFree(device_values));
  CUDA_CHECK(cudaFree(device_scores));
  CUDA_CHECK(cudaFree(device_update_keys));
  CUDA_CHECK(cudaFree(device_update_values));
  CUDA_CHECK(cudaFree(device_update_scores));
  CUDA_CHECK(cudaFree(device_query_keys));
  CUDA_CHECK(cudaFree(device_query_values));
  CUDA_CHECK(cudaFree(device_query_scores));
  CUDA_CHECK(cudaFree(device_found));
}

template <int strategy>
void check_compatible_value_score_completion() {
  const size_t paths[][2] = {{128, 48}, {128, 112}, {4, 112}, {4, 48}};
  for (const auto& path : paths) {
    const size_t dimension = path[0];
    const size_t populated = path[1];
    for (size_t count : {1U, 17U, 33U}) {
      check_compatible_value_score_completion_case<strategy>(dimension,
                                                             populated, count);
    }
  }
}

}  // namespace

TEST(CorrectnessStress, CompatibleValueScoreWritersCompleteBothEffects) {
  check_compatible_value_score_completion<EvictStrategy::kCustomized>();
}

#if defined(HKV_ENABLE_P1_PARTITION_TESTS) && HKV_ENABLE_P1_PARTITION_TESTS
TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop48_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 48, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop48_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 48, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop48_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 48, 33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop112_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 112, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop112_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 112, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim128_Pop112_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      128, 112, 33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop112_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 112, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop112_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 112, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop112_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 112, 33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop48_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 48, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop48_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 48, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersCompleteBothEffects_Dim4_Pop48_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kCustomized>(
      4, 48, 33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop48_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 48, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop48_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 48,
                                                                    17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop48_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 48,
                                                                    33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop112_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 112,
                                                                    1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop112_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 112,
                                                                    17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim128_Pop112_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(128, 112,
                                                                    33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop112_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 112, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop112_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 112, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop112_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 112, 33);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop48_Count1) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 48, 1);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop48_Count17) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 48, 17);
}

TEST(CorrectnessStress,
     CompatibleValueScoreWritersApplyLfuExactlyOnce_Dim4_Pop48_Count33) {
  check_compatible_value_score_completion_case<EvictStrategy::kLfu>(4, 48, 33);
}

#endif

TEST(CorrectnessStress, CompatibleValueScoreWritersApplyLfuExactlyOnce) {
  check_compatible_value_score_completion<EvictStrategy::kLfu>();
}

TEST(CorrectnessStress, PureHbmCompatibleValueWritersPublishWholeRows) {
  for (size_t dimension : {4U, 16U, 64U, 513U}) {
    for (bool unique : {false, true}) {
      for (size_t count :
           {1U, 15U, 16U, 17U, 31U, 32U, 33U, 127U, 128U, 129U}) {
        SCOPED_TRACE(dimension);
        SCOPED_TRACE(unique);
        SCOPED_TRACE(count);
        TableOptions options;
        options.init_capacity = 2048;
        options.max_capacity = 2048;
        options.max_hbm_for_vectors = 2048 * dimension * sizeof(V);
        options.dim = dimension;
        options.max_bucket_size = 128;
        options.block_size = count % 2 ? 33 : 65;
        options.api_lock = true;
        Table table;
        table.init(options);
        std::vector<K> keys(count);
        std::iota(keys.begin(), keys.end(), 1);
        std::vector<S> scores(count, 100);
        std::vector<V> initial(count * dimension);
        for (size_t row = 0; row < count; ++row) {
          for (size_t component = 0; component < dimension; ++component) {
            initial[row * dimension + component] = row * 1024 + component;
          }
        }
        K* device_keys = copy_to_device(keys);
        S* device_scores = copy_to_device(scores);
        V* device_values = copy_to_device(initial);
        table.insert_or_assign(count, device_keys, device_values, device_scores,
                               0, unique);
        CUDA_CHECK(cudaStreamSynchronize(0));
        struct Writer {
          cudaStream_t stream;
          size_t count;
          K* keys;
          V* values;
        };
        std::vector<Writer> writers(2);
        const size_t middle = count / 2;
        for (size_t worker = 0; worker < writers.size(); ++worker) {
          const size_t begin = worker == 0 ? 0 : middle;
          const size_t end = worker == 0 ? middle + 1 : count;
          std::vector<K> update_keys(keys.begin() + begin, keys.begin() + end);
          std::vector<V> values((end - begin) * dimension);
          for (size_t row = begin; row < end; ++row) {
            for (size_t component = 0; component < dimension; ++component) {
              values[(row - begin) * dimension + component] =
                  (worker + 1) * 100000 + row * 1024 + component;
            }
          }
          auto& writer = writers[worker];
          CUDA_CHECK(
              cudaStreamCreateWithFlags(&writer.stream, cudaStreamNonBlocking));
          writer.count = update_keys.size();
          writer.keys = copy_to_device(update_keys);
          writer.values = copy_to_device(values);
        }
        bool* device_found = nullptr;
        CUDA_CHECK(cudaMalloc(&device_found, count * sizeof(bool)));
        cudaStream_t consumer;
        CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        std::vector<std::exception_ptr> failures(2);
        std::vector<std::thread> threads;
        for (size_t worker = 0; worker < writers.size(); ++worker) {
          threads.emplace_back([&, worker] {
            ready.fetch_add(1);
            while (!start.load()) std::this_thread::yield();
            try {
              CUDA_CHECK(cudaSetDevice(device));
              auto& writer = writers[worker];
              table.assign_values(writer.count, writer.keys, writer.values,
                                  writer.stream, unique);
            } catch (...) {
              failures[worker] = std::current_exception();
            }
          });
        }
        while (ready.load() != writers.size()) std::this_thread::yield();
        start.store(true);
        for (auto& thread : threads) thread.join();
        for (auto failure : failures) {
          if (failure) std::rethrow_exception(failure);
        }
        CUDA_CHECK(cudaMemsetAsync(device_found, 0xff, count * sizeof(bool),
                                   consumer));
        table.find(count, device_keys, device_values, device_found,
                   device_scores, consumer);
        CUDA_CHECK(cudaStreamSynchronize(consumer));
        std::vector<uint8_t> found(count);
        CUDA_CHECK(cudaMemcpy(found.data(), device_found, count * sizeof(bool),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(initial.data(), device_values,
                              initial.size() * sizeof(V),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scores.data(), device_scores, count * sizeof(S),
                              cudaMemcpyDeviceToHost));
        for (size_t row = 0; row < count; ++row) {
          EXPECT_EQ(found[row], 1);
          EXPECT_EQ(scores[row], 100);
          const V base = initial[row * dimension] - row * 1024;
          if (row < middle) EXPECT_EQ(base, 100000);
          if (row > middle) EXPECT_EQ(base, 200000);
          if (row == middle) EXPECT_TRUE(base == 100000 || base == 200000);
          for (size_t component = 0; component < dimension; ++component) {
            EXPECT_EQ(initial[row * dimension + component],
                      base + row * 1024 + component);
          }
        }
        for (auto& writer : writers) {
          CUDA_CHECK(cudaStreamSynchronize(writer.stream));
          CUDA_CHECK(cudaStreamDestroy(writer.stream));
          CUDA_CHECK(cudaFree(writer.keys));
          CUDA_CHECK(cudaFree(writer.values));
        }
        CUDA_CHECK(cudaStreamDestroy(consumer));
        CUDA_CHECK(cudaFree(device_keys));
        CUDA_CHECK(cudaFree(device_scores));
        CUDA_CHECK(cudaFree(device_values));
        CUDA_CHECK(cudaFree(device_found));
      }
    }
  }
}

TEST(CorrectnessStress, MemoryModeScoreAdmissionAccounting) {
  constexpr size_t kCapacity = 128 * 128;
  constexpr size_t kFill = kCapacity;
  constexpr size_t kBurst = 512;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> fill_keys(kFill);
  std::iota(fill_keys.begin(), fill_keys.end(), 1);
  std::vector<V> fill_values(kFill * kDim, 1.0f);
  std::vector<S> fill_scores(kFill);
  for (size_t i = 0; i < kFill; i++) {
    fill_scores[i] = 1000000 + i;
  }

  insert_or_assign(table, fill_keys, fill_values, fill_scores, true, true);
  const size_t size_after_fill = table.size(0);
  ASSERT_GT(size_after_fill, static_cast<size_t>(kFill * 0.95))
      << "admission accounting test requires a near-full table";

  std::vector<K> low_keys(kBurst);
  std::iota(low_keys.begin(), low_keys.end(), kFill + 1);
  std::vector<V> low_values(kBurst * kDim, 2.0f);
  std::vector<S> low_scores(kBurst, 1);

  insert_or_assign(table, low_keys, low_values, low_scores, true, false);
  const size_t size_after_low = table.size(0);
  EXPECT_LE(size_after_low, kCapacity);

  auto low_found = find_keys(table, low_keys);
  const size_t low_accepted = count_found(low_found);

  std::vector<K> high_keys(kBurst);
  std::iota(high_keys.begin(), high_keys.end(), kFill + 1 + kBurst);
  std::vector<V> high_values(kBurst * kDim, 3.0f);
  std::vector<S> high_scores(kBurst, 1000000000);

  insert_or_assign(table, high_keys, high_values, high_scores, true, false);
  EXPECT_LE(table.size(0), kCapacity);

  auto high_found = find_keys(table, high_keys);
  const size_t high_accepted = count_found(high_found);

  EXPECT_LT(low_accepted, kBurst)
      << "low-score burst should not be fully admitted near capacity";
  EXPECT_GT(high_accepted, low_accepted)
      << "high-score burst should admit more keys than low-score burst";
}

TEST(CorrectnessStress, HybridPendingPayloadMatchesFinalMetadata) {
  constexpr size_t dim = 33;
  for (bool io_by_cpu : {false, true}) {
    for (bool equal_scores : {false, true}) {
      for (size_t count :
           {size_t{2}, size_t{31}, size_t{32}, size_t{33}, size_t{1025}}) {
        SCOPED_TRACE(::testing::Message()
                     << "cpu_io=" << io_by_cpu << " equal=" << equal_scores
                     << " count=" << count);
        Table table;
        create_table(table, TableMode::kThroughput, 512, dim, false, io_by_cpu);
        std::vector<K> keys(count + 2, K{42});
        keys.front() = nv::merlin::DEFAULT_EMPTY_KEY;
        keys.back() = nv::merlin::DEFAULT_EMPTY_KEY;
        std::vector<V> values(keys.size() * dim);
        std::vector<S> scores(keys.size());
        for (size_t index = 0; index < keys.size(); ++index) {
          scores[index] = UINT64_MAX - (equal_scores ? 0 : index);
          for (size_t component = 0; component < dim; ++component) {
            values[index * dim + component] =
                static_cast<V>(1000 + index * 64 + component);
          }
        }
        insert_or_assign(table, keys, values, scores, false);
        EXPECT_EQ(table.size(0), size_t{1});
        std::vector<V> found_values;
        std::vector<S> found_scores;
        auto found = find_keys(table, {K{42}}, &found_values, &found_scores);
        ASSERT_EQ(count_found(found), size_t{1});
        ASSERT_GE(found_values[0], V{1064});
        ASSERT_LE(found_values[0], static_cast<V>(1000 + count * 64));
        const size_t winner = static_cast<size_t>(found_values[0] - 1000) / 64;
        ASSERT_GE(winner, size_t{1});
        ASSERT_LE(winner, count);
        EXPECT_EQ(found_scores[0], scores[winner]);
        for (size_t component = 0; component < dim; ++component) {
          EXPECT_EQ(found_values[component], values[winner * dim + component]);
        }
      }
    }
  }
}

TEST(CorrectnessStress, HybridPendingPayloadPreservesEveryLfuIncrement) {
  constexpr size_t count = 33;
  constexpr size_t dim = 33;
  using LfuTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLfu>;
  for (bool io_by_cpu : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "cpu_io=" << io_by_cpu);
    TableOptions options;
    options.init_capacity = 512;
    options.max_capacity = 512;
    options.max_hbm_for_vectors = 0;
    options.max_bucket_size = 128;
    options.dim = dim;
    options.io_by_cpu = io_by_cpu;
    LfuTable table;
    table.init(options);
    std::vector<K> keys(count, K{42});
    std::vector<S> scores(count, S{1});
    std::vector<V> values(count * dim);
    for (size_t index = 0; index < count; ++index) {
      for (size_t component = 0; component < dim; ++component) {
        values[index * dim + component] =
            static_cast<V>(1000 + index * 64 + component);
      }
    }
    K* device_keys = copy_to_device(keys);
    V* device_values = copy_to_device(values);
    S* device_scores = copy_to_device(scores);
    bool* device_found = nullptr;
    CUDA_CHECK(cudaMalloc(&device_found, sizeof(bool)));
    for (size_t batch = 0; batch < 2; ++batch) {
      table.insert_or_assign(count, device_keys, device_values, device_scores,
                             0, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(device_found, 0, sizeof(bool)));
    table.find(1, device_keys, device_values, device_found, device_scores, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool found = false;
    S found_score = 0;
    std::vector<V> found_values(dim);
    CUDA_CHECK(
        cudaMemcpy(&found, device_found, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&found_score, device_scores, sizeof(S),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(found_values.data(), device_values, dim * sizeof(V),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_keys));
    CUDA_CHECK(cudaFree(device_values));
    CUDA_CHECK(cudaFree(device_scores));
    CUDA_CHECK(cudaFree(device_found));
    ASSERT_TRUE(found);
    EXPECT_EQ(table.size(0), size_t{1});
    EXPECT_EQ(found_score, S{2 * count});
    ASSERT_GE(found_values[0], V{1000});
    ASSERT_LE(found_values[0], static_cast<V>(1000 + (count - 1) * 64));
    const size_t winner = static_cast<size_t>(found_values[0] - 1000) / 64;
    ASSERT_LT(winner, count);
    for (size_t component = 0; component < dim; ++component) {
      EXPECT_EQ(found_values[component], values[winner * dim + component]);
    }
  }
}

namespace {

template <typename Score>
__global__ void lfu_saturation_boundary_kernel(const Score* a, const Score* b,
                                               Score* result, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    result[index] = nv::merlin::ScoreFunctor<
        K, V, Score,
        nv::merlin::EvictStrategyInternal::kLfu>::saturating_add(a[index],
                                                                 b[index]);
  }
}

template <typename Score>
void check_lfu_saturation_boundaries() {
  constexpr Score maximum = std::numeric_limits<Score>::max();
  constexpr Score cap = maximum - 1;
  const std::vector<Score> a = {0, 1, cap - 1, cap - 1, cap, 0, 7, maximum};
  const std::vector<Score> b = {0, 2, 1, 2, 1, maximum, maximum, 0};
  const std::vector<Score> expected = {0, 3, cap, cap, cap, cap, cap, cap};
  Score* device_a = copy_to_device(a);
  Score* device_b = copy_to_device(b);
  Score* device_result = nullptr;
  CUDA_CHECK(cudaMalloc(&device_result, expected.size() * sizeof(Score)));
  lfu_saturation_boundary_kernel<Score>
      <<<1, 32>>>(device_a, device_b, device_result, expected.size());
  CUDA_CHECK(cudaGetLastError());
  std::vector<Score> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), device_result,
                        actual.size() * sizeof(Score), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_result));
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(actual[index], expected[index]) << "case=" << index;
  }
}

}  // namespace

TEST(CorrectnessStress, LfuSaturationAvoidsUnsignedWrap) {
  check_lfu_saturation_boundaries<uint64_t>();
  // Internal arithmetic helper coverage; public uint32_t tables still only
  // support the customized strategy, as required by HashTable's static checks.
  check_lfu_saturation_boundaries<uint32_t>();
}

TEST(CorrectnessStress, LfuMaximumIncrementSaturatesOnInsertAndUpdate) {
  using LfuTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLfu>;
  constexpr S maximum = std::numeric_limits<S>::max();
  constexpr S cap = maximum - 1;
  for (auto mode : {TableMode::kThroughput, TableMode::kMemory}) {
    SCOPED_TRACE(::testing::Message() << "mode=" << static_cast<int>(mode));
    TableOptions options;
    options.init_capacity = 512;
    options.max_capacity = 512;
    options.max_hbm_for_vectors =
        mode == TableMode::kMemory ? 0 : 512 * kDim * sizeof(V);
    options.max_bucket_size = 128;
    options.dim = kDim;
    options.table_mode = mode;
    LfuTable table;
    table.init(options);
    std::vector<K> keys = {17, 29};
    std::vector<V> values(keys.size() * kDim, V{3});
    std::vector<S> scores = {S{1}, maximum};
    K* device_keys = copy_to_device(keys);
    V* device_values = copy_to_device(values);
    S* device_scores = copy_to_device(scores);
    bool* device_found = nullptr;
    CUDA_CHECK(cudaMalloc(&device_found, keys.size() * sizeof(bool)));
    table.insert_or_assign(keys.size(), device_keys, device_values,
                           device_scores, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(device_found, 0, keys.size() * sizeof(bool)));
    table.find(keys.size(), device_keys, device_values, device_found,
               device_scores, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> found(keys.size());
    CUDA_CHECK(cudaMemcpy(found.data(), device_found, found.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scores.data(), device_scores,
                          scores.size() * sizeof(S), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(found[0] && found[1]);
    EXPECT_EQ(scores[0], S{1});
    EXPECT_EQ(scores[1], cap);
    std::fill(scores.begin(), scores.end(), maximum);
    CUDA_CHECK(cudaMemcpy(device_scores, scores.data(),
                          scores.size() * sizeof(S), cudaMemcpyHostToDevice));
    table.insert_or_assign(keys.size(), device_keys, device_values,
                           device_scores, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    table.find(keys.size(), device_keys, device_values, device_found,
               device_scores, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(scores.data(), device_scores,
                          scores.size() * sizeof(S), cudaMemcpyDeviceToHost));
    EXPECT_EQ(scores[0], cap);
    EXPECT_EQ(scores[1], cap);
    CUDA_CHECK(cudaFree(device_keys));
    CUDA_CHECK(cudaFree(device_values));
    CUDA_CHECK(cudaFree(device_scores));
    CUDA_CHECK(cudaFree(device_found));
  }
}
