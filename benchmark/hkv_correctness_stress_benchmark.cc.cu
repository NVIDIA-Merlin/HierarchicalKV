/*
 * Review-response correctness/adversarial matrix benchmark.
 *
 * The benchmark emits one CSV row per scenario.  It is intentionally more
 * audit-oriented than throughput-oriented: every row reports denominators for
 * attempted operations, distinct final input residency (not API admission),
 * and post-run findability checks. Counter meanings are recorded in notes.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"

namespace {

constexpr size_t kDim = 16;
constexpr size_t kBucketSize = 128;

using K = uint64_t;
using V = float;
using S = uint64_t;
using TableMode = nv::merlin::TableMode;
using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

struct Config {
  std::string mode = "quick";
  std::string table_mode = "both";
  std::string scenario = "all";
  size_t capacity = 128UL * 1024;
  size_t batch = 32UL * 1024;
  int streams = 4;
  double alpha = 1.25;
};

struct Metrics {
  std::string scenario;
  std::string table_mode;
  uint64_t attempted = 0;
  uint64_t accepted = 0;
  uint64_t rejected = 0;
  uint64_t evicted = 0;
  uint64_t final_size = 0;
  uint64_t missing = 0;
  uint64_t duplicate_keys = 0;
  uint64_t torn_vectors = 0;
  uint64_t semantic_errors = 0;
  std::string notes;
};

struct FindResult {
  uint64_t found = 0;
  std::vector<uint8_t> founds;
  std::vector<V> values;
  std::vector<S> scores;
};

struct Snapshot {
  uint64_t exported = 0;
  uint64_t duplicate_keys = 0;
  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }

  void copy_from_host(const std::vector<T>& host,
                      cudaStream_t stream = 0) const {
    if (!host.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(ptr_, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

  void copy_to_host(std::vector<T>& host, size_t count,
                    cudaStream_t stream = 0) const {
    host.resize(count);
    if (count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(host.data(), ptr_, count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream));
    }
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

std::string table_mode_name(TableMode mode) {
  return mode == TableMode::kMemory ? "memory" : "throughput";
}

std::string sanitize_note(std::string note) {
  for (char& c : note) {
    if (c == ',' || c == '\n' || c == '\r') {
      c = ';';
    }
  }
  return note;
}

void print_header() {
  std::cout << "scenario,table_mode,attempted,accepted,rejected,evicted,"
               "final_size,missing,duplicate_keys,torn_vectors,notes\n";
}

void print_metrics(const Metrics& m) {
  std::cout << m.scenario << ',' << m.table_mode << ',' << m.attempted << ','
            << m.accepted << ',' << m.rejected << ',' << m.evicted << ','
            << m.final_size << ',' << m.missing << ',' << m.duplicate_keys
            << ',' << m.torn_vectors << ',' << sanitize_note(m.notes)
            << ";semantic_errors=" << m.semantic_errors << std::endl;
}

bool metrics_failed(const Metrics& metrics) {
  return metrics.missing != 0 || metrics.duplicate_keys != 0 ||
         metrics.torn_vectors != 0 || metrics.semantic_errors != 0;
}

template <typename Oracle>
uint64_t generation_errors(const std::vector<K>& keys,
                           const std::vector<V>& values,
                           const std::vector<S>& scores,
                           const std::vector<uint8_t>& founds, Oracle oracle) {
  uint64_t errors = 0;
  for (size_t row = 0; row < keys.size(); ++row) {
    if (!founds[row]) continue;
    bool valid = oracle(keys[row], scores[row], values[row * kDim]);
    for (size_t dim = 1; dim < kDim; ++dim) {
      valid = valid && values[row * kDim + dim] == values[row * kDim];
    }
    errors += !valid;
  }
  return errors;
}

bool want_table_mode(const Config& cfg, TableMode mode) {
  return cfg.table_mode == "both" || cfg.table_mode == table_mode_name(mode);
}

bool want_scenario(const Config& cfg, const std::string& scenario) {
  return cfg.scenario == "all" || cfg.scenario == scenario;
}

void create_table(Table& table, TableMode mode, size_t capacity,
                  bool pure_hbm = false, bool api_lock = true) {
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = pure_hbm ? capacity * kDim * sizeof(V) : 0;
  options.max_bucket_size = kBucketSize;
  options.max_load_factor = 1.0f;
  options.dim = kDim;
  options.table_mode = mode;
  options.api_lock = api_lock;
  table.init(options);
}

std::vector<K> continuous_keys(size_t n, K start = 1) {
  std::vector<K> keys(n);
  std::iota(keys.begin(), keys.end(), start);
  return keys;
}

std::vector<S> scores_with_base(size_t n, S base) {
  std::vector<S> scores(n);
  for (size_t i = 0; i < n; i++) {
    scores[i] = base + static_cast<S>(i);
  }
  return scores;
}

std::vector<V> sentinel_values(size_t n, V sentinel) {
  return std::vector<V>(n * kDim, sentinel);
}

void set_row_sentinel(std::vector<V>& values, size_t row, V sentinel) {
  for (size_t d = 0; d < kDim; d++) {
    values[row * kDim + d] = sentinel;
  }
}

void insert_or_assign(Table& table, const std::vector<K>& keys,
                      const std::vector<V>& values,
                      const std::vector<S>& scores, bool unique_key,
                      bool ignore_evict_strategy = false,
                      cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(values.size());
  DeviceBuffer<S> d_scores(scores.size());
  d_keys.copy_from_host(keys, stream);
  d_values.copy_from_host(values, stream);
  d_scores.copy_from_host(scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  table.insert_or_assign(keys.size(), d_keys.get(), d_values.get(),
                         d_scores.get(), stream, unique_key,
                         ignore_evict_strategy);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void assign_values(Table& table, const std::vector<K>& keys,
                   const std::vector<V>& values, const std::vector<S>& scores,
                   cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(values.size());
  DeviceBuffer<S> d_scores(scores.size());
  d_keys.copy_from_host(keys, stream);
  d_values.copy_from_host(values, stream);
  d_scores.copy_from_host(scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  table.assign(keys.size(), d_keys.get(), d_values.get(), d_scores.get(),
               stream, true);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

FindResult find_keys(Table& table, const std::vector<K>& keys,
                     bool copy_scores = false, cudaStream_t stream = 0) {
  FindResult result;
  if (keys.empty()) {
    return result;
  }
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(keys.size() * kDim);
  DeviceBuffer<bool> d_founds(keys.size());
  DeviceBuffer<S> d_scores(copy_scores ? keys.size() : 0);
  d_keys.copy_from_host(keys, stream);
  CUDA_CHECK(
      cudaMemsetAsync(d_founds.get(), 0, keys.size() * sizeof(bool), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  table.find(keys.size(), d_keys.get(), d_values.get(), d_founds.get(),
             copy_scores ? d_scores.get() : nullptr, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint8_t> found_bytes(keys.size());
  if (!keys.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(found_bytes.data(), d_founds.get(),
                               keys.size() * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
  }
  d_values.copy_to_host(result.values, keys.size() * kDim, stream);
  if (copy_scores) {
    d_scores.copy_to_host(result.scores, keys.size(), stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  result.founds.resize(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    result.founds[i] = found_bytes[i] ? 1 : 0;
    result.found += result.founds[i] ? 1 : 0;
  }
  return result;
}

Snapshot export_snapshot(Table& table, cudaStream_t stream = 0) {
  Snapshot snapshot;
  const size_t n = table.capacity();
  if (n == 0) {
    return snapshot;
  }

  DeviceBuffer<K> d_keys(n);
  DeviceBuffer<V> d_values(n * kDim);
  DeviceBuffer<S> d_scores(n);
  snapshot.exported = table.export_batch(n, 0, d_keys.get(), d_values.get(),
                                         d_scores.get(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  d_keys.copy_to_host(snapshot.keys, snapshot.exported, stream);
  d_values.copy_to_host(snapshot.values, snapshot.exported * kDim, stream);
  d_scores.copy_to_host(snapshot.scores, snapshot.exported, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::unordered_set<K> unique;
  unique.reserve(snapshot.keys.size() * 2 + 1);
  for (K key : snapshot.keys) {
    if (!unique.insert(key).second) {
      snapshot.duplicate_keys++;
    }
  }
  return snapshot;
}

uint64_t count_torn_vectors(const std::vector<V>& values,
                            const std::vector<uint8_t>& founds,
                            const std::unordered_set<int64_t>& legal = {}) {
  uint64_t torn = 0;
  const size_t rows = founds.size();
  for (size_t i = 0; i < rows; i++) {
    if (!founds[i]) {
      continue;
    }
    const V first = values[i * kDim];
    bool ok = true;
    for (size_t d = 1; d < kDim; d++) {
      if (values[i * kDim + d] != first) {
        ok = false;
        break;
      }
    }
    if (ok && !legal.empty()) {
      ok = legal.find(static_cast<int64_t>(std::llround(first))) != legal.end();
    }
    torn += ok ? 0 : 1;
  }
  return torn;
}

uint64_t count_torn_snapshot(const Snapshot& snapshot,
                             const std::unordered_set<int64_t>& legal = {}) {
  std::vector<uint8_t> all_found(snapshot.exported, 1);
  return count_torn_vectors(snapshot.values, all_found, legal);
}

size_t throughput_bucket(K key, size_t capacity) {
  const uint64_t hash = nv::merlin::Murmur3HashHost(key);
  const size_t global_idx = hash & (capacity - 1);
  return global_idx / kBucketSize;
}

std::pair<size_t, size_t> dual_bucket_pair(K key, size_t buckets_num) {
  const uint64_t hash = nv::merlin::Murmur3HashHost(key);
  size_t b1 = static_cast<uint32_t>(hash) % buckets_num;
  size_t b2 = static_cast<uint32_t>(hash >> 32) % buckets_num;
  if (b2 == b1) {
    b2 = (b2 + 1) % buckets_num;
  }
  return {b1, b2};
}

std::vector<K> keys_in_throughput_bucket(size_t count, size_t capacity,
                                         size_t bucket, K start) {
  std::vector<K> keys;
  keys.reserve(count);
  for (K candidate = start; keys.size() < count; candidate++) {
    if (throughput_bucket(candidate, capacity) == bucket) {
      keys.push_back(candidate);
    }
  }
  return keys;
}

std::vector<K> keys_in_dual_bucket_pair(size_t count, size_t capacity,
                                        size_t target_b1, size_t target_b2,
                                        K start) {
  const size_t buckets_num = capacity / kBucketSize;
  std::vector<K> keys;
  keys.reserve(count);
  for (K candidate = start; keys.size() < count; candidate++) {
    const auto pair = dual_bucket_pair(candidate, buckets_num);
    if (pair.first == target_b1 && pair.second == target_b2) {
      keys.push_back(candidate);
    }
  }
  return keys;
}

std::vector<K> zipf_like_keys(size_t n, size_t hot_keys, double alpha,
                              K start) {
  std::vector<double> weights(hot_keys);
  for (size_t i = 0; i < hot_keys; i++) {
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
  }
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  std::mt19937_64 rng(20260610);

  std::vector<K> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = start + static_cast<K>(dist(rng));
  }
  return keys;
}

uint64_t count_export_missing(Table& table, const Snapshot& snapshot) {
  if (snapshot.keys.empty()) {
    return 0;
  }
  return snapshot.keys.size() - find_keys(table, snapshot.keys).found;
}

void validate_snapshot(Table& table, const Snapshot& snapshot,
                       Metrics& metrics) {
  metrics.final_size = table.size(0);
  metrics.duplicate_keys = snapshot.duplicate_keys;
  metrics.semantic_errors += metrics.final_size != snapshot.exported;
  const auto found = find_keys(table, snapshot.keys, true);
  metrics.missing += snapshot.exported - found.found;
  for (size_t row = 0; row < snapshot.keys.size(); ++row) {
    if (!found.founds[row]) continue;
    metrics.semantic_errors += found.scores[row] != snapshot.scores[row];
    for (size_t dim = 0; dim < kDim; ++dim) {
      metrics.semantic_errors +=
          found.values[row * kDim + dim] != snapshot.values[row * kDim + dim];
    }
  }
}

uint64_t expected_single_residents(const std::vector<K>& keys,
                                   size_t capacity) {
  std::vector<size_t> occupancy(capacity / kBucketSize, 0);
  std::unordered_set<K> unique(keys.begin(), keys.end());
  for (K key : unique) ++occupancy[throughput_bucket(key, capacity)];
  uint64_t expected = 0;
  for (size_t count : occupancy) expected += std::min(count, kBucketSize);
  return expected;
}

uint64_t high_score_admission_errors(const std::vector<K>& keys,
                                     const FindResult& found, size_t capacity,
                                     TableMode mode) {
  if (mode == TableMode::kThroughput)
    return found.found != expected_single_residents(keys, capacity);
  uint64_t errors = found.found < std::min(keys.size(), kBucketSize);
  std::vector<size_t> candidates(capacity / kBucketSize, 0);
  for (K key : keys) {
    const auto pair = dual_bucket_pair(key, candidates.size());
    ++candidates[pair.first];
    ++candidates[pair.second];
  }
  for (size_t row = 0; row < keys.size(); ++row) {
    const auto pair = dual_bucket_pair(keys[row], candidates.size());
    if (candidates[pair.first] <= kBucketSize &&
        candidates[pair.second] <= kBucketSize)
      errors += !found.founds[row];
  }
  return errors;
}

uint64_t eviction_return_errors(const std::vector<K>& prior_keys,
                                const std::vector<K>& input_keys,
                                const std::vector<K>& final_keys,
                                const std::vector<K>& returned_keys) {
  std::unordered_set<K> expected(prior_keys.begin(), prior_keys.end());
  uint64_t errors = 0;
  for (K key : input_keys) errors += !expected.insert(key).second;
  for (K key : final_keys) errors += expected.erase(key) != 1;
  errors += returned_keys.size() != expected.size();
  for (K key : returned_keys) errors += expected.erase(key) != 1;
  return errors + expected.size();
}

const char* residency_note() {
  return "accepted=final_distinct_input_residents;"
         "rejected=final_distinct_input_nonresidents_not_API_rejections;"
         "within_batch_evictions=unobserved";
}

Metrics duplicate_same_key_throughput(const Config& cfg) {
  const size_t attempted = cfg.batch;
  Table table;
  create_table(table, TableMode::kThroughput, cfg.capacity, true);

  std::vector<K> keys(attempted, 42);
  std::vector<V> values(attempted * kDim);
  std::vector<S> scores(attempted);
  std::unordered_set<int64_t> legal;
  for (size_t i = 0; i < attempted; i++) {
    const int64_t sentinel = 1000 + static_cast<int64_t>(i);
    legal.insert(sentinel);
    scores[i] = static_cast<S>(i + 1);
    set_row_sentinel(values, i, static_cast<V>(sentinel));
  }

  insert_or_assign(table, keys, values, scores, false);
  auto found = find_keys(table, std::vector<K>{42}, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "duplicate_same_key_no_torn";
  m.table_mode = "throughput";
  m.attempted = attempted;
  m.accepted = found.found;
  m.rejected = 1 - found.found;
  m.final_size = table.size(0);
  m.missing = 1 - found.found;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, legal);
  m.semantic_errors = generation_errors(
      std::vector<K>{42}, found.values, found.scores, found.founds,
      [&](K key, S score, V value) {
        return key == 42 && score >= 1 && score <= attempted &&
               value == static_cast<V>(999 + score);
      });
  validate_snapshot(table, snapshot, m);
  m.notes = std::string(residency_note()) + ";unique_key=false";
  return m;
}

Metrics duplicate_groups_throughput(const Config& cfg) {
  const size_t groups = std::max<size_t>(1, cfg.batch / 8);
  const size_t repeat = 4;
  const size_t attempted = groups * repeat;
  Table table;
  create_table(table, TableMode::kThroughput, cfg.capacity, true);

  std::vector<K> keys(attempted);
  std::vector<V> values(attempted * kDim);
  std::vector<S> scores(attempted);
  std::unordered_set<int64_t> legal;
  for (size_t g = 0; g < groups; g++) {
    for (size_t r = 0; r < repeat; r++) {
      const size_t row = g * repeat + r;
      keys[row] = 1000000 + static_cast<K>(g);
      scores[row] = static_cast<S>(row + 1);
      const int64_t sentinel = 2000000 + static_cast<int64_t>(row);
      legal.insert(sentinel);
      set_row_sentinel(values, row, static_cast<V>(sentinel));
    }
  }

  insert_or_assign(table, keys, values, scores, false);
  std::vector<K> unique_keys = continuous_keys(groups, 1000000);
  auto found = find_keys(table, unique_keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "duplicate_groups";
  m.table_mode = "throughput";
  m.attempted = attempted;
  m.accepted = found.found;
  m.rejected = groups - found.found;
  m.final_size = table.size(0);
  m.semantic_errors =
      found.found != expected_single_residents(keys, cfg.capacity);
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, legal);
  m.semantic_errors += generation_errors(
      unique_keys, found.values, found.scores, found.founds,
      [&](K key, S score, V value) {
        return score >= 1 && score <= attempted && key == keys[score - 1] &&
               value == values[(score - 1) * kDim];
      });
  validate_snapshot(table, snapshot, m);
  m.notes = std::string(residency_note()) + ";repeat_factor=4";
  return m;
}

Metrics duplicate_contract_memory(const Config& cfg) {
  Metrics m;
  m.scenario = "duplicate_same_key_contract";
  m.table_mode = "memory";
  m.attempted = std::min<size_t>(cfg.batch, 1024);

  Table table;
  create_table(table, TableMode::kMemory, cfg.capacity, false);
  std::vector<K> keys(m.attempted, 7);
  std::vector<V> values = sentinel_values(m.attempted, 1.0f);
  std::vector<S> scores = scores_with_base(m.attempted, 1);

  try {
    insert_or_assign(table, keys, values, scores, false);
    m.accepted = table.size(0);
    ++m.semantic_errors;
    m.notes = "unexpected;memory mode should reject unique_key=false";
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("requires unique_key=true") ==
        std::string::npos)
      throw;
    m.rejected = m.attempted;
    m.notes = "documented API contract rejects unique_key=false";
  }
  m.final_size = table.size(0);
  m.semantic_errors += m.final_size != 0;
  return m;
}

Metrics memory_repeated_update(const Config& cfg) {
  const size_t n = std::min(cfg.batch, cfg.capacity / 2);
  Table table;
  create_table(table, TableMode::kMemory, cfg.capacity, false);

  std::vector<K> keys = continuous_keys(n, 100);
  std::vector<S> scores = scores_with_base(n, 1);
  insert_or_assign(table, keys, sentinel_values(n, 1.0f), scores, true, true);
  scores = scores_with_base(n, n + 1);
  insert_or_assign(table, keys, sentinel_values(n, 2.0f), scores, true, true);

  auto found = find_keys(table, keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "memory_repeated_update";
  m.table_mode = "memory";
  m.attempted = n * 2;
  m.accepted = found.found;
  m.final_size = table.size(0);
  m.missing = n - found.found;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, {2});
  m.semantic_errors =
      generation_errors(keys, found.values, found.scores, found.founds,
                        [&](K key, S score, V value) {
                          return key >= 100 && key < 100 + n &&
                                 score == n + 1 + key - 100 && value == 2;
                        });
  validate_snapshot(table, snapshot, m);
  m.notes =
      "two legal unique-key batches over same key "
      "set;accepted=final_distinct_input_residents";
  return m;
}

Metrics admission_denominator(const Config& cfg, TableMode mode,
                              bool high_score) {
  const size_t capacity = cfg.capacity;
  const size_t fill_count = capacity;
  const size_t burst = std::min(cfg.batch, capacity / 4);
  const bool throughput = mode == TableMode::kThroughput;
  Table table;
  create_table(table, mode, capacity, throughput);

  std::vector<K> fill_keys = continuous_keys(fill_count, 1);
  std::vector<S> fill_scores = scores_with_base(fill_count, 1000000);
  insert_or_assign(table, fill_keys, sentinel_values(fill_count, 1.0f),
                   fill_scores, true, true);
  const Snapshot before = export_snapshot(table);
  const uint64_t size_after_fill = before.exported;

  std::vector<K> burst_keys = continuous_keys(burst, capacity + 1000000);
  std::vector<S> burst_scores(
      burst, high_score ? static_cast<S>(9000000000ULL) : static_cast<S>(1));
  insert_or_assign(table, burst_keys,
                   sentinel_values(burst, high_score ? 3.0f : 2.0f),
                   burst_scores, true, false);

  auto burst_found = find_keys(table, burst_keys);
  auto fill_found = find_keys(table, before.keys);
  const uint64_t resident_evicted = before.exported - fill_found.found;
  const Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = high_score ? "high_score_admission" : "low_score_admission";
  m.table_mode = table_mode_name(mode);
  m.attempted = burst;
  m.accepted = burst_found.found;
  m.rejected = burst - burst_found.found;
  m.evicted = resident_evicted;
  m.final_size = table.size(0);
  m.torn_vectors = count_torn_vectors(burst_found.values, burst_found.founds,
                                      {high_score ? 3 : 2});
  std::ostringstream note;
  note << "size_after_fill=" << size_after_fill << ';' << residency_note()
       << ";evicted=prior_resident_loss";
  m.notes = note.str();
  validate_snapshot(table, snapshot, m);
  m.semantic_errors +=
      m.final_size != before.exported - resident_evicted + burst_found.found;
  m.semantic_errors += !high_score && resident_evicted != 0;
  if (high_score) {
    m.semantic_errors +=
        high_score_admission_errors(burst_keys, burst_found, capacity, mode);
    for (S score : before.scores)
      m.semantic_errors += score >= burst_scores.front();
  }
  m.semantic_errors += generation_errors(
      snapshot.keys, snapshot.values, snapshot.scores,
      std::vector<uint8_t>(snapshot.exported, 1), [&](K key, S score, V value) {
        if (key >= 1 && key <= capacity)
          return score == 999999 + key && value == 1;
        return key >= capacity + 1000000 && key < capacity + 1000000 + burst &&
               score == (high_score ? S{9000000000ULL} : S{1}) &&
               value == (high_score ? V{3} : V{2});
      });
  return m;
}

Metrics evicted_resident_memory(const Config& cfg) {
  const size_t capacity = cfg.capacity;
  const size_t burst = std::min(cfg.batch, capacity / 4);
  Table table;
  create_table(table, TableMode::kMemory, capacity, false);

  std::vector<K> fill_keys = continuous_keys(capacity, 1);
  insert_or_assign(table, fill_keys, sentinel_values(capacity, 1.0f),
                   scores_with_base(capacity, 1000), true, true);
  const Snapshot before = export_snapshot(table);

  std::vector<K> high_keys = continuous_keys(burst, capacity + 2000000);
  insert_or_assign(table, high_keys, sentinel_values(burst, 9.0f),
                   std::vector<S>(burst, 9000000000ULL), true, false);

  auto new_found = find_keys(table, high_keys);
  auto old_found = find_keys(table, before.keys);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "evicted_resident_findability";
  m.table_mode = "memory";
  m.attempted = burst;
  m.accepted = new_found.found;
  m.rejected = burst - new_found.found;
  m.evicted = before.exported - old_found.found;
  m.final_size = table.size(0);
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_snapshot(snapshot, {1, 9});
  m.notes = std::string(residency_note()) + ";evicted=prior_resident_loss";
  validate_snapshot(table, snapshot, m);
  m.semantic_errors += m.final_size != old_found.found + new_found.found;
  m.semantic_errors += high_score_admission_errors(
      high_keys, new_found, capacity, TableMode::kMemory);
  for (S score : before.scores) m.semantic_errors += score >= 9000000000ULL;
  m.semantic_errors += generation_errors(
      snapshot.keys, snapshot.values, snapshot.scores,
      std::vector<uint8_t>(snapshot.exported, 1), [&](K key, S score, V value) {
        if (key >= 1 && key <= capacity)
          return score == 999 + key && value == 1;
        return key >= capacity + 2000000 && key < capacity + 2000000 + burst &&
               score == 9000000000ULL && value == 9;
      });
  return m;
}

Metrics explicit_eviction_throughput(const Config& cfg) {
  const size_t capacity = cfg.capacity;
  const size_t burst = std::min(cfg.batch, capacity / 4);
  Table table;
  create_table(table, TableMode::kThroughput, capacity, true);

  std::vector<K> fill_keys = continuous_keys(capacity, 1);
  insert_or_assign(table, fill_keys, sentinel_values(capacity, 1.0f),
                   scores_with_base(capacity, 1000), true, true);
  const Snapshot before = export_snapshot(table);

  std::vector<K> high_keys = continuous_keys(burst, capacity + 3000000);
  std::vector<V> high_values = sentinel_values(burst, 8.0f);
  std::vector<S> high_scores(burst, 9000000000ULL);
  DeviceBuffer<K> d_keys(burst);
  DeviceBuffer<V> d_values(burst * kDim);
  DeviceBuffer<S> d_scores(burst);
  DeviceBuffer<K> d_evicted_keys(burst);
  DeviceBuffer<V> d_evicted_values(burst * kDim);
  DeviceBuffer<S> d_evicted_scores(burst);
  d_keys.copy_from_host(high_keys);
  d_values.copy_from_host(high_values);
  d_scores.copy_from_host(high_scores);
  CUDA_CHECK(cudaDeviceSynchronize());

  const uint64_t evicted = table.insert_and_evict(
      burst, d_keys.get(), d_values.get(), d_scores.get(), d_evicted_keys.get(),
      d_evicted_values.get(), d_evicted_scores.get(), 0, true, false);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (evicted > burst)
    throw std::runtime_error("eviction return count exceeds output capacity");

  std::vector<K> evicted_keys;
  std::vector<V> evicted_values;
  std::vector<S> evicted_scores;
  d_evicted_keys.copy_to_host(evicted_keys, evicted);
  d_evicted_values.copy_to_host(evicted_values, evicted * kDim);
  d_evicted_scores.copy_to_host(evicted_scores, evicted);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto new_found = find_keys(table, high_keys);
  const auto old_found = find_keys(table, before.keys);
  auto evicted_found = find_keys(table, evicted_keys);
  Snapshot snapshot = export_snapshot(table);
  const uint64_t export_missing = count_export_missing(table, snapshot);

  Metrics m;
  m.scenario = "explicit_eviction_findability";
  m.table_mode = "throughput";
  m.attempted = burst;
  m.accepted = new_found.found;
  m.rejected = burst - new_found.found;
  m.evicted = before.exported - old_found.found;
  m.final_size = table.size(0);
  m.missing = evicted_found.found;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_snapshot(snapshot, {1, 8});
  std::ostringstream note;
  note << "evicted_still_found=" << evicted_found.found
       << ";export_missing=" << export_missing << ";api_returned=" << evicted
       << ';' << residency_note() << ";evicted=prior_resident_loss";
  m.notes = note.str();
  validate_snapshot(table, snapshot, m);
  m.semantic_errors += m.final_size != old_found.found + new_found.found;
  m.semantic_errors += high_score_admission_errors(
      high_keys, new_found, capacity, TableMode::kThroughput);
  for (S score : before.scores) m.semantic_errors += score >= 9000000000ULL;
  m.semantic_errors += eviction_return_errors(before.keys, high_keys,
                                              snapshot.keys, evicted_keys);
  m.semantic_errors += generation_errors(
      evicted_keys, evicted_values, evicted_scores,
      std::vector<uint8_t>(evicted, 1), [&](K key, S score, V value) {
        if (key >= 1 && key <= capacity)
          return score == 999 + key && value == 1;
        return key >= capacity + 3000000 && key < capacity + 3000000 + burst &&
               score == 9000000000ULL && value == 8;
      });
  m.semantic_errors += generation_errors(
      snapshot.keys, snapshot.values, snapshot.scores,
      std::vector<uint8_t>(snapshot.exported, 1), [&](K key, S score, V value) {
        if (key >= 1 && key <= capacity)
          return score == 999 + key && value == 1;
        return key >= capacity + 3000000 && key < capacity + 3000000 + burst &&
               score == 9000000000ULL && value == 8;
      });
  return m;
}

Metrics adversarial_bucket_throughput(const Config& cfg) {
  const size_t buckets = cfg.mode == "quick" ? 8 : 256;
  const size_t capacity = buckets * kBucketSize;
  const size_t attempted = kBucketSize * 2;
  Table table;
  create_table(table, TableMode::kThroughput, capacity, true);

  std::vector<K> keys =
      keys_in_throughput_bucket(attempted, capacity, 0, 5000000);
  std::vector<S> scores = scores_with_base(attempted, 1);
  insert_or_assign(table, keys, sentinel_values(attempted, 4.0f), scores, true,
                   false);

  auto found = find_keys(table, keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "adversarial_single_bucket";
  m.table_mode = "throughput";
  m.attempted = attempted;
  m.accepted = found.found;
  m.rejected = attempted - found.found;
  m.final_size = table.size(0);
  m.semantic_errors = found.found != kBucketSize;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, {4});
  validate_snapshot(table, snapshot, m);
  m.semantic_errors +=
      generation_errors(keys, found.values, found.scores, found.founds,
                        [&](K key, S score, V value) {
                          return score >= 1 && score <= attempted &&
                                 keys[score - 1] == key && value == 4;
                        });
  m.notes = std::string(residency_note()) + ";single_bucket_capacity=128";
  return m;
}

Metrics adversarial_dual_pair_memory(const Config& cfg) {
  const size_t buckets = cfg.mode == "quick" ? 8 : 256;
  const size_t capacity = buckets * kBucketSize;
  const size_t attempted = kBucketSize * 2;
  Table table;
  create_table(table, TableMode::kMemory, capacity, false);

  std::vector<K> keys =
      keys_in_dual_bucket_pair(attempted, capacity, 0, 1, 6000000);
  std::vector<S> scores = scores_with_base(attempted, 1);
  insert_or_assign(table, keys, sentinel_values(attempted, 5.0f), scores, true,
                   false);

  auto found = find_keys(table, keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "adversarial_dual_bucket_pair";
  m.table_mode = "memory";
  m.attempted = attempted;
  m.accepted = found.found;
  m.rejected = attempted - found.found;
  m.final_size = table.size(0);
  m.missing = attempted - found.found;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_snapshot(snapshot, {5});
  validate_snapshot(table, snapshot, m);
  m.semantic_errors +=
      generation_errors(keys, found.values, found.scores, found.founds,
                        [&](K key, S score, V value) {
                          return score >= 1 && score <= attempted &&
                                 keys[score - 1] == key && value == 5;
                        });
  m.notes = std::string(residency_note()) + ";pair_capacity=256";
  return m;
}

Metrics skew_hotset_throughput(const Config& cfg) {
  const size_t hot_keys =
      std::max<size_t>(1, std::min(cfg.batch / 16, cfg.capacity / 4));
  std::vector<K> keys = zipf_like_keys(cfg.batch, hot_keys, cfg.alpha, 7000000);
  std::vector<V> values(cfg.batch * kDim);
  std::vector<S> scores(cfg.batch);
  std::unordered_set<int64_t> legal;
  for (size_t i = 0; i < cfg.batch; i++) {
    const int64_t sentinel = 6000 + static_cast<int64_t>(i % 1024);
    legal.insert(sentinel);
    scores[i] = i + 1;
    set_row_sentinel(values, i, static_cast<V>(sentinel));
  }

  Table table;
  create_table(table, TableMode::kThroughput, cfg.capacity, true);
  insert_or_assign(table, keys, values, scores, false);

  std::unordered_set<K> unique(keys.begin(), keys.end());
  std::vector<K> unique_keys(unique.begin(), unique.end());
  auto found = find_keys(table, unique_keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "skew_hotset_zipf";
  m.table_mode = "throughput";
  m.attempted = cfg.batch;
  m.accepted = found.found;
  m.rejected = unique_keys.size() - found.found;
  m.final_size = table.size(0);
  m.semantic_errors =
      found.found != expected_single_residents(keys, cfg.capacity);
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, legal);
  std::ostringstream note;
  note << "alpha=" << cfg.alpha << ";unique_inputs=" << unique_keys.size()
       << ';' << residency_note();
  m.notes = note.str();
  m.semantic_errors += generation_errors(
      unique_keys, found.values, found.scores, found.founds,
      [&](K key, S score, V value) {
        return score >= 1 && score <= cfg.batch && keys[score - 1] == key &&
               value == values[(score - 1) * kDim];
      });
  validate_snapshot(table, snapshot, m);
  return m;
}

Metrics skew_repeated_memory(const Config& cfg) {
  const size_t hot_keys =
      std::max<size_t>(1, std::min(cfg.batch / 16, cfg.capacity / 4));
  const int rounds = 4;
  std::vector<K> keys = continuous_keys(hot_keys, 8000000);
  Table table;
  create_table(table, TableMode::kMemory, cfg.capacity, false);

  for (int r = 0; r < rounds; r++) {
    insert_or_assign(
        table, keys, sentinel_values(hot_keys, static_cast<V>(10 + r)),
        scores_with_base(hot_keys, 1000 + r * hot_keys), true, true);
  }

  auto found = find_keys(table, keys, true);
  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "skew_hotset_repeated_update";
  m.table_mode = "memory";
  m.attempted = hot_keys * rounds;
  m.accepted = found.found;
  m.final_size = table.size(0);
  m.missing = hot_keys - found.found;
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_vectors(found.values, found.founds, {13});
  m.semantic_errors = generation_errors(
      keys, found.values, found.scores, found.founds,
      [&](K key, S score, V value) {
        return key >= 8000000 && key < 8000000 + hot_keys &&
               score == 1000 + 3 * hot_keys + key - 8000000 && value == 13;
      });
  validate_snapshot(table, snapshot, m);
  m.notes =
      "hot set replayed as four legal unique "
      "batches;accepted=final_distinct_input_residents";
  return m;
}

struct Barrier {
  explicit Barrier(int target) : target_(target) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    arrived_++;
    if (arrived_ == target_) {
      released_ = true;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [this] { return released_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int target_;
  int arrived_ = 0;
  bool released_ = false;
};

Metrics rui_overlap_throughput(const Config& cfg) {
  const size_t initial = std::min(cfg.capacity / 2, cfg.batch * 2);
  const int workers = std::max(3, cfg.streams);
  const size_t op_batch =
      std::min(initial, std::max<size_t>(1, cfg.batch / workers));
  Table table;
  create_table(table, TableMode::kThroughput, cfg.capacity, true, true);

  std::vector<K> initial_keys = continuous_keys(initial, 1);
  insert_or_assign(table, initial_keys, sentinel_values(initial, 1.0f),
                   scores_with_base(initial, 1000), true, true);

  Barrier barrier(workers);
  std::atomic<uint64_t> attempted{0};
  std::atomic<bool> failed{false};
  std::atomic<uint64_t> read_missing{0};
  std::atomic<uint64_t> read_pair_errors{0};
  std::vector<cudaStream_t> streams(workers);
  for (auto& stream : streams) CUDA_CHECK(cudaStreamCreate(&stream));
  std::unordered_map<K, std::vector<std::pair<S, V>>> generations;
  for (size_t row = 0; row < initial; ++row)
    generations[initial_keys[row]].emplace_back(1000 + row, 1.0f);
  for (int worker_id = 0; worker_id < workers; ++worker_id) {
    if (worker_id % 3 == 0) continue;
    for (size_t row = 0; row < op_batch; ++row) {
      K key = 1 + static_cast<K>((row + worker_id * op_batch) % initial);
      if (worker_id % 3 == 2 && row >= op_batch / 2)
        key = 9000000 + static_cast<K>(worker_id * op_batch + row);
      const S score =
          (worker_id % 3 == 1 ? 5000 : 9000) + worker_id * op_batch + row;
      const V value =
          static_cast<V>((worker_id % 3 == 1 ? 20 : 30) + worker_id % 10);
      generations[key].emplace_back(score, value);
    }
  }
  std::vector<size_t> candidate_counts(cfg.capacity / kBucketSize, 0);
  for (const auto& entry : generations)
    ++candidate_counts[throughput_bucket(entry.first, cfg.capacity)];
  auto must_remain_resident = [&](K key) {
    return candidate_counts[throughput_bucket(key, cfg.capacity)] <=
           kBucketSize;
  };
  auto legal_generation = [&](K key, S score, V value, bool completed) {
    const auto found = generations.find(key);
    if (found == generations.end()) return false;
    auto first = found->second.begin();
    if (completed && key >= 1 && key <= initial && found->second.size() > 1)
      ++first;
    return std::find(first, found->second.end(),
                     std::make_pair(score, value)) != found->second.end();
  };

  auto worker = [&](int worker_id) {
    barrier.wait();
    try {
      cudaStream_t stream = streams[worker_id];
      if (worker_id % 3 == 0) {
        std::vector<K> keys(op_batch);
        for (size_t i = 0; i < op_batch; i++) {
          keys[i] = 1 + static_cast<K>((i + worker_id * op_batch) % initial);
        }
        const auto found = find_keys(table, keys, true, stream);
        for (size_t row = 0; row < keys.size(); ++row)
          read_missing.fetch_add(must_remain_resident(keys[row]) &&
                                 !found.founds[row]);
        read_pair_errors.fetch_add(generation_errors(
            keys, found.values, found.scores, found.founds,
            [&](K key, S score, V value) {
              return legal_generation(key, score, value, false);
            }));
      } else if (worker_id % 3 == 1) {
        std::vector<K> keys(op_batch);
        for (size_t i = 0; i < op_batch; i++) {
          keys[i] = 1 + static_cast<K>((i + worker_id * op_batch) % initial);
        }
        assign_values(
            table, keys,
            sentinel_values(op_batch, static_cast<V>(20 + worker_id % 10)),
            scores_with_base(op_batch, 5000 + worker_id * op_batch), stream);
      } else {
        std::vector<K> keys(op_batch);
        for (size_t i = 0; i < op_batch / 2; i++) {
          keys[i] = 1 + static_cast<K>((i + worker_id * op_batch) % initial);
        }
        for (size_t i = op_batch / 2; i < op_batch; i++) {
          keys[i] = 9000000 + static_cast<K>(worker_id * op_batch + i);
        }
        insert_or_assign(
            table, keys,
            sentinel_values(op_batch, static_cast<V>(30 + worker_id % 10)),
            scores_with_base(op_batch, 9000 + worker_id * op_batch), true,
            false, stream);
      }
      attempted.fetch_add(op_batch);
    } catch (...) {
      failed = true;
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int worker_id = 0; worker_id < workers; worker_id++) {
    threads.emplace_back(worker, worker_id);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto stream : streams) CUDA_CHECK(cudaStreamDestroy(stream));

  Snapshot snapshot = export_snapshot(table);

  Metrics m;
  m.scenario = "rui_overlap";
  m.table_mode = "throughput";
  m.attempted = attempted.load();
  m.accepted = snapshot.exported;
  m.final_size = table.size(0);
  m.duplicate_keys = snapshot.duplicate_keys;
  m.torn_vectors = count_torn_snapshot(snapshot);
  m.missing = read_missing.load();
  m.semantic_errors = failed.load() + read_pair_errors.load();
  const std::unordered_set<K> final_keys(snapshot.keys.begin(),
                                         snapshot.keys.end());
  for (const auto& entry : generations)
    m.missing +=
        must_remain_resident(entry.first) && !final_keys.count(entry.first);
  m.semantic_errors += generation_errors(
      snapshot.keys, snapshot.values, snapshot.scores,
      std::vector<uint8_t>(snapshot.exported, 1), [&](K key, S score, V value) {
        return legal_generation(key, score, value, true);
      });
  validate_snapshot(table, snapshot, m);
  m.notes = failed
                ? "thread_exception"
                : "api_lock=true;host_R/U/"
                  "I_requests_overlap;cross_role_kernels_serialized;accepted="
                  "final_table_residents;rejected_and_evicted=unobserved;"
                  "read_hits_required_when_bucket_candidates_fit;"
                  "read_pairs_checked=true;final_writer_generation_required";
  return m;
}

uint64_t parse_positive(const std::string& value) {
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected a positive decimal integer");
  size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed == 0)
    throw std::invalid_argument("expected a positive decimal integer");
  return parsed;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) -> std::string {
      return arg.substr(prefix.size());
    };
    if (arg == "--mode=full") {
      cfg.mode = "full";
      cfg.capacity = 16UL * 1024 * 1024;
      cfg.batch = 1024UL * 1024;
    } else if (arg == "--mode=quick") {
      cfg.mode = "quick";
      cfg.capacity = 128UL * 1024;
      cfg.batch = 32UL * 1024;
    } else if (arg.rfind("--table_mode=", 0) == 0) {
      cfg.table_mode = value("--table_mode=");
    } else if (arg.rfind("--scenario=", 0) == 0) {
      cfg.scenario = value("--scenario=");
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity = parse_positive(value("--capacity="));
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch = parse_positive(value("--batch="));
    } else if (arg.rfind("--streams=", 0) == 0) {
      const auto streams = parse_positive(value("--streams="));
      if (streams > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("--streams exceeds int range");
      cfg.streams = static_cast<int>(streams);
    } else if (arg.rfind("--alpha=", 0) == 0) {
      size_t consumed = 0;
      const auto text = value("--alpha=");
      cfg.alpha = std::stod(text, &consumed);
      if (consumed != text.size() || !std::isfinite(cfg.alpha) ||
          cfg.alpha <= 0)
        throw std::invalid_argument("--alpha must be positive and finite");
    } else if (arg == "--help" || arg == "-h") {
      std::cerr
          << "Usage: hkv_correctness_stress_benchmark [--mode=quick|full]\n"
          << "       [--table_mode=throughput|memory|both]\n"
          << "       [--scenario=all|duplicates|updates|admission|eviction|"
             "bucket_skew|skew|rui]\n"
          << "       [--capacity=N] [--batch=N] [--streams=N] [--alpha=A]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  if (cfg.capacity < kBucketSize * 2) {
    std::cerr << "--capacity must be at least " << (kBucketSize * 2) << "\n";
    std::exit(2);
  }
  if ((cfg.capacity & (cfg.capacity - 1)) != 0)
    throw std::invalid_argument("--capacity must be a power of two");
  const std::unordered_set<std::string> modes{"throughput", "memory", "both"};
  const std::unordered_set<std::string> scenarios{
      "all",      "duplicates",  "updates", "admission",
      "eviction", "bucket_skew", "skew",    "rui"};
  if (!modes.count(cfg.table_mode) || !scenarios.count(cfg.scenario))
    throw std::invalid_argument("invalid table_mode or scenario filter");
  if ((cfg.scenario == "updates" && cfg.table_mode == "throughput") ||
      (cfg.scenario == "rui" && cfg.table_mode == "memory"))
    throw std::invalid_argument("filter selects zero supported cases");
  if (cfg.batch == 0) {
    std::cerr << "--batch must be positive\n";
    std::exit(2);
  }
  if (cfg.streams < 1) {
    std::cerr << "--streams must be positive\n";
    std::exit(2);
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    CUDA_CHECK(cudaFree(0));
    print_header();
    size_t case_count = 0;
    size_t failed_cases = 0;
    auto run_and_print = [&](const Metrics& metrics) {
      print_metrics(metrics);
      ++case_count;
      failed_cases += metrics_failed(metrics);
    };

    if (want_scenario(cfg, "duplicates") &&
        want_table_mode(cfg, TableMode::kThroughput)) {
      run_and_print(duplicate_same_key_throughput(cfg));
      run_and_print(duplicate_groups_throughput(cfg));
    }
    if (want_scenario(cfg, "duplicates") &&
        want_table_mode(cfg, TableMode::kMemory)) {
      run_and_print(duplicate_contract_memory(cfg));
    }
    if (want_scenario(cfg, "updates") &&
        want_table_mode(cfg, TableMode::kMemory)) {
      run_and_print(memory_repeated_update(cfg));
    }
    if (want_scenario(cfg, "admission")) {
      if (want_table_mode(cfg, TableMode::kThroughput)) {
        run_and_print(
            admission_denominator(cfg, TableMode::kThroughput, false));
        run_and_print(admission_denominator(cfg, TableMode::kThroughput, true));
      }
      if (want_table_mode(cfg, TableMode::kMemory)) {
        run_and_print(admission_denominator(cfg, TableMode::kMemory, false));
        run_and_print(admission_denominator(cfg, TableMode::kMemory, true));
      }
    }
    if (want_scenario(cfg, "eviction")) {
      if (want_table_mode(cfg, TableMode::kThroughput)) {
        run_and_print(explicit_eviction_throughput(cfg));
      }
      if (want_table_mode(cfg, TableMode::kMemory)) {
        run_and_print(evicted_resident_memory(cfg));
      }
    }
    if (want_scenario(cfg, "bucket_skew")) {
      if (want_table_mode(cfg, TableMode::kThroughput)) {
        run_and_print(adversarial_bucket_throughput(cfg));
      }
      if (want_table_mode(cfg, TableMode::kMemory)) {
        run_and_print(adversarial_dual_pair_memory(cfg));
      }
    }
    if (want_scenario(cfg, "skew")) {
      if (want_table_mode(cfg, TableMode::kThroughput)) {
        run_and_print(skew_hotset_throughput(cfg));
      }
      if (want_table_mode(cfg, TableMode::kMemory)) {
        run_and_print(skew_repeated_memory(cfg));
      }
    }
    if (want_scenario(cfg, "rui") &&
        want_table_mode(cfg, TableMode::kThroughput)) {
      run_and_print(rui_overlap_throughput(cfg));
    }

    std::cerr << "cases=" << case_count << ";failed_cases=" << failed_cases
              << '\n';
    return case_count == 0 ? 2 : (failed_cases == 0 ? 0 : 1);
  } catch (const std::invalid_argument& error) {
    std::cerr << "invalid_configuration: " << error.what() << std::endl;
    return 2;
  } catch (const std::exception& error) {
    std::cout.flush();
    std::cerr << "scenario_exception: " << error.what() << std::endl;
    return 1;
  }
}
