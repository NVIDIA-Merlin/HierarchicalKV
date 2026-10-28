/*
 * HKV-level hash sensitivity benchmark for review-response experiments.
 *
 * Unlike hash_sensitivity_benchmark.cc.cu, this target exercises real HKV
 * kernels.  The runner rebuilds this binary with HKV_HASH_VARIANT set to
 * Murmur3, SplitMix64-style, xxHash-style avalanche, or wyhash-style finalizer.
 *
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#ifndef HKV_HASH_ACCOUNTING_TEST
#include "benchmark_util.cuh"
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"
#ifndef HKV_HASH_OUTCOMES_AVAILABLE
#define HKV_HASH_OUTCOMES_AVAILABLE 0
#endif
#endif

namespace {

using K = uint64_t;
using V = float;
using S = uint64_t;

enum class Distribution {
  kSequential,
  kUniform64,
  kStridedLowbits,
  kZipfUnique,
};

struct Accounting {
  size_t attempted = 0;
  size_t batch_distinct_attempts = 0;
  size_t observed_admissions = 0;
  size_t observed_evictions = 0;
  size_t unresolved_absent = 0;
  double first_observed_eviction_prebatch_lf =
      std::numeric_limits<double>::quiet_NaN();
  std::unordered_set<K> seen;
  std::unordered_set<K> resident;

  void observe(const std::vector<K>& keys, const std::vector<K>& snapshot,
               size_t capacity) {
    std::unordered_set<K> batch(keys.begin(), keys.end());
    std::unordered_set<K> current(snapshot.begin(), snapshot.end());
    if (capacity == 0 || current.size() > capacity ||
        current.size() != snapshot.size()) {
      throw std::runtime_error(
          "invalid resident snapshot size or duplicate key");
    }
    for (K key : current) {
      if (resident.count(key) == 0 && batch.count(key) == 0) {
        throw std::runtime_error(
            "resident key absent from prior state and batch");
      }
    }
    size_t lost = 0;
    for (K key : resident) lost += current.count(key) == 0;
    if (lost > 0 && observed_evictions == 0) {
      first_observed_eviction_prebatch_lf =
          static_cast<double>(resident.size()) / capacity;
    }
    for (K key : current) observed_admissions += resident.count(key) == 0;
    for (K key : batch) {
      unresolved_absent += resident.count(key) == 0 && current.count(key) == 0;
    }
    attempted += keys.size();
    batch_distinct_attempts += batch.size();
    observed_evictions += lost;
    seen.insert(batch.begin(), batch.end());
    resident = std::move(current);
  }
};

#ifndef HKV_HASH_ACCOUNTING_TEST
using Table =
    nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kCustomized>;

constexpr size_t kDim = 32;
constexpr size_t kBucketSize = 128;

using Outcome = nv::merlin::OccupyResult;
constexpr Outcome kOutcomeUnwritten = static_cast<Outcome>(0x6a6a6a6a);
constexpr Outcome kOutcomeCanary = static_cast<Outcome>(0x5b5b5b5b);

struct OutcomeCounts {
  size_t empty = 0;
  size_t reclaimed = 0;
  size_t updated = 0;
  size_t evicted = 0;
  size_t refused = 0;
  size_t illegal = 0;

  size_t admitted() const { return empty + reclaimed + evicted; }
  size_t attempted() const { return admitted() + updated + refused + illegal; }
  void add(const OutcomeCounts& batch) {
    empty += batch.empty;
    reclaimed += batch.reclaimed;
    updated += batch.updated;
    evicted += batch.evicted;
    refused += batch.refused;
    illegal += batch.illegal;
  }
};

OutcomeCounts checked_outcomes(const std::vector<Outcome>& guarded,
                               const std::vector<K>& keys,
                               const std::unordered_set<K>& before,
                               const std::vector<K>& snapshot) {
  if (guarded.size() < keys.size() + 2 || guarded.front() != kOutcomeCanary)
    throw std::runtime_error("outcome prefix canary/length failure");
  for (size_t index = keys.size() + 1; index < guarded.size(); ++index)
    if (guarded[index] != kOutcomeCanary)
      throw std::runtime_error("outcome suffix/padding canary failure");
  std::unordered_set<K> after(snapshot.begin(), snapshot.end());
  if (after.size() != snapshot.size() ||
      std::unordered_set<K>(keys.begin(), keys.end()).size() != keys.size())
    throw std::runtime_error("outcome oracle requires unique keys/snapshot");
  OutcomeCounts counts;
  for (size_t index = 0; index < keys.size(); ++index) {
    switch (guarded[index + 1]) {
      case Outcome::OCCUPIED_EMPTY:
        ++counts.empty;
        break;
      case Outcome::OCCUPIED_RECLAIMED:
        ++counts.reclaimed;
        break;
      case Outcome::DUPLICATE:
        if (before.count(keys[index]) == 0)
          throw std::runtime_error("outcome update without prior resident");
        ++counts.updated;
        break;
      case Outcome::EVICT:
        ++counts.evicted;
        break;
      case Outcome::REFUSED:
        if (after.count(keys[index]) != 0)
          throw std::runtime_error("refused unique request became resident");
        ++counts.refused;
        break;
      case Outcome::ILLEGAL:
        ++counts.illegal;
        break;
      default:
        throw std::runtime_error("nonterminal/unknown/unwritten outcome");
    }
  }
  if (counts.attempted() != keys.size() ||
      before.size() + counts.empty + counts.reclaimed != after.size())
    throw std::runtime_error("outcome request/occupancy conservation failure");
  size_t observed_added = 0;
  size_t observed_lost = 0;
  for (K key : after) observed_added += before.count(key) == 0;
  for (K key : before) observed_lost += after.count(key) == 0;
  if (observed_added > counts.admitted() || observed_lost > counts.evicted)
    throw std::runtime_error("outcomes smaller than snapshot lower bounds");
  if (counts.illegal != 0)
    throw std::runtime_error("legal input produced illegal outcomes: " +
                             std::to_string(counts.illegal));
  return counts;
}

struct Config {
  std::string mode = "quick";
  size_t capacity = 128UL * 1024;
  size_t batch = 32UL * 1024;
  size_t query = 32UL * 1024;
  double overfill = 1.125;
  uint64_t seed = 0;
  bool exact_events = false;
  bool hash_only = false;
  size_t hash_warmup = 3;
  size_t hash_runs = 5;
  std::vector<Distribution> distributions;
  std::vector<nv::merlin::TableMode> table_modes;
};

struct BucketStats {
  double cv = 0.0;
  double p99 = 0.0;
  uint32_t max = 0;
};

struct Result {
  std::string distribution;
  std::string table_mode;
  size_t capacity = 0;
  size_t requested_capacity = 0;
  size_t batch = 0;
  size_t query = 0;
  uint64_t seed = 0;
  double overfill = 0.0;
  Accounting accounting;
  bool exact_events = false;
  OutcomeCounts events;
  double first_exact_eviction_prebatch_lf =
      std::numeric_limits<double>::quiet_NaN();
  size_t final_size = 0;
  double achieved_lf = 0.0;
  double old_found_rate = 0.0;
  double recent_found_rate = 0.0;
  double insert_throughput_bkvs = 0.0;
  double find_throughput_bkvs = 0.0;
  double insert_seconds = 0.0;
  size_t primary_bucket_sample_size = 0;
  BucketStats primary_bucket_stats;
  std::string notes;
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
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  void copy_from_host(const std::vector<T>& host, size_t count,
                      cudaStream_t stream = 0) const {
    if (count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(ptr_, host.data(), count * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};
#endif

uint64_t splitmix64_host(uint64_t key) {
  uint64_t z = key + UINT64_C(0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

const char* distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::kSequential:
      return "sequential";
    case Distribution::kUniform64:
      return "uniform64";
    case Distribution::kStridedLowbits:
      return "strided_lowbits";
    case Distribution::kZipfUnique:
      return "zipf_unique";
  }
  return "unknown";
}

#ifndef HKV_HASH_ACCOUNTING_TEST
const char* table_mode_name(nv::merlin::TableMode mode) {
  return mode == nv::merlin::TableMode::kMemory ? "memory" : "throughput";
}
#endif

K sanitize_key(uint64_t key, uint64_t salt) {
  constexpr uint64_t kFirstReserved = std::numeric_limits<K>::max() - 3;
  if (key >= kFirstReserved || key == 0) {
    key ^= UINT64_C(0x9e3779b97f4a7c15) + salt;
  }
  if (key >= kFirstReserved || key == 0) {
    key = salt % (kFirstReserved - 1) + 1;
  }
  return key;
}

class ZipfGenerator {
 public:
  ZipfGenerator(uint64_t n, double theta, uint64_t seed)
      : n_(n), theta_(theta), rng_(seed) {
    zeta_n_ = zeta(n_, theta_);
    zeta_2_ = zeta(2, theta_);
    alpha_ = 1.0 / (1.0 - theta_);
    eta_ = (1.0 - std::pow(2.0 / n_, 1.0 - theta_)) / (1.0 - zeta_2_ / zeta_n_);
  }

  uint64_t next() {
    double u = dist_(rng_);
    double uz = u * zeta_n_;
    if (uz < 1.0) return 0;
    if (uz < 1.0 + std::pow(0.5, theta_)) return 1;
    uint64_t val =
        static_cast<uint64_t>(n_ * std::pow(eta_ * u - eta_ + 1.0, alpha_));
    return std::min(val, n_ - 1);
  }

 private:
  static double zeta(uint64_t n, double theta) {
    constexpr uint64_t kExact = 10000;
    double sum = 0.0;
    uint64_t exact = std::min(n, kExact);
    for (uint64_t i = 1; i <= exact; i++) {
      sum += 1.0 / std::pow(static_cast<double>(i), theta);
    }
    if (n > exact && theta != 1.0) {
      sum += (std::pow(static_cast<double>(n), 1.0 - theta) -
              std::pow(static_cast<double>(exact), 1.0 - theta)) /
             (1.0 - theta);
    }
    return sum;
  }

  uint64_t n_;
  double theta_;
  double zeta_n_ = 0.0;
  double zeta_2_ = 0.0;
  double alpha_ = 0.0;
  double eta_ = 0.0;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

std::vector<K> make_keys(Distribution distribution, size_t offset, size_t count,
                         size_t key_range, uint64_t seed) {
  std::vector<K> keys;
  keys.reserve(count);

  if (distribution == Distribution::kZipfUnique) {
    ZipfGenerator zipf(std::max<size_t>(key_range, count * 8), 1.1,
                       (0xBAD5EEDULL ^ seed) + offset);
    std::unordered_set<K> seen;
    seen.reserve(count * 2 + 1);
    while (keys.size() < count) {
      uint64_t rank = zipf.next();
      K key = sanitize_key(rank + 1, offset + keys.size());
      if (seen.insert(key).second) keys.push_back(key);
    }
    return keys;
  }

  for (size_t i = 0; i < count; i++) {
    uint64_t index = offset + i;
    uint64_t key = 0;
    switch (distribution) {
      case Distribution::kSequential:
        key = index + 1;
        break;
      case Distribution::kUniform64:
        key = splitmix64_host(index + UINT64_C(0xD1B54A32D192ED03) + seed);
        break;
      case Distribution::kStridedLowbits:
        key = (index + 1) << 16;
        break;
      case Distribution::kZipfUnique:
        break;
    }
    keys.push_back(sanitize_key(key, index + 1));
  }
  return keys;
}

uint64_t parse_unsigned(const std::string& value) {
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected unsigned decimal integer: " + value);
  }
  return std::stoull(value);
}

double parse_overfill(const std::string& value) {
  size_t parsed = 0;
  double ratio = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(ratio) || ratio <= 0.0) {
    throw std::invalid_argument("expected positive finite overfill: " + value);
  }
  return ratio;
}

#ifndef HKV_HASH_ACCOUNTING_TEST
std::vector<S> make_scores(size_t offset, size_t count) {
  std::vector<S> scores(count);
  for (size_t i = 0; i < count; i++) {
    scores[i] = static_cast<S>(offset + i + 1);
  }
  return scores;
}

BucketStats compute_primary_bucket_stats(const std::vector<K>& keys,
                                         size_t capacity,
                                         nv::merlin::TableMode mode) {
  size_t bucket_count = std::max<size_t>(1, capacity / kBucketSize);
  std::vector<uint32_t> occupancy(bucket_count, 0);

  for (K key : keys) {
    uint64_t hash = nv::merlin::Murmur3HashHost(key);
    size_t bucket = 0;
    if (mode == nv::merlin::TableMode::kMemory) {
      bucket = static_cast<uint32_t>(hash) % bucket_count;
    } else {
      size_t global_idx = static_cast<size_t>(hash % capacity);
      bucket = global_idx / kBucketSize;
    }
    occupancy[bucket]++;
  }

  std::vector<uint32_t> sorted = occupancy;
  std::sort(sorted.begin(), sorted.end());

  double mean = 0.0;
  for (uint32_t value : occupancy) mean += value;
  mean /= static_cast<double>(occupancy.size());

  double variance = 0.0;
  for (uint32_t value : occupancy) {
    double delta = static_cast<double>(value) - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(occupancy.size());

  BucketStats stats;
  stats.cv = mean == 0.0 ? 0.0 : std::sqrt(variance) / mean;
  size_t p99_index =
      std::min(sorted.size() - 1,
               static_cast<size_t>(std::ceil(sorted.size() * 0.99)) - 1);
  stats.p99 = static_cast<double>(sorted[p99_index]);
  stats.max = sorted.back();
  return stats;
}

double found_rate(Table& table, const std::vector<K>& keys,
                  DeviceBuffer<K>& d_keys, DeviceBuffer<V>& d_values,
                  DeviceBuffer<bool>& d_found, cudaStream_t stream,
                  const std::unordered_set<K>& resident,
                  double* throughput_bkvs = nullptr) {
  if (keys.empty()) return 0.0;

  d_keys.copy_from_host(keys, keys.size(), stream);
  CUDA_CHECK(
      cudaMemsetAsync(d_found.get(), 0, keys.size() * sizeof(bool), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto timer = benchmark::Timer<double>();
  timer.start();
  table.find(keys.size(), d_keys.get(), d_values.get(), d_found.get(), nullptr,
             stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.end();

  std::vector<uint8_t> found(keys.size());
  CUDA_CHECK(cudaMemcpyAsync(found.data(), d_found.get(),
                             keys.size() * sizeof(bool), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t count = 0;
  for (size_t index = 0; index < keys.size(); index++) {
    bool expected = resident.count(keys[index]) != 0;
    if ((found[index] != 0) != expected) {
      throw std::runtime_error("find mask disagrees with resident snapshot");
    }
    count += found[index] ? 1 : 0;
  }
  if (throughput_bkvs != nullptr) {
    *throughput_bkvs = keys.size() / timer.getResult() / 1.0e9;
  }
  return static_cast<double>(count) / static_cast<double>(keys.size());
}

void init_table(Table& table, const Config& cfg, nv::merlin::TableMode mode) {
  nv::merlin::HashTableOptions options;
  options.init_capacity = cfg.capacity;
  options.max_capacity = cfg.capacity;
  options.dim = kDim;
  options.max_bucket_size = kBucketSize;
  options.max_load_factor = 1.0f;
  options.max_hbm_for_vectors = mode == nv::merlin::TableMode::kMemory
                                    ? 0
                                    : cfg.capacity * kDim * sizeof(V);
  options.table_mode = mode;
  options.api_lock = true;

  table.init(options);
}

std::vector<K> resident_snapshot(Table& table, DeviceBuffer<K>& device_keys,
                                 DeviceBuffer<V>& device_values,
                                 size_t chunk_size, cudaStream_t stream) {
  std::vector<K> snapshot;
  snapshot.reserve(table.size(stream));
  for (size_t offset = 0; offset < table.capacity(); offset += chunk_size) {
    size_t count = table.export_batch(
        std::min(chunk_size, table.capacity() - offset), offset,
        device_keys.get(), device_values.get(), nullptr, stream);
    size_t start = snapshot.size();
    snapshot.resize(start + count);
    if (count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(snapshot.data() + start, device_keys.get(),
                                 count * sizeof(K), cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  if (snapshot.size() != table.size(stream)) {
    throw std::runtime_error("resident snapshot disagrees with table.size");
  }
  return snapshot;
}

Result run_case(const Config& cfg, Distribution distribution,
                nv::merlin::TableMode mode, cudaStream_t stream) {
  Table table;
  init_table(table, cfg, mode);
  const size_t max_attempted =
      static_cast<size_t>(std::ceil(cfg.capacity * cfg.overfill));
  const size_t key_range =
      std::max<size_t>(cfg.capacity * 64, max_attempted * 4);

  DeviceBuffer<K> d_keys(cfg.batch);
  DeviceBuffer<V> d_values(cfg.batch * kDim);
  DeviceBuffer<S> d_scores(cfg.batch);
  DeviceBuffer<Outcome> d_outcomes(cfg.exact_events ? cfg.batch + 2 : 0);
  std::vector<V> values(cfg.batch * kDim, 1.0f);

  std::vector<K> first_keys;
  std::vector<K> recent_keys;
  first_keys.reserve(std::min(cfg.query, cfg.batch));
  recent_keys.reserve(cfg.query);

  Accounting accounting;
  std::vector<K> sample_for_buckets;
  OutcomeCounts events;
  double first_exact_eviction_prebatch_lf =
      std::numeric_limits<double>::quiet_NaN();
  const size_t sample_limit = std::min(max_attempted, table.capacity());
  sample_for_buckets.reserve(sample_limit);
  double insert_seconds = 0.0;

  while (accounting.attempted < max_attempted) {
    size_t cur = std::min(cfg.batch, max_attempted - accounting.attempted);
    std::vector<K> keys =
        make_keys(distribution, accounting.attempted, cur, key_range, cfg.seed);
    if (std::unordered_set<K>(keys.begin(), keys.end()).size() != keys.size()) {
      throw std::runtime_error("generated batch violates unique_key contract");
    }
    std::vector<S> scores = make_scores(accounting.attempted, cur);
    size_t sample_take =
        std::min(sample_limit - sample_for_buckets.size(), cur);
    sample_for_buckets.insert(sample_for_buckets.end(), keys.begin(),
                              keys.begin() + sample_take);

    if (first_keys.size() < cfg.query) {
      size_t take = std::min(cfg.query - first_keys.size(), keys.size());
      first_keys.insert(first_keys.end(), keys.begin(), keys.begin() + take);
    }
    recent_keys.insert(recent_keys.end(), keys.begin(), keys.end());
    if (recent_keys.size() > cfg.query) {
      recent_keys.erase(recent_keys.begin(),
                        recent_keys.begin() + (recent_keys.size() - cfg.query));
    }

    d_keys.copy_from_host(keys, cur, stream);
    d_values.copy_from_host(values, cur * kDim, stream);
    d_scores.copy_from_host(scores, cur, stream);
    std::vector<Outcome> guarded;
    if (cfg.exact_events) {
      guarded.assign(cfg.batch + 2, kOutcomeCanary);
      std::fill(guarded.begin() + 1, guarded.begin() + cur + 1,
                kOutcomeUnwritten);
      d_outcomes.copy_from_host(guarded, guarded.size(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto timer = benchmark::Timer<double>();
    timer.start();
    if (cfg.exact_events) {
#if HKV_HASH_OUTCOMES_AVAILABLE
      table.insert_or_assign_impl<nv::merlin::EvictStrategy::kCustomized>(
          cur, d_keys.get(), d_values.get(), d_scores.get(), stream, true,
          false, d_outcomes.get() + 1);
#else
      throw std::runtime_error(
          "exact outcomes require the reviewed hook build");
#endif
    } else {
      table.insert_or_assign(cur, d_keys.get(), d_values.get(), d_scores.get(),
                             stream, true, false);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();
    insert_seconds += timer.getResult();

    if (cfg.exact_events) {
      CUDA_CHECK(cudaMemcpyAsync(guarded.data(), d_outcomes.get(),
                                 guarded.size() * sizeof(Outcome),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto snapshot =
        resident_snapshot(table, d_keys, d_values, cfg.batch, stream);
    if (cfg.exact_events) {
      auto batch_events =
          checked_outcomes(guarded, keys, accounting.resident, snapshot);
      if (events.evicted == 0 && batch_events.evicted > 0)
        first_exact_eviction_prebatch_lf =
            static_cast<double>(accounting.resident.size()) / table.capacity();
      events.add(batch_events);
    }
    accounting.observe(keys, snapshot, table.capacity());
  }

  DeviceBuffer<K> d_find_keys(cfg.query);
  DeviceBuffer<V> d_find_values(cfg.query * kDim);
  DeviceBuffer<bool> d_found(cfg.query);
  double find_tp = 0.0;
  double old_found = found_rate(table, first_keys, d_find_keys, d_find_values,
                                d_found, stream, accounting.resident);
  double recent_found =
      found_rate(table, recent_keys, d_find_keys, d_find_values, d_found,
                 stream, accounting.resident, &find_tp);

  Result result;
  result.distribution = distribution_name(distribution);
  result.table_mode = table_mode_name(mode);
  result.capacity = table.capacity();
  result.requested_capacity = cfg.capacity;
  result.batch = cfg.batch;
  result.query = cfg.query;
  result.seed = cfg.seed;
  result.overfill = cfg.overfill;
  result.accounting = std::move(accounting);
  result.exact_events = cfg.exact_events;
  result.events = events;
  result.first_exact_eviction_prebatch_lf = first_exact_eviction_prebatch_lf;
  result.final_size = table.size(stream);
  if (cfg.exact_events &&
      (events.attempted() != result.accounting.attempted ||
       events.empty + events.reclaimed != result.final_size))
    throw std::runtime_error("aggregate outcome conservation failure");
  result.achieved_lf = static_cast<double>(result.final_size) /
                       static_cast<double>(result.capacity);
  result.old_found_rate = old_found;
  result.recent_found_rate = recent_found;
  result.insert_throughput_bkvs =
      result.accounting.attempted / insert_seconds / 1.0e9;
  result.find_throughput_bkvs = find_tp;
  result.insert_seconds = insert_seconds;
  result.primary_bucket_sample_size = sample_for_buckets.size();
  result.primary_bucket_stats =
      compute_primary_bucket_stats(sample_for_buckets, result.capacity, mode);
  result.notes =
      "batch_boundary_snapshots;observed_transitions_are_event_lower_bounds;"
      "host_wall_insert_or_assign_plus_terminal_sync;"
      "untimed_generation_copies_snapshots_oracles;"
      "rates_decimal_1e9_attempts_or_queries;"
      "primary_buckets_host_mapped_actual_attempted_prefix;"
      "empty_initial_state;no_warmup;snapshot_instrumented";
  result.notes +=
      cfg.exact_events
          ? ";same_run_terminal_outcomes;diagnostic_timing_only;"
            "no_replay_attribution;evicted_event_count_no_victim_ids"
          : ";exact_admitted_rejected_evicted_unknown";
  return result;
}

__global__ void hash_only_kernel(const K* keys, K* hashes, size_t count) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; index < count; index += stride)
    hashes[index] = nv::merlin::HkvHash64Device(keys[index]);
}

uint64_t trace_fingerprint(const std::vector<K>& keys) {
  uint64_t fingerprint = UINT64_C(14695981039346656037);
  for (K key : keys) {
    for (size_t byte = 0; byte < sizeof(K); ++byte) {
      fingerprint ^= (key >> (byte * 8)) & 255;
      fingerprint *= UINT64_C(1099511628211);
    }
  }
  return fingerprint;
}

void check_hash_outputs(const std::vector<K>& keys,
                        const std::vector<K>& guarded) {
  constexpr K canary = UINT64_C(0x5b5b5b5b5b5b5b5b);
  if (guarded.size() != keys.size() + 2 || guarded.front() != canary ||
      guarded.back() != canary)
    throw std::runtime_error("hash-only output canary/length failure");
  for (size_t index = 0; index < keys.size(); ++index)
    if (guarded[index + 1] != nv::merlin::HkvHash64Host(keys[index]))
      throw std::runtime_error("hash-only device/host output mismatch");
}

void run_hash_only(const Config& cfg, Distribution distribution,
                   cudaStream_t stream) {
  const size_t total =
      static_cast<size_t>(std::ceil(cfg.capacity * cfg.overfill));
  const size_t key_range = std::max<size_t>(cfg.capacity * 64, total * 4);
  DeviceBuffer<K> device_keys(cfg.batch);
  DeviceBuffer<K> device_hashes(cfg.batch + 2);
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  for (size_t offset = 0; offset < total; offset += cfg.batch) {
    const size_t count = std::min(cfg.batch, total - offset);
    const auto keys =
        make_keys(distribution, offset, count, key_range, cfg.seed);
    device_keys.copy_from_host(keys, count, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto fingerprint = trace_fingerprint(keys);
    for (size_t run = 0; run < cfg.hash_warmup + cfg.hash_runs; ++run) {
      std::vector<K> hashes(count + 2, UINT64_C(0x5b5b5b5b5b5b5b5b));
      device_hashes.copy_from_host(hashes, hashes.size(), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto wall = benchmark::Timer<double>();
      wall.start();
      CUDA_CHECK(cudaEventRecord(begin, stream));
      hash_only_kernel<<<std::min<size_t>((count + 255) / 256, 65535), 256, 0,
                         stream>>>(device_keys.get(), device_hashes.get() + 1,
                                   count);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaEventSynchronize(end));
      wall.end();
      float gpu_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, begin, end));
      if (!std::isfinite(gpu_ms) || gpu_ms <= 0 || wall.getResult() <= 0)
        throw std::runtime_error("hash-only timing has no positive interval");
      CUDA_CHECK(cudaMemcpyAsync(hashes.data(), device_hashes.get(),
                                 hashes.size() * sizeof(K),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      check_hash_outputs(keys, hashes);
      std::cout << nv::merlin::HkvHashVariantNameHost() << ','
                << distribution_name(distribution) << ',' << cfg.seed << ','
                << cfg.capacity << ',' << cfg.batch << ',' << cfg.overfill
                << ',' << offset << ',' << count << ',' << run << ','
                << (run < cfg.hash_warmup ? 1 : 0) << ',' << cfg.hash_warmup
                << ',' << cfg.hash_runs << ',' << fingerprint << ','
                << std::setprecision(12) << gpu_ms << ',' << wall.getResult()
                << ',' << count / static_cast<double>(gpu_ms) / 1.0e6 << ','
                << count * sizeof(K) << ',' << count * sizeof(K)
                << ",hash_read_write_kernel_not_arithmetic_latency,PASS,"
                   "development_only\n";
    }
  }
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
}

void print_header() {
  std::cout
      << "hash_variant,distribution,table_mode,capacity,batch,attempted,"
         "final_size,achieved_lf,first_observed_eviction_prebatch_lf,old_found_"
         "rate,"
         "recent_found_rate,insert_throughput_gattempts_s,find_throughput_"
         "gqueries_s,"
         "primary_bucket_cv,primary_bucket_p99,primary_bucket_max,"
         "seed,seed_effect,requested_capacity,query,overfill,distinct_seen,"
         "repeated_requests,batch_distinct_attempts,observed_admissions_lower_"
         "bound,"
         "observed_evictions_lower_bound,rejected_or_transient_evicted,"
         "admitted,rejected,evicted,insert_seconds,primary_bucket_sample_size,"
         "notes,event_accounting,updated,illegal,occupied_empty,"
         "occupied_reclaimed,exact_event_attempted,"
         "first_exact_eviction_prebatch_lf,timing_class\n";
}

void print_result(const Result& result) {
  std::cout << nv::merlin::HkvHashVariantNameHost() << ','
            << result.distribution << ',' << result.table_mode << ','
            << result.capacity << ',' << result.batch << ','
            << result.accounting.attempted << ',' << result.final_size << ','
            << std::fixed << std::setprecision(6) << result.achieved_lf << ','
            << result.accounting.first_observed_eviction_prebatch_lf << ','
            << result.old_found_rate << ',' << result.recent_found_rate << ','
            << result.insert_throughput_bkvs << ','
            << result.find_throughput_bkvs << ','
            << result.primary_bucket_stats.cv << ','
            << result.primary_bucket_stats.p99 << ','
            << result.primary_bucket_stats.max << ',' << result.seed << ','
            << (result.distribution == "uniform64" ||
                        result.distribution == "zipf_unique"
                    ? "workload_rng"
                    : "fixed_structured_trace")
            << ',' << result.requested_capacity << ',' << result.query << ','
            << result.overfill << ',' << result.accounting.seen.size() << ','
            << result.accounting.attempted - result.accounting.seen.size()
            << ',' << result.accounting.batch_distinct_attempts << ','
            << result.accounting.observed_admissions << ','
            << result.accounting.observed_evictions << ','
            << result.accounting.unresolved_absent << ',';
  if (result.exact_events)
    std::cout << result.events.admitted() << ',' << result.events.refused << ','
              << result.events.evicted << ',';
  else
    std::cout << "unknown,unknown,unknown,";
  std::cout << std::setprecision(9) << result.insert_seconds << ','
            << result.primary_bucket_sample_size << ',' << result.notes << ',';
  if (result.exact_events)
    std::cout << "same_run_terminal_outcomes," << result.events.updated << ','
              << result.events.illegal << ',' << result.events.empty << ','
              << result.events.reclaimed << ',' << result.events.attempted()
              << ',' << result.first_exact_eviction_prebatch_lf
              << ",diagnostic_only\n";
  else
    std::cout
        << "snapshot_lower_bounds,unknown,unknown,unknown,unknown,unknown,"
           "unknown,snapshot_instrumented_host_wall\n";
}

Distribution parse_distribution(const std::string& value) {
  if (value == "sequential") return Distribution::kSequential;
  if (value == "uniform64") return Distribution::kUniform64;
  if (value == "strided_lowbits") return Distribution::kStridedLowbits;
  if (value == "zipf_unique") return Distribution::kZipfUnique;
  throw std::invalid_argument("unknown distribution: " + value);
}

nv::merlin::TableMode parse_table_mode(const std::string& value) {
  if (value == "throughput") return nv::merlin::TableMode::kThroughput;
  if (value == "memory") return nv::merlin::TableMode::kMemory;
  throw std::invalid_argument("unknown table mode: " + value);
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) tokens.push_back(token);
  }
  return tokens;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int index = 1; index < argc; index++) {
    std::string arg = argv[index];
    if (arg == "--mode" && index + 1 < argc) cfg.mode = argv[++index];
    if (arg.rfind("--mode=", 0) == 0) cfg.mode = arg.substr(7);
  }
  if (cfg.mode != "quick" && cfg.mode != "full") {
    throw std::invalid_argument("unknown mode: " + cfg.mode);
  }
  if (cfg.mode == "full") {
    cfg.capacity = 1024UL * 1024;
    cfg.batch = 128UL * 1024;
    cfg.query = 64UL * 1024;
  }
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + name);
      return argv[++i];
    };

    if (arg == "--mode") {
      cfg.mode = next_value(arg);
    } else if (arg.rfind("--mode=", 0) == 0) {
      cfg.mode = arg.substr(7);
    } else if (arg == "--capacity") {
      cfg.capacity = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity = parse_unsigned(arg.substr(11));
    } else if (arg == "--batch") {
      cfg.batch = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch = parse_unsigned(arg.substr(8));
    } else if (arg == "--query") {
      cfg.query = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--query=", 0) == 0) {
      cfg.query = parse_unsigned(arg.substr(8));
    } else if (arg == "--seed") {
      cfg.seed = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--seed=", 0) == 0) {
      cfg.seed = parse_unsigned(arg.substr(7));
    } else if (arg == "--exact-events") {
      cfg.exact_events = true;
    } else if (arg == "--hash-only") {
      cfg.hash_only = true;
    } else if (arg == "--hash-warmup") {
      cfg.hash_warmup = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--hash-warmup=", 0) == 0) {
      cfg.hash_warmup = parse_unsigned(arg.substr(14));
    } else if (arg == "--hash-runs") {
      cfg.hash_runs = parse_unsigned(next_value(arg));
    } else if (arg.rfind("--hash-runs=", 0) == 0) {
      cfg.hash_runs = parse_unsigned(arg.substr(12));
    } else if (arg == "--overfill") {
      cfg.overfill = parse_overfill(next_value(arg));
    } else if (arg.rfind("--overfill=", 0) == 0) {
      cfg.overfill = parse_overfill(arg.substr(11));
    } else if (arg == "--distributions") {
      cfg.distributions.clear();
      for (const auto& value : split_csv(next_value(arg))) {
        cfg.distributions.push_back(parse_distribution(value));
      }
    } else if (arg.rfind("--distributions=", 0) == 0) {
      cfg.distributions.clear();
      for (const auto& value : split_csv(arg.substr(16))) {
        cfg.distributions.push_back(parse_distribution(value));
      }
    } else if (arg == "--table_modes") {
      cfg.table_modes.clear();
      for (const auto& value : split_csv(next_value(arg))) {
        cfg.table_modes.push_back(parse_table_mode(value));
      }
    } else if (arg.rfind("--table_modes=", 0) == 0) {
      cfg.table_modes.clear();
      for (const auto& value : split_csv(arg.substr(14))) {
        cfg.table_modes.push_back(parse_table_mode(value));
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: hkv_hash_sensitivity_benchmark [--mode quick|full]\n"
                << "       [--capacity=N] [--batch=N] [--query=N]\n"
                << "       [--overfill=RATIO] [--seed=UINT64]\n"
                << "       "
                   "[--distributions=sequential,uniform64,strided_lowbits,zipf_"
                   "unique]\n"
                << "       [--table_modes=throughput,memory]\n"
                << "       [--exact-events | --hash-only]\n"
                << "       [--hash-warmup=N] [--hash-runs=N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.distributions.empty()) {
    cfg.distributions = {Distribution::kSequential,
                         Distribution::kStridedLowbits};
    if (cfg.mode == "full") {
      cfg.distributions = {Distribution::kSequential, Distribution::kUniform64,
                           Distribution::kStridedLowbits,
                           Distribution::kZipfUnique};
    }
  }
  if (cfg.table_modes.empty()) {
    cfg.table_modes = {nv::merlin::TableMode::kThroughput,
                       nv::merlin::TableMode::kMemory};
  }

  if (cfg.capacity == 0 || cfg.batch == 0 || cfg.query == 0 ||
      cfg.capacity % kBucketSize != 0) {
    throw std::invalid_argument(
        "capacity/batch/query must be positive; capacity must be divisible by "
        "bucket size");
  }
  if (!std::isfinite(cfg.overfill) || cfg.overfill <= 0.0 ||
      std::ceil(static_cast<long double>(cfg.capacity) * cfg.overfill) >
          std::numeric_limits<size_t>::max() / 128 ||
      cfg.capacity > std::numeric_limits<size_t>::max() / 128) {
    throw std::invalid_argument("overfill or capacity exceeds supported range");
  }
  cfg.batch = std::min(cfg.batch, cfg.capacity);
  cfg.query = std::min(cfg.query, cfg.batch);
  if (cfg.hash_runs == 0 ||
      cfg.hash_warmup > std::numeric_limits<size_t>::max() - cfg.hash_runs ||
      (cfg.exact_events && cfg.hash_only))
    throw std::invalid_argument("invalid or conflicting diagnostic options");
  if (cfg.exact_events && !HKV_HASH_OUTCOMES_AVAILABLE)
    throw std::invalid_argument(
        "exact outcomes require the reviewed hook build");
  if (cfg.exact_events && cfg.batch > UINT32_MAX - 127)
    throw std::invalid_argument(
        "exact outcome batch exceeds padded index range");
  return cfg;
}
#else
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct CpuOracle {
  size_t capacity;
  size_t admissions = 0;
  size_t evictions = 0;
  S ticket = 0;
  std::map<K, S> resident;

  std::vector<K> insert(const std::vector<K>& keys) {
    for (K key : keys) {
      if (resident.count(key) == 0) {
        admissions++;
        if (resident.size() == capacity) {
          auto victim =
              std::min_element(resident.begin(), resident.end(),
                               [](const auto& left, const auto& right) {
                                 return left.second < right.second;
                               });
          resident.erase(victim);
          evictions++;
        }
      }
      resident[key] = ++ticket;
    }
    std::vector<K> snapshot;
    for (const auto& entry : resident) snapshot.push_back(entry.first);
    return snapshot;
  }
};

void accounting_regression() {
  Accounting repeats;
  CpuOracle roomy{4, 0, 0, 0, {}};
  for (const std::vector<K>& batch : {std::vector<K>{1, 2}, {2, 3}, {1, 3}}) {
    repeats.observe(batch, roomy.insert(batch), roomy.capacity);
  }
  require(repeats.attempted == 6 && repeats.seen.size() == 3 &&
              repeats.resident.size() == 3 && repeats.observed_evictions == 0 &&
              repeats.observed_admissions == roomy.admissions &&
              roomy.evictions == 0 && repeats.unresolved_absent == 0 &&
              std::isnan(repeats.first_observed_eviction_prebatch_lf),
          "cross-batch repeats must not imply eviction");

  Accounting displacement;
  CpuOracle small{2, 0, 0, 0, {}};
  for (const std::vector<K>& batch : {std::vector<K>{1, 2}, {3}, {1}}) {
    displacement.observe(batch, small.insert(batch), small.capacity);
  }
  require(displacement.attempted == 4 && displacement.seen.size() == 3 &&
              displacement.resident.size() == 2 &&
              displacement.observed_evictions == 2 && small.evictions == 2 &&
              displacement.observed_admissions == 4 && small.admissions == 4 &&
              displacement.unresolved_absent == 0 &&
              displacement.first_observed_eviction_prebatch_lf == 1.0,
          "actual oracle displacement and readmission must be counted");

  Accounting hidden;
  CpuOracle transient{2, 0, 0, 0, {}};
  hidden.observe({1, 2, 3}, transient.insert({1, 2, 3}), transient.capacity);
  require(hidden.observed_admissions == 2 && transient.admissions == 3 &&
              hidden.observed_evictions == 0 && transient.evictions == 1 &&
              hidden.unresolved_absent == 1 &&
              std::isnan(hidden.first_observed_eviction_prebatch_lf),
          "within-batch transient loss must remain unresolved");

  Accounting rejection;
  rejection.observe({1, 2}, {1, 2}, 2);
  rejection.observe({3}, {1, 2}, 2);
  require(rejection.observed_admissions == 2 &&
              rejection.observed_evictions == 0 &&
              rejection.unresolved_absent == 1,
          "absent rejected candidate must not fabricate an eviction");

  for (const std::vector<K>& invalid : {std::vector<K>{1, 1}, {4}, {1, 2, 3}}) {
    bool rejected = false;
    try {
      Accounting bad;
      bad.observe({1, 2}, invalid, 2);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "invalid snapshot must fail closed");
  }

  for (Distribution distribution :
       {Distribution::kSequential, Distribution::kUniform64,
        Distribution::kStridedLowbits, Distribution::kZipfUnique}) {
    auto first = make_keys(distribution, 0, 256, 8192, 23);
    auto same = make_keys(distribution, 0, 256, 8192, 23);
    auto different = make_keys(distribution, 0, 256, 8192, 24);
    require(first == same &&
                std::unordered_set<K>(first.begin(), first.end()).size() == 256,
            std::string(distribution_name(distribution)) +
                " not reproducible/unique");
    bool randomized = distribution == Distribution::kUniform64 ||
                      distribution == Distribution::kZipfUnique;
    require(randomized ? first != different : first == different,
            "seed effect must match declared distribution semantics");
  }
  Accounting zipf_repeats;
  CpuOracle zipf_oracle{1024, 0, 0, 0, {}};
  for (size_t offset : {size_t{0}, size_t{256}}) {
    auto batch = make_keys(Distribution::kZipfUnique, offset, 256, 8192, 23);
    zipf_repeats.observe(batch, zipf_oracle.insert(batch),
                         zipf_oracle.capacity);
  }
  require(zipf_repeats.seen.size() < zipf_repeats.attempted &&
              zipf_repeats.observed_evictions == 0 &&
              zipf_oracle.evictions == 0,
          "real Zipf generator cross-batch repetition must not imply eviction");

  require(parse_unsigned("18446744073709551615") ==
              std::numeric_limits<uint64_t>::max(),
          "full uint64 seed must be supported");
  for (const char* invalid : {"-1", "+1", "1x", "", "18446744073709551616"}) {
    bool rejected = false;
    try {
      parse_unsigned(invalid);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "invalid integer must fail closed");
  }
  for (K key :
       {K{0}, std::numeric_limits<K>::max(), std::numeric_limits<K>::max() - 1,
        std::numeric_limits<K>::max() - 2, std::numeric_limits<K>::max() - 3}) {
    K sanitized = sanitize_key(key, std::numeric_limits<K>::max());
    require(sanitized > 0 && sanitized < std::numeric_limits<K>::max() - 3,
            "reserved keys must not reach HKV");
  }
  require(parse_overfill("1.125") == 1.125, "valid overfill must be accepted");
  for (const char* invalid : {"0", "-1", "nan", "inf", "1x", ""}) {
    bool rejected = false;
    try {
      parse_overfill(invalid);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "invalid overfill must fail closed");
  }
  std::cout << "PASS: 10 CPU accounting/generator regression groups; "
               "GPU execution NOT_RUN\n";
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef HKV_HASH_ACCOUNTING_TEST
    static_cast<void>(argc);
    static_cast<void>(argv);
    accounting_regression();
#else
    Config cfg = parse_args(argc, argv);
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cerr << "GPU: " << props.name << "\n";
    std::cerr << "HKV hash variant: " << nv::merlin::HkvHashVariantNameHost()
              << " (" << HKV_HASH_VARIANT << ")\n";
    std::cerr << "mode=" << cfg.mode << " capacity=" << cfg.capacity
              << " batch=" << cfg.batch << " query=" << cfg.query
              << " overfill=" << cfg.overfill << " seed=" << cfg.seed << "\n";

    CUDA_CHECK(cudaFree(0));
    if (cfg.hash_only)
      std::cout << "hash_variant,distribution,seed,requested_capacity,batch,"
                   "overfill,key_offset,count,run,warmup,configured_warmup,"
                   "configured_runs,input_trace_fnv64,gpu_ms,host_wall_seconds,"
                   "hash_throughput_ghashes_s,input_bytes,output_bytes,"
                   "measurement,oracle_status,timing_class\n";
    else
      print_header();
    for (Distribution distribution : cfg.distributions) {
      if (cfg.hash_only) {
        run_hash_only(cfg, distribution, 0);
        continue;
      }
      for (auto table_mode : cfg.table_modes) {
        std::cerr << "running distribution=" << distribution_name(distribution)
                  << " table_mode=" << table_mode_name(table_mode) << std::endl;
        print_result(run_case(cfg, distribution, table_mode, 0));
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  } catch (const std::exception& e) {
    std::cerr << "hkv_hash_sensitivity_benchmark: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
