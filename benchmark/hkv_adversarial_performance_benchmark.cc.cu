#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"

namespace {

constexpr size_t kDim = 32;
constexpr size_t kBucketSize = 128;
constexpr size_t kLogicalLanes = 8;
using K = uint64_t;
using V = float;
using S = uint64_t;
using Clock = std::chrono::steady_clock;
using Table =
    nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kCustomized>;

struct Config {
  std::string mode = "quick";
  std::string scenario = "all";
  std::string roles = "RUI";
  std::string overlap_kind = "hot_key";
  size_t capacity = 1UL << 20;
  size_t batch = 1UL << 16;
  size_t adversarial_batch = 1UL << 15;
  size_t total_requests = 0;
  size_t request_batch = 128;
  uint64_t seed = 20260611;
  int streams = 6;
  int warmup = 1;
  int runs = 3;
  int overlap_pct = 50;
  int ingress_pct = 50;
  double alpha = 1.25;
  double load_factor = 0.5;
  bool streams_set = false;
  bool overlap_set = false;
  bool kind_set = false;
};

void require(bool valid, const std::string& message) {
  if (!valid) throw std::runtime_error(message);
}

void check_cuda(cudaError_t result) {
  require(result == cudaSuccess, cudaGetErrorString(result));
}

template <typename Element>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    require(count <= std::numeric_limits<size_t>::max() / sizeof(Element),
            "Device allocation size overflow");
    if (count != 0)
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                            count * sizeof(Element)));
  }
  ~DeviceBuffer() {
    if (data_) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  Element* get() const { return data_; }
  void upload(const Element* source, size_t count, cudaStream_t stream) {
    require(count <= count_, "Upload exceeds buffer");
    if (count)
      check_cuda(cudaMemcpyAsync(data_, source, count * sizeof(Element),
                                 cudaMemcpyHostToDevice, stream));
  }
  void download(Element* destination, size_t count, cudaStream_t stream) {
    require(count <= count_, "Download exceeds buffer");
    if (count)
      check_cuda(cudaMemcpyAsync(destination, data_, count * sizeof(Element),
                                 cudaMemcpyDeviceToHost, stream));
  }

 private:
  Element* data_ = nullptr;
  size_t count_;
};

class Stream {
 public:
  Stream() { check_cuda(cudaStreamCreate(&stream_)); }
  ~Stream() { cudaStreamDestroy(stream_); }
  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

struct Buffers {
  explicit Buffers(size_t count)
      : keys(count), values(count * kDim), scores(count), founds(count) {}
  DeviceBuffer<K> keys;
  DeviceBuffer<V> values;
  DeviceBuffer<S> scores;
  DeviceBuffer<uint8_t> founds;
};

struct Returned {
  explicit Returned(size_t count = 0)
      : values(count * kDim), scores(count), founds(count) {}
  std::vector<V> values;
  std::vector<S> scores;
  std::vector<uint8_t> founds;
};

struct Snapshot {
  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
};

struct Batch {
  char role = 'R';
  S generation = 0;
  std::vector<K> keys;
  Returned returned;
  bool unique = true;
  bool completed = false;
  double latency_us = 0;
  double service_us = 0;
  Clock::time_point finished;
};

struct Metrics {
  std::string scenario;
  std::string operation;
  std::string roles;
  std::string overlap_kind = "none";
  std::string timing_scope;
  std::string latency_scope;
  uint64_t planned = 0;
  uint64_t completed = 0;
  uint64_t read_requests = 0;
  uint64_t update_requests = 0;
  uint64_t insert_requests = 0;
  uint64_t read_hits = 0;
  uint64_t read_misses = 0;
  uint64_t checked_rows = 0;
  uint64_t errors = 0;
  uint64_t worker_errors = 0;
  uint64_t initial_size = 0;
  uint64_t final_size = 0;
  uint64_t final_new_keys = 0;
  uint64_t required_writer_residents = 0;
  uint64_t writer_residents = 0;
  uint64_t cross_worker_key_requests = 0;
  uint64_t cross_worker_bucket_requests = 0;
  uint64_t trace_hash = 1469598103934665603ULL;
  size_t request_batch = 0;
  size_t hot_buckets = 0;
  size_t device_worker_bytes = 0;
  int workers = 1;
  int overlap_pct = 0;
  int ingress_pct = 0;
  int run = 0;
  double seconds = 0;
  std::vector<double> latency_us;
  std::vector<double> service_us;
};

size_t throughput_bucket(K key, size_t capacity) {
  return (nv::merlin::Murmur3HashHost(key) & (capacity - 1)) / kBucketSize;
}

V payload_component(K key, S score, size_t column) {
  if (column < 4) return static_cast<V>((key >> (column * 16)) & 65535);
  if (column < 8) return static_cast<V>((score >> ((column - 4) * 16)) & 65535);
  return static_cast<V>(((key ^ score) >> ((column % 4) * 16) & 65535) +
                        column * 65536);
}

std::vector<V> payloads(const std::vector<K>& keys,
                        const std::vector<S>& scores) {
  require(keys.size() == scores.size(), "Payload size mismatch");
  std::vector<V> values(keys.size() * kDim);
  for (size_t row = 0; row < keys.size(); ++row)
    for (size_t column = 0; column < kDim; ++column)
      values[row * kDim + column] =
          payload_component(keys[row], scores[row], column);
  return values;
}

bool valid_payload(K key, S score, const V* values) {
  for (size_t column = 0; column < kDim; ++column)
    if (values[column] != payload_component(key, score, column)) return false;
  return true;
}

void create_table(Table& table, size_t capacity) {
  nv::merlin::HashTableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_bucket_size = kBucketSize;
  options.max_load_factor = 1.0f;
  options.dim = kDim;
  options.table_mode = nv::merlin::TableMode::kThroughput;
  options.max_hbm_for_vectors = capacity * kDim * sizeof(V);
  options.api_lock = true;
  table.init(options);
  require(table.capacity() == capacity,
          "Actual capacity differs from requested capacity");
}

bool unique_keys(const std::vector<K>& keys) {
  return std::unordered_set<K>(keys.begin(), keys.end()).size() == keys.size();
}

double insert_batch(Table& table, const std::vector<K>& keys,
                    const std::vector<S>& scores, bool unique) {
  Stream stream;
  Buffers buffers(keys.size());
  const auto values = payloads(keys, scores);
  buffers.keys.upload(keys.data(), keys.size(), stream);
  buffers.scores.upload(scores.data(), scores.size(), stream);
  buffers.values.upload(values.data(), values.size(), stream);
  check_cuda(cudaStreamSynchronize(stream));
  const auto start = Clock::now();
  table.insert_or_assign(keys.size(), buffers.keys.get(), buffers.values.get(),
                         buffers.scores.get(), stream, unique, false);
  check_cuda(cudaStreamSynchronize(stream));
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Returned lookup(Table& table, const std::vector<K>& keys) {
  Returned returned(keys.size());
  Stream stream;
  Buffers buffers(keys.size());
  buffers.keys.upload(keys.data(), keys.size(), stream);
  table.find(keys.size(), buffers.keys.get(), buffers.values.get(),
             reinterpret_cast<bool*>(buffers.founds.get()),
             buffers.scores.get(), stream);
  buffers.values.download(returned.values.data(), returned.values.size(),
                          stream);
  buffers.scores.download(returned.scores.data(), keys.size(), stream);
  buffers.founds.download(returned.founds.data(), keys.size(), stream);
  check_cuda(cudaStreamSynchronize(stream));
  return returned;
}

Snapshot snapshot(Table& table) {
  const size_t capacity = table.capacity();
  Stream stream;
  Buffers buffers(capacity);
  const size_t count =
      table.export_batch(capacity, 0, buffers.keys.get(), buffers.values.get(),
                         buffers.scores.get(), stream);
  check_cuda(cudaStreamSynchronize(stream));
  require(count <= capacity, "Export count exceeds capacity");
  Snapshot result;
  result.keys.resize(count);
  result.values.resize(count * kDim);
  result.scores.resize(count);
  buffers.keys.download(result.keys.data(), count, stream);
  buffers.values.download(result.values.data(), count * kDim, stream);
  buffers.scores.download(result.scores.data(), count, stream);
  check_cuda(cudaStreamSynchronize(stream));
  return result;
}

using ScoresByKey = std::unordered_map<K, S>;

ScoresByKey expected_top(const std::vector<K>& keys,
                         const std::vector<S>& scores, size_t capacity) {
  ScoresByKey distinct;
  for (size_t row = 0; row < keys.size(); ++row) {
    const auto inserted = distinct.emplace(keys[row], scores[row]);
    require(inserted.second || inserted.first->second == scores[row],
            "Static oracle requires one score per distinct key");
  }
  std::vector<std::vector<std::pair<S, K>>> buckets(capacity / kBucketSize);
  for (const auto& entry : distinct)
    buckets[throughput_bucket(entry.first, capacity)].emplace_back(entry.second,
                                                                   entry.first);
  ScoresByKey expected;
  for (auto& bucket : buckets) {
    std::sort(bucket.begin(), bucket.end(), std::greater<std::pair<S, K>>());
    for (size_t slot = 0; slot < std::min(kBucketSize, bucket.size()); ++slot)
      expected.emplace(bucket[slot].second, bucket[slot].first);
  }
  return expected;
}

void verify_exact(const std::vector<K>& keys, const Returned& returned,
                  const ScoresByKey& expected, Metrics& metrics,
                  bool count_reads) {
  for (size_t row = 0; row < keys.size(); ++row) {
    const auto found = expected.find(keys[row]);
    const bool should_find = found != expected.end();
    const bool did_find = returned.founds[row] != 0;
    metrics.errors += should_find != did_find;
    if (count_reads) {
      metrics.read_hits += did_find;
      metrics.read_misses += !did_find;
    }
    if (did_find) {
      metrics.errors += !should_find || returned.scores[row] != found->second;
      metrics.errors += !valid_payload(keys[row], returned.scores[row],
                                       returned.values.data() + row * kDim);
      ++metrics.checked_rows;
    }
  }
}

void verify_snapshot_exact(Table& table, const ScoresByKey& expected,
                           Metrics& metrics) {
  const auto exported = snapshot(table);
  metrics.final_size = table.size(0);
  metrics.errors += metrics.final_size != exported.keys.size();
  metrics.errors += exported.keys.size() != expected.size();
  std::unordered_set<K> seen;
  for (size_t row = 0; row < exported.keys.size(); ++row) {
    const auto found = expected.find(exported.keys[row]);
    metrics.errors += !seen.insert(exported.keys[row]).second;
    metrics.errors +=
        found == expected.end() || exported.scores[row] != found->second;
    metrics.errors += !valid_payload(exported.keys[row], exported.scores[row],
                                     exported.values.data() + row * kDim);
    ++metrics.checked_rows;
  }
  for (const auto& entry : expected)
    metrics.errors += seen.count(entry.first) == 0;
  for (size_t offset = 0; offset < exported.keys.size(); offset += 16384) {
    const size_t end = std::min(offset + 16384, exported.keys.size());
    std::vector<K> keys(exported.keys.begin() + offset,
                        exported.keys.begin() + end);
    verify_exact(keys, lookup(table, keys), expected, metrics, false);
  }
}

double percentile(std::vector<double> samples, double quantile) {
  if (samples.empty()) return 0;
  std::sort(samples.begin(), samples.end());
  const size_t index =
      static_cast<size_t>(std::ceil(quantile * samples.size())) - 1;
  return samples[std::min(index, samples.size() - 1)];
}

bool metrics_failed(const Metrics& metrics) {
  return metrics.errors || metrics.worker_errors || !metrics.planned ||
         metrics.completed != metrics.planned || !(metrics.seconds > 0) ||
         !std::isfinite(metrics.seconds) || metrics.latency_us.empty() ||
         metrics.read_hits + metrics.read_misses != metrics.read_requests ||
         metrics.read_requests + metrics.update_requests +
                 metrics.insert_requests !=
             metrics.completed;
}

void print_header() {
  std::cout
      << "scenario,operation,mode,table_mode,capacity,request_batch,hot_"
         "buckets,"
         "streams,run,seed,roles,overlap_kind,overlap_pct,ingress_pct,target_"
         "load_factor,"
         "initial_size,final_size,attempted,completed_requests,read_requests,"
         "update_requests,"
         "insert_requests,read_hits,read_misses,seconds,throughput_grequest_s,"
         "latency_samples,"
         "p50_us,p95_us,p99_us,service_p50_us,service_p95_us,service_p99_us,"
         "cross_worker_key_requests,cross_worker_bucket_requests,final_"
         "resident_new_keys,"
         "required_writer_residents,writer_residents,oracle_checked_rows,"
         "oracle_errors,"
         "worker_errors,trace_hash,device_payload_capacity_bytes,device_worker_"
         "buffer_bytes,"
         "timing_scope,latency_scope,completion_scope,outcome_scope,"
         "attribution_scope,status\n";
}

void print_metrics(const Config& cfg, const Metrics& metrics) {
  const double rate =
      metrics.seconds > 0 ? metrics.completed / metrics.seconds / 1e9 : 0;
  std::cout << std::setprecision(17) << metrics.scenario << ','
            << metrics.operation << ',' << cfg.mode << ",throughput,"
            << cfg.capacity << ',' << metrics.request_batch << ','
            << metrics.hot_buckets << ',' << metrics.workers << ','
            << metrics.run << ',' << cfg.seed << ',' << metrics.roles << ','
            << metrics.overlap_kind << ',' << metrics.overlap_pct << ','
            << metrics.ingress_pct << ',' << cfg.load_factor << ','
            << metrics.initial_size << ',' << metrics.final_size << ','
            << metrics.planned << ',' << metrics.completed << ','
            << metrics.read_requests << ',' << metrics.update_requests << ','
            << metrics.insert_requests << ',' << metrics.read_hits << ','
            << metrics.read_misses << ',' << std::fixed << std::setprecision(9)
            << metrics.seconds << ',' << rate << ','
            << metrics.latency_us.size() << ','
            << percentile(metrics.latency_us, 0.50) << ','
            << percentile(metrics.latency_us, 0.95) << ','
            << percentile(metrics.latency_us, 0.99) << ','
            << percentile(metrics.service_us, 0.50) << ','
            << percentile(metrics.service_us, 0.95) << ','
            << percentile(metrics.service_us, 0.99) << ','
            << metrics.cross_worker_key_requests << ','
            << metrics.cross_worker_bucket_requests << ','
            << metrics.final_new_keys << ','
            << metrics.required_writer_residents << ','
            << metrics.writer_residents << ',' << metrics.checked_rows << ','
            << metrics.errors << ',' << metrics.worker_errors << ','
            << metrics.trace_hash << ',' << cfg.capacity * kDim * sizeof(V)
            << ',' << metrics.device_worker_bytes << ',' << metrics.timing_scope
            << ',' << metrics.latency_scope
            << ",api_requests_stream_synchronized,final_residency_not_event_"
               "admissions,"
               "cpu_lock_wait_and_gpu_overlap_unmeasured,"
            << (metrics_failed(metrics) ? "FAIL" : "PASS") << std::endl;
}

void complete_run(const Config& cfg, Metrics& metrics, int run) {
  metrics.run = run + 1;
  if (run >= 0 || metrics_failed(metrics)) print_metrics(cfg, metrics);
  require(!metrics_failed(metrics),
          "Benchmark completion/oracle gate failed: " + metrics.scenario);
}

void fingerprint(uint64_t& hash, uint64_t value) {
  hash ^= value;
  hash *= 1099511628211ULL;
}

std::vector<K> continuous_keys(size_t count, K start) {
  std::vector<K> keys(count);
  std::iota(keys.begin(), keys.end(), start);
  return keys;
}

std::vector<K> constrained_keys(size_t count, size_t capacity,
                                size_t hot_buckets, K start) {
  std::vector<K> keys;
  for (K candidate = start; keys.size() < count; ++candidate) {
    require(candidate < std::numeric_limits<K>::max() - 1024,
            "Key generation overflow");
    if (throughput_bucket(candidate, capacity) < hot_buckets)
      keys.push_back(candidate);
  }
  return keys;
}

std::vector<S> static_scores(const std::vector<K>& keys) {
  std::vector<S> scores;
  scores.reserve(keys.size());
  for (K key : keys) scores.push_back(key + 1000);
  return scores;
}

void run_static_insert(const Config& cfg, const std::string& scenario,
                       const std::vector<K>& keys, size_t hot_buckets) {
  const auto scores = static_scores(keys);
  const auto expected = expected_top(keys, scores, cfg.capacity);
  for (int run = -cfg.warmup; run < cfg.runs; ++run) {
    Table table;
    create_table(table, cfg.capacity);
    Metrics metrics;
    metrics.scenario = scenario;
    metrics.operation = "insert_or_assign";
    metrics.roles = "I";
    metrics.hot_buckets = hot_buckets;
    metrics.planned = keys.size();
    metrics.request_batch = keys.size();
    metrics.device_worker_bytes =
        keys.size() * (sizeof(K) + sizeof(S) + kDim * sizeof(V) + 1);
    metrics.timing_scope =
        "api_submit_to_stream_complete_input_copies_excluded";
    metrics.latency_scope = "one_batch_no_queue_no_network";
    metrics.seconds = insert_batch(table, keys, scores, unique_keys(keys));
    metrics.completed = keys.size();
    metrics.insert_requests = keys.size();
    metrics.latency_us.push_back(metrics.seconds * 1e6);
    metrics.service_us = metrics.latency_us;
    for (K key : keys) fingerprint(metrics.trace_hash, key);
    verify_snapshot_exact(table, expected, metrics);
    metrics.final_new_keys = metrics.final_size;
    metrics.required_writer_residents = expected.size();
    metrics.writer_residents = metrics.final_size;
    complete_run(cfg, metrics, run);
  }
}

void run_static_find(const Config& cfg, const std::vector<K>& keys) {
  const size_t fill =
      std::min(cfg.capacity / 2, std::max(cfg.batch, keys.size()));
  const auto initial = continuous_keys(fill, 1);
  const auto scores = static_scores(initial);
  const auto expected = expected_top(initial, scores, cfg.capacity);
  for (int run = -cfg.warmup; run < cfg.runs; ++run) {
    Table table;
    create_table(table, cfg.capacity);
    insert_batch(table, initial, scores, true);
    Metrics metrics;
    metrics.scenario = "uniform";
    metrics.operation = "find";
    metrics.roles = "R";
    metrics.planned = keys.size();
    metrics.request_batch = keys.size();
    metrics.device_worker_bytes =
        keys.size() * (sizeof(K) + sizeof(S) + kDim * sizeof(V) + 1);
    metrics.initial_size = table.size(0);
    metrics.timing_scope =
        "host_lookup_including_alloc_copies_stream_completion";
    metrics.latency_scope = "one_batch_no_queue_no_network";
    const auto start = Clock::now();
    const auto returned = lookup(table, keys);
    metrics.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    metrics.completed = keys.size();
    metrics.read_requests = keys.size();
    metrics.latency_us.push_back(metrics.seconds * 1e6);
    metrics.service_us = metrics.latency_us;
    for (K key : initial) fingerprint(metrics.trace_hash, key);
    for (K key : keys) fingerprint(metrics.trace_hash, key);
    verify_exact(keys, returned, expected, metrics, true);
    verify_snapshot_exact(table, expected, metrics);
    complete_run(cfg, metrics, run);
  }
}

class StartGate {
 public:
  explicit StartGate(size_t target) : target_(target) {}
  bool ready_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++ready_;
    condition_.notify_all();
    condition_.wait(lock, [&] { return released_ || cancelled_; });
    return !cancelled_;
  }
  bool release() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return ready_ == target_ || cancelled_; });
    if (cancelled_) return false;
    epoch_ = Clock::now();
    released_ = true;
    condition_.notify_all();
    return true;
  }
  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    condition_.notify_all();
  }
  Clock::time_point epoch() const { return epoch_; }

 private:
  size_t target_;
  size_t ready_ = 0;
  bool released_ = false;
  bool cancelled_ = false;
  Clock::time_point epoch_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

struct Workload {
  std::vector<K> initial;
  std::vector<Batch> batches;
  std::unordered_map<K, std::vector<S>> writers;
  std::unordered_set<K> upserts;
  std::vector<size_t> distinct_per_bucket;
};

Workload make_workload(const Config& cfg, int overlap, const std::string& kind,
                       bool streaming) {
  Workload work;
  const size_t bucket_count = cfg.capacity / kBucketSize;
  const size_t per_bucket = static_cast<size_t>(cfg.load_factor * kBucketSize);
  require(per_bucket >= kLogicalLanes,
          "V3 requires at least eight initial rows per bucket");
  std::vector<std::vector<K>> resident(bucket_count);
  const size_t initial_count = per_bucket * bucket_count;
  K candidate = 1 + cfg.seed % 1048576;
  while (work.initial.size() < initial_count) {
    const K key = candidate++;
    auto& bucket = resident[throughput_bucket(key, cfg.capacity)];
    if (bucket.size() < per_bucket) {
      bucket.push_back(key);
      work.initial.push_back(key);
    }
  }
  const size_t shared_buckets =
      (cfg.request_batch + per_bucket / kLogicalLanes - 1) /
      (per_bucket / kLogicalLanes);
  const size_t private_buckets =
      (cfg.request_batch + per_bucket - 1) / per_bucket;
  if (kind == "same_bucket")
    require(bucket_count >= shared_buckets + kLogicalLanes * private_buckets,
            "V3 capacity/load_factor too small for disjoint same-bucket lanes");
  else
    require(work.initial.size() >= (kLogicalLanes + 1) * cfg.request_batch,
            "V3 capacity/load_factor too small for disjoint hot-key lanes");
  std::vector<size_t> needed(bucket_count, 0);
  for (size_t offset = 0, batch_id = 0; offset < cfg.total_requests;
       ++batch_id) {
    const size_t count =
        std::min(cfg.request_batch, cfg.total_requests - offset);
    const size_t lane = batch_id % kLogicalLanes;
    Batch batch;
    batch.role = cfg.roles[batch_id % cfg.roles.size()];
    batch.generation = batch_id + 2;
    batch.keys.resize(count);
    if (batch.role == 'R') batch.returned = Returned(count);
    const size_t shared = count * overlap / 100;
    const size_t ingress =
        streaming && batch.role == 'I' ? count * cfg.ingress_pct / 100 : 0;
    for (size_t row = 0; row < count; ++row) {
      K key;
      if (kind == "hot_key") {
        const size_t index =
            row < shared ? row : (lane + 1) * cfg.request_batch + row;
        key = work.initial[index];
      } else if (row < shared) {
        const size_t slots_per_lane = per_bucket / kLogicalLanes;
        key = resident[row / slots_per_lane]
                      [(row % slots_per_lane) * kLogicalLanes + lane];
      } else {
        const size_t bucket =
            shared_buckets + lane * private_buckets + row / per_bucket;
        key = resident[bucket][row % per_bucket];
      }
      batch.keys[row] = key;
      if (row < ingress) ++needed[throughput_bucket(key, cfg.capacity)];
    }
    offset += count;
    work.batches.push_back(std::move(batch));
  }
  std::vector<std::vector<K>> new_keys(bucket_count);
  size_t outstanding = std::accumulate(needed.begin(), needed.end(), size_t{0});
  while (outstanding) {
    require(candidate < std::numeric_limits<K>::max() - 1024,
            "Ingress key generation overflow");
    const K key = candidate++;
    const size_t bucket = throughput_bucket(key, cfg.capacity);
    if (new_keys[bucket].size() < needed[bucket]) {
      new_keys[bucket].push_back(key);
      --outstanding;
    }
  }
  work.distinct_per_bucket.assign(bucket_count, per_bucket);
  for (auto& batch : work.batches) {
    const size_t ingress = streaming && batch.role == 'I'
                               ? batch.keys.size() * cfg.ingress_pct / 100
                               : 0;
    for (size_t row = 0; row < batch.keys.size(); ++row) {
      if (row < ingress) {
        const size_t bucket = throughput_bucket(batch.keys[row], cfg.capacity);
        batch.keys[row] = new_keys[bucket].back();
        new_keys[bucket].pop_back();
        ++work.distinct_per_bucket[bucket];
      }
      if (batch.role != 'R')
        work.writers[batch.keys[row]].push_back(batch.generation);
      if (batch.role == 'I') work.upserts.insert(batch.keys[row]);
    }
    batch.unique = unique_keys(batch.keys);
  }
  return work;
}

bool allowed_generation(const Workload& work, K key, S score,
                        bool initial_allowed) {
  if (initial_allowed && score == 1) return true;
  const auto writers = work.writers.find(key);
  return writers != work.writers.end() &&
         std::find(writers->second.begin(), writers->second.end(), score) !=
             writers->second.end();
}

void verify_unique_admission(const Workload& work, const Config& cfg,
                             const std::unordered_set<K>& initial,
                             const ScoresByKey& exported, Metrics& metrics) {
  if (metrics.workers != 1 ||
      !std::all_of(work.batches.begin(), work.batches.end(),
                   [](const Batch& batch) {
                     return batch.role == 'I' && batch.completed;
                   }))
    return;
  for (const auto& entry : work.writers)
    if (initial.count(entry.first) || entry.second.size() != 1) return;

  const size_t bucket_count = cfg.capacity / kBucketSize;
  std::vector<std::vector<S>> expected(bucket_count);
  std::vector<std::vector<S>> actual(bucket_count);
  for (K key : initial)
    expected[throughput_bucket(key, cfg.capacity)].push_back(1);
  for (const auto& entry : work.writers)
    expected[throughput_bucket(entry.first, cfg.capacity)].push_back(
        entry.second.front());
  for (const auto& entry : exported)
    actual[throughput_bucket(entry.first, cfg.capacity)].push_back(
        entry.second);
  for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
    auto& scores = expected[bucket];
    std::sort(scores.begin(), scores.end(), std::greater<S>());
    scores.resize(std::min(kBucketSize, scores.size()));
    std::sort(actual[bucket].begin(), actual[bucket].end(), std::greater<S>());
    metrics.errors += actual[bucket] != scores;
  }
  for (const auto& entry : work.writers) {
    const auto& scores = expected[throughput_bucket(entry.first, cfg.capacity)];
    if (entry.second.front() > scores.back())
      metrics.errors += exported.count(entry.first) == 0;
  }
}

void verify_workload(Table& table, const Workload& work, const Config& cfg,
                     Metrics& metrics) {
  std::unordered_set<K> initial(work.initial.begin(), work.initial.end());
  std::unordered_set<K> prior_insertions = initial;
  std::vector<size_t> prior_distinct(work.distinct_per_bucket.size(), 0);
  for (K key : initial) ++prior_distinct[throughput_bucket(key, cfg.capacity)];
  for (const auto& batch : work.batches) {
    if (metrics.workers == 1 && batch.role == 'I' && batch.completed)
      for (K key : batch.keys)
        if (prior_insertions.insert(key).second)
          ++prior_distinct[throughput_bucket(key, cfg.capacity)];
    if (batch.role != 'R' || !batch.completed) continue;
    for (size_t row = 0; row < batch.keys.size(); ++row) {
      const K key = batch.keys[row];
      const bool found = batch.returned.founds[row] != 0;
      metrics.read_hits += found;
      metrics.read_misses += !found;
      if (!found) {
        const auto& distinct =
            metrics.workers == 1 ? prior_distinct : work.distinct_per_bucket;
        metrics.errors +=
            distinct[throughput_bucket(key, cfg.capacity)] <= kBucketSize;
      } else {
        const S score = batch.returned.scores[row];
        metrics.errors +=
            !allowed_generation(work, key, score, initial.count(key) != 0);
        metrics.errors += !valid_payload(
            key, score, batch.returned.values.data() + row * kDim);
        ++metrics.checked_rows;
      }
    }
  }
  const auto exported = snapshot(table);
  metrics.final_size = table.size(0);
  metrics.errors += metrics.final_size != exported.keys.size();
  std::unordered_set<K> seen;
  std::vector<size_t> resident_count(work.distinct_per_bucket.size(), 0);
  std::vector<size_t> high_count(resident_count.size(), 0);
  std::vector<size_t> writer_count(resident_count.size(), 0);
  ScoresByKey exported_scores;
  for (const auto& entry : work.writers) {
    const size_t bucket = throughput_bucket(entry.first, cfg.capacity);
    if (work.upserts.count(entry.first) ||
        work.distinct_per_bucket[bucket] <= kBucketSize)
      ++writer_count[bucket];
  }
  for (size_t row = 0; row < exported.keys.size(); ++row) {
    const K key = exported.keys[row];
    const S score = exported.scores[row];
    const bool was_initial = initial.count(key) != 0;
    const bool written = work.writers.count(key) != 0;
    const size_t bucket = throughput_bucket(key, cfg.capacity);
    ++resident_count[bucket];
    high_count[bucket] += score > 1;
    metrics.writer_residents += written;
    metrics.final_new_keys += !was_initial;
    metrics.errors += !seen.insert(key).second;
    metrics.errors += !was_initial && !written;
    metrics.errors +=
        !allowed_generation(work, key, score, was_initial && !written);
    metrics.errors +=
        !valid_payload(key, score, exported.values.data() + row * kDim);
    exported_scores[key] = score;
    ++metrics.checked_rows;
  }
  for (size_t bucket = 0; bucket < resident_count.size(); ++bucket) {
    metrics.errors += resident_count[bucket] !=
                      std::min(kBucketSize, work.distinct_per_bucket[bucket]);
    const size_t required = std::min(kBucketSize, writer_count[bucket]);
    metrics.required_writer_residents += required;
    metrics.errors += high_count[bucket] < required;
  }
  for (K key : initial)
    if (work.distinct_per_bucket[throughput_bucket(key, cfg.capacity)] <=
        kBucketSize)
      metrics.errors += seen.count(key) == 0;
  for (const auto& entry : work.writers)
    if (work.distinct_per_bucket[throughput_bucket(
            entry.first, cfg.capacity)] <= kBucketSize)
      metrics.errors += seen.count(entry.first) == 0;
  verify_unique_admission(work, cfg, initial, exported_scores, metrics);
  for (size_t offset = 0; offset < exported.keys.size(); offset += 16384) {
    const size_t end = std::min(offset + 16384, exported.keys.size());
    std::vector<K> keys(exported.keys.begin() + offset,
                        exported.keys.begin() + end);
    verify_exact(keys, lookup(table, keys), exported_scores, metrics, false);
  }
}

Metrics run_fixed_once(const Config& cfg, const std::string& scenario,
                       int workers, int overlap, const std::string& kind,
                       bool streaming) {
  Workload work = make_workload(cfg, overlap, kind, streaming);
  Table table;
  create_table(table, cfg.capacity);
  for (size_t offset = 0; offset < work.initial.size(); offset += 16384) {
    const size_t end = std::min(offset + 16384, work.initial.size());
    std::vector<K> keys(work.initial.begin() + offset,
                        work.initial.begin() + end);
    insert_batch(table, keys, std::vector<S>(keys.size(), 1), true);
  }
  Metrics metrics;
  metrics.scenario = scenario;
  metrics.operation = "fixed_work_find_assign_insert";
  metrics.roles = cfg.roles;
  metrics.overlap_kind = kind;
  metrics.overlap_pct = overlap;
  metrics.ingress_pct = streaming ? cfg.ingress_pct : 0;
  metrics.workers = workers;
  metrics.planned = cfg.total_requests;
  metrics.request_batch = cfg.request_batch;
  metrics.initial_size = table.size(0);
  metrics.errors += metrics.initial_size != work.initial.size();
  ScoresByKey initial_scores;
  for (K key : work.initial) initial_scores.emplace(key, 1);
  verify_snapshot_exact(table, initial_scores, metrics);
  require(metrics.errors == 0, "Prefill key/value/score oracle failed");
  metrics.device_worker_bytes = workers * cfg.request_batch *
                                (sizeof(K) + sizeof(S) + kDim * sizeof(V) + 1);
  metrics.timing_scope =
      "common_go_to_last_batch_completion_including_host_pack_copies_api_wait";
  metrics.latency_scope =
      "batch_ready_at_common_go_to_host_completion_includes_backlog_no_network";
  std::unordered_map<K, uint64_t> key_workers;
  std::unordered_map<size_t, uint64_t> bucket_workers;
  for (K key : work.initial) fingerprint(metrics.trace_hash, key);
  for (size_t batch_id = 0; batch_id < work.batches.size(); ++batch_id) {
    const auto& batch = work.batches[batch_id];
    fingerprint(metrics.trace_hash, batch.role);
    fingerprint(metrics.trace_hash, batch.generation);
    for (K key : batch.keys) {
      fingerprint(metrics.trace_hash, key);
      const uint64_t mask = uint64_t{1} << (batch_id % workers);
      key_workers[key] |= mask;
      bucket_workers[throughput_bucket(key, cfg.capacity)] |= mask;
    }
  }
  for (const auto& batch : work.batches)
    for (K key : batch.keys) {
      const uint64_t key_mask = key_workers.at(key);
      const uint64_t bucket_mask =
          bucket_workers.at(throughput_bucket(key, cfg.capacity));
      metrics.cross_worker_key_requests += (key_mask & (key_mask - 1)) != 0;
      metrics.cross_worker_bucket_requests +=
          (bucket_mask & (bucket_mask - 1)) != 0;
    }
  metrics.hot_buckets = bucket_workers.size();
  int device = 0;
  check_cuda(cudaGetDevice(&device));
  StartGate gate(workers);
  std::vector<std::string> failures(workers);
  std::vector<std::thread> threads;
  bool released = false;
  try {
    for (int worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
        try {
          check_cuda(cudaSetDevice(device));
          Stream stream;
          Buffers buffers(cfg.request_batch);
          std::vector<S> scores(cfg.request_batch);
          std::vector<V> values(cfg.request_batch * kDim);
          if (!gate.ready_and_wait()) return;
          for (size_t batch_id = worker; batch_id < work.batches.size();
               batch_id += workers) {
            auto& batch = work.batches[batch_id];
            const size_t count = batch.keys.size();
            const auto start = Clock::now();
            buffers.keys.upload(batch.keys.data(), count, stream);
            if (batch.role == 'R') {
              table.find(count, buffers.keys.get(), buffers.values.get(),
                         reinterpret_cast<bool*>(buffers.founds.get()),
                         buffers.scores.get(), stream);
              buffers.values.download(batch.returned.values.data(),
                                      count * kDim, stream);
              buffers.scores.download(batch.returned.scores.data(), count,
                                      stream);
              buffers.founds.download(batch.returned.founds.data(), count,
                                      stream);
            } else {
              std::fill(scores.begin(), scores.begin() + count,
                        batch.generation);
              for (size_t row = 0; row < count; ++row)
                for (size_t column = 0; column < kDim; ++column)
                  values[row * kDim + column] = payload_component(
                      batch.keys[row], batch.generation, column);
              buffers.scores.upload(scores.data(), count, stream);
              buffers.values.upload(values.data(), count * kDim, stream);
              if (batch.role == 'U')
                table.assign(count, buffers.keys.get(), buffers.values.get(),
                             buffers.scores.get(), stream, batch.unique);
              else
                table.insert_or_assign(
                    count, buffers.keys.get(), buffers.values.get(),
                    buffers.scores.get(), stream, batch.unique, false);
            }
            check_cuda(cudaStreamSynchronize(stream));
            batch.finished = Clock::now();
            batch.service_us = std::chrono::duration<double, std::micro>(
                                   batch.finished - start)
                                   .count();
            batch.latency_us = std::chrono::duration<double, std::micro>(
                                   batch.finished - gate.epoch())
                                   .count();
            batch.completed = true;
          }
        } catch (const std::exception& error) {
          failures[worker] = error.what();
          gate.cancel();
        } catch (...) {
          failures[worker] = "Unknown worker exception";
          gate.cancel();
        }
      });
    }
    released = gate.release();
  } catch (...) {
    gate.cancel();
    for (auto& thread : threads) thread.join();
    throw;
  }
  for (auto& thread : threads) thread.join();
  for (const auto& failure : failures)
    if (!failure.empty()) {
      ++metrics.worker_errors;
      std::cerr << "Worker failed: " << failure << '\n';
    }
  auto end = gate.epoch();
  for (const auto& batch : work.batches) {
    if (!batch.completed) continue;
    metrics.completed += batch.keys.size();
    metrics.read_requests += batch.role == 'R' ? batch.keys.size() : 0;
    metrics.update_requests += batch.role == 'U' ? batch.keys.size() : 0;
    metrics.insert_requests += batch.role == 'I' ? batch.keys.size() : 0;
    metrics.latency_us.push_back(batch.latency_us);
    metrics.service_us.push_back(batch.service_us);
    end = std::max(end, batch.finished);
  }
  if (released)
    metrics.seconds = std::chrono::duration<double>(end - gate.epoch()).count();
  verify_workload(table, work, cfg, metrics);
  return metrics;
}

void run_fixed(const Config& cfg, const std::string& scenario, int workers,
               int overlap, const std::string& kind, bool streaming) {
  for (int run = -cfg.warmup; run < cfg.runs; ++run) {
    auto metrics =
        run_fixed_once(cfg, scenario, workers, overlap, kind, streaming);
    complete_run(cfg, metrics, run);
  }
}

uint64_t parse_unsigned(const std::string& value) {
  require(!value.empty() && std::all_of(value.begin(), value.end(),
                                        [](char digit) {
                                          return digit >= '0' && digit <= '9';
                                        }),
          "Expected unsigned integer: " + value);
  return std::stoull(value);
}

double parse_real(const std::string& value) {
  size_t end = 0;
  const double result = std::stod(value, &end);
  require(end == value.size() && std::isfinite(result),
          "Expected finite number: " + value);
  return result;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option(argv[argument]);
    if (option == "--help" || option == "-h") {
      std::cout << "--mode=quick|full "
                   "--scenario=all|uniform|bucket_skew|single_bucket|zipf|rui|"
                   "fixed_work|streaming\n"
                   "--capacity=N --batch=N --adversarial_batch=N "
                   "--total_requests=N --request_batch=N\n"
                   "--streams=N --roles=R|U|I|RU|RI|UI|RUI "
                   "--overlap_pct=0|50|100 --overlap_kind=hot_key|same_bucket\n"
                   "--ingress_pct=0..100 --load_factor=0..1 --seed=N "
                   "--warmup=N --runs=N --alpha=A\n"
                   "Table mode is throughput only as in this original harness; "
                   "no memory-mode substitution.\n";
      std::exit(0);
    }
    if (option == "--mode=full")
      cfg.mode = "full";
    else if (option == "--mode=quick")
      cfg.mode = "quick";
  }
  if (cfg.mode == "full") {
    cfg.capacity = 16UL * 1024 * 1024;
    cfg.batch = 1024UL * 1024;
    cfg.adversarial_batch = 256UL * 1024;
    cfg.warmup = 2;
    cfg.runs = 5;
  }
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option(argv[argument]);
    const auto delimiter = option.find('=');
    require(delimiter != std::string::npos, "Expected --name=value: " + option);
    const std::string name = option.substr(0, delimiter);
    const std::string value = option.substr(delimiter + 1);
    if (name == "--mode")
      require(value == "quick" || value == "full", "Invalid mode");
    else if (name == "--scenario")
      cfg.scenario = value;
    else if (name == "--roles")
      cfg.roles = value;
    else if (name == "--overlap_kind") {
      cfg.overlap_kind = value;
      cfg.kind_set = true;
    } else if (name == "--capacity")
      cfg.capacity = parse_unsigned(value);
    else if (name == "--batch")
      cfg.batch = parse_unsigned(value);
    else if (name == "--adversarial_batch")
      cfg.adversarial_batch = parse_unsigned(value);
    else if (name == "--total_requests") {
      cfg.total_requests = parse_unsigned(value);
      require(cfg.total_requests > 0, "Empty fixed work");
    } else if (name == "--request_batch")
      cfg.request_batch = parse_unsigned(value);
    else if (name == "--seed")
      cfg.seed = parse_unsigned(value);
    else if (name == "--alpha")
      cfg.alpha = parse_real(value);
    else if (name == "--load_factor")
      cfg.load_factor = parse_real(value);
    else if (name == "--table_mode")
      require(value == "throughput",
              "This original harness has no memory-mode contract");
    else {
      const uint64_t number = parse_unsigned(value);
      require(number <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
              "Integer option overflow");
      if (name == "--streams") {
        cfg.streams = number;
        cfg.streams_set = true;
      } else if (name == "--warmup")
        cfg.warmup = number;
      else if (name == "--runs")
        cfg.runs = number;
      else if (name == "--overlap_pct") {
        cfg.overlap_pct = number;
        cfg.overlap_set = true;
      } else if (name == "--ingress_pct")
        cfg.ingress_pct = number;
      else
        throw std::runtime_error("Unknown option: " + name);
    }
  }
  const std::unordered_set<std::string> scenarios = {
      "all",  "uniform", "bucket_skew", "single_bucket",
      "zipf", "rui",     "fixed_work",  "streaming"};
  require(scenarios.count(cfg.scenario), "Unknown scenario");
  require(cfg.capacity >= 256 && (cfg.capacity & (cfg.capacity - 1)) == 0 &&
              cfg.capacity <=
                  std::numeric_limits<size_t>::max() / (kDim * sizeof(V)),
          "Invalid power-of-two capacity");
  require(
      cfg.batch > 0 && cfg.adversarial_batch > 0 && cfg.request_batch > 0 &&
          cfg.batch <= std::numeric_limits<size_t>::max() / (kDim * sizeof(V)),
      "Invalid batch size");
  require(cfg.streams > 0 && cfg.streams <= 64 && cfg.runs > 0,
          "Invalid streams/runs");
  require(cfg.alpha > 0 && cfg.load_factor > 0 && cfg.load_factor <= 1,
          "Invalid alpha/load_factor");
  require(
      cfg.overlap_pct == 0 || cfg.overlap_pct == 50 || cfg.overlap_pct == 100,
      "Invalid overlap percentage");
  require(cfg.ingress_pct >= 0 && cfg.ingress_pct <= 100,
          "Invalid ingress percentage");
  require(cfg.overlap_kind == "hot_key" || cfg.overlap_kind == "same_bucket",
          "Invalid overlap kind");
  require(!cfg.roles.empty() &&
              cfg.roles.find_first_not_of("RUI") == std::string::npos,
          "Invalid role sequence");
  if (!cfg.total_requests) cfg.total_requests = cfg.batch;
  if (cfg.scenario == "all" || cfg.scenario == "rui" ||
      cfg.scenario == "fixed_work" || cfg.scenario == "streaming")
    require(cfg.total_requests >= cfg.request_batch &&
                cfg.total_requests <=
                    std::numeric_limits<size_t>::max() / (kDim * sizeof(V)),
            "Invalid fixed work size");
  return cfg;
}

bool want(const Config& cfg, const std::string& scenario) {
  return cfg.scenario == "all" || cfg.scenario == scenario;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    check_cuda(cudaFree(nullptr));
    print_header();
    if (want(cfg, "uniform")) {
      const auto keys = continuous_keys(cfg.batch, 1);
      run_static_insert(cfg, "uniform", keys, cfg.capacity / kBucketSize);
      run_static_find(cfg, keys);
    }
    if (want(cfg, "bucket_skew")) {
      for (size_t divisor : {4, 16, 64, 256}) {
        const size_t hot =
            std::max<size_t>(1, cfg.capacity / kBucketSize / divisor);
        const auto keys =
            constrained_keys(cfg.batch, cfg.capacity, hot,
                             10000000 + divisor * 1000 + cfg.seed % 1000);
        run_static_insert(cfg, "bucket_skew_1over" + std::to_string(divisor),
                          keys, hot);
      }
    }
    if (want(cfg, "single_bucket")) {
      const auto keys =
          constrained_keys(std::min(cfg.adversarial_batch, kBucketSize * 4),
                           cfg.capacity, 1, 20000000 + cfg.seed % 1000);
      run_static_insert(cfg, "single_bucket_pressure", keys, 1);
    }
    if (want(cfg, "zipf")) {
      const size_t hot = std::max<size_t>(1, cfg.batch / 16);
      std::vector<double> weights(hot);
      for (size_t index = 0; index < hot; ++index)
        weights[index] = 1.0 / std::pow(index + 1, cfg.alpha);
      std::discrete_distribution<size_t> distribution(weights.begin(),
                                                      weights.end());
      std::mt19937_64 random(cfg.seed);
      std::vector<K> keys(cfg.batch);
      for (auto& key : keys) key = 30000000 + distribution(random);
      std::unordered_set<size_t> active_buckets;
      for (K key : keys)
        active_buckets.insert(throughput_bucket(key, cfg.capacity));
      run_static_insert(cfg, "zipf_hotset", keys, active_buckets.size());
    }
    if (want(cfg, "rui")) {
      run_fixed(cfg, "rui_serial", 1, cfg.overlap_pct, cfg.overlap_kind, true);
      run_fixed(cfg, "rui_overlap", cfg.streams, cfg.overlap_pct,
                cfg.overlap_kind, true);
    }
    for (const std::string scenario : {"fixed_work", "streaming"}) {
      if (!want(cfg, scenario)) continue;
      const std::vector<int> workers = cfg.streams_set
                                           ? std::vector<int>{cfg.streams}
                                           : std::vector<int>{1, 2, 4, 8};
      const std::vector<int> overlaps = cfg.overlap_set
                                            ? std::vector<int>{cfg.overlap_pct}
                                            : std::vector<int>{0, 50, 100};
      const std::vector<std::string> kinds =
          cfg.kind_set ? std::vector<std::string>{cfg.overlap_kind}
                       : std::vector<std::string>{"hot_key", "same_bucket"};
      for (int count : workers)
        for (int overlap : overlaps)
          for (const auto& kind : kinds)
            run_fixed(cfg, scenario, count, overlap, kind,
                      scenario == "streaming");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Adversarial benchmark failed: " << error.what() << '\n';
    return 2;
  }
}
