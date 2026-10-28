#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using HKVTable = nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kLru,
                                       nv::merlin::Sm80>;

constexpr size_t kBucketSize = 128;
constexpr V kMissingValue = -1.0f;
constexpr S kMissingScore = std::numeric_limits<S>::max();

struct Config {
  std::string name;
  size_t dim;
  size_t capacity;
};

struct Options {
  std::string config = "all";
  std::string source_id;
  size_t capacity = 0;
  size_t batch_size = 1024 * 1024;
  size_t runs = 5;
  size_t warmup = 3;
  uint64_t seed = 20261019;
  double load_factor = 0;
  double hit_rate = 0;
  bool return_scores = false;
  bool self_test = false;
  bool help = false;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

uint64_t parse_unsigned(const std::string& text) {
  require(!text.empty() &&
              text.find_first_not_of("0123456789") == std::string::npos,
          "Expected an unsigned decimal integer: " + text);
  size_t consumed = 0;
  const auto result = std::stoull(text, &consumed);
  require(consumed == text.size(), "Invalid integer: " + text);
  return result;
}

double parse_fraction(const std::string& text, bool allow_zero) {
  if (allow_zero && (text == "0" || text == "0.0")) return 0;
  if (text == "0.5" || text == "0.50") return 0.5;
  if (text == "1" || text == "1.0" || text == "1.00") return 1;
  throw std::runtime_error("Expected 0.5 or 1, plus 0 for hit-rate");
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      options.help = true;
      continue;
    }
    if (argument == "--cpu-self-test") {
      options.self_test = true;
      continue;
    }
    require(index + 1 < argc, "Missing value for " + argument);
    const std::string value = argv[++index];
    if (argument == "--config")
      options.config = value;
    else if (argument == "--source-id")
      options.source_id = value;
    else if (argument == "--capacity")
      options.capacity = parse_unsigned(value);
    else if (argument == "--batch-size")
      options.batch_size = parse_unsigned(value);
    else if (argument == "--runs")
      options.runs = parse_unsigned(value);
    else if (argument == "--warmup")
      options.warmup = parse_unsigned(value);
    else if (argument == "--seed")
      options.seed = parse_unsigned(value);
    else if (argument == "--load-factor")
      options.load_factor = parse_fraction(value, false);
    else if (argument == "--hit-rate")
      options.hit_rate = parse_fraction(value, true);
    else if (argument == "--return-scores") {
      require(value == "0" || value == "1", "return-scores must be 0 or 1");
      options.return_scores = value == "1";
    } else
      throw std::runtime_error("Unknown option: " + argument);
  }
  require(options.config == "all" || options.config == "A" ||
              options.config == "B" || options.config == "C",
          "Config selection is empty or unsupported");
  require(options.batch_size > 0 && options.batch_size <= (1UL << 26),
          "batch-size must be in [1, 2^26]");
  require(options.runs > 0 && options.runs <= 1000 && options.warmup <= 1000,
          "runs must be in [1,1000], warmup in [0,1000]");
  require(
      options.capacity == 0 ||
          (options.capacity >= kBucketSize && options.capacity <= (1UL << 30) &&
           (options.capacity & (options.capacity - 1)) == 0),
      "capacity must be a power of two in [128,2^30]");
  require(options.seed <= std::numeric_limits<uint64_t>::max() - options.runs,
          "Seed sequence overflows");
  if (!options.help && !options.self_test) {
    require(options.source_id.size() == 64 &&
                options.source_id.find_first_not_of("0123456789abcdef") ==
                    std::string::npos,
            "--source-id requires the SHA256 of the frozen source manifest");
  }
  return options;
}

V expected_value(K key, size_t component) {
  return static_cast<V>(((key >> (16 * (component % 4))) & 65535) +
                        (component / 4) * 65536);
}

uint64_t fingerprint(const std::vector<K>& keys) {
  uint64_t result = 14695981039346656037ULL;
  for (K key : keys) {
    for (size_t byte = 0; byte < sizeof(K); ++byte) {
      result = (result ^ ((key >> (byte * 8)) & 255)) * 1099511628211ULL;
    }
  }
  return result;
}

std::vector<K> make_queries(const std::vector<K>& residents, size_t count,
                            double hit_rate, uint64_t seed) {
  require(!residents.empty() && count > 0, "Empty query or resident set");
  std::mt19937_64 random(seed);
  std::vector<K> queries(count);
  const size_t hits = static_cast<size_t>(count * hit_rate);
  for (size_t index = 0; index < count; ++index) {
    queries[index] = index < hits ? residents[random() % residents.size()]
                                  : 2 * (index + (seed & 0xffffffffULL)) + 1;
  }
  for (size_t remaining = count; remaining > 1; --remaining) {
    std::swap(queries[remaining - 1], queries[random() % remaining]);
  }
  return queries;
}

void validate_result(const std::vector<K>& queries, size_t dim,
                     const std::unordered_map<K, S>& resident_scores,
                     const std::vector<unsigned char>& found,
                     const std::vector<V>& values, const std::vector<S>& scores,
                     bool check_scores) {
  require(!queries.empty() && found.size() == queries.size() &&
              values.size() == queries.size() * dim &&
              scores.size() == queries.size(),
          "Incomplete or empty lookup output");
  for (size_t index = 0; index < queries.size(); ++index) {
    const auto resident = resident_scores.find(queries[index]);
    const bool expected_found = resident != resident_scores.end();
    require(found[index] == static_cast<unsigned char>(expected_found),
            "Found mismatch at query " + std::to_string(index));
    if (expected_found && check_scores) {
      require(scores[index] == resident->second,
              "Score mismatch at query " + std::to_string(index));
    }
    for (size_t component = 0; component < dim; ++component) {
      const V expected = expected_found
                             ? expected_value(queries[index], component)
                             : kMissingValue;
      require(values[index * dim + component] == expected,
              "Payload/miss-preservation mismatch at query " +
                  std::to_string(index));
    }
  }
}

template <typename Element>
struct DeviceBuffer {
  Element* data = nullptr;
  explicit DeviceBuffer(size_t count) {
    CUDA_CHECK(cudaMalloc(&data, count * sizeof(Element)));
  }
  ~DeviceBuffer() {
    if (data) cudaFree(data);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreate(&value)); }
  ~Stream() {
    if (value) cudaStreamDestroy(value);
  }
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() {
    if (value) cudaEventDestroy(value);
  }
};

struct Measurement {
  double host_seconds;
  float stream_ms;
  size_t residents;
  size_t hits;
  uint64_t population_sum;
  uint64_t query_hash;
};

Measurement measure_find(const Config& config, const Options& options,
                         double load_factor, uint64_t seed) {
  Stream stream;
  nv::merlin::HashTableOptions table_options;
  table_options.init_capacity = config.capacity;
  table_options.max_capacity = config.capacity;
  table_options.max_bucket_size = kBucketSize;
  table_options.dim = config.dim;
  table_options.max_hbm_for_vectors = config.capacity * config.dim * sizeof(V);
  table_options.table_mode = nv::merlin::TableMode::kThroughput;
  table_options.api_lock = true;
  HKVTable table;
  table.init(table_options);
  require(table.capacity() == config.capacity,
          "Actual capacity differs from requested capacity");

  const size_t batch = options.batch_size;
  const size_t target = static_cast<size_t>(config.capacity * load_factor);
  const size_t per_bucket = static_cast<size_t>(kBucketSize * load_factor);
  std::vector<size_t> bucket_counts(config.capacity / kBucketSize, 0);
  std::vector<K> samples;
  std::vector<K> keys;
  std::vector<V> values(batch * config.dim);
  DeviceBuffer<K> device_keys(batch);
  DeviceBuffer<V> device_values(batch * config.dim);
  DeviceBuffer<S> device_scores(batch);
  DeviceBuffer<bool> device_found(batch);
  size_t inserted = 0;
  size_t attempts = 0;
  K ordinal = seed & 0xffffffffULL;
  uint64_t population_sum = 0;
  uint64_t population_xor = 0;
  while (inserted < target) {
    keys.clear();
    while (keys.size() < batch && inserted + keys.size() < target) {
      require(++attempts <= config.capacity * 64,
              "Balanced population attempt limit exceeded");
      const K key = 2 * (++ordinal);
      const size_t bucket =
          (nv::merlin::Murmur3HashHost(key) % config.capacity) / kBucketSize;
      if (bucket_counts[bucket] == per_bucket) continue;
      ++bucket_counts[bucket];
      keys.push_back(key);
      if (samples.size() < batch) samples.push_back(key);
      population_sum += key;
      population_xor ^= key;
    }
    require(!keys.empty(), "Population made no progress");
    for (size_t index = 0; index < keys.size(); ++index) {
      for (size_t component = 0; component < config.dim; ++component) {
        values[index * config.dim + component] =
            expected_value(keys[index], component);
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(device_keys.data, keys.data(),
                               keys.size() * sizeof(K), cudaMemcpyHostToDevice,
                               stream.value));
    CUDA_CHECK(cudaMemcpyAsync(device_values.data, values.data(),
                               keys.size() * config.dim * sizeof(V),
                               cudaMemcpyHostToDevice, stream.value));
    table.insert_or_assign(keys.size(), device_keys.data, device_values.data,
                           nullptr, stream.value);
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    inserted += keys.size();
  }
  require(table.size(stream.value) == target,
          "Population lost or added entries");

  std::unordered_map<K, S> resident_scores;
  std::unordered_map<K, bool> sampled;
  for (K key : samples) sampled.emplace(key, true);
  keys.resize(batch);
  std::vector<S> scores(batch);
  size_t exported = 0;
  uint64_t exported_sum = 0;
  uint64_t exported_xor = 0;
  for (size_t offset = 0; offset < config.capacity; offset += batch) {
    const size_t count =
        table.export_batch(batch, offset, device_keys.data, device_values.data,
                           device_scores.data, stream.value);
    require(count <= batch, "Export count exceeds buffer");
    CUDA_CHECK(cudaMemcpyAsync(keys.data(), device_keys.data, count * sizeof(K),
                               cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(values.data(), device_values.data,
                               count * config.dim * sizeof(V),
                               cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(scores.data(), device_scores.data,
                               count * sizeof(S), cudaMemcpyDeviceToHost,
                               stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    for (size_t index = 0; index < count; ++index) {
      require(keys[index] % 2 == 0, "Export contains a non-population key");
      exported_sum += keys[index];
      exported_xor ^= keys[index];
      for (size_t component = 0; component < config.dim; ++component) {
        require(values[index * config.dim + component] ==
                    expected_value(keys[index], component),
                "Exported payload mismatch");
      }
      if (sampled.count(keys[index])) {
        require(resident_scores.emplace(keys[index], scores[index]).second,
                "Duplicate sampled key in export");
      }
    }
    exported += count;
  }
  require(exported == target && exported_sum == population_sum &&
              exported_xor == population_xor &&
              resident_scores.size() == samples.size(),
          "Export/population identity mismatch");

  std::vector<unsigned char> found(batch);
  auto prepare = [&](const std::vector<K>& queries) {
    std::fill(values.begin(), values.end(), kMissingValue);
    std::fill(scores.begin(), scores.end(), kMissingScore);
    CUDA_CHECK(cudaMemcpyAsync(device_keys.data, queries.data(),
                               queries.size() * sizeof(K),
                               cudaMemcpyHostToDevice, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(device_values.data, values.data(),
                               queries.size() * config.dim * sizeof(V),
                               cudaMemcpyHostToDevice, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(device_scores.data, scores.data(),
                               queries.size() * sizeof(S),
                               cudaMemcpyHostToDevice, stream.value));
    CUDA_CHECK(cudaMemsetAsync(device_found.data, 0xff,
                               queries.size() * sizeof(bool), stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto validate = [&](const std::vector<K>& queries, bool check_scores) {
    const size_t count = queries.size();
    found.resize(count);
    values.resize(count * config.dim);
    scores.resize(count);
    CUDA_CHECK(cudaMemcpyAsync(found.data(), device_found.data,
                               count * sizeof(bool), cudaMemcpyDeviceToHost,
                               stream.value));
    CUDA_CHECK(cudaMemcpyAsync(values.data(), device_values.data,
                               count * config.dim * sizeof(V),
                               cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(scores.data(), device_scores.data,
                               count * sizeof(S), cudaMemcpyDeviceToHost,
                               stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    validate_result(queries, config.dim, resident_scores, found, values, scores,
                    check_scores);
  };
  auto control = [&](double hit_rate, bool return_scores) {
    const auto queries = make_queries(samples, batch, hit_rate, seed);
    prepare(queries);
    table.find(batch, device_keys.data, device_values.data, device_found.data,
               return_scores ? device_scores.data : nullptr, stream.value);
    validate(queries, return_scores);
  };
  auto controls = [&]() {
    control(1, options.return_scores);
    control(0, options.return_scores);
    if (!options.return_scores) {
      control(1, true);
      control(0, true);
    }
  };
  controls();
  const auto queries = make_queries(samples, batch, options.hit_rate, seed);
  for (size_t warmup = 0; warmup < options.warmup; ++warmup) {
    prepare(queries);
    table.find(batch, device_keys.data, device_values.data, device_found.data,
               options.return_scores ? device_scores.data : nullptr,
               stream.value);
    validate(queries, options.return_scores);
  }
  prepare(queries);
  Event start;
  Event stop;
  const auto host_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start.value, stream.value));
  table.find(batch, device_keys.data, device_values.data, device_found.data,
             options.return_scores ? device_scores.data : nullptr,
             stream.value);
  CUDA_CHECK(cudaEventRecord(stop.value, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  const double host_seconds = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - host_start)
                                  .count();
  float stream_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&stream_ms, start.value, stop.value));
  require(std::isfinite(host_seconds) && host_seconds > 0 &&
              std::isfinite(stream_ms) && stream_ms > 0,
          "Invalid completed timing interval");
  validate(queries, options.return_scores);
  const size_t hits =
      std::count(found.begin(), found.end(), static_cast<unsigned char>(1));
  require(hits == static_cast<size_t>(batch * options.hit_rate),
          "Completed hit count mismatch");
  controls();
  return {host_seconds, stream_ms,      exported,
          hits,         population_sum, fingerprint(queries)};
}

void cpu_self_test() {
  const std::vector<K> residents = {2, 4, 6, 8};
  const std::unordered_map<K, S> scores_by_key = {
      {2, 11}, {4, 12}, {6, 13}, {8, 14}};
  require(make_queries(residents, 32, 0.5, 7) ==
              make_queries(residents, 32, 0.5, 7),
          "Nondeterministic queries");
  for (double hit_rate : {0.0, 0.5, 1.0}) {
    const auto queries = make_queries(residents, 32, hit_rate, 7);
    std::vector<unsigned char> found(queries.size());
    std::vector<V> values(queries.size() * 8, kMissingValue);
    std::vector<S> scores(queries.size(), kMissingScore);
    for (size_t index = 0; index < queries.size(); ++index) {
      if (scores_by_key.count(queries[index])) {
        found[index] = 1;
        scores[index] = scores_by_key.at(queries[index]);
        for (size_t component = 0; component < 8; ++component)
          values[index * 8 + component] =
              expected_value(queries[index], component);
      }
    }
    validate_result(queries, 8, scores_by_key, found, values, scores, true);
    for (size_t mutation = 0; mutation < 4; ++mutation) {
      auto bad_found = found;
      auto bad_values = values;
      auto bad_scores = scores;
      if (mutation == 0) bad_found[0] ^= 1;
      if (mutation == 1) bad_values[0] = std::numeric_limits<V>::quiet_NaN();
      if (mutation == 2) bad_found.clear();
      if (mutation == 3 && hit_rate == 0) continue;
      if (mutation == 3) {
        const size_t index =
            std::find(found.begin(), found.end(), 1) - found.begin();
        bad_scores[index] ^= 1;
      }
      bool rejected = false;
      try {
        validate_result(queries, 8, scores_by_key, bad_found, bad_values,
                        bad_scores, true);
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected, "Negative output fixture falsely passed");
    }
  }
  std::cout << "CPU_SELF_TEST_PASS positive=3 negative=11 host_only=1\n";
}

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.help) {
      std::cout
          << "Digest find control: throughput/LRU, balanced real-key "
             "population.\n"
             "--source-id SHA256 --config all|A|B|C --capacity N --batch-size "
             "N\n"
             "--load-factor 0.5|1 --hit-rate 0|0.5|1 --return-scores 0|1\n"
             "--runs N --warmup N --seed N --cpu-self-test\n";
      return 0;
    }
    if (options.self_test) {
      cpu_self_test();
      return 0;
    }
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int runtime = 0;
    int driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::cerr << "GPU=" << properties.name << " CUDA_runtime=" << runtime
              << " driver_api=" << driver << '\n';
#ifdef DISABLE_DIGEST
    const char* mode = "digest_constant";
#else
    const char* mode = "with_digest";
#endif
    std::vector<Config> configs = {{"A", 8, 128UL * 1024 * 1024},
                                   {"B", 32, 128UL * 1024 * 1024},
                                   {"C", 64, 64UL * 1024 * 1024}};
    const std::vector<double> load_factors =
        options.load_factor == 0 ? std::vector<double>{0.5, 1.0}
                                 : std::vector<double>{options.load_factor};
    size_t requested = 0;
    size_t completed = 0;
    for (const auto& config : configs)
      if (options.config == "all" || options.config == config.name)
        requested += load_factors.size() * options.runs;
    require(requested > 0, "No benchmark cases selected");
    std::cout << "mode,config,dim,load_factor,run,throughput_bkvs,source_id,"
                 "workload,policy,table_mode,capacity,hbm_vector_bytes,batch_"
                 "size,seed,hit_rate,return_scores,residents,completed_queries,"
                 "hits,host_seconds,stream_interval_ms,population_key_sum,"
                 "query_fnv64,hash_variant,oracle_pass\n";
    for (auto config : configs) {
      if (options.config != "all" && options.config != config.name) continue;
      if (options.capacity) config.capacity = options.capacity;
      for (double load_factor : load_factors) {
        for (size_t run = 0; run < options.runs; ++run) {
          const uint64_t seed = options.seed + run;
          const auto result = measure_find(config, options, load_factor, seed);
          std::cout << std::setprecision(17) << mode << ',' << config.name
                    << ',' << config.dim << ',' << load_factor << ',' << run + 1
                    << ',' << options.batch_size / result.host_seconds / 1e9
                    << ',' << options.source_id
                    << ",balanced_even_keys_v1,lru,throughput,"
                    << config.capacity << ','
                    << config.capacity * config.dim * sizeof(V) << ','
                    << options.batch_size << ',' << seed << ','
                    << options.hit_rate << ',' << options.return_scores << ','
                    << result.residents << ',' << options.batch_size << ','
                    << result.hits << ',' << result.host_seconds << ','
                    << result.stream_ms << ',' << result.population_sum << ','
                    << result.query_hash << ',' << HKV_HASH_VARIANT << ",1\n";
          ++completed;
        }
      }
    }
    require(completed == requested, "Incomplete benchmark selection");
    std::cerr << "completed_cases=" << completed
              << " requested_cases=" << requested << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "digest_ablation_failure: " << error.what() << '\n';
    return 1;
  }
}
