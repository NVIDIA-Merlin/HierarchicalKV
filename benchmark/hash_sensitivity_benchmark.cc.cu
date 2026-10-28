/*
 * Hash sensitivity microbenchmark for standalone comparison.
 *
 * This benchmark intentionally does not change HKV's core hash path.  It
 * replays HKV's bucket-index and digest derivation rules in a standalone
 * benchmark-local table so standard avalanche finalizers can be compared
 * without recompiling the library kernels.
 *
 * Output CSV:
 *   hash,distribution,table_mode,load_factor,occupancy_mean,
 *   occupancy_variance,occupancy_cv,occupancy_p99,occupancy_max,
 *   digest_fp_per_query,operation,throughput_bkvs,resident_found_rate,
 *   miss_false_found_rate,...
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class HashKind {
  kMurmur3,
  kSplitmix64,
  kXxhashAvalanche,
  kWyhashFinal,
  kIdentity,
};

enum class Distribution {
  kSequential,
  kUniform64,
  kStridedLowbits,
};

enum class TableMode {
  kThroughput,
  kMemory,
};

struct Options {
  std::string mode = "quick";
  std::vector<HashKind> hashes;
  std::vector<Distribution> distributions;
  std::vector<TableMode> table_modes;
  std::vector<double> load_factors;
  size_t capacity_slots = 128 * 1024;
  size_t bucket_size = 128;
  size_t query_count = 32 * 1024;
};

struct Slot {
  uint64_t key = 0;
  uint8_t digest = 0;
  bool occupied = false;
};

struct OccupancyStats {
  double mean = 0.0;
  double variance = 0.0;
  double cv = 0.0;
  double p99 = 0.0;
  uint32_t max = 0;
};

struct ProbeStats {
  double digest_fp_per_query = 0.0;
  double digest_fp_per_slot = 0.0;
  double resident_found_rate = 0.0;
  double miss_false_found_rate = 0.0;
  double throughput_bkvs = 0.0;
};

struct InsertResult {
  size_t attempted = 0;
  size_t inserted = 0;
  size_t dropped = 0;
};

static inline uint64_t murmur3_fmix64(uint64_t key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

static inline uint64_t splitmix64_final(uint64_t key) {
  uint64_t z = key + UINT64_C(0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static inline uint64_t xxhash64_avalanche(uint64_t key) {
  uint64_t h = key;
  h ^= h >> 33;
  h *= UINT64_C(0xc2b2ae3d27d4eb4f);
  h ^= h >> 29;
  h *= UINT64_C(0x165667b19e3779f9);
  h ^= h >> 32;
  return h;
}

static inline uint64_t wyhash_final(uint64_t key) {
  uint64_t h = key;
  h ^= h >> 32;
  h *= UINT64_C(0xd6e8feb86659fd93);
  h ^= h >> 32;
  h *= UINT64_C(0xd6e8feb86659fd93);
  h ^= h >> 32;
  return h;
}

static inline uint64_t apply_hash(HashKind hash, uint64_t key) {
  switch (hash) {
    case HashKind::kMurmur3:
      return murmur3_fmix64(key);
    case HashKind::kSplitmix64:
      return splitmix64_final(key);
    case HashKind::kXxhashAvalanche:
      return xxhash64_avalanche(key);
    case HashKind::kWyhashFinal:
      return wyhash_final(key);
    case HashKind::kIdentity:
      return key;
  }
  return key;
}

static const char* to_string(HashKind hash) {
  switch (hash) {
    case HashKind::kMurmur3:
      return "murmur3";
    case HashKind::kSplitmix64:
      return "splitmix64";
    case HashKind::kXxhashAvalanche:
      return "xxhash_avalanche";
    case HashKind::kWyhashFinal:
      return "wyhash_final";
    case HashKind::kIdentity:
      return "identity";
  }
  return "unknown";
}

static const char* to_string(Distribution distribution) {
  switch (distribution) {
    case Distribution::kSequential:
      return "sequential";
    case Distribution::kUniform64:
      return "uniform64";
    case Distribution::kStridedLowbits:
      return "strided_lowbits";
  }
  return "unknown";
}

static const char* to_string(TableMode mode) {
  switch (mode) {
    case TableMode::kThroughput:
      return "kThroughput";
    case TableMode::kMemory:
      return "kMemory";
  }
  return "unknown";
}

static HashKind parse_hash(const std::string& value) {
  if (value == "murmur3" || value == "murmur3_fmix64") {
    return HashKind::kMurmur3;
  }
  if (value == "splitmix64") return HashKind::kSplitmix64;
  if (value == "xxhash_avalanche" || value == "xxhash-style" ||
      value == "xxhash") {
    return HashKind::kXxhashAvalanche;
  }
  if (value == "wyhash_final" || value == "wyhash-style" || value == "wyhash") {
    return HashKind::kWyhashFinal;
  }
  if (value == "identity") return HashKind::kIdentity;
  throw std::invalid_argument("unknown hash: " + value);
}

static Distribution parse_distribution(const std::string& value) {
  if (value == "sequential") return Distribution::kSequential;
  if (value == "uniform64") return Distribution::kUniform64;
  if (value == "strided_lowbits" || value == "strided") {
    return Distribution::kStridedLowbits;
  }
  throw std::invalid_argument("unknown distribution: " + value);
}

static TableMode parse_table_mode(const std::string& value) {
  if (value == "kThroughput" || value == "throughput") {
    return TableMode::kThroughput;
  }
  if (value == "kMemory" || value == "memory" || value == "dual") {
    return TableMode::kMemory;
  }
  throw std::invalid_argument("unknown table mode: " + value);
}

template <typename T, typename ParseFn>
static std::vector<T> parse_csv(const std::string& value, ParseFn parse_one) {
  std::vector<T> result;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) result.push_back(parse_one(token));
  }
  return result;
}

static std::vector<double> parse_load_factors(const std::string& value) {
  return parse_csv<double>(
      value, [](const std::string& token) { return std::stod(token); });
}

static uint64_t make_key(Distribution distribution, size_t index,
                         bool miss_key) {
  const uint64_t i = static_cast<uint64_t>(index + 1);
  switch (distribution) {
    case Distribution::kSequential:
      return miss_key ? (UINT64_C(1) << 62) + i : i;
    case Distribution::kUniform64:
      return splitmix64_final((miss_key ? UINT64_C(0xa51ce55eed5eed00)
                                        : UINT64_C(0xc0ffee1234567890)) +
                              i);
    case Distribution::kStridedLowbits:
      return (miss_key ? (UINT64_C(1) << 36) + i : i) << 16;
  }
  return i;
}

static uint8_t digest_from_hash(uint64_t hash, TableMode mode) {
  if (mode == TableMode::kMemory) return static_cast<uint8_t>(hash >> 56);
  return static_cast<uint8_t>(hash >> 32);
}

class MicroTable {
 public:
  MicroTable(HashKind hash, TableMode mode, size_t capacity_slots,
             size_t bucket_size)
      : hash_(hash),
        mode_(mode),
        capacity_slots_(capacity_slots),
        bucket_size_(bucket_size),
        bucket_count_(capacity_slots / bucket_size),
        slots_(bucket_count_ * bucket_size),
        occupancy_(bucket_count_, 0) {
    if (bucket_size_ == 0) {
      throw std::invalid_argument("bucket size must be non-zero");
    }
    if (bucket_count_ == 0) {
      throw std::invalid_argument("capacity must contain at least one bucket");
    }
  }

  bool insert(uint64_t key) {
    uint64_t hashed = apply_hash(hash_, key);
    uint8_t digest = digest_from_hash(hashed, mode_);
    size_t bucket = choose_insert_bucket(hashed);
    if (bucket == npos()) return false;
    size_t offset = bucket * bucket_size_ + occupancy_[bucket];
    slots_[offset].key = key;
    slots_[offset].digest = digest;
    slots_[offset].occupied = true;
    occupancy_[bucket]++;
    inserted_keys_.push_back(key);
    return true;
  }

  bool contains(uint64_t key) const {
    uint64_t hashed = apply_hash(hash_, key);
    uint8_t digest = digest_from_hash(hashed, mode_);
    size_t b1 = first_bucket(hashed);
    if (bucket_contains(b1, digest, key)) return true;
    if (mode_ == TableMode::kMemory) {
      size_t b2 = second_bucket(hashed, b1);
      if (bucket_contains(b2, digest, key)) return true;
    }
    return false;
  }

  size_t count_digest_matches(uint64_t key, size_t* checked_slots) const {
    uint64_t hashed = apply_hash(hash_, key);
    uint8_t digest = digest_from_hash(hashed, mode_);
    size_t matches = 0;
    size_t checked = 0;
    size_t b1 = first_bucket(hashed);
    matches += bucket_digest_matches(b1, digest, key, &checked);
    if (mode_ == TableMode::kMemory) {
      size_t b2 = second_bucket(hashed, b1);
      matches += bucket_digest_matches(b2, digest, key, &checked);
    }
    if (checked_slots != nullptr) *checked_slots = checked;
    return matches;
  }

  OccupancyStats occupancy_stats() const {
    OccupancyStats stats;
    if (occupancy_.empty()) return stats;

    std::vector<uint32_t> sorted = occupancy_;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (uint32_t count : occupancy_) sum += static_cast<double>(count);
    stats.mean = sum / static_cast<double>(occupancy_.size());

    double sq = 0.0;
    for (uint32_t count : occupancy_) {
      double delta = static_cast<double>(count) - stats.mean;
      sq += delta * delta;
    }
    stats.variance = sq / static_cast<double>(occupancy_.size());
    stats.cv = stats.mean == 0.0 ? 0.0 : std::sqrt(stats.variance) / stats.mean;

    size_t p99_index = static_cast<size_t>(std::ceil(0.99 * sorted.size())) - 1;
    p99_index = std::min(p99_index, sorted.size() - 1);
    stats.p99 = static_cast<double>(sorted[p99_index]);
    stats.max = sorted.back();
    return stats;
  }

  const std::vector<uint64_t>& inserted_keys() const { return inserted_keys_; }

 private:
  static constexpr size_t npos() { return std::numeric_limits<size_t>::max(); }

  size_t first_bucket(uint64_t hashed) const {
    if (mode_ == TableMode::kMemory) {
      return static_cast<size_t>(hashed & UINT64_C(0xffffffff)) % bucket_count_;
    }
    size_t global_slot = static_cast<size_t>(hashed % capacity_slots_);
    return global_slot / bucket_size_;
  }

  size_t second_bucket(uint64_t hashed, size_t b1) const {
    size_t b2 = static_cast<size_t>((hashed >> 32) & UINT64_C(0xffffffff)) %
                bucket_count_;
    if (b2 == b1) b2 = (b2 + 1) % bucket_count_;
    return b2;
  }

  size_t choose_insert_bucket(uint64_t hashed) const {
    size_t b1 = first_bucket(hashed);
    if (mode_ == TableMode::kThroughput) {
      return occupancy_[b1] < bucket_size_ ? b1 : npos();
    }

    size_t b2 = second_bucket(hashed, b1);
    bool b1_has_room = occupancy_[b1] < bucket_size_;
    bool b2_has_room = occupancy_[b2] < bucket_size_;
    if (!b1_has_room && !b2_has_room) return npos();
    if (!b2_has_room) return b1;
    if (!b1_has_room) return b2;
    return occupancy_[b1] <= occupancy_[b2] ? b1 : b2;
  }

  bool bucket_contains(size_t bucket, uint8_t digest, uint64_t key) const {
    size_t base = bucket * bucket_size_;
    for (uint32_t i = 0; i < occupancy_[bucket]; i++) {
      const Slot& slot = slots_[base + i];
      if (slot.occupied && slot.digest == digest && slot.key == key) {
        return true;
      }
    }
    return false;
  }

  size_t bucket_digest_matches(size_t bucket, uint8_t digest, uint64_t key,
                               size_t* checked_slots) const {
    size_t matches = 0;
    size_t base = bucket * bucket_size_;
    for (uint32_t i = 0; i < occupancy_[bucket]; i++) {
      const Slot& slot = slots_[base + i];
      if (!slot.occupied) continue;
      if (checked_slots != nullptr) (*checked_slots)++;
      if (slot.digest == digest && slot.key != key) matches++;
    }
    return matches;
  }

  HashKind hash_;
  TableMode mode_;
  size_t capacity_slots_;
  size_t bucket_size_;
  size_t bucket_count_;
  std::vector<Slot> slots_;
  std::vector<uint32_t> occupancy_;
  std::vector<uint64_t> inserted_keys_;
};

static InsertResult populate_table(MicroTable* table, Distribution distribution,
                                   size_t requested) {
  InsertResult result;
  result.attempted = requested;
  for (size_t i = 0; i < requested; i++) {
    if (table->insert(make_key(distribution, i, false))) {
      result.inserted++;
    } else {
      result.dropped++;
    }
  }
  return result;
}

static ProbeStats run_probe(const MicroTable& table, Distribution distribution,
                            size_t query_count) {
  ProbeStats stats;
  const auto& resident_keys = table.inserted_keys();
  size_t resident_queries = std::min(query_count, resident_keys.size());
  size_t miss_queries = query_count;
  if (resident_queries == 0 && miss_queries == 0) return stats;

  size_t resident_found = 0;
  size_t miss_false_found = 0;
  size_t digest_matches = 0;
  size_t checked_slots = 0;

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < resident_queries; i++) {
    if (table.contains(resident_keys[i])) resident_found++;
  }
  for (size_t i = 0; i < miss_queries; i++) {
    uint64_t key = make_key(distribution, i, true);
    if (table.contains(key)) miss_false_found++;
    digest_matches += table.count_digest_matches(key, &checked_slots);
  }
  auto end = std::chrono::steady_clock::now();

  double seconds = std::chrono::duration<double>(end - start).count();
  double total_queries = static_cast<double>(resident_queries + miss_queries);
  stats.throughput_bkvs =
      seconds == 0.0 ? 0.0
                     : total_queries / seconds / (1024.0 * 1024.0 * 1024.0);
  stats.resident_found_rate = resident_queries == 0
                                  ? 0.0
                                  : static_cast<double>(resident_found) /
                                        static_cast<double>(resident_queries);
  stats.miss_false_found_rate = miss_queries == 0
                                    ? 0.0
                                    : static_cast<double>(miss_false_found) /
                                          static_cast<double>(miss_queries);
  stats.digest_fp_per_query = miss_queries == 0
                                  ? 0.0
                                  : static_cast<double>(digest_matches) /
                                        static_cast<double>(miss_queries);
  stats.digest_fp_per_slot = checked_slots == 0
                                 ? 0.0
                                 : static_cast<double>(digest_matches) /
                                       static_cast<double>(checked_slots);
  return stats;
}

static void print_csv_header() {
  std::cout << "hash,distribution,table_mode,load_factor,"
            << "occupancy_mean,occupancy_variance,occupancy_cv,"
            << "occupancy_p99,occupancy_max,digest_fp_per_query,"
            << "operation,throughput_bkvs,resident_found_rate,"
            << "miss_false_found_rate,resident_count,capacity_slots,"
            << "inserted_load_factor,dropped_insert_rate,digest_fp_per_slot"
            << std::endl;
}

static void print_result(HashKind hash, Distribution distribution,
                         TableMode table_mode, double load_factor,
                         const OccupancyStats& occupancy,
                         const ProbeStats& probe, const InsertResult& insert,
                         size_t capacity_slots) {
  double inserted_load_factor = capacity_slots == 0
                                    ? 0.0
                                    : static_cast<double>(insert.inserted) /
                                          static_cast<double>(capacity_slots);
  double dropped_insert_rate = insert.attempted == 0
                                   ? 0.0
                                   : static_cast<double>(insert.dropped) /
                                         static_cast<double>(insert.attempted);

  std::cout << to_string(hash) << "," << to_string(distribution) << ","
            << to_string(table_mode) << "," << std::fixed
            << std::setprecision(4) << load_factor << ","
            << std::setprecision(6) << occupancy.mean << ","
            << occupancy.variance << "," << occupancy.cv << "," << occupancy.p99
            << "," << occupancy.max << "," << probe.digest_fp_per_query
            << ",micro_lookup," << probe.throughput_bkvs << ","
            << probe.resident_found_rate << "," << probe.miss_false_found_rate
            << "," << insert.inserted << "," << capacity_slots << ","
            << inserted_load_factor << "," << dropped_insert_rate << ","
            << probe.digest_fp_per_slot << std::endl;
}

static void set_mode_defaults(Options* options) {
  if (options->mode == "quick") {
    options->capacity_slots = 128 * 1024;
    options->query_count = 32 * 1024;
    options->load_factors = {0.50, 0.90};
    options->table_modes = {TableMode::kThroughput, TableMode::kMemory};
  } else if (options->mode == "full") {
    options->capacity_slots = 4 * 1024 * 1024;
    options->query_count = 1024 * 1024;
    options->load_factors = {0.25, 0.50, 0.75, 0.90, 0.95, 1.00};
    options->table_modes = {TableMode::kThroughput, TableMode::kMemory};
  } else {
    throw std::invalid_argument("unknown mode: " + options->mode);
  }
}

static void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --mode quick|full              Preset sizes and LF grid.\n"
      << "  --hash all|csv                 Hashes: murmur3,splitmix64,\n"
      << "                                 "
         "xxhash_avalanche,wyhash_final,identity.\n"
      << "  --distribution all|csv         Distributions: "
         "sequential,uniform64,\n"
      << "                                 strided_lowbits.\n"
      << "  --table-mode all|csv           Modes: kThroughput,kMemory.\n"
      << "  --load-factors csv             Override LF grid.\n"
      << "  --capacity-slots n             Override logical slot capacity.\n"
      << "  --bucket-size n                Override bucket size, default 128.\n"
      << "  --query-count n                Override resident/miss query "
         "count.\n";
}

static Options parse_args(int argc, char** argv) {
  Options options;
  options.hashes = {HashKind::kMurmur3, HashKind::kSplitmix64,
                    HashKind::kXxhashAvalanche, HashKind::kWyhashFinal,
                    HashKind::kIdentity};
  options.distributions = {Distribution::kSequential, Distribution::kUniform64,
                           Distribution::kStridedLowbits};
  set_mode_defaults(&options);

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--mode") {
      options.mode = require_value("--mode");
      set_mode_defaults(&options);
    } else if (arg == "--hash") {
      std::string value = require_value("--hash");
      if (value == "all") {
        options.hashes = {HashKind::kMurmur3, HashKind::kSplitmix64,
                          HashKind::kXxhashAvalanche, HashKind::kWyhashFinal,
                          HashKind::kIdentity};
      } else {
        options.hashes = parse_csv<HashKind>(value, parse_hash);
      }
    } else if (arg == "--distribution") {
      std::string value = require_value("--distribution");
      if (value == "all") {
        options.distributions = {Distribution::kSequential,
                                 Distribution::kUniform64,
                                 Distribution::kStridedLowbits};
      } else {
        options.distributions =
            parse_csv<Distribution>(value, parse_distribution);
      }
    } else if (arg == "--table-mode") {
      std::string value = require_value("--table-mode");
      if (value == "all") {
        options.table_modes = {TableMode::kThroughput, TableMode::kMemory};
      } else {
        options.table_modes = parse_csv<TableMode>(value, parse_table_mode);
      }
    } else if (arg == "--load-factors") {
      options.load_factors =
          parse_load_factors(require_value("--load-factors"));
    } else if (arg == "--capacity-slots") {
      options.capacity_slots =
          static_cast<size_t>(std::stoull(require_value("--capacity-slots")));
    } else if (arg == "--bucket-size") {
      options.bucket_size =
          static_cast<size_t>(std::stoull(require_value("--bucket-size")));
    } else if (arg == "--query-count") {
      options.query_count =
          static_cast<size_t>(std::stoull(require_value("--query-count")));
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (options.capacity_slots % options.bucket_size != 0) {
    throw std::invalid_argument(
        "capacity-slots must be divisible by bucket-size");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    print_csv_header();

    for (HashKind hash : options.hashes) {
      for (Distribution distribution : options.distributions) {
        for (TableMode table_mode : options.table_modes) {
          for (double load_factor : options.load_factors) {
            size_t requested = static_cast<size_t>(std::llround(
                load_factor * static_cast<double>(options.capacity_slots)));
            MicroTable table(hash, table_mode, options.capacity_slots,
                             options.bucket_size);
            InsertResult insert =
                populate_table(&table, distribution, requested);
            OccupancyStats occupancy = table.occupancy_stats();
            ProbeStats probe =
                run_probe(table, distribution, options.query_count);
            print_result(hash, distribution, table_mode, load_factor, occupancy,
                         probe, insert, options.capacity_slots);
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "hash_sensitivity_benchmark: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
