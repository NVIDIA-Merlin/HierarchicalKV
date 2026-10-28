/*
 * Synthetic device-only TBE-like set-associative simulation.
 *
 * This benchmark models the FBGEMM/TBE cache shape without depending on the
 * embedding-bag API:
 *   - 32-way set-associative cache by default.
 *   - 128B value rows by default (32 float values).
 *   - Batched lookup/snapshot, miss compaction, and protected replacement.
 *   - LRU and LFU policies.
 *
 * Output: CSV to stdout; progress/configuration to stderr.
 */

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda/atomic>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"

using K = uint64_t;
using Meta = unsigned long long;

enum class Policy { kLru, kLfu };

struct Config {
  std::string mode = "quick";
  std::string policy = "lru";
  std::vector<double> load_factors{0.25, 0.50, 0.75, 1.00};
  size_t capacity_slots = 1UL << 20;
  size_t batch_size = 1UL << 16;
  int associativity = 32;
  int value_bytes = 128;
  int warmup = 1;
  int runs = 2;
  double miss_rate = 0.25;
  uint64_t seed = 0x5eed1234ULL;
  bool self_test = false;
};

struct DeviceCache {
  K* keys = nullptr;
  int* valid = nullptr;
  float* values = nullptr;
  Meta* age = nullptr;
  Meta* freq = nullptr;
  int* locks = nullptr;

  void allocate(size_t capacity_slots, size_t value_dim, size_t num_sets) {
    CUDA_CHECK(cudaMalloc(&keys, capacity_slots * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&valid, capacity_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&values, capacity_slots * value_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&age, capacity_slots * sizeof(Meta)));
    CUDA_CHECK(cudaMalloc(&freq, capacity_slots * sizeof(Meta)));
    CUDA_CHECK(cudaMalloc(&locks, num_sets * sizeof(int)));
  }

  void free() {
    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(valid));
    CUDA_CHECK(cudaFree(values));
    CUDA_CHECK(cudaFree(age));
    CUDA_CHECK(cudaFree(freq));
    CUDA_CHECK(cudaFree(locks));
  }
};

struct RequestBuffers {
  K* keys = nullptr;
  float* incoming_values = nullptr;
  float* output_values = nullptr;
  float* writeback_values = nullptr;
  int* hit_flags = nullptr;
  int* slot_for_request = nullptr;
  int* miss_indices = nullptr;
  int* victim_slots = nullptr;
  int* evicted_valid = nullptr;
  K* evicted_keys = nullptr;
  K* returned_keys = nullptr;
  unsigned long long* hit_count = nullptr;
  unsigned long long* miss_count = nullptr;
  unsigned long long* replacement_count = nullptr;
  unsigned long long* writeback_bytes = nullptr;
  unsigned long long* global_clock = nullptr;
  unsigned long long* completed_count = nullptr;
  Meta* tickets = nullptr;

  void allocate(size_t batch_size, size_t value_dim) {
    CUDA_CHECK(cudaMalloc(&keys, batch_size * sizeof(K)));
    CUDA_CHECK(
        cudaMalloc(&incoming_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&output_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&writeback_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(writeback_values, 0,
                               batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hit_flags, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&slot_for_request, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&miss_indices, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&victim_slots, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&evicted_valid, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&evicted_keys, batch_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&returned_keys, batch_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&hit_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&miss_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&replacement_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&writeback_bytes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&global_clock, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&completed_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&tickets, batch_size * sizeof(Meta)));
  }

  void free() {
    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(incoming_values));
    CUDA_CHECK(cudaFree(output_values));
    CUDA_CHECK(cudaFree(writeback_values));
    CUDA_CHECK(cudaFree(hit_flags));
    CUDA_CHECK(cudaFree(slot_for_request));
    CUDA_CHECK(cudaFree(miss_indices));
    CUDA_CHECK(cudaFree(victim_slots));
    CUDA_CHECK(cudaFree(evicted_valid));
    CUDA_CHECK(cudaFree(evicted_keys));
    CUDA_CHECK(cudaFree(returned_keys));
    CUDA_CHECK(cudaFree(hit_count));
    CUDA_CHECK(cudaFree(miss_count));
    CUDA_CHECK(cudaFree(replacement_count));
    CUDA_CHECK(cudaFree(writeback_bytes));
    CUDA_CHECK(cudaFree(global_clock));
    CUDA_CHECK(cudaFree(completed_count));
    CUDA_CHECK(cudaFree(tickets));
  }
};

struct Counters {
  unsigned long long hits = 0;
  unsigned long long misses = 0;
  unsigned long long replacements = 0;
  unsigned long long writeback_bytes = 0;
  unsigned long long completed = 0;
  size_t final_occupancy = 0;
};

struct StageTimes {
  float lookup_ms = 0.0f;
  float miss_ms = 0.0f;
  float service_ms = 0.0f;
  float gpu_ms = 0.0f;
  double wall_ms = 0.0;
};

static size_t div_up(size_t n, size_t d) { return (n + d - 1) / d; }

static int grid_for(size_t n, int block) {
  size_t grid = div_up(n, static_cast<size_t>(block));
  return static_cast<int>(std::min<size_t>(grid, 65535));
}

static std::vector<double> parse_lf_list(const std::string& input) {
  std::vector<double> values;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::stod(item));
  }
  return values;
}

static std::string policy_name(Policy policy) {
  return policy == Policy::kLru ? "lru" : "lfu";
}

static void apply_mode_defaults(Config* cfg) {
  if (cfg->mode == "quick") {
    cfg->capacity_slots = 1UL << 20;
    cfg->batch_size = 1UL << 16;
    cfg->warmup = 1;
    cfg->runs = 2;
  } else if (cfg->mode == "full") {
    cfg->capacity_slots = 128UL * 1024 * 1024;
    cfg->batch_size = 1024UL * 1024;
    cfg->warmup = 3;
    cfg->runs = 5;
  } else {
    std::cerr << "Unknown mode: " << cfg->mode << std::endl;
    std::exit(1);
  }
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  bool capacity_overridden = false;
  bool batch_overridden = false;
  bool warmup_overridden = false;
  bool runs_overridden = false;

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(1);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--mode") {
      cfg.mode = require_value("--mode");
    } else if (arg.rfind("--mode=", 0) == 0) {
      cfg.mode = arg.substr(7);
    } else if (arg == "--quick") {
      cfg.mode = "quick";
    } else if (arg == "--full") {
      cfg.mode = "full";
    } else if (arg == "--policy") {
      cfg.policy = require_value("--policy");
    } else if (arg.rfind("--policy=", 0) == 0) {
      cfg.policy = arg.substr(9);
    } else if (arg == "--lf") {
      cfg.load_factors = parse_lf_list(require_value("--lf"));
    } else if (arg.rfind("--lf=", 0) == 0) {
      cfg.load_factors = parse_lf_list(arg.substr(5));
    } else if (arg == "--capacity") {
      cfg.capacity_slots = std::stoull(require_value("--capacity"));
      capacity_overridden = true;
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity_slots = std::stoull(arg.substr(11));
      capacity_overridden = true;
    } else if (arg == "--batch") {
      cfg.batch_size = std::stoull(require_value("--batch"));
      batch_overridden = true;
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch_size = std::stoull(arg.substr(8));
      batch_overridden = true;
    } else if (arg == "--warmup") {
      cfg.warmup = std::stoi(require_value("--warmup"));
      warmup_overridden = true;
    } else if (arg.rfind("--warmup=", 0) == 0) {
      cfg.warmup = std::stoi(arg.substr(9));
      warmup_overridden = true;
    } else if (arg == "--runs") {
      cfg.runs = std::stoi(require_value("--runs"));
      runs_overridden = true;
    } else if (arg.rfind("--runs=", 0) == 0) {
      cfg.runs = std::stoi(arg.substr(7));
      runs_overridden = true;
    } else if (arg == "--assoc") {
      cfg.associativity = std::stoi(require_value("--assoc"));
    } else if (arg.rfind("--assoc=", 0) == 0) {
      cfg.associativity = std::stoi(arg.substr(8));
    } else if (arg == "--value-bytes") {
      cfg.value_bytes = std::stoi(require_value("--value-bytes"));
    } else if (arg.rfind("--value-bytes=", 0) == 0) {
      cfg.value_bytes = std::stoi(arg.substr(14));
    } else if (arg == "--miss-rate") {
      cfg.miss_rate = std::stod(require_value("--miss-rate"));
    } else if (arg.rfind("--miss-rate=", 0) == 0) {
      cfg.miss_rate = std::stod(arg.substr(12));
    } else if (arg == "--seed") {
      cfg.seed = std::stoull(require_value("--seed"));
    } else if (arg.rfind("--seed=", 0) == 0) {
      cfg.seed = std::stoull(arg.substr(7));
    } else if (arg == "--self-test") {
      cfg.self_test = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: tbe_like_cache_baseline [options]\n"
          << "  --mode quick|full          Default: quick\n"
          << "  --policy lru|lfu|all       Default: lru\n"
          << "  --lf 0.25,0.50,0.75,1.00  Load factors to sweep\n"
          << "  --capacity N               Override cache slots\n"
          << "  --batch N                  Override request batch size\n"
          << "  --assoc N                  Default: 32\n"
          << "  --value-bytes N            Default: 128\n"
          << "  --miss-rate P              Default: 0.25\n"
          << "  --warmup N --runs N --seed N\n"
          << "  --self-test                Both-policy adversarial gate\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(1);
    }
  }

  Config mode_defaults;
  mode_defaults.mode = cfg.mode;
  apply_mode_defaults(&mode_defaults);
  if (!capacity_overridden) cfg.capacity_slots = mode_defaults.capacity_slots;
  if (!batch_overridden) cfg.batch_size = mode_defaults.batch_size;
  if (!warmup_overridden) cfg.warmup = mode_defaults.warmup;
  if (!runs_overridden) cfg.runs = mode_defaults.runs;

  if (cfg.associativity <= 0 || cfg.associativity > 128) {
    std::cerr << "Associativity must be in [1, 128]" << std::endl;
    std::exit(1);
  }
  if (cfg.value_bytes <= 0 ||
      cfg.value_bytes % static_cast<int>(sizeof(float))) {
    std::cerr << "value-bytes must be a positive multiple of sizeof(float)"
              << std::endl;
    std::exit(1);
  }
  if (cfg.capacity_slots < static_cast<size_t>(cfg.associativity) ||
      cfg.capacity_slots % static_cast<size_t>(cfg.associativity) != 0) {
    std::cerr << "capacity must be a multiple of associativity" << std::endl;
    std::exit(1);
  }
  if (cfg.batch_size == 0 || cfg.batch_size > INT_MAX ||
      cfg.capacity_slots > INT_MAX || cfg.warmup < 0 || cfg.runs <= 0 ||
      cfg.warmup > INT_MAX - cfg.runs) {
    throw std::runtime_error("Invalid batch/capacity/warmup/runs");
  }
  if (!std::isfinite(cfg.miss_rate) || cfg.miss_rate < 0.0 ||
      cfg.miss_rate > 1.0) {
    std::cerr << "miss-rate must be in [0, 1]" << std::endl;
    std::exit(1);
  }
  if (cfg.load_factors.empty()) {
    std::cerr << "At least one load factor is required" << std::endl;
    std::exit(1);
  }
  for (double lf : cfg.load_factors) {
    if (!std::isfinite(lf) || lf <= 0.0 || lf > 1.0) {
      std::cerr << "Load factor must be in (0, 1]: " << lf << std::endl;
      std::exit(1);
    }
  }
  if (cfg.policy != "lru" && cfg.policy != "lfu" && cfg.policy != "all") {
    std::cerr << "policy must be lru, lfu, or all" << std::endl;
    std::exit(1);
  }
  return cfg;
}

__host__ __device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__host__ __device__ __forceinline__ float deterministic_value(K key, int dim) {
  return static_cast<float>(
      dim < 4 ? ((key >> (16 * dim)) & 0xffffULL)
              : (mix64(key ^ static_cast<K>(dim)) & 0xffffffULL));
}

__global__ void init_cache_kernel(K* keys, int* valid, float* values, Meta* age,
                                  Meta* freq, int* locks, size_t capacity_slots,
                                  size_t num_sets, int associativity,
                                  int resident_ways, int value_dim) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t slot = tid; slot < capacity_slots; slot += stride) {
    size_t set = slot / associativity;
    int way = static_cast<int>(slot % associativity);
    int is_valid = way < resident_ways ? 1 : 0;
    K key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    keys[slot] = key;
    valid[slot] = is_valid;
    age[slot] = is_valid ? static_cast<Meta>(way + 1) : 0;
    freq[slot] = is_valid ? 1 : 0;
  }

  size_t total_values = capacity_slots * static_cast<size_t>(value_dim);
  for (size_t idx = tid; idx < total_values; idx += stride) {
    size_t slot = idx / value_dim;
    int dim = static_cast<int>(idx % value_dim);
    size_t set = slot / associativity;
    int way = static_cast<int>(slot % associativity);
    K key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    values[idx] = deterministic_value(key, dim);
  }

  for (size_t set = tid; set < num_sets; set += stride) {
    locks[set] = 0;
  }
}

__global__ void prepare_requests_kernel(K* request_keys, float* incoming_values,
                                        size_t batch_size, size_t num_sets,
                                        int resident_ways, int value_dim,
                                        double miss_rate, uint64_t seed,
                                        uint64_t miss_key_base) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    uint64_t r = mix64(seed ^ i);
    double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
    K key;
    if (u < miss_rate) {
      key = static_cast<K>(miss_key_base + i);
    } else {
      size_t set = mix64(seed + i * 17ULL) % num_sets;
      int way = static_cast<int>(mix64(seed + i * 31ULL) %
                                 static_cast<uint64_t>(resident_ways));
      key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    }
    request_keys[i] = key;
    for (int dim = 0; dim < value_dim; dim++) {
      incoming_values[i * static_cast<size_t>(value_dim) + dim] =
          deterministic_value(key, dim);
    }
  }
}

__global__ void lookup_stage_kernel(
    const K* request_keys, K* cache_keys, const int* valid, Meta* age,
    Meta* freq, int* hit_flags, int* slot_for_request,
    unsigned long long* hit_count, unsigned long long* global_clock,
    size_t batch_size, size_t num_sets, int associativity, int policy,
    const float* cache_values, float* output_values, int value_dim,
    Meta* completed, Meta* tickets, K* returned_keys) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    K key = request_keys[i];
    size_t set = key % num_sets;
    size_t base = set * static_cast<size_t>(associativity);
    int found_slot = -1;

    for (int way = 0; way < associativity; way++) {
      size_t slot = base + static_cast<size_t>(way);
      if (valid[slot] && cache_keys[slot] == key) {
        found_slot = static_cast<int>(slot);
        break;
      }
    }

    hit_flags[i] = found_slot >= 0 ? 1 : 0;
    slot_for_request[i] = found_slot;
    if (found_slot >= 0) {
      atomicAdd(hit_count, 1ULL);
      Meta ticket = atomicAdd(global_clock, 1ULL) + 1ULL;
      tickets[i] = ticket;
      returned_keys[i] = cache_keys[found_slot];
      atomicMax(&age[found_slot], ticket);
      if (policy != 0) {
        atomicAdd(&freq[found_slot], 1ULL);
      }
      for (int dim = 0; dim < value_dim; ++dim) {
        output_values[i * value_dim + dim] =
            cache_values[static_cast<size_t>(found_slot) * value_dim + dim];
      }
      atomicAdd(completed, 1ULL);
    }
  }
}

__global__ void miss_compact_stage_kernel(const int* hit_flags,
                                          int* miss_indices,
                                          unsigned long long* miss_count,
                                          size_t batch_size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    if (!hit_flags[i]) {
      unsigned long long pos = atomicAdd(miss_count, 1ULL);
      miss_indices[pos] = static_cast<int>(i);
    }
  }
}

__device__ __forceinline__ void lock_set(int* locks, size_t set) {
  cuda::atomic_ref<int, cuda::thread_scope_device> lock(locks[set]);
  int expected = 0;
  while (!lock.compare_exchange_weak(expected, 1, cuda::memory_order_acquire,
                                     cuda::memory_order_relaxed)) {
    expected = 0;
  }
}

__device__ __forceinline__ void unlock_set(int* locks, size_t set) {
  cuda::atomic_ref<int, cuda::thread_scope_device> lock(locks[set]);
  lock.store(0, cuda::memory_order_release);
}

__global__ void admission_stage_kernel(
    const K* request_keys, K* cache_keys, int* valid, Meta* age, Meta* freq,
    int* locks, const int* miss_indices, int* victim_slots, int* evicted_valid,
    K* evicted_keys, int* slot_for_request,
    const unsigned long long* miss_count, unsigned long long* replacement_count,
    unsigned long long* writeback_bytes, unsigned long long* global_clock,
    size_t num_sets, int associativity, int policy, int value_dim,
    const float* incoming_values, float* cache_values, float* output_values,
    float* writeback_values, Meta* completed, Meta* tickets, K* returned_keys) {
  const int lane = threadIdx.x % warpSize;
  size_t tid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  size_t stride = blockDim.x * gridDim.x / warpSize;
  unsigned long long total_misses = *miss_count;

  for (unsigned long long pos = tid; pos < total_misses; pos += stride) {
    int request_idx = miss_indices[pos];
    K key = request_keys[request_idx];
    size_t set = key % num_sets;
    size_t base = set * static_cast<size_t>(associativity);

    int existing_slot = -1;
    int victim = -1;
    int was_valid = 0;
    if (lane == 0) {
      lock_set(locks, set);
      for (int way = 0; way < associativity; way++) {
        size_t slot = base + static_cast<size_t>(way);
        if (valid[slot] && cache_keys[slot] == key) {
          existing_slot = static_cast<int>(slot);
          break;
        }
      }

      if (existing_slot >= 0) {
        victim_slots[pos] = -1;
        evicted_valid[pos] = 0;
        evicted_keys[pos] = 0;
        victim = existing_slot;
      } else {
        for (int way = 0; way < associativity; way++) {
          size_t slot = base + static_cast<size_t>(way);
          if (!valid[slot]) {
            victim = static_cast<int>(slot);
            break;
          }
        }

        if (victim < 0) {
          victim = static_cast<int>(base);
          for (int way = 1; way < associativity; way++) {
            size_t slot = base + static_cast<size_t>(way);
            if (policy == 0) {
              if (age[slot] < age[victim]) victim = static_cast<int>(slot);
            } else {
              if (freq[slot] < freq[victim] ||
                  (freq[slot] == freq[victim] && age[slot] < age[victim])) {
                victim = static_cast<int>(slot);
              }
            }
          }
        }

        victim_slots[pos] = victim;
        was_valid = valid[victim];
        evicted_valid[pos] = was_valid;
        evicted_keys[pos] = cache_keys[victim];
      }
    }
    victim = __shfl_sync(0xffffffff, victim, 0);
    existing_slot = __shfl_sync(0xffffffff, existing_slot, 0);
    was_valid = __shfl_sync(0xffffffff, was_valid, 0);
    __syncwarp();
    for (int dim = lane; dim < value_dim; dim += warpSize) {
      size_t cache_idx = static_cast<size_t>(victim) * value_dim + dim;
      size_t request_idx_value =
          static_cast<size_t>(request_idx) * value_dim + dim;
      if (existing_slot < 0) {
        if (was_valid)
          writeback_values[pos * value_dim + dim] = cache_values[cache_idx];
        cache_values[cache_idx] = incoming_values[request_idx_value];
      }
      output_values[request_idx_value] = cache_values[cache_idx];
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      Meta ticket = atomicAdd(global_clock, 1ULL) + 1ULL;
      tickets[request_idx] = ticket;
      age[victim] = ticket;
      if (existing_slot < 0) {
        cache_keys[victim] = key;
        valid[victim] = 1;
        freq[victim] = 1;
        atomicAdd(replacement_count, 1ULL);
        if (was_valid)
          atomicAdd(writeback_bytes,
                    static_cast<Meta>(value_dim) * sizeof(float));
      } else if (policy != 0) {
        ++freq[victim];
      }
      slot_for_request[request_idx] = victim;
      returned_keys[request_idx] = cache_keys[victim];
      atomicAdd(completed, 1ULL);
      unlock_set(locks, set);
    }
    __syncwarp();
  }
}

static float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

static StageTimes run_once(const Config& cfg, Policy policy, DeviceCache* cache,
                           RequestBuffers* req, size_t num_sets,
                           int resident_ways, int value_dim, int run_index,
                           const std::vector<K>* explicit_requests = nullptr) {
  const int block = 256;
  const int request_grid = grid_for(cfg.batch_size, block);
  uint64_t seed = cfg.seed + static_cast<uint64_t>(run_index) * 101ULL;
  uint64_t miss_key_base = static_cast<uint64_t>(cfg.capacity_slots) *
                               static_cast<uint64_t>(run_index + 2ULL) +
                           1ULL;
  for (Meta* counter : {req->hit_count, req->miss_count, req->replacement_count,
                        req->writeback_bytes, req->completed_count}) {
    CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(Meta)));
  }
  CUDA_CHECK(cudaMemsetAsync(req->output_values, 0xff,
                             cfg.batch_size * cfg.value_bytes));
  if (explicit_requests) {
    std::vector<float> incoming(cfg.batch_size * value_dim);
    for (size_t row = 0; row < cfg.batch_size; ++row)
      for (int dim = 0; dim < value_dim; ++dim)
        incoming[row * value_dim + dim] =
            deterministic_value((*explicit_requests)[row], dim);
    CUDA_CHECK(cudaMemcpy(req->keys, explicit_requests->data(),
                          cfg.batch_size * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(req->incoming_values, incoming.data(),
                          incoming.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  } else {
    prepare_requests_kernel<<<request_grid, block>>>(
        req->keys, req->incoming_values, cfg.batch_size, num_sets,
        resident_ways, value_dim, cfg.miss_rate, seed, miss_key_base);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaEvent_t events[4];
  for (auto& event : events) CUDA_CHECK(cudaEventCreate(&event));
  CUDA_CHECK(cudaStreamSynchronize(0));
  const auto wall_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(events[0]));
  lookup_stage_kernel<<<request_grid, block>>>(
      req->keys, cache->keys, cache->valid, cache->age, cache->freq,
      req->hit_flags, req->slot_for_request, req->hit_count, req->global_clock,
      cfg.batch_size, num_sets, cfg.associativity,
      policy == Policy::kLru ? 0 : 1, cache->values, req->output_values,
      value_dim, req->completed_count, req->tickets, req->returned_keys);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(events[1]));
  miss_compact_stage_kernel<<<request_grid, block>>>(
      req->hit_flags, req->miss_indices, req->miss_count, cfg.batch_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(events[2]));
  admission_stage_kernel<<<grid_for(cfg.batch_size, block / 32), block>>>(
      req->keys, cache->keys, cache->valid, cache->age, cache->freq,
      cache->locks, req->miss_indices, req->victim_slots, req->evicted_valid,
      req->evicted_keys, req->slot_for_request, req->miss_count,
      req->replacement_count, req->writeback_bytes, req->global_clock, num_sets,
      cfg.associativity, policy == Policy::kLru ? 0 : 1, value_dim,
      req->incoming_values, cache->values, req->output_values,
      req->writeback_values, req->completed_count, req->tickets,
      req->returned_keys);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(events[3]));
  CUDA_CHECK(cudaStreamSynchronize(0));
  const auto wall_stop = std::chrono::steady_clock::now();
  StageTimes times;
  times.wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
  times.lookup_ms = elapsed(events[0], events[1]);
  times.miss_ms = elapsed(events[1], events[2]);
  times.service_ms = elapsed(events[2], events[3]);
  times.gpu_ms = elapsed(events[0], events[3]);
  for (auto event : events) CUDA_CHECK(cudaEventDestroy(event));
  return times;
}

template <typename Value>
static std::vector<Value> read_device(const Value* pointer, size_t count) {
  std::vector<Value> values(count);
  if (count)
    CUDA_CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(Value),
                          cudaMemcpyDeviceToHost));
  return values;
}

static void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("oracle: " + message);
}

static void check_payload(K key, const float* values, int dim) {
  for (int component = 0; component < dim; ++component)
    require(values[component] == deterministic_value(key, component),
            "key/payload mismatch key=" + std::to_string(key) +
                " component=" + std::to_string(component));
}

static size_t verify_batch(const Config& cfg, const DeviceCache& cache,
                           const RequestBuffers& req, const Counters& counts,
                           size_t previous_occupancy) {
  require(counts.completed == cfg.batch_size, "incomplete service");
  require(counts.hits + counts.misses == counts.completed, "hit/miss counts");
  require(counts.replacements <= counts.misses, "admission count");
  const int dim = cfg.value_bytes / sizeof(float);
  const size_t chunk_rows = 2048;
  for (size_t begin = 0; begin < cfg.batch_size; begin += chunk_rows) {
    size_t count = std::min(chunk_rows, cfg.batch_size - begin);
    auto keys = read_device(req.keys + begin, count);
    auto returned = read_device(req.returned_keys + begin, count);
    require(keys == returned, "returned key mismatch");
    auto values = read_device(req.output_values + begin * dim, count * dim);
    for (size_t row = 0; row < count; ++row)
      check_payload(keys[row], &values[row * dim], dim);
  }
  size_t evictions = 0;
  size_t admissions = 0;
  for (size_t begin = 0; begin < counts.misses; begin += chunk_rows) {
    size_t count =
        std::min(chunk_rows, static_cast<size_t>(counts.misses) - begin);
    auto valid = read_device(req.evicted_valid + begin, count);
    auto keys = read_device(req.evicted_keys + begin, count);
    auto victims = read_device(req.victim_slots + begin, count);
    auto values = read_device(req.writeback_values + begin * dim, count * dim);
    for (size_t row = 0; row < count; ++row) {
      require(valid[row] == 0 || valid[row] == 1, "eviction validity");
      if (victims[row] >= 0) ++admissions;
      if (valid[row]) {
        require(victims[row] >= 0, "eviction without admission");
        ++evictions;
        check_payload(keys[row], &values[row * dim], dim);
      }
    }
  }
  require(admissions == counts.replacements, "admission records");
  require(evictions * cfg.value_bytes == counts.writeback_bytes,
          "writeback bytes");
  size_t occupancy = 0;
  const size_t cache_chunk = std::max<size_t>(
      cfg.associativity, chunk_rows / cfg.associativity * cfg.associativity);
  const size_t num_sets = cfg.capacity_slots / cfg.associativity;
  for (size_t begin = 0; begin < cfg.capacity_slots; begin += cache_chunk) {
    size_t count = std::min(cache_chunk, cfg.capacity_slots - begin);
    auto keys = read_device(cache.keys + begin, count);
    auto valid = read_device(cache.valid + begin, count);
    auto values = read_device(cache.values + begin * dim, count * dim);
    std::unordered_set<K> seen;
    for (size_t row = 0; row < count; ++row) {
      if (row % cfg.associativity == 0) seen.clear();
      require(valid[row] == 0 || valid[row] == 1, "resident validity");
      if (valid[row]) {
        ++occupancy;
        require(keys[row] % num_sets == (begin + row) / cfg.associativity,
                "resident key/set");
        require(seen.insert(keys[row]).second, "duplicate resident key");
        check_payload(keys[row], &values[row * dim], dim);
      }
    }
  }
  require(occupancy == previous_occupancy + admissions - evictions,
          "occupancy conservation");
  return occupancy;
}

struct CpuCache {
  std::vector<K> keys;
  std::vector<int> valid;
  std::vector<Meta> age;
  std::vector<Meta> freq;
};

static CpuCache snapshot(const DeviceCache& cache, size_t slots) {
  return {read_device(cache.keys, slots), read_device(cache.valid, slots),
          read_device(cache.age, slots), read_device(cache.freq, slots)};
}

static void verify_policy(const Config& cfg, Policy policy, CpuCache before,
                          const DeviceCache& cache, const RequestBuffers& req,
                          const std::vector<K>& requests,
                          const Counters& counts) {
  auto hits = read_device(req.hit_flags, cfg.batch_size);
  auto slots = read_device(req.slot_for_request, cfg.batch_size);
  auto tickets = read_device(req.tickets, cfg.batch_size);
  auto misses = read_device(req.miss_indices, counts.misses);
  auto victims = read_device(req.victim_slots, counts.misses);
  auto evicted = read_device(req.evicted_keys, counts.misses);
  auto evicted_valid = read_device(req.evicted_valid, counts.misses);
  std::vector<int> position(cfg.batch_size, -1);
  for (size_t index = 0; index < misses.size(); ++index) {
    require(misses[index] >= 0 &&
                static_cast<size_t>(misses[index]) < cfg.batch_size,
            "miss index");
    require(position[misses[index]] == -1, "duplicate compacted request");
    position[misses[index]] = static_cast<int>(index);
  }
  const size_t sets = cfg.capacity_slots / cfg.associativity;
  std::vector<size_t> order;
  size_t hit_count = 0;
  for (size_t row = 0; row < requests.size(); ++row) {
    size_t base = requests[row] % sets * cfg.associativity;
    bool present = false;
    for (int way = 0; way < cfg.associativity; ++way)
      present |=
          before.valid[base + way] && before.keys[base + way] == requests[row];
    require(hits[row] == static_cast<int>(present), "initial hit mask");
    require(present == (position[row] == -1), "compaction membership");
    hit_count += present;
    order.push_back(row);
  }
  require(hit_count == counts.hits, "observed hit count");
  std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    return tickets[left] < tickets[right];
  });
  Meta previous = *std::max_element(before.age.begin(), before.age.end());
  bool saw_miss = false;
  for (size_t row : order) {
    require(tickets[row] > previous, "nonmonotonic access ticket");
    previous = tickets[row];
    require(!saw_miss || !hits[row], "lookup/mutation stage overlap");
    saw_miss |= !hits[row];
    size_t base = requests[row] % sets * cfg.associativity;
    size_t chosen = cfg.capacity_slots;
    bool present = false;
    for (int way = 0; way < cfg.associativity; ++way) {
      size_t slot = base + way;
      if (before.valid[slot] && before.keys[slot] == requests[row]) {
        chosen = slot;
        present = true;
        break;
      }
    }
    if (!present) {
      for (int way = 0; way < cfg.associativity; ++way)
        if (!before.valid[base + way]) {
          chosen = base + way;
          break;
        }
      if (chosen == cfg.capacity_slots) {
        chosen = base;
        for (int way = 1; way < cfg.associativity; ++way) {
          size_t slot = base + way;
          bool older = before.age[slot] < before.age[chosen];
          if (policy == Policy::kLru
                  ? older
                  : (before.freq[slot] < before.freq[chosen] ||
                     (before.freq[slot] == before.freq[chosen] && older)))
            chosen = slot;
        }
      }
    }
    require(slots[row] == static_cast<int>(chosen), "policy slot");
    if (!hits[row]) {
      int pos = position[row];
      require(victims[pos] == (present ? -1 : static_cast<int>(chosen)),
              "policy admission");
      require(evicted_valid[pos] == (present ? 0 : before.valid[chosen]),
              "policy eviction");
      if (evicted_valid[pos])
        require(evicted[pos] == before.keys[chosen], "evicted key");
    }
    if (!present) {
      before.keys[chosen] = requests[row];
      before.valid[chosen] = 1;
      before.freq[chosen] = 1;
    } else if (policy == Policy::kLfu)
      ++before.freq[chosen];
    before.age[chosen] = tickets[row];
  }
  auto after = snapshot(cache, cfg.capacity_slots);
  require(before.keys == after.keys && before.valid == after.valid &&
              before.freq == after.freq && before.age == after.age,
          "final policy state");
}

static Counters read_counters(const RequestBuffers& req) {
  Counters c;
  CUDA_CHECK(cudaMemcpy(&c.hits, req.hit_count, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.misses, req.miss_count, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.replacements, req.replacement_count,
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.writeback_bytes, req.writeback_bytes,
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.completed, req.completed_count, sizeof(Meta),
                        cudaMemcpyDeviceToHost));
  return c;
}

static void emit_row(const std::string& mode, Policy policy, double lf,
                     const Config& cfg, size_t resident_slots, int run,
                     const std::string& stage, double elapsed_ms,
                     const Counters& counters) {
  double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  require(std::isfinite(seconds) && seconds >= 0.0 &&
              (stage != "total_wall" || seconds > 0.0),
          "invalid timing");
  const size_t payload_bytes = cfg.capacity_slots * cfg.value_bytes;
  const size_t metadata_bytes =
      cfg.capacity_slots * (sizeof(K) + sizeof(int) + 2 * sizeof(Meta)) +
      cfg.capacity_slots / cfg.associativity * sizeof(int);
  const size_t request_bytes =
      cfg.batch_size * (3 * sizeof(K) + 5 * sizeof(int) + sizeof(Meta) +
                        3 * static_cast<size_t>(cfg.value_bytes)) +
      6 * sizeof(Meta);
  std::cout << "TBE_like," << policy_name(policy) << "," << mode << ","
            << std::fixed << std::setprecision(9) << lf << ","
            << cfg.associativity << "," << cfg.value_bytes << ","
            << cfg.capacity_slots << "," << resident_slots << ","
            << cfg.batch_size << "," << run << "," << stage << ","
            << std::setprecision(9) << elapsed_ms << ",";
  if (stage == "total_wall")
    std::cout << static_cast<double>(counters.completed) / seconds / 1e9;
  std::cout << "," << counters.hits << "," << counters.misses << ","
            << counters.replacements << "," << counters.writeback_bytes << ","
            << counters.completed << ","
            << counters.writeback_bytes / cfg.value_bytes << ","
            << counters.final_occupancy << "," << payload_bytes << ","
            << metadata_bytes << "," << request_bytes << ","
            << payload_bytes + metadata_bytes + request_bytes << "," << cfg.seed
            << ","
            << cfg.seed + static_cast<uint64_t>(run - 1 + cfg.warmup) * 101ULL
            << ",PASS,"
            << (stage == "total_wall" ? "device_cache_service_host_wall"
                                      : "gpu_event_diagnostic")
            << ",immutable_row_snapshot,device_snapshot_only,0,0,0,batch_total_"
               "after_completion,per_batch_full_oracle"
            << std::endl;
}

static void run_policy_lf(const Config& cfg, Policy policy, double lf) {
  int value_dim = cfg.value_bytes / static_cast<int>(sizeof(float));
  size_t num_sets = cfg.capacity_slots / cfg.associativity;
  int resident_ways =
      std::max(1, static_cast<int>(std::llround(cfg.associativity * lf)));
  resident_ways = std::min(resident_ways, cfg.associativity);
  size_t resident_slots = num_sets * static_cast<size_t>(resident_ways);
  double actual_lf = static_cast<double>(resident_slots) /
                     static_cast<double>(cfg.capacity_slots);

  std::cerr << "[TBE-like] policy=" << policy_name(policy)
            << " mode=" << cfg.mode << " LF=" << std::fixed
            << std::setprecision(9) << actual_lf
            << " assoc=" << cfg.associativity
            << " value_bytes=" << cfg.value_bytes
            << " capacity=" << cfg.capacity_slots << " batch=" << cfg.batch_size
            << std::endl;

  DeviceCache cache;
  RequestBuffers req;
  cache.allocate(cfg.capacity_slots, value_dim, num_sets);
  req.allocate(cfg.batch_size, value_dim);

  const int block = 256;
  int init_grid =
      grid_for(std::max(cfg.capacity_slots,
                        cfg.capacity_slots * static_cast<size_t>(value_dim)),
               block);
  init_cache_kernel<<<init_grid, block>>>(
      cache.keys, cache.valid, cache.values, cache.age, cache.freq, cache.locks,
      cfg.capacity_slots, num_sets, cfg.associativity, resident_ways,
      value_dim);
  CUDA_CHECK(cudaGetLastError());
  const Meta initial_clock = static_cast<Meta>(resident_ways);
  CUDA_CHECK(cudaMemcpy(req.global_clock, &initial_clock, sizeof(Meta),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t occupancy = resident_slots;
  for (int run = 0; run < cfg.warmup + cfg.runs; run++) {
    StageTimes times = run_once(cfg, policy, &cache, &req, num_sets,
                                resident_ways, value_dim, run);
    Counters counters = read_counters(req);
    occupancy = verify_batch(cfg, cache, req, counters, occupancy);
    counters.final_occupancy = occupancy;
    if (run >= cfg.warmup) {
      int visible_run = run - cfg.warmup + 1;
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "lookup_snapshot", times.lookup_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "miss_compact", times.miss_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "admission_snapshot", times.service_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "total_gpu", times.gpu_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "total_wall", times.wall_ms, counters);
    }
  }

  req.free();
  cache.free();
}

static void self_test() {
  size_t cases = 0;
  for (Policy policy : {Policy::kLru, Policy::kLfu}) {
    for (int ways : {2, 32}) {
      std::vector<std::vector<K>> traces{
          {static_cast<K>(ways), static_cast<K>(ways + 1)},
          {static_cast<K>(ways), static_cast<K>(ways),
           static_cast<K>(ways + 1)},
          {0, 1, static_cast<K>(ways), static_cast<K>(ways + 1), 0},
          {65536, 1ULL << 48, 65536}};
      std::vector<K> overflow;
      for (int index = 0; index < ways * 2 + 3; ++index)
        overflow.push_back(ways + index);
      traces.push_back(overflow);
      for (const auto& trace : traces) {
        Config cfg;
        cfg.capacity_slots = ways;
        cfg.associativity = ways;
        cfg.batch_size = trace.size();
        cfg.value_bytes = 128;
        DeviceCache cache;
        RequestBuffers req;
        cache.allocate(ways, 32, 1);
        req.allocate(trace.size(), 32);
        init_cache_kernel<<<1, 256>>>(cache.keys, cache.valid, cache.values,
                                      cache.age, cache.freq, cache.locks, ways,
                                      1, ways, ways, 32);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        std::vector<Meta> frequencies(ways, 2);
        CUDA_CHECK(cudaMemcpy(cache.freq, frequencies.data(),
                              ways * sizeof(Meta), cudaMemcpyHostToDevice));
        Meta clock = ways;
        CUDA_CHECK(cudaMemcpy(req.global_clock, &clock, sizeof(Meta),
                              cudaMemcpyHostToDevice));
        for (int replay = 0; replay < 2; ++replay) {
          auto before = snapshot(cache, ways);
          run_once(cfg, policy, &cache, &req, 1, ways, 32, replay, &trace);
          auto counts = read_counters(req);
          verify_batch(cfg, cache, req, counts, ways);
          verify_policy(cfg, policy, before, cache, req, trace, counts);
          ++cases;
        }
        req.free();
        cache.free();
      }
    }
  }
  std::cerr << "SELF_TEST PASS cases=" << cases
            << " policies=lru,lfu oracle=exact_payload_key_writeback_policy\n";
}

int main(int argc, char** argv) try {
  Config cfg = parse_args(argc, argv);

  if (cfg.self_test) {
    self_test();
    return 0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review-response TBE-like baseline" << std::endl;
  std::cerr
      << "Device-only simulation; not FBGEMM. Excludes allocation, request "
         "generation, "
         "oracle/readback, backing fetch, host writeback, optimizer and CSV. "
         "Allocated bytes exclude CUDA/runtime overhead; rows within a process "
         "are not independent runs."
      << std::endl;

  std::cout
      << "baseline,policy,mode,initial_load_factor,associativity,value_bytes,"
         "capacity_slots,initial_resident_slots,batch_size,run,stage,elapsed_"
         "ms,"
         "throughput_grequests_s,hits,misses,admissions,writeback_snapshot_"
         "bytes,"
         "completed_requests,evictions,final_resident_slots,cache_payload_"
         "bytes,cache_metadata_bytes,"
         "request_device_bytes,total_device_allocation_bytes,seed,batch_seed,"
         "oracle_status,timing_scope,"
         "output_contract,writeback_scope,rejected,failed,backing_store_bytes,"
         "counter_scope,measurement_regime"
      << std::endl;

  std::vector<Policy> policies;
  if (cfg.policy == "all") {
    policies = {Policy::kLru, Policy::kLfu};
  } else if (cfg.policy == "lfu") {
    policies = {Policy::kLfu};
  } else {
    policies = {Policy::kLru};
  }

  for (Policy policy : policies) {
    for (double lf : cfg.load_factors) {
      run_policy_lf(cfg, policy, lf);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << std::endl;
  return 1;
}
