/*
 * E19: Cache Quality Sweep — Hit Rate vs Zipfian Alpha for All Scoring Policies
 *
 * Config B: dim=32, capacity=128M, batch=1M, bucket_size=128,
 *           Pure HBM (max_hbm_for_vectors=0), kThroughput mode.
 *
 * Scoring policies tested:
 *   kLru        — score = device timestamp (managed internally)
 *   kLfu        — score = access frequency (caller provides increment)
 *   kEpochLru   — epoch-based LRU (caller provides global_epoch)
 *   kEpochLfu   — epoch-based LFU (caller provides global_epoch + increment)
 *   kCustomized — LRU-style monotonic counter provided by caller
 *
 * Zipfian alpha values: {0.50, 0.75, 0.99, 1.25}
 *
 * Protocol per (policy, alpha):
 *   1. Create fresh table.
 *   2. Pre-populate to capacity with sequential keys [0, CAPACITY).
 *   3. Run 5x CAPACITY Zipfian steady-state inserts from key_range=10x
 * CAPACITY.
 *   4. Run 5 rounds of 1M Zipfian find queries (different seed).
 *   5. Compute hit ratio.
 *
 * Output: CSV to stdout, progress to stderr.
 *
 * Usage:
 *   ./e19_cache_quality_sweep [policy] [alpha]
 *     policy: all (default), kLru, kLfu, kEpochLru, kEpochLfu, kCustomized
 *     alpha:  all (default), or a specific float value matching one of the
 * sweep
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using namespace nv::merlin;
using namespace benchmark;

/* ================================================================
 * Constants — Config B
 * ================================================================ */

static constexpr size_t DIM = 32;
static constexpr size_t CAPACITY = 128UL * 1024 * 1024;  // 128M slots
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;      // 1M keys per batch
static constexpr size_t BUCKET_SIZE = 128;
static constexpr size_t STEADY_STATE_BATCHES = 5UL * CAPACITY / BATCH_SIZE;
static constexpr int FIND_ROUNDS = 5;
static constexpr uint64_t KEY_RANGE = 10UL * CAPACITY;
static constexpr uint64_t INSERT_SEED = 42;
static constexpr uint64_t FIND_SEED = 12345;

static const double ALPHA_VALUES[] = {0.50, 0.75, 0.99, 1.25};
static constexpr int NUM_ALPHAS = 4;

/* ================================================================
 * Zipfian Generator (YCSB-style, consistent with other E-benchmarks)
 * ================================================================ */

class ZipfianGenerator {
 public:
  ZipfianGenerator(uint64_t n, double theta, uint64_t seed = 42)
      : n_(n), theta_(theta), rng_(seed) {
    zeta_n_ = zetaApprox(n_, theta_);
    zeta_2_ = zetaApprox(2, theta_);
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

  void fill(K* keys, size_t count) {
    for (size_t i = 0; i < count; i++) keys[i] = next();
  }

 private:
  uint64_t n_;
  double theta_;
  double zeta_n_, zeta_2_, alpha_, eta_;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> dist_{0.0, 1.0};

  static double zetaApprox(uint64_t n, double theta) {
    const uint64_t EXACT = 10000;
    double sum = 0.0;
    uint64_t e = std::min(n, EXACT);
    for (uint64_t i = 1; i <= e; i++)
      sum += 1.0 / std::pow(static_cast<double>(i), theta);
    if (n > EXACT && theta != 1.0)
      sum += (std::pow(static_cast<double>(n), 1.0 - theta) -
              std::pow(static_cast<double>(EXACT), 1.0 - theta)) /
             (1.0 - theta);
    return sum;
  }
};

/* ================================================================
 * Shared device/host buffers
 * ================================================================ */

struct BenchBuffers {
  K* h_keys = nullptr;
  S* h_scores = nullptr;
  bool* h_found = nullptr;
  K* d_keys = nullptr;
  S* d_scores = nullptr;
  V* d_vectors = nullptr;
  bool* d_found = nullptr;

  void alloc() {
    CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
    CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));
    CUDA_CHECK(cudaMallocHost(&h_found, BATCH_SIZE * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S)));
    CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
    CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));
  }

  void free() {
    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_scores));
    CUDA_CHECK(cudaFreeHost(h_found));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_found));
  }
};

/* ================================================================
 * Templated benchmark — one instantiation per EvictStrategy
 *
 * Score conventions:
 *   kLru      — scores=nullptr; table assigns device timestamp internally
 *   kLfu      — scores=frequency increment (1 per access)
 *   kEpochLru — scores=nullptr; caller must call set_global_epoch each batch
 *   kEpochLfu — scores=frequency increment (1 per access) + set_global_epoch
 *   kCustomized — scores=monotonic counter (caller-managed LRU-style)
 * ================================================================ */

template <int Strategy>
void run_policy(const char* policy_name, double alpha, BenchBuffers& buf,
                cudaStream_t stream) {
  using HKVTable = HashTable<K, V, S, Strategy>;

  std::cerr << "[E19] policy=" << policy_name << " alpha=" << std::fixed
            << std::setprecision(2) << alpha << " — creating table..."
            << std::flush;

  HashTableOptions options;
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = 0;  // Pure HBM: all vectors in GPU DRAM
  options.max_bucket_size = BUCKET_SIZE;
  options.table_mode = TableMode::kThroughput;  // single-bucket mode

  auto table = std::make_shared<HKVTable>();
  table->init(options);

  // Global epoch counter (incremented once per batch for epoch-based policies)
  uint64_t global_epoch = 0;

  // Monotonic score counter for kCustomized (continues across populate + steady
  // state so that steady-state scores are always higher than populate scores)
  uint64_t global_counter = 0;

  /* -----------------------------------------------------------------
   * Phase 1: Pre-populate to capacity with sequential keys [0, CAPACITY)
   * ----------------------------------------------------------------- */
  std::cerr << " populate..." << std::flush;

  size_t populated = 0;
  while (populated < CAPACITY) {
    size_t cur = std::min(BATCH_SIZE, CAPACITY - populated);

    // Build key batch: sequential
    for (size_t i = 0; i < cur; i++) buf.h_keys[i] = populated + i;
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));

    if constexpr (Strategy == EvictStrategy::kEpochLru ||
                  Strategy == EvictStrategy::kEpochLfu) {
      table->set_global_epoch(global_epoch++);
    }

    if constexpr (Strategy == EvictStrategy::kLru ||
                  Strategy == EvictStrategy::kEpochLru) {
      // LRU-based: table manages timestamps internally; no scores needed
      table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, nullptr, stream);
    } else if constexpr (Strategy == EvictStrategy::kLfu ||
                         Strategy == EvictStrategy::kEpochLfu) {
      // LFU-based: scores = frequency increment (1 per access)
      for (size_t i = 0; i < cur; i++) buf.h_scores[i] = 1;
      CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, buf.d_scores,
                              stream);
    } else {
      // kCustomized: monotonic counter (more-recently-touched => higher score)
      for (size_t i = 0; i < cur; i++) buf.h_scores[i] = global_counter++;
      CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, buf.d_scores,
                              stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    populated += cur;
  }

  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << " LF=" << std::fixed << std::setprecision(4) << real_lf;

  /* -----------------------------------------------------------------
   * Phase 2: Steady-state Zipfian insertions (5x CAPACITY)
   * ----------------------------------------------------------------- */
  std::cerr << " steady-state..." << std::flush;

  ZipfianGenerator zipf_insert(KEY_RANGE, alpha, INSERT_SEED);

  for (size_t b = 0; b < STEADY_STATE_BATCHES; b++) {
    zipf_insert.fill(buf.h_keys, BATCH_SIZE);
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));

    if constexpr (Strategy == EvictStrategy::kEpochLru ||
                  Strategy == EvictStrategy::kEpochLfu) {
      table->set_global_epoch(global_epoch++);
    }

    if constexpr (Strategy == EvictStrategy::kLru ||
                  Strategy == EvictStrategy::kEpochLru) {
      table->insert_or_assign(BATCH_SIZE, buf.d_keys, buf.d_vectors, nullptr,
                              stream);
    } else if constexpr (Strategy == EvictStrategy::kLfu ||
                         Strategy == EvictStrategy::kEpochLfu) {
      for (size_t i = 0; i < BATCH_SIZE; i++) buf.h_scores[i] = 1;
      CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, BATCH_SIZE * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BATCH_SIZE, buf.d_keys, buf.d_vectors,
                              buf.d_scores, stream);
    } else {
      // kCustomized: monotonic counter continues from populate phase
      for (size_t i = 0; i < BATCH_SIZE; i++)
        buf.h_scores[i] = global_counter++;
      CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, BATCH_SIZE * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BATCH_SIZE, buf.d_keys, buf.d_vectors,
                              buf.d_scores, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Progress report every 10% of steady-state batches
    if (STEADY_STATE_BATCHES >= 10 &&
        (b + 1) % (STEADY_STATE_BATCHES / 10) == 0) {
      std::cerr << " " << ((b + 1) * 100 / STEADY_STATE_BATCHES) << "%"
                << std::flush;
    }
  }

  /* -----------------------------------------------------------------
   * Phase 3: Measure hit ratio AND throughput with find queries
   * ----------------------------------------------------------------- */
  std::cerr << " find..." << std::flush;

  ZipfianGenerator zipf_find(KEY_RANGE, alpha, FIND_SEED);
  size_t total_found = 0;
  size_t total_queries = 0;
  double total_find_ms = 0.0;

  for (int r = 0; r < FIND_ROUNDS; r++) {
    zipf_find.fill(buf.h_keys, BATCH_SIZE);
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));

    // Time the find kernel
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    table->find(BATCH_SIZE, buf.d_keys, buf.d_vectors, buf.d_found, nullptr,
                stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    total_find_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaMemcpy(buf.h_found, buf.d_found, BATCH_SIZE * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      if (buf.h_found[j]) total_found++;
    }
    total_queries += BATCH_SIZE;
  }

  double hit_ratio =
      static_cast<double>(total_found) / static_cast<double>(total_queries);
  double find_throughput_bkvs =
      static_cast<double>(total_queries) / (total_find_ms * 1e-3) / 1e9;

  std::cerr << " hit=" << std::fixed << std::setprecision(4) << hit_ratio
            << " find=" << std::setprecision(3) << find_throughput_bkvs
            << " B-KV/s" << std::endl;

  // CSV output:
  // policy,alpha,hit_ratio,total_found,total_queries,find_throughput_bkvs
  std::cout << policy_name << "," << std::fixed << std::setprecision(2) << alpha
            << "," << std::setprecision(4) << hit_ratio << "," << total_found
            << "," << total_queries << "," << std::setprecision(4)
            << find_throughput_bkvs << std::endl;
}

/* ================================================================
 * Dispatch helper — runs one policy across all selected alphas
 * ================================================================ */

template <int Strategy>
void sweep_alphas(const char* policy_name, const std::vector<double>& alphas,
                  BenchBuffers& buf, cudaStream_t stream) {
  for (double alpha : alphas) {
    run_policy<Strategy>(policy_name, alpha, buf, stream);
    // Allow GPU to cool and release all table memory before next run
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E19: Cache Quality Sweep (Hit Rate vs Zipfian Alpha)"
            << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << CAPACITY
            << ", batch=" << BATCH_SIZE << ", bucket_size=" << BUCKET_SIZE
            << ", Pure HBM, kThroughput" << std::endl;
  std::cerr << "Key range: " << KEY_RANGE
            << ", steady-state: " << (STEADY_STATE_BATCHES * BATCH_SIZE)
            << " inserts" << std::endl;

  // Parse optional arguments: [policy] [alpha]
  std::string target_policy = (argc > 1) ? argv[1] : "all";
  std::string target_alpha_str = (argc > 2) ? argv[2] : "all";

  // Build the set of alpha values to test
  std::vector<double> alphas;
  if (target_alpha_str == "all") {
    for (int i = 0; i < NUM_ALPHAS; i++) alphas.push_back(ALPHA_VALUES[i]);
  } else {
    double a = std::stod(target_alpha_str);
    bool found = false;
    for (int i = 0; i < NUM_ALPHAS; i++) {
      if (std::fabs(ALPHA_VALUES[i] - a) < 1e-6) {
        alphas.push_back(ALPHA_VALUES[i]);
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "Error: alpha=" << target_alpha_str
                << " is not in the sweep set {0.50, 0.75, 0.99, 1.25}"
                << std::endl;
      return 1;
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  BenchBuffers buf;
  buf.alloc();

  // CSV header
  std::cout
      << "policy,alpha,hit_ratio,total_found,total_queries,find_throughput_bkvs"
      << std::endl;

  try {
    if (target_policy == "all" || target_policy == "kLru") {
      std::cerr << "=== kLru ===" << std::endl;
      sweep_alphas<EvictStrategy::kLru>("kLru", alphas, buf, stream);
    }
    if (target_policy == "all" || target_policy == "kLfu") {
      std::cerr << "=== kLfu ===" << std::endl;
      sweep_alphas<EvictStrategy::kLfu>("kLfu", alphas, buf, stream);
    }
    if (target_policy == "all" || target_policy == "kEpochLru") {
      std::cerr << "=== kEpochLru ===" << std::endl;
      sweep_alphas<EvictStrategy::kEpochLru>("kEpochLru", alphas, buf, stream);
    }
    if (target_policy == "all" || target_policy == "kEpochLfu") {
      std::cerr << "=== kEpochLfu ===" << std::endl;
      sweep_alphas<EvictStrategy::kEpochLfu>("kEpochLfu", alphas, buf, stream);
    }
    if (target_policy == "all" || target_policy == "kCustomized") {
      std::cerr << "=== kCustomized ===" << std::endl;
      sweep_alphas<EvictStrategy::kCustomized>("kCustomized", alphas, buf,
                                               stream);
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    buf.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    buf.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }

  buf.free();
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cerr << "=== E19 Complete ===" << std::endl;
  return 0;
}
