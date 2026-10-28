/*
 * E17: Single-Bucket vs Dual-Bucket Comparison Benchmark
 *
 * Extends E10 to compare kThroughput (single-bucket) vs kMemory (dual-bucket)
 * side-by-side under identical conditions.
 *
 * Config B: dim=32, capacity=128M, Pure HBM (max_hbm_for_vectors=0),
 *           EvictStrategy::kLru, bucket_size=128
 *
 * Four measurements:
 *   Part 1: First-eviction load factor for BOTH modes
 *   Part 2: Steady-state cache hit ratio with Zipfian workload
 *   Part 3: insert_or_assign + find throughput at various load factors
 *   Part 4: Top-K score retention (correctness under LF=1.0 steady state)
 *
 * Output: CSV to stdout, progress to stderr.
 * Usage: ./e17_single_vs_dual [part]
 *   part=0 (default): run all parts
 *   part=1/2/3/4: run specific part only
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using namespace nv::merlin;
using namespace benchmark;

static constexpr size_t DIM = 32;
static constexpr size_t CAPACITY = 128UL * 1024 * 1024;  // 128M
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;      // 1M
static constexpr size_t BUCKET_SIZE = 128;

/* ================================================================
 * Zipfian Generator (same as E10)
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
    double sum = 0;
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
 * Helper: allocate device/host buffers used across all parts
 * ================================================================ */

struct BenchBuffers {
  K* h_keys;
  S* h_scores;
  bool* h_found;
  K* d_keys;
  S* d_scores;
  V* d_vectors;
  bool* d_found;

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
 * Mode descriptor
 * ================================================================ */

struct ModeConfig {
  const char* name;      // "single" or "dual"
  TableMode table_mode;  // kThroughput or kMemory
};

static const ModeConfig MODES[] = {
    {"single", TableMode::kThroughput},
    {"dual", TableMode::kMemory},
};
static constexpr int NUM_MODES = 2;

/* ================================================================
 * Helper: create table for a given mode
 * ================================================================ */

using HKVTable = HashTable<K, V, S, EvictStrategy::kLru>;

std::shared_ptr<HKVTable> create_table(TableMode mode) {
  HashTableOptions options;
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = 0;  // Pure HBM for both modes
  options.max_bucket_size = BUCKET_SIZE;
  options.table_mode = mode;

  auto table = std::make_shared<HKVTable>();
  table->init(options);
  return table;
}

/* ================================================================
 * Helper: fill table with sequential keys up to target_count
 * ================================================================ */

void fill_sequential(HKVTable& table, size_t target_count, BenchBuffers& buf,
                     cudaStream_t stream) {
  size_t inserted = 0;
  while (inserted < target_count) {
    size_t cur = std::min(BATCH_SIZE, target_count - inserted);
    for (size_t i = 0; i < cur; i++) {
      buf.h_keys[i] = inserted + i;
      buf.h_scores[i] = inserted + i;  // score = key (ignored under kLru)
    }
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    // kLru mode: pass nullptr for scores (auto-generates LRU timestamps)
    table.insert_or_assign(cur, buf.d_keys, buf.d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    inserted += cur;
  }
}

/* ================================================================
 * Part 1: First-eviction load factor (single vs dual)
 * ================================================================ */

void part1_first_eviction_lf(const ModeConfig& mode, BenchBuffers& buf,
                             cudaStream_t stream) {
  std::cerr << "[Part1] mode=" << mode.name << " first-eviction LF..."
            << std::flush;

  auto table = create_table(mode.table_mode);

  size_t total_inserted = 0;
  size_t prev_size = 0;
  double first_eviction_lf = 1.0;  // default if no eviction detected

  // Insert sequential keys in batches of 1M until eviction or full
  while (total_inserted < CAPACITY * 2) {
    size_t cur = BATCH_SIZE;
    for (size_t i = 0; i < cur; i++) {
      buf.h_keys[i] = total_inserted + i;
      buf.h_scores[i] = total_inserted + i;
    }
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    total_inserted += cur;

    size_t current_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // If size < total_inserted, eviction has occurred
    if (current_size < total_inserted) {
      // The load factor BEFORE this batch triggered eviction
      first_eviction_lf =
          static_cast<double>(prev_size) / static_cast<double>(CAPACITY);
      std::cerr << " eviction at batch " << (total_inserted / BATCH_SIZE)
                << ", LF_before=" << std::fixed << std::setprecision(6)
                << first_eviction_lf << std::endl;
      break;
    }

    prev_size = current_size;

    // If we filled the table fully without eviction
    if (current_size >= CAPACITY) {
      first_eviction_lf =
          static_cast<double>(current_size) / static_cast<double>(CAPACITY);
      std::cerr << " table full without eviction, LF=" << std::fixed
                << std::setprecision(6) << first_eviction_lf << std::endl;
      break;
    }
  }

  std::cout << mode.name << ",first_eviction_lf," << BUCKET_SIZE << ","
            << std::fixed << std::setprecision(6) << first_eviction_lf
            << std::endl;
}

/* ================================================================
 * Part 2: Steady-state cache hit ratio (single vs dual)
 * ================================================================ */

void part2_hit_ratio(const ModeConfig& mode, BenchBuffers& buf,
                     cudaStream_t stream) {
  std::cerr << "[Part2] mode=" << mode.name << " hit ratio..." << std::flush;

  auto table = create_table(mode.table_mode);

  // Pre-populate to lambda=1.0 with sequential keys (score = key)
  std::cerr << " populate..." << std::flush;
  fill_sequential(*table, CAPACITY, buf, stream);

  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << " LF=" << std::fixed << std::setprecision(4) << real_lf;

  // Zipfian steady-state insertions: 5 * capacity keys
  // Use monotonically increasing scores (LRU simulation):
  // recently accessed keys get high scores and stay in cache,
  // infrequently accessed keys get evicted.
  uint64_t key_range = 10UL * CAPACITY;
  double zipf_alpha = 0.99;
  size_t steady_state_count = 5UL * CAPACITY;

  std::cerr << " steady-state..." << std::flush;
  ZipfianGenerator zipf_insert(key_range, zipf_alpha, 42);
  uint64_t global_score = CAPACITY;  // continue from the pre-populate scores
  size_t inserted = 0;
  while (inserted < steady_state_count) {
    size_t cur = std::min(BATCH_SIZE, steady_state_count - inserted);
    zipf_insert.fill(buf.h_keys, cur);
    // LRU-style: score = global counter (most recently touched -> highest
    // score)
    for (size_t i = 0; i < cur; i++) buf.h_scores[i] = global_score++;
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    inserted += cur;
  }

  // 5 rounds of find with 1M Zipfian keys each
  std::cerr << " find..." << std::flush;
  ZipfianGenerator zipf_find(key_range, zipf_alpha, 12345);
  size_t total_found = 0;
  size_t total_queries = 0;

  for (int r = 0; r < 5; r++) {
    zipf_find.fill(buf.h_keys, BATCH_SIZE);
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));
    table->find(BATCH_SIZE, buf.d_keys, buf.d_vectors, buf.d_found, nullptr,
                stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(buf.h_found, buf.d_found, BATCH_SIZE * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      if (buf.h_found[j]) total_found++;
    }
    total_queries += BATCH_SIZE;
  }

  double hit_ratio =
      static_cast<double>(total_found) / static_cast<double>(total_queries);

  std::cerr << " hit=" << std::fixed << std::setprecision(4) << hit_ratio
            << std::endl;

  std::cout << mode.name << ",hit_ratio," << BUCKET_SIZE << "," << std::fixed
            << std::setprecision(4) << hit_ratio << std::endl;
}

/* ================================================================
 * Part 3: Throughput at various load factors (single vs dual)
 * ================================================================ */

void part3_throughput(const ModeConfig& mode, BenchBuffers& buf,
                      cudaStream_t stream) {
  const double load_factors[] = {0.50, 0.75, 0.95, 1.00};
  const int NUM_LF = 4;
  const int WARMUP_RUNS = 3;
  const int MEASURE_RUNS = 5;
  const int TOTAL_RUNS = WARMUP_RUNS + MEASURE_RUNS;

  for (int lf_idx = 0; lf_idx < NUM_LF; lf_idx++) {
    double target_lf = load_factors[lf_idx];
    size_t fill_count =
        static_cast<size_t>(static_cast<double>(CAPACITY) * target_lf);
    // Clamp to capacity
    if (fill_count > CAPACITY) fill_count = CAPACITY;

    std::cerr << "[Part3] mode=" << mode.name << " LF=" << std::fixed
              << std::setprecision(2) << target_lf << "..." << std::flush;

    for (int run = 0; run < TOTAL_RUNS; run++) {
      // Create fresh table for each run
      auto table = create_table(mode.table_mode);

      // Fill to target load factor with sequential keys
      if (fill_count > 0) {
        fill_sequential(*table, fill_count, buf, stream);
      }

      // Prepare insert batch: keys in the range [fill_count,
      // fill_count+BATCH_SIZE) These keys do NOT already exist, so they insert
      // (and may evict at high LF)
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        buf.h_keys[i] = fill_count + i;
        buf.h_scores[i] = fill_count + i;
      }
      CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, BATCH_SIZE * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, BATCH_SIZE * sizeof(S),
                            cudaMemcpyHostToDevice));

      // Measure insert throughput
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto t_insert = Timer<double>();
      t_insert.start();
      // kLru mode: pass nullptr for scores (auto-generates LRU timestamps)
      table->insert_or_assign(BATCH_SIZE, buf.d_keys, buf.d_vectors, nullptr,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      t_insert.end();
      double insert_tp = static_cast<double>(BATCH_SIZE) /
                         t_insert.getResult() /
                         (1024.0 * 1024.0 * 1024.0);  // billion keys/s

      // Prepare find batch: keys in [0, BATCH_SIZE) -- guaranteed 100% hit
      // (they were inserted during the fill phase, assuming fill_count >=
      // BATCH_SIZE)
      size_t find_count = BATCH_SIZE;
      if (fill_count < BATCH_SIZE) {
        find_count = fill_count > 0 ? fill_count : BATCH_SIZE;
      }
      for (size_t i = 0; i < find_count; i++) {
        buf.h_keys[i] = i;  // sequential keys from 0
      }
      CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, find_count * sizeof(K),
                            cudaMemcpyHostToDevice));

      // Measure find throughput
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto t_find = Timer<double>();
      t_find.start();
      table->find(find_count, buf.d_keys, buf.d_vectors, buf.d_found, nullptr,
                  stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      t_find.end();
      double find_tp = static_cast<double>(find_count) / t_find.getResult() /
                       (1024.0 * 1024.0 * 1024.0);  // billion keys/s

      // Only output measured runs (skip warmup)
      if (run >= WARMUP_RUNS) {
        int measured_run = run - WARMUP_RUNS + 1;
        std::cout << mode.name << ",insert_throughput," << BUCKET_SIZE << ","
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << measured_run << "," << std::setprecision(6) << insert_tp
                  << std::endl;
        std::cout << mode.name << ",find_throughput," << BUCKET_SIZE << ","
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << measured_run << "," << std::setprecision(6) << find_tp
                  << std::endl;
      }
    }

    std::cerr << " done" << std::endl;
  }
}

/* ================================================================
 * Part 4: Top-K score retention (single vs dual)
 *
 * Measures "correctness" of eviction: after a Zipfian steady-state
 * workload, what fraction of the ideal top-N (N=capacity) highest-
 * scored keys are actually present in the table?
 *
 * Protocol:
 *   1. Fill table to capacity with sequential keys, score = key.
 *   2. Run 5× capacity Zipfian inserts with LRU scoring
 *      (monotonically increasing global counter → recent keys score higher).
 *   3. Track all inserted keys and their latest scores on the host side.
 *   4. Export the table contents (keys + scores).
 *   5. Compute the ideal top-N set from the host-side score map.
 *   6. Report what fraction of ideal top-N are present in the table.
 * ================================================================ */

void part4_score_retention(const ModeConfig& mode, BenchBuffers& buf,
                           cudaStream_t stream) {
  std::cerr << "[Part4] mode=" << mode.name << " score retention..."
            << std::flush;

  auto table = create_table(mode.table_mode);

  // --- Phase 1: Pre-populate to capacity with sequential keys ---
  std::cerr << " populate..." << std::flush;
  fill_sequential(*table, CAPACITY, buf, stream);

  // Host-side score map: track the latest score for every key we've inserted
  std::unordered_map<K, S> key_score_map;
  key_score_map.reserve(CAPACITY * 6);  // expect ~5x capacity unique keys
  for (size_t i = 0; i < CAPACITY; i++) {
    key_score_map[static_cast<K>(i)] = static_cast<S>(i);
  }

  // --- Phase 2: Zipfian steady-state insertions (5× capacity) ---
  uint64_t key_range = 10UL * CAPACITY;
  double zipf_alpha = 0.99;
  size_t steady_state_count = 5UL * CAPACITY;

  std::cerr << " steady-state..." << std::flush;
  ZipfianGenerator zipf_insert(key_range, zipf_alpha, 42);
  uint64_t global_score = CAPACITY;  // continue from pre-populate scores
  size_t inserted = 0;
  while (inserted < steady_state_count) {
    size_t cur = std::min(BATCH_SIZE, steady_state_count - inserted);
    zipf_insert.fill(buf.h_keys, cur);
    for (size_t i = 0; i < cur; i++) {
      buf.h_scores[i] = global_score;
      key_score_map[buf.h_keys[i]] = global_score;  // track latest score
      global_score++;
    }
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    inserted += cur;
  }

  // --- Phase 3: Export table contents ---
  std::cerr << " export..." << std::flush;
  K* h_export_keys;
  S* h_export_scores;
  K* d_export_keys;
  V* d_export_values;
  S* d_export_scores;

  CUDA_CHECK(cudaMallocHost(&h_export_keys, CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_export_scores, CAPACITY * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_export_keys, CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_export_values, CAPACITY * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_export_scores, CAPACITY * sizeof(S)));

  size_t exported = table->export_batch(
      CAPACITY, 0, d_export_keys, d_export_values, d_export_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_export_keys, d_export_keys, exported * sizeof(K),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_export_scores, d_export_scores, exported * sizeof(S),
                        cudaMemcpyDeviceToHost));

  // Build set of keys actually in the table
  std::unordered_set<K> table_keys(h_export_keys, h_export_keys + exported);

  // --- Phase 4: Compute ideal top-N set ---
  // Sort all ever-inserted keys by their latest score (descending)
  std::cerr << " compute ideal..." << std::flush;
  std::vector<std::pair<S, K>> scored_keys;
  scored_keys.reserve(key_score_map.size());
  for (auto& kv : key_score_map) {
    scored_keys.emplace_back(kv.second, kv.first);
  }
  // Sort descending by score
  std::sort(scored_keys.begin(), scored_keys.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  // The ideal top-N: the CAPACITY keys with highest scores
  size_t ideal_n = std::min(static_cast<size_t>(CAPACITY), scored_keys.size());
  size_t retained = 0;
  for (size_t i = 0; i < ideal_n; i++) {
    if (table_keys.count(scored_keys[i].second)) {
      retained++;
    }
  }

  double retention_rate =
      static_cast<double>(retained) / static_cast<double>(ideal_n);

  std::cerr << " retention=" << retained << "/" << ideal_n << " (" << std::fixed
            << std::setprecision(4) << (retention_rate * 100.0) << "%)"
            << " table_size=" << exported << std::endl;

  std::cout << mode.name << ",score_retention," << BUCKET_SIZE << ","
            << std::fixed << std::setprecision(6) << retention_rate << ","
            << retained << "," << ideal_n << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFreeHost(h_export_keys));
  CUDA_CHECK(cudaFreeHost(h_export_scores));
  CUDA_CHECK(cudaFree(d_export_keys));
  CUDA_CHECK(cudaFree(d_export_values));
  CUDA_CHECK(cudaFree(d_export_scores));
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E17: Single-Bucket vs Dual-Bucket Comparison" << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << CAPACITY
            << ", bucket_size=" << BUCKET_SIZE << ", Pure HBM, kLru"
            << std::endl;

  // Usage: ./e17_single_vs_dual [part]
  // part: 0 for all (default), 1/2/3/4 for specific part
  int target_part = 0;
  if (argc > 1) {
    target_part = std::atoi(argv[1]);
    if (target_part < 0 || target_part > 4) {
      std::cerr << "Error: part must be 0 (all), 1, 2, 3, or 4" << std::endl;
      return 1;
    }
    if (target_part > 0)
      std::cerr << "Running only Part " << target_part << std::endl;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  BenchBuffers buf;
  buf.alloc();

  // CSV header
  std::cout << "mode,metric,bucket_size,load_factor,run,value" << std::endl;

  try {
    if (target_part == 0 || target_part == 1) {
      std::cerr << "=== Part 1: First-eviction load factor ===" << std::endl;
      for (int m = 0; m < NUM_MODES; m++) {
        part1_first_eviction_lf(MODES[m], buf, stream);
      }
    }

    if (target_part == 0 || target_part == 2) {
      std::cerr << "=== Part 2: Steady-state cache hit ratio ===" << std::endl;
      for (int m = 0; m < NUM_MODES; m++) {
        part2_hit_ratio(MODES[m], buf, stream);
      }
    }

    if (target_part == 0 || target_part == 3) {
      std::cerr << "=== Part 3: Throughput at various load factors ==="
                << std::endl;
      for (int m = 0; m < NUM_MODES; m++) {
        part3_throughput(MODES[m], buf, stream);
      }
    }

    if (target_part == 0 || target_part == 4) {
      std::cerr << "=== Part 4: Top-K score retention ===" << std::endl;
      for (int m = 0; m < NUM_MODES; m++) {
        part4_score_retention(MODES[m], buf, stream);
      }
    }

  } catch (const CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    buf.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }

  buf.free();
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cerr << "=== E17 Complete ===" << std::endl;
  return 0;
}
