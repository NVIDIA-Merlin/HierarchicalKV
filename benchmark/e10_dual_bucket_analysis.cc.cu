/*
 * E10: Dual-Bucket Analysis Benchmark
 *
 * Config: dim=32, capacity=128M, Pure HBM (max_hbm_for_vectors=0),
 *         EvictStrategy::kCustomized, TableMode::kMemory
 *
 * Three measurements:
 *   Part 1: First-eviction load factor (bucket_size 64 vs 128)
 *   Part 2: Steady-state cache hit ratio with Zipfian workload
 *   Part 3: Throughput at various load factors (insert + find)
 *
 * Output: CSV to stdout, progress to stderr.
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

/* ================================================================
 * Zipfian Generator (copied from score_strategy_benchmark.cc.cu)
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
 * Helper: create a dual-bucket table with the given bucket size
 * ================================================================ */

using HKVTable = HashTable<K, V, S, EvictStrategy::kCustomized>;

std::shared_ptr<HKVTable> create_table(size_t bucket_size) {
  HashTableOptions options;
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = 0;
  options.max_bucket_size = bucket_size;
  options.table_mode = TableMode::kMemory;

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
      buf.h_scores[i] = inserted + i;  // score = key
    }
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table.insert_or_assign(cur, buf.d_keys, buf.d_vectors, buf.d_scores,
                           stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    inserted += cur;
  }
}

/* ================================================================
 * Part 1: First-eviction load factor
 * ================================================================ */

void part1_first_eviction_lf(size_t bucket_size, BenchBuffers& buf,
                             cudaStream_t stream) {
  std::cerr << "[Part1] bucket_size=" << bucket_size << " first-eviction LF..."
            << std::flush;

  auto table = create_table(bucket_size);

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
    table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, buf.d_scores,
                            stream);
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

  std::cout << "first_eviction_lf," << bucket_size << "," << std::fixed
            << std::setprecision(6) << first_eviction_lf << std::endl;
}

/* ================================================================
 * Part 2: Steady-state cache hit ratio
 * ================================================================ */

void part2_hit_ratio(size_t bucket_size, BenchBuffers& buf,
                     cudaStream_t stream) {
  std::cerr << "[Part2] bucket_size=" << bucket_size << " hit ratio..."
            << std::flush;

  auto table = create_table(bucket_size);

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
    // LRU-style: score = global counter (most recently touched → highest score)
    for (size_t i = 0; i < cur; i++) buf.h_scores[i] = global_score++;
    CUDA_CHECK(cudaMemcpy(buf.d_keys, buf.h_keys, cur * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_scores, buf.h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, buf.d_keys, buf.d_vectors, buf.d_scores,
                            stream);
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

  std::cout << "hit_ratio," << bucket_size << "," << std::fixed
            << std::setprecision(4) << hit_ratio << std::endl;
}

/* ================================================================
 * Part 3: Throughput at various load factors
 * ================================================================ */

void part3_throughput(size_t bucket_size, BenchBuffers& buf,
                      cudaStream_t stream) {
  const double load_factors[] = {0.25, 0.50, 0.75, 0.90, 0.95, 1.00};
  const int NUM_LF = 6;
  const int WARMUP_RUNS = 3;
  const int MEASURE_RUNS = 5;
  const int TOTAL_RUNS = WARMUP_RUNS + MEASURE_RUNS;

  for (int lf_idx = 0; lf_idx < NUM_LF; lf_idx++) {
    double target_lf = load_factors[lf_idx];
    size_t fill_count =
        static_cast<size_t>(static_cast<double>(CAPACITY) * target_lf);
    // Clamp to capacity
    if (fill_count > CAPACITY) fill_count = CAPACITY;

    std::cerr << "[Part3] bucket_size=" << bucket_size << " LF=" << std::fixed
              << std::setprecision(2) << target_lf << "..." << std::flush;

    for (int run = 0; run < TOTAL_RUNS; run++) {
      // Create fresh table for each run
      auto table = create_table(bucket_size);

      // Fill to target load factor with sequential keys
      if (fill_count > 0) {
        fill_sequential(*table, fill_count, buf, stream);
      }

      // Prepare insert batch: keys in the range [fill_count,
      // fill_count+BATCH_SIZE) These keys do NOT already exist, so they insert
      // (and may evict at high LF) For LF=1.0, these will cause evictions
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
      table->insert_or_assign(BATCH_SIZE, buf.d_keys, buf.d_vectors,
                              buf.d_scores, stream);
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
        // If fill_count < 1M, use all filled keys for find
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
        std::cout << "insert_throughput," << bucket_size << "," << std::fixed
                  << std::setprecision(2) << target_lf << "," << measured_run
                  << "," << std::setprecision(6) << insert_tp << std::endl;
        std::cout << "find_throughput," << bucket_size << "," << std::fixed
                  << std::setprecision(2) << target_lf << "," << measured_run
                  << "," << std::setprecision(6) << find_tp << std::endl;
      }
    }

    std::cerr << " done" << std::endl;
  }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E10: Dual-Bucket Analysis" << std::endl;
  std::cerr << "Config: dim=" << DIM << ", capacity=" << CAPACITY
            << ", Pure HBM, kCustomized, kMemory" << std::endl;

  // Usage: ./e10_dual_bucket_analysis [bucket_size] [part]
  // bucket_size: 64, 128, or 0 for both (default: 0)
  // part: 1, 2, 3, or 0 for all (default: 0)
  int target_bucket_size = 0;
  int target_part = 0;
  if (argc > 1) {
    target_bucket_size = std::atoi(argv[1]);
    if (target_bucket_size != 0 && target_bucket_size != 64 &&
        target_bucket_size != 128) {
      std::cerr << "Error: bucket_size must be 0 (both), 64, or 128, got "
                << target_bucket_size << std::endl;
      return 1;
    }
    if (target_bucket_size > 0)
      std::cerr << "Running only bucket_size=" << target_bucket_size
                << std::endl;
  }
  if (argc > 2) {
    target_part = std::atoi(argv[2]);
    if (target_part < 0 || target_part > 3) {
      std::cerr << "Error: part must be 0 (all), 1, 2, or 3" << std::endl;
      return 1;
    }
    if (target_part > 0)
      std::cerr << "Running only Part " << target_part << std::endl;
  }

  std::vector<size_t> bucket_sizes;
  if (target_bucket_size > 0) {
    bucket_sizes.push_back(static_cast<size_t>(target_bucket_size));
  } else {
    bucket_sizes = {64, 128};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  BenchBuffers buf;
  buf.alloc();

  // CSV header
  std::cout << "metric,bucket_size,load_factor,run,throughput_bkvs"
            << std::endl;

  try {
    if (target_part == 0 || target_part == 1) {
      std::cerr << "=== Part 1: First-eviction load factor ===" << std::endl;
      for (size_t bs : bucket_sizes) {
        part1_first_eviction_lf(bs, buf, stream);
      }
    }

    if (target_part == 0 || target_part == 2) {
      std::cerr << "=== Part 2: Steady-state cache hit ratio ===" << std::endl;
      for (size_t bs : bucket_sizes) {
        part2_hit_ratio(bs, buf, stream);
      }
    }

    if (target_part == 0 || target_part == 3) {
      std::cerr << "=== Part 3: Throughput at various load factors ==="
                << std::endl;
      for (size_t bs : bucket_sizes) {
        part3_throughput(bs, buf, stream);
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
  std::cerr << "=== E10 Complete ===" << std::endl;
  return 0;
}
