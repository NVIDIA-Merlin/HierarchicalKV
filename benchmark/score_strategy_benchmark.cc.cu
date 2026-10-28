/*
 * E5: Score Strategy Comparison (Eviction Strategy — Exp #4d)
 *
 * Config B: dim=32, capacity=128M, Pure HBM, λ=1.0
 * Strategies: LRU, LFU, EpochLRU, EpochLFU, Customized
 * Workload: Zipfian (α=0.99) over 10× capacity key range
 * Measure: insert throughput + find hit rate after steady-state
 *
 * Output: CSV with per-strategy insert throughput, find throughput, hit rate.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using namespace nv::merlin;
using namespace benchmark;

static constexpr size_t DIM = 32;
static constexpr size_t INIT_CAPACITY = 128UL * 1024 * 1024;
static constexpr size_t HBM_GB = 16;
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;
static constexpr int STEADY_STATE_ROUNDS = 10;
static constexpr int FIND_ROUNDS = 5;

/* ─── Zipfian Generator (YCSB-style) ─── */

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

/* ─── Strategy Benchmark ─── */

template <int Strategy>
void run_strategy(const char* name, uint64_t key_range, double zipf_alpha,
                  cudaStream_t stream) {
  using HKVTable = HashTable<K, V, S, Strategy, Sm80>;

  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);

  auto table = std::make_shared<HKVTable>();
  table->init(options);

  K* h_keys;
  S* h_scores;
  K* d_keys;
  S* d_scores;
  V* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  // Set all scores to 1 (for LFU/EpochLFU/Custom: unit frequency)
  for (size_t j = 0; j < BATCH_SIZE; j++) h_scores[j] = 1;
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, BATCH_SIZE * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Pre-populate to λ=1.0 with sequential keys
  std::cerr << "  Pre-populating..." << std::flush;
  K start = 0;
  int epoch = 0;
  while (start < INIT_CAPACITY) {
    size_t cur = std::min(BATCH_SIZE, INIT_CAPACITY - start);
    table->set_global_epoch(epoch++);
    create_continuous_keys<K, S>(h_keys, h_scores, cur, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    // LRU/EpochLRU: nullptr scores; LFU/EpochLFU/Custom: pass scores
    if (Strategy == EvictStrategy::kLru ||
        Strategy == EvictStrategy::kEpochLru) {
      table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    } else {
      // Reset scores to 1 for each batch
      for (size_t j = 0; j < cur; j++) h_scores[j] = 1;
      CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << " LF=" << std::fixed << std::setprecision(4) << real_lf;

  // Zipfian steady-state insertions (10× capacity)
  std::cerr << " steady-state..." << std::flush;
  ZipfianGenerator zipf(key_range, zipf_alpha, 42);
  size_t total_inserts = STEADY_STATE_ROUNDS * INIT_CAPACITY;
  size_t inserted = 0;
  auto timer = benchmark::Timer<double>();
  timer.start();
  while (inserted < total_inserts) {
    size_t cur = std::min(BATCH_SIZE, total_inserts - inserted);
    table->set_global_epoch(epoch++);
    zipf.fill(h_keys, cur);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    if (Strategy == EvictStrategy::kLru ||
        Strategy == EvictStrategy::kEpochLru) {
      table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    } else {
      for (size_t j = 0; j < cur; j++) h_scores[j] = 1;
      CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    inserted += cur;
  }
  timer.end();
  float insert_tp =
      total_inserts / timer.getResult() / (1024.0 * 1024.0 * 1024.0);

  // Measure find hit rate with Zipfian queries
  std::cerr << " hit-rate..." << std::flush;
  ZipfianGenerator zipf_find(key_range, zipf_alpha, 12345);
  bool* h_found;
  CUDA_CHECK(cudaMallocHost(&h_found, BATCH_SIZE * sizeof(bool)));

  size_t total_found = 0;
  size_t total_queries = 0;
  float find_tp_sum = 0;

  for (int r = 0; r < FIND_ROUNDS; r++) {
    zipf_find.fill(h_keys, BATCH_SIZE);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));
    auto ft = benchmark::Timer<double>();
    ft.start();
    table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ft.end();
    find_tp_sum += BATCH_SIZE / ft.getResult() / (1024.0 * 1024.0 * 1024.0);
    CUDA_CHECK(cudaMemcpy(h_found, d_found, BATCH_SIZE * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < BATCH_SIZE; j++)
      if (h_found[j]) total_found++;
    total_queries += BATCH_SIZE;
  }

  float hit_rate = static_cast<float>(total_found) / total_queries;
  float find_tp = find_tp_sum / FIND_ROUNDS;

  std::cout << name << "," << std::fixed << std::setprecision(6) << insert_tp
            << "," << find_tp << "," << std::setprecision(4) << hit_rate
            << std::endl;
  std::cerr << " insert=" << std::fixed << std::setprecision(3) << insert_tp
            << " find=" << find_tp << " hit=" << std::setprecision(1)
            << (hit_rate * 100) << "%" << std::endl;

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
}

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E5: Score Strategy Comparison" << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", LF=1.0, Zipfian α=0.99" << std::endl;

  // Accept strategy name as argument; run all if no argument given
  std::string target = (argc > 1) ? argv[1] : "all";

  // Print header only when running all or when explicitly requested
  if (target == "all" || target == "header") {
    std::cout << "strategy,insert_throughput_bkvs,find_throughput_bkvs,hit_rate"
              << std::endl;
    if (target == "header") return 0;
  }

  uint64_t key_range = 10UL * INIT_CAPACITY;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  try {
    if (target == "all" || target == "LRU") {
      std::cerr << "LRU:" << std::endl;
      run_strategy<EvictStrategy::kLru>("LRU", key_range, 0.99, stream);
    }
    if (target == "all" || target == "LFU") {
      std::cerr << "LFU:" << std::endl;
      run_strategy<EvictStrategy::kLfu>("LFU", key_range, 0.99, stream);
    }
    if (target == "all" || target == "EpochLRU") {
      std::cerr << "EpochLRU:" << std::endl;
      run_strategy<EvictStrategy::kEpochLru>("EpochLRU", key_range, 0.99,
                                             stream);
    }
    if (target == "all" || target == "EpochLFU") {
      std::cerr << "EpochLFU:" << std::endl;
      run_strategy<EvictStrategy::kEpochLfu>("EpochLFU", key_range, 0.99,
                                             stream);
    }
    if (target == "all" || target == "Custom") {
      std::cerr << "Custom:" << std::endl;
      run_strategy<EvictStrategy::kCustomized>("Custom", key_range, 0.99,
                                               stream);
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
