/*
 * E8: HKV Baseline Benchmark
 *
 * Config B: dim=32, capacity=128M, HBM=16GB, EvictStrategy::kLru
 * Measures insert_or_assign and find throughput at load factors {0.50, 0.75}
 * using BATCH_SIZE=1M keys, 5 runs with 3 warmup iterations.
 *
 * Output: CSV  library,operation,load_factor,run,throughput_bkvs
 * (library = "HKV")
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;  // 1M
static constexpr int WARMUP = 3;
static constexpr int RUNS = 5;
static constexpr float EPSILON = 0.001f;

/* ─── Pre-populate Table to Target LF ─── */

static K prepopulate(
    std::shared_ptr<HashTable<K, V, S, EvictStrategy::kLru, Sm80>>& table,
    size_t target_count, cudaStream_t stream) {
  K* h_keys;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));

  K* d_keys;
  V* d_vectors;
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  K start = 0;
  while (start < target_count) {
    size_t cur = std::min(BATCH_SIZE, target_count - start);
    for (size_t i = 0; i < cur; i++) h_keys[i] = start + i;
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));

  return start;
}

/* ─── Insert Benchmark ─── */

static void bench_insert(float target_lf, cudaStream_t stream) {
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  const size_t prefill_n = target_n - BATCH_SIZE;

  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create fresh table
    HashTableOptions options;
    options.init_capacity = INIT_CAPACITY;
    options.max_capacity = INIT_CAPACITY;
    options.dim = DIM;
    options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);

    using Table = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;
    auto table = std::make_shared<Table>();
    table->init(options);

    // Pre-populate to (target_n - BATCH_SIZE)
    K start = prepopulate(table, prefill_n, stream);

    // Prepare the timed batch
    K* h_keys;
    CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
    for (size_t i = 0; i < BATCH_SIZE; i++) h_keys[i] = start + i;

    K* d_keys;
    V* d_vectors;
    CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
    CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));

    // Timed insert_or_assign (kLru: no scores, uses internal timestamps)
    auto timer = benchmark::KernelTimer<double>();
    timer.start();
    table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();

    if (run >= WARMUP) {
      double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
      std::cout << "HKV,insert," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  insert run " << (run - WARMUP + 1) << ": " << std::fixed
                << std::setprecision(3) << tp << " B-KV/s" << std::endl;
    }

    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_vectors));
  }
}

/* ─── Find Benchmark ─── */

static void bench_find(float target_lf, cudaStream_t stream) {
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);

  // Create and fully populate table (once per load factor, reused across runs)
  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);

  using Table = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;
  auto table = std::make_shared<Table>();
  table->init(options);

  K start = prepopulate(table, target_n, stream);

  // Prepare query keys (100% hit): randomly sample from [0, target_n)
  K* h_keys;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<K> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_keys[i] = dist(rng);
    }
  }

  K* d_keys;
  V* d_vectors;
  bool* d_found;
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));

  for (int run = 0; run < WARMUP + RUNS; run++) {
    auto timer = benchmark::KernelTimer<double>();
    timer.start();
    table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();

    if (run >= WARMUP) {
      double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
      std::cout << "HKV,find," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  find run " << (run - WARMUP + 1) << ": " << std::fixed
                << std::setprecision(3) << tp << " B-KV/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
}

/* ─── Main ─── */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E16: HKV LF Degradation (kThroughput, eviction-enabled)"
            << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", HBM=" << HBM_GB << "GB, kLru" << std::endl;

  std::cout << "library,operation,load_factor,run,throughput_bkvs" << std::endl;

  std::vector<float> load_factors;
  if (argc > 1) {
    // Single LF mode: ./e8_hkv_baseline 0.85
    load_factors.push_back(std::stof(argv[1]));
  } else {
    // E16 default: full sweep (no 0.10)
    load_factors = {0.25f, 0.50f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f, 1.00f};
  }
  for (float lf : load_factors) {
    std::cerr << "--- LF=" << std::fixed << std::setprecision(2) << lf << " ---"
              << std::endl;
    bench_insert(lf, 0);
    bench_find(lf, 0);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
