/*
 * Bucket Size Sweep Benchmark
 *
 * Config B: dim=32, capacity=128M, Pure HBM, LRU
 * λ = 0.50
 * Bucket sizes: {32, 64, 128, 256}
 * Operations: find, insert_or_assign
 * Runs: 3 per (bucket_size, api) pair — reports median.
 * Shares one pre-populated table across runs for the same bucket_size.
 *
 * Output: CSV to stdout for easy parsing.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
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
static constexpr int NUM_RUNS = 3;
static constexpr int NUM_WARMUP = 5;
static constexpr float LOAD_FACTOR = 0.50f;
static constexpr float EPSILON = 0.001f;

using HKVTable = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

void pre_populate(std::shared_ptr<HKVTable>& table, float load_factor,
                  cudaStream_t stream, K& key_end) {
  const size_t fill_batch = 1024 * 1024UL;
  K* h_keys;
  S* h_scores;
  K* d_keys;
  S* d_scores;
  V* d_vectors;

  CUDA_CHECK(cudaMallocHost(&h_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, fill_batch * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, fill_batch * sizeof(V) * DIM));

  size_t target = static_cast<size_t>(INIT_CAPACITY * load_factor);
  K start = 0;
  int epoch = 0;
  while (start < target) {
    size_t cur = std::min(fill_batch, target - start);
    table->set_global_epoch(epoch++);
    create_continuous_keys<K, S>(h_keys, h_scores, cur, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  // Fine-tune
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (load_factor - real_lf > EPSILON) {
    auto append = static_cast<int64_t>((load_factor - real_lf) * INIT_CAPACITY);
    if (append <= 0) break;
    append = std::min(static_cast<int64_t>(fill_batch), append);
    create_continuous_keys<K, S>(h_keys, h_scores, append, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, append * sizeof(K), cudaMemcpyHostToDevice));
    table->insert_or_assign(append, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += append;
    real_lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  key_end = start;

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
}

// Measure one (api) run. Returns throughput in B-KV/s.
float measure_once(std::shared_ptr<HKVTable>& table, API_Select api,
                   cudaStream_t stream, K key_end, int run_idx) {
  K* h_keys;
  S* h_scores;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));

  K* d_keys;
  V* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  if (api == API_Select::find) {
    // Use existing keys for 100% hit rate
    K offset = static_cast<K>(run_idx * BATCH_SIZE) % key_end;
    create_continuous_keys<K, S>(h_keys, h_scores, BATCH_SIZE, offset);
  } else {
    // For insert_or_assign: use existing keys (overwrite, no growth at LF=0.50)
    K offset = static_cast<K>(run_idx * BATCH_SIZE) % key_end;
    create_continuous_keys<K, S>(h_keys, h_scores, BATCH_SIZE, offset);
  }
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));

  // Warmup
  for (int w = 0; w < NUM_WARMUP; w++) {
    if (api == API_Select::find) {
      table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
    } else {
      table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timed run (use CUDA events for precision)
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));
  if (api == API_Select::find) {
    table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
  } else {
    table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
  }
  CUDA_CHECK(cudaEventRecord(ev_stop, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

  float throughput =
      BATCH_SIZE / (elapsed_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));

  return throughput;
}

int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", batch=" << BATCH_SIZE << ", LF=" << LOAD_FACTOR << std::endl;
  std::cerr << "Runs per point: " << NUM_RUNS << ", Warmup: " << NUM_WARMUP
            << std::endl;

  std::vector<size_t> bucket_sizes = {32, 64, 128, 256};
  std::vector<API_Select> apis = {API_Select::find,
                                  API_Select::insert_or_assign};
  std::vector<std::string> api_names = {"find", "insert_or_assign"};

  // CSV header
  std::cout << "api,bucket_size,load_factor,run,throughput_bkvs" << std::endl;

  try {
    for (size_t bucket_size : bucket_sizes) {
      // Validate: capacity / bucket_size >= 2 (dual-bucket requirement)
      if (INIT_CAPACITY / bucket_size < 2) {
        std::cerr << "SKIP bucket_size=" << bucket_size
                  << " (capacity/bucket_size < 2)" << std::endl;
        continue;
      }

      // Create and pre-populate ONE table per bucket_size
      HashTableOptions options;
      options.init_capacity = INIT_CAPACITY;
      options.max_capacity = INIT_CAPACITY;
      options.dim = DIM;
      options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
      options.max_bucket_size = bucket_size;

      auto table = std::make_shared<HKVTable>();
      table->init(options);

      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      std::cerr << "bucket=" << bucket_size
                << " pre-populating to LF=" << LOAD_FACTOR << "..."
                << std::flush;

      K key_end = 0;
      pre_populate(table, LOAD_FACTOR, stream, key_end);

      float real_lf = table->load_factor(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::cerr << " actual LF=" << std::fixed << std::setprecision(4)
                << real_lf << std::endl;

      // Run all (api, run) combos on this table
      for (size_t ai = 0; ai < apis.size(); ai++) {
        auto api = apis[ai];
        const auto& api_name = api_names[ai];
        std::vector<float> results;

        for (int run = 0; run < NUM_RUNS; run++) {
          std::cerr << "  " << api_name << " bucket=" << bucket_size
                    << " run=" << (run + 1) << "/" << NUM_RUNS
                    << " measuring..." << std::flush;

          float tp = measure_once(table, api, stream, key_end, run);
          results.push_back(tp);

          std::cout << api_name << "," << bucket_size << "," << std::fixed
                    << std::setprecision(2) << LOAD_FACTOR << "," << (run + 1)
                    << "," << std::setprecision(6) << tp << std::endl;

          std::cerr << " " << std::fixed << std::setprecision(3) << tp
                    << " B-KV/s" << std::endl;
        }

        // Print median
        std::sort(results.begin(), results.end());
        float median = results[NUM_RUNS / 2];
        std::cerr << "  => " << api_name << " bucket=" << bucket_size
                  << " median=" << std::setprecision(3) << median << " B-KV/s"
                  << std::endl;
      }

      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
