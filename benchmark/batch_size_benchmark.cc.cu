/*
 * E2: Batch Size Sensitivity Benchmark
 *
 * Config B: dim=32, capacity=128M, Pure HBM, LRU
 * λ = 1.0
 * Batch sizes: {1K, 10K, 100K, 500K, 1M, 2M, 5M, 10M}
 * Operations: find, insert_or_assign
 * Runs: 5 per (batch_size, api) pair
 *
 * Output: CSV to stdout for easy parsing.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
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
static constexpr float LOAD_FACTOR = 1.0f;
static constexpr int NUM_RUNS = 5;
static constexpr int NUM_WARMUP = 3;
static constexpr float EPSILON = 0.001f;

// Pre-populate to target load factor using large batch (1M).
void pre_populate(
    std::shared_ptr<HashTable<K, V, S, EvictStrategy::kLru, Sm80>>& table,
    cudaStream_t stream) {
  const size_t fill_batch = 1024 * 1024UL;
  K* h_keys;
  S* h_scores;
  V* h_vectors;
  K* d_keys;
  S* d_scores;
  V* d_vectors;

  CUDA_CHECK(cudaMallocHost(&h_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, fill_batch * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, fill_batch * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, fill_batch * sizeof(V) * DIM));

  size_t target_count = static_cast<size_t>(INIT_CAPACITY * LOAD_FACTOR);
  K start = 0;
  int epoch = 0;

  while (start < target_count) {
    size_t cur = std::min(fill_batch, target_count - start);
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
  while (LOAD_FACTOR - real_lf > EPSILON) {
    auto append = static_cast<int64_t>((LOAD_FACTOR - real_lf) * INIT_CAPACITY);
    if (append <= 0) break;
    append = std::min(static_cast<int64_t>(fill_batch), append);
    create_continuous_keys<K, S>(h_keys, h_scores, append, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, append * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, append * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(append, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += append;
    real_lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_vectors));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
}

// Measure one (api, batch_size) combo. Returns throughput in B-KV/s.
float measure_once(
    std::shared_ptr<HashTable<K, V, S, EvictStrategy::kLru, Sm80>>& table,
    API_Select api, size_t batch_size, cudaStream_t stream, K key_start) {
  K* h_keys;
  S* h_scores;
  CUDA_CHECK(cudaMallocHost(&h_keys, batch_size * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, batch_size * sizeof(S)));

  K* d_keys;
  S* d_scores;
  V* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, batch_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, batch_size * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, batch_size * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, batch_size * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, batch_size * sizeof(V) * DIM));

  // Generate keys: for find, use existing keys (hitrate=1.0);
  // for insert_or_assign at LF=1.0, use new keys to force eviction.
  if (api == API_Select::find) {
    // Use keys within existing range for 100% hit
    create_continuous_keys<K, S>(h_keys, h_scores, batch_size, 0);
  } else {
    // Use keys beyond current range to trigger eviction path
    create_continuous_keys<K, S>(h_keys, h_scores, batch_size, key_start);
  }
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, batch_size * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, batch_size * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Warmup
  for (int w = 0; w < NUM_WARMUP; w++) {
    if (api == API_Select::find) {
      table->find(batch_size, d_keys, d_vectors, d_found, nullptr, stream);
    } else {
      table->insert_or_assign(batch_size, d_keys, d_vectors, nullptr, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timed run
  auto timer = benchmark::Timer<double>();
  if (api == API_Select::find) {
    timer.start();
    table->find(batch_size, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();
  } else {
    timer.start();
    table->insert_or_assign(batch_size, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();
  }

  float throughput =
      batch_size / timer.getResult() / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));

  return throughput;
}

int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", HBM=" << HBM_GB << "GB, LF=" << LOAD_FACTOR << std::endl;
  std::cerr << "Runs per point: " << NUM_RUNS << ", Warmup: " << NUM_WARMUP
            << std::endl;

  std::vector<size_t> batch_sizes = {
      1024,             // 1K
      10 * 1024,        // 10K
      100 * 1024,       // 100K
      500 * 1024,       // 500K
      1024 * 1024,      // 1M
      2048 * 1024,      // 2M
      5 * 1024 * 1024,  // 5M
      10 * 1024 * 1024  // 10M
  };

  std::vector<API_Select> apis = {API_Select::find,
                                  API_Select::insert_or_assign};
  std::vector<std::string> api_names = {"find", "insert_or_assign"};

  // CSV header
  std::cout << "api,batch_size,run,throughput_bkvs" << std::endl;

  try {
    for (size_t ai = 0; ai < apis.size(); ai++) {
      auto api = apis[ai];
      const auto& api_name = api_names[ai];

      for (auto batch_size : batch_sizes) {
        std::vector<float> results;

        for (int run = 0; run < NUM_RUNS; run++) {
          // Fresh table for each run
          HashTableOptions options;
          options.init_capacity = INIT_CAPACITY;
          options.max_capacity = INIT_CAPACITY;
          options.dim = DIM;
          options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);

          using Table = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;
          auto table = std::make_shared<Table>();
          table->init(options);

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));

          std::cerr << "  " << api_name << " batch=" << batch_size
                    << " run=" << (run + 1) << "/" << NUM_RUNS
                    << " pre-populating..." << std::flush;

          pre_populate(table, stream);

          std::cerr << " LF=" << std::fixed << std::setprecision(4)
                    << table->load_factor(stream) << " measuring..."
                    << std::flush;

          K key_start = static_cast<K>(INIT_CAPACITY) + run * batch_size;
          float tp = measure_once(table, api, batch_size, stream, key_start);
          results.push_back(tp);

          std::cout << api_name << "," << batch_size << "," << (run + 1) << ","
                    << std::fixed << std::setprecision(6) << tp << std::endl;

          std::cerr << " " << std::fixed << std::setprecision(3) << tp
                    << " B-KV/s" << std::endl;

          CUDA_CHECK(cudaStreamDestroy(stream));
        }

        // Print median as comment
        std::sort(results.begin(), results.end());
        float median = results[NUM_RUNS / 2];
        std::cerr << "  => " << api_name << " batch=" << batch_size
                  << " median=" << std::fixed << std::setprecision(3) << median
                  << " B-KV/s" << std::endl;
      }
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
