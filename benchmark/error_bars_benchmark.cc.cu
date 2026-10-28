/*
 * E3: Statistical Rigor — Error Bars Benchmark
 *
 * Config A/B/C at λ=0.50, 5 runs per (config, api) pair.
 * APIs: find, find*, insert_or_assign, insert_and_evict, assign, contains
 * Output: CSV with all individual runs for computing median/P25/P75.
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

static constexpr size_t BATCH_SIZE = 1024 * 1024UL;  // 1M
static constexpr float LOAD_FACTOR = 0.50f;
static constexpr int NUM_RUNS = 5;
static constexpr float EPSILON = 0.001f;

struct Config {
  const char* name;
  size_t dim;
  size_t capacity;
  size_t hbm_gb;
};

// Reuse test_one_api logic from the main benchmark but simplified.
float measure_api(size_t dim, size_t init_capacity, size_t hbm_gb,
                  API_Select api, cudaStream_t stream) {
  HashTableOptions options;
  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(hbm_gb);

  using Table = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;
  auto table = std::make_shared<Table>();
  table->init(options);

  K* h_keys;
  S* h_scores;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));

  K* d_keys;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  S* d_scores;
  K* d_evict_keys;
  S* d_evict_scores;

  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, BATCH_SIZE * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, BATCH_SIZE * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_evict_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_scores, BATCH_SIZE * sizeof(S)));

  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, BATCH_SIZE * sizeof(V) * dim));

  // Pre-populate to target LF
  size_t target_count = static_cast<size_t>(init_capacity * LOAD_FACTOR);
  K start = 0;
  int epoch = 0;
  while (start < target_count) {
    size_t cur = std::min(BATCH_SIZE, target_count - start);
    table->set_global_epoch(epoch++);
    create_continuous_keys<K, S>(h_keys, h_scores, cur, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->find_or_insert(cur, d_keys, d_vectors_ptr, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  // Fine-tune LF
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (LOAD_FACTOR - real_lf > EPSILON) {
    auto append = static_cast<int64_t>((LOAD_FACTOR - real_lf) * init_capacity);
    if (append <= 0) break;
    append = std::min(static_cast<int64_t>(BATCH_SIZE), append);
    create_continuous_keys<K, S>(h_keys, h_scores, append, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, append * sizeof(K), cudaMemcpyHostToDevice));
    table->insert_or_assign(append, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += append;
    real_lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Prepare keys for measurement (60% hit rate for realistic workload)
  create_keys_for_hitrate<K, S>(h_keys, h_scores, BATCH_SIZE, 0.6f,
                                Hit_Mode::last_insert, start, true);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, BATCH_SIZE * sizeof(S),
                        cudaMemcpyHostToDevice));

  // Warmup (3 iterations)
  for (int w = 0; w < 3; w++) {
    table->set_global_epoch(epoch++);
    switch (api) {
      case API_Select::find:
        table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
        break;
      case API_Select::find_ptr: {
        benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, BATCH_SIZE, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        table->find(BATCH_SIZE, d_keys, d_vectors_ptr, d_found, nullptr,
                    stream);
        break;
      }
      case API_Select::insert_or_assign:
        table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
        break;
      case API_Select::insert_and_evict:
        table->insert_and_evict(BATCH_SIZE, d_keys, d_vectors, nullptr,
                                d_evict_keys, d_def_val, d_evict_scores,
                                stream);
        break;
      case API_Select::assign:
        table->assign(BATCH_SIZE, d_keys, d_def_val, nullptr, stream);
        break;
      case API_Select::contains:
        table->contains(BATCH_SIZE, d_keys, d_found, stream);
        break;
      default:
        break;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timed run
  table->set_global_epoch(epoch++);
  auto timer = benchmark::Timer<double>();

  switch (api) {
    case API_Select::find:
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    case API_Select::find_ptr: {
      benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, BATCH_SIZE, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_vectors_ptr, d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_or_assign:
      timer.start();
      table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    case API_Select::insert_and_evict:
      timer.start();
      table->insert_and_evict(BATCH_SIZE, d_keys, d_vectors, nullptr,
                              d_evict_keys, d_def_val, d_evict_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    case API_Select::assign:
      timer.start();
      table->assign(BATCH_SIZE, d_keys, d_def_val, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    case API_Select::contains:
      timer.start();
      table->contains(BATCH_SIZE, d_keys, d_found, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    default:
      break;
  }

  float throughput =
      BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_scores));

  return throughput;
}

int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "LF=" << LOAD_FACTOR << ", batch=" << BATCH_SIZE
            << ", runs=" << NUM_RUNS << std::endl;

  std::vector<Config> configs = {
      {"A", 8, 128UL * 1024 * 1024, 4},
      {"B", 32, 128UL * 1024 * 1024, 16},
      {"C", 64, 64UL * 1024 * 1024, 16},
  };

  struct ApiInfo {
    API_Select api;
    const char* name;
  };
  std::vector<ApiInfo> apis = {
      {API_Select::find, "find"},
      {API_Select::find_ptr, "find_ptr"},
      {API_Select::insert_or_assign, "insert_or_assign"},
      {API_Select::insert_and_evict, "insert_and_evict"},
      {API_Select::assign, "assign"},
      {API_Select::contains, "contains"},
  };

  // CSV header
  std::cout << "config,dim,capacity,api,load_factor,run,throughput_bkvs"
            << std::endl;

  try {
    for (auto& cfg : configs) {
      for (auto& ai : apis) {
        for (int run = 0; run < NUM_RUNS; run++) {
          std::cerr << "  " << cfg.name << " " << ai.name
                    << " run=" << (run + 1) << "/" << NUM_RUNS << " ..."
                    << std::flush;

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));

          float tp =
              measure_api(cfg.dim, cfg.capacity, cfg.hbm_gb, ai.api, stream);

          std::cout << cfg.name << "," << cfg.dim << "," << cfg.capacity << ","
                    << ai.name << "," << std::fixed << std::setprecision(2)
                    << LOAD_FACTOR << "," << (run + 1) << ","
                    << std::setprecision(6) << tp << std::endl;

          std::cerr << " " << std::fixed << std::setprecision(3) << tp
                    << " B-KV/s" << std::endl;

          CUDA_CHECK(cudaStreamDestroy(stream));
        }
      }
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
