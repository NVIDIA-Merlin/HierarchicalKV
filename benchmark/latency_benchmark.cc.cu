/*
 * E4: Latency Distribution Benchmark (P50/P95/P99)
 *
 * Config B: dim=32, capacity=128M, Pure HBM, LRU
 * λ ∈ {0.50, 1.00}
 * Operations: find, insert_or_assign, assign
 * 1000 batches per (api, λ) combo, CUDA event timing per batch.
 *
 * Output: CSV with per-batch latency in milliseconds.
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
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;  // 1M
static constexpr int NUM_BATCHES = 1000;
static constexpr int NUM_WARMUP = 10;
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
  V** d_vptrs;
  bool* d_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, fill_batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, fill_batch * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, fill_batch * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_vptrs, fill_batch * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, fill_batch * sizeof(bool)));
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
    table->find_or_insert(cur, d_keys, d_vptrs, d_found, nullptr, stream);
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
  CUDA_CHECK(cudaFree(d_vptrs));
  CUDA_CHECK(cudaFree(d_found));
}

void run_latency_test(std::shared_ptr<HKVTable>& table, API_Select api,
                      const char* api_name, float load_factor, K key_end,
                      cudaStream_t stream) {
  K* h_keys;
  S* h_scores;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));

  K* d_keys;
  V* d_vectors;
  V* d_def_val;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_def_val, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_def_val, 2, BATCH_SIZE * sizeof(V) * DIM));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Prepare keys: for find/contains, use existing keys; for write ops, use
  // new keys at LF=1.0 (eviction path) or existing at LF=0.50.
  K insert_key_base = key_end;

  auto prepare_keys = [&](int batch_idx) {
    if (api == API_Select::find || api == API_Select::assign) {
      // Use keys within existing range
      K offset = static_cast<K>(batch_idx * BATCH_SIZE) % key_end;
      create_continuous_keys<K, S>(h_keys, h_scores, BATCH_SIZE, offset);
    } else {
      // insert_or_assign: use new keys to trigger eviction at LF=1.0
      K base = insert_key_base + static_cast<K>(batch_idx) * BATCH_SIZE;
      create_continuous_keys<K, S>(h_keys, h_scores, BATCH_SIZE, base);
    }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                          cudaMemcpyHostToDevice));
  };

  // Warmup
  for (int w = 0; w < NUM_WARMUP; w++) {
    prepare_keys(w);
    table->set_global_epoch(w);
    switch (api) {
      case API_Select::find:
        table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
        break;
      case API_Select::insert_or_assign:
        table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
        break;
      case API_Select::assign:
        table->assign(BATCH_SIZE, d_keys, d_def_val, nullptr, stream);
        break;
      default:
        break;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timed batches
  std::vector<float> latencies;
  latencies.reserve(NUM_BATCHES);

  for (int i = 0; i < NUM_BATCHES; i++) {
    prepare_keys(NUM_WARMUP + i);
    table->set_global_epoch(NUM_WARMUP + i);

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    switch (api) {
      case API_Select::find:
        table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
        break;
      case API_Select::insert_or_assign:
        table->insert_or_assign(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
        break;
      case API_Select::assign:
        table->assign(BATCH_SIZE, d_keys, d_def_val, nullptr, stream);
        break;
      default:
        break;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    latencies.push_back(elapsed_ms);

    // CSV output: one line per batch
    std::cout << api_name << "," << std::fixed << std::setprecision(2)
              << load_factor << "," << (i + 1) << "," << std::setprecision(4)
              << elapsed_ms << std::endl;
  }

  // Summary to stderr
  std::sort(latencies.begin(), latencies.end());
  int n = latencies.size();
  float p50 = latencies[n / 2];
  float p95 = latencies[static_cast<int>(n * 0.95)];
  float p99 = latencies[static_cast<int>(n * 0.99)];
  float p999 = latencies[std::min(static_cast<int>(n * 0.999), n - 1)];
  std::cerr << "  " << api_name << " LF=" << std::fixed << std::setprecision(2)
            << load_factor << ": P50=" << std::setprecision(3) << p50
            << "ms P95=" << p95 << "ms P99=" << p99 << "ms P99.9=" << p999
            << "ms" << std::endl;

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_found));
}

int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", batch=" << BATCH_SIZE << std::endl;
  std::cerr << "Batches per combo: " << NUM_BATCHES
            << ", Warmup: " << NUM_WARMUP << std::endl;

  struct ApiInfo {
    API_Select api;
    const char* name;
  };
  std::vector<ApiInfo> apis = {
      {API_Select::find, "find"},
      {API_Select::insert_or_assign, "insert_or_assign"},
      {API_Select::assign, "assign"},
  };
  std::vector<float> load_factors = {0.50f, 1.00f};

  // CSV header
  std::cout << "api,load_factor,batch_idx,latency_ms" << std::endl;

  try {
    for (float lf : load_factors) {
      for (auto& ai : apis) {
        std::cerr << "Running " << ai.name << " at LF=" << std::fixed
                  << std::setprecision(2) << lf << " ..." << std::endl;

        // Fresh table for each (api, lf) combo
        HashTableOptions options;
        options.init_capacity = INIT_CAPACITY;
        options.max_capacity = INIT_CAPACITY;
        options.dim = DIM;
        options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);

        auto table = std::make_shared<HKVTable>();
        table->init(options);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::cerr << "  Pre-populating to LF=" << lf << "..." << std::endl;
        K key_end = 0;
        pre_populate(table, lf, stream, key_end);

        float real_lf = table->load_factor(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cerr << "  Actual LF=" << std::fixed << std::setprecision(4)
                  << real_lf << std::endl;

        run_latency_test(table, ai.api, ai.name, lf, key_end, stream);

        CUDA_CHECK(cudaStreamDestroy(stream));
      }
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
