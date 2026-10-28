/*
 * E9: Eviction Comparison Benchmark
 *
 * Compares HKV's built-in eviction (insert_and_evict) against a simulated
 * "external eviction" approach representing how cuCollections/other GPU hash
 * tables would handle eviction when the table fills up.
 *
 * Benchmark 1 (hkv_builtin):
 *   - Config B: dim=32, capacity=128M, pure HBM, kCustomized, LF=1.0
 *   - Pre-populate to capacity (score = key)
 *   - Zipfian (alpha=0.99) steady-state: insert_and_evict, 1M batches
 *   - Total insertions: 5x capacity
 *
 * Benchmark 2 (external_sweep):
 *   - Same table but simulating external eviction approach
 *   - Insert until 90% full, then sweep: export all -> sort by score ->
 *     erase bottom 20% -> continue inserting
 *   - Total insertions: 5x capacity
 *
 * Output CSV (stdout):
 *   method,total_ops,wall_time_s,throughput_bkvs,p50_ms,p95_ms,p99_ms
 *
 * Progress (stderr): reported every ~10% of workload.
 */

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using namespace nv::merlin;
using namespace benchmark;

/* ─── Config B constants ─── */
static constexpr size_t DIM = 32;
static constexpr size_t INIT_CAPACITY = 128UL * 1024 * 1024;  // 128M slots
static constexpr size_t HBM_GB = 16;
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;       // 1M keys per batch
static constexpr size_t TOTAL_OPS = 5UL * INIT_CAPACITY;  // 5x capacity
static constexpr double ZIPF_ALPHA = 0.99;
static constexpr uint64_t KEY_RANGE = 10UL * INIT_CAPACITY;

/* ─── Zipfian Generator (YCSB-style, copied from score_strategy_benchmark) ───
 */

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

/* ─── Helpers ─── */

struct LatencyStats {
  float p50;
  float p95;
  float p99;
};

static LatencyStats compute_percentiles(std::vector<float>& latencies) {
  std::sort(latencies.begin(), latencies.end());
  size_t n = latencies.size();
  LatencyStats s;
  s.p50 = latencies[n / 2];
  s.p95 = latencies[static_cast<size_t>(n * 0.95)];
  s.p99 = latencies[static_cast<size_t>(n * 0.99)];
  return s;
}

/* ─── Benchmark 1: HKV Built-in Eviction ─── */

void benchmark_hkv_builtin_eviction(cudaStream_t stream) {
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized, Sm80>;

  std::cerr << "[hkv_builtin] Initializing table..." << std::endl;

  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
  // max_load_factor defaults to 0.5 but insert_and_evict handles eviction
  // internally when the table is full. We set it high to avoid spurious rehash.
  options.max_load_factor = 1.0f;

  auto table = std::make_shared<Table>();
  table->init(options);

  /* Allocate host and device buffers */
  K* h_keys;
  S* h_scores;
  K* d_keys;
  S* d_scores;
  V* d_vectors;
  K* d_evicted_keys;
  V* d_evicted_values;
  S* d_evicted_scores;

  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_evicted_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evicted_values, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_evicted_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  /* Pre-populate to full capacity with sequential keys, score = key */
  std::cerr << "[hkv_builtin] Pre-populating to LF=1.0..." << std::flush;
  K start = 0;
  while (start < INIT_CAPACITY) {
    size_t cur = std::min(BATCH_SIZE, INIT_CAPACITY - start);
    for (size_t j = 0; j < cur; j++) {
      h_keys[j] = start + static_cast<K>(j);
      h_scores[j] = h_keys[j];  // score = key value
    }
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << " LF=" << std::fixed << std::setprecision(4) << real_lf
            << std::endl;

  /* Zipfian steady-state: insert_and_evict */
  std::cerr << "[hkv_builtin] Running " << TOTAL_OPS << " insert_and_evict ops"
            << " (" << (TOTAL_OPS / BATCH_SIZE) << " batches)..." << std::endl;

  ZipfianGenerator zipf(KEY_RANGE, ZIPF_ALPHA, 42);
  size_t total_batches = (TOTAL_OPS + BATCH_SIZE - 1) / BATCH_SIZE;
  size_t progress_interval = total_batches / 10;
  if (progress_interval == 0) progress_interval = 1;

  std::vector<float> batch_latencies;
  batch_latencies.reserve(total_batches);

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  auto wall_start = std::chrono::steady_clock::now();

  size_t ops_done = 0;
  for (size_t b = 0; b < total_batches; b++) {
    size_t cur = std::min(BATCH_SIZE, TOTAL_OPS - ops_done);

    // Generate Zipfian keys; score = key (higher key => higher score => hotter)
    zipf.fill(h_keys, cur);
    for (size_t j = 0; j < cur; j++) {
      h_scores[j] = h_keys[j];
    }
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    table->insert_and_evict(cur, d_keys, d_vectors, d_scores, d_evicted_keys,
                            d_evicted_values, d_evicted_scores, stream);

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    batch_latencies.push_back(elapsed_ms);

    ops_done += cur;

    if ((b + 1) % progress_interval == 0 || b + 1 == total_batches) {
      float pct = static_cast<float>(b + 1) / total_batches * 100.0f;
      std::cerr << "  [hkv_builtin] " << std::fixed << std::setprecision(0)
                << pct << "% (" << ops_done << "/" << TOTAL_OPS << ")"
                << std::endl;
    }
  }

  auto wall_end = std::chrono::steady_clock::now();
  double wall_time_s =
      std::chrono::duration<double>(wall_end - wall_start).count();
  double throughput_bkvs =
      static_cast<double>(ops_done) / wall_time_s / (1024.0 * 1024.0 * 1024.0);

  auto stats = compute_percentiles(batch_latencies);

  std::cout << "hkv_builtin," << ops_done << "," << std::fixed
            << std::setprecision(3) << wall_time_s << ","
            << std::setprecision(6) << throughput_bkvs << ","
            << std::setprecision(3) << stats.p50 << "," << stats.p95 << ","
            << stats.p99 << std::endl;

  std::cerr << "[hkv_builtin] Done: wall=" << std::fixed << std::setprecision(3)
            << wall_time_s << "s throughput=" << std::setprecision(6)
            << throughput_bkvs << " B-KV/s P50=" << std::setprecision(3)
            << stats.p50 << "ms P95=" << stats.p95 << "ms P99=" << stats.p99
            << "ms" << std::endl;

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_evicted_keys));
  CUDA_CHECK(cudaFree(d_evicted_values));
  CUDA_CHECK(cudaFree(d_evicted_scores));
}

/* ─── Benchmark 2: Simulated External Eviction ─── */

void benchmark_external_eviction(cudaStream_t stream) {
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized, Sm80>;

  std::cerr << "[external_sweep] Initializing table..." << std::endl;

  /*
   * Simulate what cuCollections/other GPU hash tables must do:
   * They have no built-in eviction, so when the table approaches capacity
   * the application must:
   *   1. Export all entries
   *   2. Sort by score
   *   3. Delete the bottom 20% (lowest-score entries)
   *   4. Resume inserting
   *
   * We use HKV itself but only call insert_or_assign (no eviction) and
   * manually orchestrate the sweep cycle.
   */

  // Use a large enough capacity so that at LF=0.90 we hold the data.
  // We set max_load_factor=0.95 (slightly above trigger) to avoid auto-rehash.
  // We manually trigger the sweep when size approaches 90% of capacity.
  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
  options.max_load_factor = 1.0f;  // Prevent auto-rehash; we manage eviction

  auto table = std::make_shared<Table>();
  table->init(options);

  const size_t FILL_THRESHOLD =
      static_cast<size_t>(INIT_CAPACITY * 0.90);  // 90% capacity
  const size_t EVICT_COUNT =
      static_cast<size_t>(INIT_CAPACITY * 0.20);  // Delete bottom 20%

  /* Allocate host and device buffers */
  K* h_keys;
  S* h_scores;
  K* d_keys;
  S* d_scores;
  V* d_vectors;

  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  /* Export/sort/erase buffers -- sized for full capacity */
  K* d_export_keys;
  V* d_export_values;
  S* d_export_scores;
  CUDA_CHECK(cudaMalloc(&d_export_keys, INIT_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_export_values, INIT_CAPACITY * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_export_scores, INIT_CAPACITY * sizeof(S)));

  /* Pre-populate to ~90% capacity with sequential keys, score = key */
  std::cerr << "[external_sweep] Pre-populating to ~90% capacity..."
            << std::flush;
  K start = 0;
  while (start < FILL_THRESHOLD) {
    size_t cur = std::min(BATCH_SIZE, FILL_THRESHOLD - start);
    for (size_t j = 0; j < cur; j++) {
      h_keys[j] = start + static_cast<K>(j);
      h_scores[j] = h_keys[j];
    }
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << " LF=" << std::fixed << std::setprecision(4) << real_lf
            << std::endl;

  /* Zipfian workload with external sweep */
  std::cerr << "[external_sweep] Running " << TOTAL_OPS
            << " insert_or_assign ops"
            << " (" << (TOTAL_OPS / BATCH_SIZE) << " batches) with sweeps..."
            << std::endl;

  ZipfianGenerator zipf(KEY_RANGE, ZIPF_ALPHA, 42);  // Same seed as builtin
  size_t total_batches = (TOTAL_OPS + BATCH_SIZE - 1) / BATCH_SIZE;
  size_t progress_interval = total_batches / 10;
  if (progress_interval == 0) progress_interval = 1;

  std::vector<float> batch_latencies;
  batch_latencies.reserve(total_batches);

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  int sweep_count = 0;
  auto wall_start = std::chrono::steady_clock::now();

  size_t ops_done = 0;
  for (size_t b = 0; b < total_batches; b++) {
    size_t cur = std::min(BATCH_SIZE, TOTAL_OPS - ops_done);

    // Check if we need to sweep before inserting
    size_t current_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float batch_time_ms = 0.0f;

    if (current_size + cur > FILL_THRESHOLD) {
      /*
       * SWEEP: export all -> sort by score -> erase bottom 20%
       * The entire sweep is timed as part of this batch's latency,
       * which is the key insight: external eviction creates huge
       * latency spikes.
       */
      CUDA_CHECK(cudaEventRecord(ev_start, stream));

      // Step 1: Export all entries
      size_t table_sz = current_size;
      size_t exported = 0;
      size_t export_batch_size =
          std::min(BATCH_SIZE, static_cast<size_t>(INIT_CAPACITY));
      size_t offset = 0;
      // Export in chunks using export_batch
      while (offset < table->capacity()) {
        size_t chunk = std::min(export_batch_size, table->capacity() - offset);
        size_t got =
            table->export_batch(chunk, offset, d_export_keys + exported,
                                d_export_values + exported * DIM,
                                d_export_scores + exported, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        exported += got;
        offset += chunk;
      }

      // Step 2: Sort by score using thrust (ascending: lowest scores first)
      // We need to sort keys alongside scores to know which keys to erase.
      // Use thrust with raw pointers.
      thrust::device_ptr<S> t_scores(d_export_scores);
      thrust::device_ptr<K> t_keys(d_export_keys);
      // Sort by score ascending (lowest score = eviction candidates first)
      thrust::sort_by_key(t_scores, t_scores + exported, t_keys);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Step 3: Erase bottom 20% (the first EVICT_COUNT entries after sorting)
      size_t to_erase = std::min(EVICT_COUNT, exported);
      // Erase in batches
      size_t erased = 0;
      while (erased < to_erase) {
        size_t erase_batch = std::min(BATCH_SIZE, to_erase - erased);
        table->erase(erase_batch, d_export_keys + erased, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        erased += erase_batch;
      }

      // Now insert the current batch
      zipf.fill(h_keys, cur);
      for (size_t j = 0; j < cur; j++) {
        h_scores[j] = h_keys[j];
      }
      CUDA_CHECK(
          cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);

      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      CUDA_CHECK(cudaEventElapsedTime(&batch_time_ms, ev_start, ev_stop));

      sweep_count++;
      std::cerr << "  [external_sweep] Sweep #" << sweep_count << " at batch "
                << (b + 1) << ": exported=" << exported
                << " erased=" << to_erase << " sweep_time=" << std::fixed
                << std::setprecision(1) << batch_time_ms << "ms" << std::endl;
    } else {
      // Normal insert (no sweep needed)
      zipf.fill(h_keys, cur);
      for (size_t j = 0; j < cur; j++) {
        h_scores[j] = h_keys[j];
      }
      CUDA_CHECK(
          cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_scores, h_scores, cur * sizeof(S),
                            cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaEventRecord(ev_start, stream));
      table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      CUDA_CHECK(cudaEventElapsedTime(&batch_time_ms, ev_start, ev_stop));
    }

    batch_latencies.push_back(batch_time_ms);
    ops_done += cur;

    if ((b + 1) % progress_interval == 0 || b + 1 == total_batches) {
      float pct = static_cast<float>(b + 1) / total_batches * 100.0f;
      std::cerr << "  [external_sweep] " << std::fixed << std::setprecision(0)
                << pct << "% (" << ops_done << "/" << TOTAL_OPS << ")"
                << std::endl;
    }
  }

  auto wall_end = std::chrono::steady_clock::now();
  double wall_time_s =
      std::chrono::duration<double>(wall_end - wall_start).count();
  double throughput_bkvs =
      static_cast<double>(ops_done) / wall_time_s / (1024.0 * 1024.0 * 1024.0);

  auto stats = compute_percentiles(batch_latencies);

  std::cout << "external_sweep," << ops_done << "," << std::fixed
            << std::setprecision(3) << wall_time_s << ","
            << std::setprecision(6) << throughput_bkvs << ","
            << std::setprecision(3) << stats.p50 << "," << stats.p95 << ","
            << stats.p99 << std::endl;

  std::cerr << "[external_sweep] Done: wall=" << std::fixed
            << std::setprecision(3) << wall_time_s
            << "s throughput=" << std::setprecision(6) << throughput_bkvs
            << " B-KV/s P50=" << std::setprecision(3) << stats.p50
            << "ms P95=" << stats.p95 << "ms P99=" << stats.p99 << "ms"
            << " sweeps=" << sweep_count << std::endl;

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_export_keys));
  CUDA_CHECK(cudaFree(d_export_values));
  CUDA_CHECK(cudaFree(d_export_scores));
}

/* ─── Main ─── */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E9: Eviction Comparison Benchmark" << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", batch=" << BATCH_SIZE << std::endl;
  std::cerr << "Zipfian alpha=" << ZIPF_ALPHA << ", key_range=" << KEY_RANGE
            << ", total_ops=" << TOTAL_OPS << std::endl;

  std::string target = (argc > 1) ? argv[1] : "all";

  // CSV header
  if (target == "all" || target == "header") {
    std::cout << "method,total_ops,wall_time_s,throughput_bkvs,"
              << "p50_ms,p95_ms,p99_ms" << std::endl;
    if (target == "header") return 0;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  try {
    if (target == "all" || target == "hkv_builtin") {
      benchmark_hkv_builtin_eviction(stream);
      // Sync and let GPU cool between benchmarks
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (target == "all" || target == "external_sweep") {
      benchmark_external_eviction(stream);
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  } catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
