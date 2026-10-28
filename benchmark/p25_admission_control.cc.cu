/*
 * P2-5: Admission Control Micro-Experiment
 *
 * Validates that HKV's built-in admission control preserves cache quality
 * under adversarial random-key bursts (uniform distribution, α=0.0).
 *
 * Protocol:
 *   1. Fill table to LF≈1.0 with "valuable" keys (high scores).
 *   2. Run A: Insert burst of random NEW keys with score=1 (low).
 *      → Built-in admission rejects (score < bucket_min), cache preserved.
 *   3. Run B: Insert burst of random NEW keys with score=MAX (high).
 *      → All insertions accepted, evict original high-value entries.
 *   4. Compare hit rates on original keys: A vs B.
 *
 * This demonstrates that HKV's admission policy (refuse if incoming
 * score < bucket minimum) defends cache quality against adversarial
 * low-score insertions.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using Table =
    nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kCustomized>;
using TableOptions = nv::merlin::HashTableOptions;

constexpr uint64_t DIM = 8;
constexpr uint64_t INIT_CAPACITY = 16 * 1024 * 1024;  // 16M keys
constexpr uint64_t BUCKET_SIZE = 128;
constexpr uint64_t BATCH_SIZE = 1024 * 1024;  // 1M per batch

// Fill with high scores so bucket minimums are high
constexpr S HIGH_SCORE_BASE = 1000000;
// Adversarial low score (will be rejected by admission)
constexpr S LOW_SCORE = 1;
// Very high score to force eviction (bypass admission)
constexpr S VERY_HIGH_SCORE = 999999999ULL;

void fill_table(Table* table, uint64_t n, K* d_keys, V* d_values, S* d_scores,
                cudaStream_t stream) {
  std::vector<K> h_keys(BATCH_SIZE);
  std::vector<V> h_values(BATCH_SIZE * DIM);
  std::vector<S> h_scores(BATCH_SIZE);

  uint64_t inserted = 0;
  while (inserted < n) {
    uint64_t batch = std::min(BATCH_SIZE, n - inserted);
    for (uint64_t i = 0; i < batch; i++) {
      h_keys[i] = inserted + i + 1;  // keys 1..n
      h_scores[i] = HIGH_SCORE_BASE + inserted + i;
      for (uint64_t d = 0; d < DIM; d++) {
        h_values[i * DIM + d] = static_cast<float>(h_keys[i]);
      }
    }
    cudaMemcpyAsync(d_keys, h_keys.data(), batch * sizeof(K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, h_values.data(), batch * DIM * sizeof(V),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores, h_scores.data(), batch * sizeof(S),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Use ignore_evict_strategy=true during fill to ensure all go in
    table->insert_or_assign(batch, d_keys, d_values, d_scores, stream, true,
                            true);
    cudaStreamSynchronize(stream);
    inserted += batch;
  }
}

// Insert burst of random new keys with given score
void insert_burst(Table* table, uint64_t n, S score, K* d_keys, V* d_values,
                  S* d_scores, cudaStream_t stream) {
  std::vector<K> h_keys(BATCH_SIZE);
  std::vector<V> h_values(BATCH_SIZE * DIM);
  std::vector<S> h_scores(BATCH_SIZE);
  std::mt19937_64 rng(42);

  uint64_t sent = 0;
  while (sent < n) {
    uint64_t batch = std::min(BATCH_SIZE, n - sent);
    for (uint64_t i = 0; i < batch; i++) {
      // Ensure keys don't overlap with original 1..INIT_CAPACITY
      h_keys[i] = INIT_CAPACITY + 10000000ULL + rng() % (1ULL << 50);
      h_scores[i] = score;
      for (uint64_t d = 0; d < DIM; d++) {
        h_values[i * DIM + d] = 0.0f;
      }
    }
    cudaMemcpyAsync(d_keys, h_keys.data(), batch * sizeof(K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, h_values.data(), batch * DIM * sizeof(V),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores, h_scores.data(), batch * sizeof(S),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // ignore_evict_strategy=false → admission control active
    // (reject if incoming score < bucket minimum)
    table->insert_or_assign(batch, d_keys, d_values, d_scores, stream, true,
                            false);
    cudaStreamSynchronize(stream);
    sent += batch;
  }
}

// Look up original keys and count hits
double measure_hit_rate(Table* table, uint64_t n, K* d_keys, V* d_values,
                        S* d_scores, bool* d_founds, cudaStream_t stream) {
  std::vector<K> h_keys(BATCH_SIZE);
  bool* h_founds = new bool[BATCH_SIZE];

  uint64_t total_found = 0;
  uint64_t checked = 0;

  while (checked < n) {
    uint64_t batch = std::min(BATCH_SIZE, n - checked);
    for (uint64_t i = 0; i < batch; i++) {
      h_keys[i] = checked + i + 1;
    }
    cudaMemcpyAsync(d_keys, h_keys.data(), batch * sizeof(K),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    table->find(batch, d_keys, d_values, d_founds, d_scores, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_founds, d_founds, batch * sizeof(bool),
               cudaMemcpyDeviceToHost);
    for (uint64_t i = 0; i < batch; i++) {
      if (h_founds[i]) total_found++;
    }
    checked += batch;
  }
  delete[] h_founds;
  return static_cast<double>(total_found) / static_cast<double>(n) * 100.0;
}

void run_experiment(const char* label, S burst_score, K* d_keys, V* d_values,
                    S* d_scores, bool* d_founds, cudaStream_t stream) {
  printf("--- %s (burst score = %lu) ---\n", label, burst_score);

  TableOptions opt;
  opt.init_capacity = INIT_CAPACITY;
  opt.max_capacity = INIT_CAPACITY;
  opt.max_bucket_size = BUCKET_SIZE;
  opt.dim = DIM;
  opt.max_hbm_for_vectors = INIT_CAPACITY * DIM * sizeof(V);

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  // Step 1: Fill to LF≈1.0
  printf("  Step 1: Filling table with %luM high-score keys...\n",
         INIT_CAPACITY / (1024 * 1024));
  fill_table(table.get(), INIT_CAPACITY, d_keys, d_values, d_scores, stream);
  uint64_t size_after_fill = table->size(stream);
  double lf_fill = static_cast<double>(size_after_fill) / INIT_CAPACITY;
  printf("  Table size after fill: %lu (LF=%.4f)\n", size_after_fill, lf_fill);

  // Step 2: Hit rate before burst (baseline)
  double hit_rate_before = measure_hit_rate(
      table.get(), INIT_CAPACITY, d_keys, d_values, d_scores, d_founds, stream);
  printf("  Hit rate before burst: %.2f%%\n", hit_rate_before);

  // Step 3: Adversarial burst
  uint64_t burst_count = INIT_CAPACITY / 4;  // 25% of capacity
  printf("  Step 2: Inserting %luM random keys (score=%lu)...\n",
         burst_count / (1024 * 1024), burst_score);
  insert_burst(table.get(), burst_count, burst_score, d_keys, d_values,
               d_scores, stream);
  uint64_t size_after_burst = table->size(stream);
  printf(
      "  Table size after burst: %lu (delta=%ld)\n", size_after_burst,
      static_cast<long>(size_after_burst) - static_cast<long>(size_after_fill));

  // Step 4: Hit rate after burst
  double hit_rate_after = measure_hit_rate(
      table.get(), INIT_CAPACITY, d_keys, d_values, d_scores, d_founds, stream);
  printf("  Hit rate after burst: %.2f%%\n", hit_rate_after);
  printf("  Hit rate delta: %+.2f pp\n\n", hit_rate_after - hit_rate_before);
}

int main() {
  printf("=== P2-5: Admission Control Micro-Experiment ===\n");
  printf("GPU: ");
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("%s\n", prop.name);
  printf("Capacity: %luM, DIM=%lu, Bucket=%lu\n", INIT_CAPACITY / (1024 * 1024),
         DIM, BUCKET_SIZE);
  printf("Original keys: score = %lu + key_idx\n", HIGH_SCORE_BASE);
  printf("Burst size: %luM (25%% of capacity)\n\n",
         (INIT_CAPACITY / 4) / (1024 * 1024));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  K* d_keys;
  V* d_values;
  S* d_scores;
  bool* d_founds;

  cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K));
  cudaMalloc(&d_values, BATCH_SIZE * DIM * sizeof(V));
  cudaMalloc(&d_scores, BATCH_SIZE * sizeof(S));
  cudaMalloc(&d_founds, BATCH_SIZE * sizeof(bool));

  // Run A: Low-score burst (admission control REJECTS these)
  run_experiment("Run A: Low-score burst (admission rejects)", LOW_SCORE,
                 d_keys, d_values, d_scores, d_founds, stream);

  // Run B: High-score burst (admission ACCEPTS, evicts originals)
  run_experiment("Run B: High-score burst (no admission, evicts originals)",
                 VERY_HIGH_SCORE, d_keys, d_values, d_scores, d_founds, stream);

  printf("=== Conclusion ===\n");
  printf("Compare Run A vs Run B hit rates to quantify admission control's\n");
  printf("cache-quality preservation under adversarial insertion patterns.\n");

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_scores);
  cudaFree(d_founds);
  cudaStreamDestroy(stream);

  return 0;
}
