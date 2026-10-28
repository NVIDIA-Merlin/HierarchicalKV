/*
 * E8: BGHT Baseline Benchmark
 *
 * BGHT stores (key -> index) pairs; actual values live in a separate
 * flat array.  Insert = table.insert + scatter_values; Find = table.find +
 * gather_values.  Both phases are timed end-to-end.
 *
 * Sentinel: key = ~0ULL, value = ~0ULL
 *
 * Output: CSV  library,operation,load_factor,run,throughput_bkvs
 */

#include <bght/bcht.hpp>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common.cuh"

/* ─── Types ─── */

using key_type = uint64_t;
using index_type = uint64_t;
using pair_type = bght::pair<key_type, index_type>;

static constexpr key_type SENTINEL_KEY = ~0ULL;
static constexpr index_type SENTINEL_VAL = ~0ULL;

/* ─── Key / Value Generation ─── */

static void generate_sequential_keys(std::vector<key_type>& keys, size_t n,
                                     key_type start = 1) {
  keys.resize(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = start + static_cast<key_type>(i);
  }
}

static void generate_random_float_values(std::vector<float>& vals, size_t n,
                                         unsigned seed = 42) {
  vals.resize(n * DIM);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n * DIM; i++) {
    vals[i] = dist(rng);
  }
}

/* ─── Build Pairs: (key_i, index_i) ─── */

static void build_pairs(std::vector<pair_type>& pairs,
                        const std::vector<key_type>& keys, size_t offset) {
  pairs.resize(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    pairs[i] = {keys[i], static_cast<index_type>(offset + i)};
  }
}

/* ─── Benchmark ─── */

void run_bght(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  const size_t table_capacity = static_cast<size_t>(
      std::ceil(static_cast<double>(target_n) / target_lf));

  std::cerr << "BGHT LF=" << std::fixed << std::setprecision(2) << target_lf
            << " target_n=" << target_n << " table_cap=" << table_capacity
            << std::endl;

  // Generate all keys and float values
  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, /*start=*/1);

  std::vector<float> h_all_float_vals;
  generate_random_float_values(h_all_float_vals, target_n);

  const size_t prefill_n = target_n - BATCH_SIZE;

  // Build prefill pairs: (key_i, i) for i in [0, prefill_n)
  std::vector<pair_type> h_prefill_pairs;
  {
    std::vector<key_type> prefill_keys(h_all_keys.begin(),
                                       h_all_keys.begin() + prefill_n);
    build_pairs(h_prefill_pairs, prefill_keys, 0);
  }

  // Build timed batch pairs: (key_i, prefill_n + i) for i in [0, BATCH_SIZE)
  std::vector<pair_type> h_batch_pairs;
  {
    std::vector<key_type> batch_keys(h_all_keys.begin() + prefill_n,
                                     h_all_keys.end());
    build_pairs(h_batch_pairs, batch_keys, prefill_n);
  }

  // Indices for scatter
  std::vector<uint64_t> h_batch_indices(BATCH_SIZE);
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    h_batch_indices[i] = prefill_n + i;
  }

  // Allocate device memory
  // Flat value array: CAPACITY * DIM floats
  float* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, CAPACITY * DIM * sizeof(float)));

  // Prefill pairs
  pair_type* d_prefill_pairs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_prefill_pairs, prefill_n * sizeof(pair_type)));
  CUDA_CHECK(cudaMemcpy(d_prefill_pairs, h_prefill_pairs.data(),
                         prefill_n * sizeof(pair_type),
                         cudaMemcpyHostToDevice));

  // Prefill float values -> scatter into d_values
  float* d_prefill_floats = nullptr;
  uint64_t* d_prefill_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_prefill_floats, prefill_n * DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_prefill_indices, prefill_n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_prefill_floats, h_all_float_vals.data(),
                         prefill_n * DIM * sizeof(float),
                         cudaMemcpyHostToDevice));
  {
    std::vector<uint64_t> prefill_idx(prefill_n);
    for (size_t i = 0; i < prefill_n; i++) prefill_idx[i] = i;
    CUDA_CHECK(cudaMemcpy(d_prefill_indices, prefill_idx.data(),
                           prefill_n * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
  }

  // Batch pairs and float values
  pair_type* d_batch_pairs = nullptr;
  float* d_batch_floats = nullptr;
  uint64_t* d_batch_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_batch_pairs, BATCH_SIZE * sizeof(pair_type)));
  CUDA_CHECK(cudaMalloc(&d_batch_floats, BATCH_SIZE * DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_batch_indices, BATCH_SIZE * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_batch_pairs, h_batch_pairs.data(),
                         BATCH_SIZE * sizeof(pair_type),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_batch_floats,
                         h_all_float_vals.data() + prefill_n * DIM,
                         BATCH_SIZE * DIM * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_batch_indices, h_batch_indices.data(),
                         BATCH_SIZE * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));

  // Query keys for find benchmark
  key_type* d_query_keys = nullptr;
  index_type* d_result_indices = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query_keys, BATCH_SIZE * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_result_indices, BATCH_SIZE * sizeof(index_type)));
  CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * DIM * sizeof(float)));

  // Prepare random query keys (100% hit)
  std::vector<key_type> h_query_keys(BATCH_SIZE);
  {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_query_keys[i] = h_all_keys[dist(rng)];
    }
  }
  CUDA_CHECK(cudaMemcpy(d_query_keys, h_query_keys.data(),
                         BATCH_SIZE * sizeof(key_type),
                         cudaMemcpyHostToDevice));

  // Device array for batch keys (for find-back verification)
  key_type* d_batch_keys = nullptr;
  CUDA_CHECK(cudaMalloc(&d_batch_keys, BATCH_SIZE * sizeof(key_type)));
  {
    std::vector<key_type> h_batch_keys(h_all_keys.begin() + prefill_n,
                                       h_all_keys.end());
    CUDA_CHECK(cudaMemcpy(d_batch_keys, h_batch_keys.data(),
                           BATCH_SIZE * sizeof(key_type),
                           cudaMemcpyHostToDevice));
  }

  // Device array for find-back verification results
  index_type* d_verify_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_verify_indices, BATCH_SIZE * sizeof(index_type)));

  /* ─── INSERT Benchmark ─── */
  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create fresh table
    bght::bcht<key_type, index_type> table(table_capacity, SENTINEL_KEY,
                                           SENTINEL_VAL);

    // Pre-fill: insert pairs + scatter values
    // Insert in chunks to avoid exceeding BGHT limits
    {
      const size_t chunk = 4UL * 1024 * 1024;
      for (size_t off = 0; off < prefill_n; off += chunk) {
        size_t cur = std::min(chunk, prefill_n - off);
        table.insert(d_prefill_pairs + off, d_prefill_pairs + off + cur);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      scatter_values(d_values, d_prefill_floats, d_prefill_indices, DIM,
                     prefill_n);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed: insert batch pairs + scatter batch values
    CudaTimer timer;
    timer.start();
    table.insert(d_batch_pairs, d_batch_pairs + BATCH_SIZE);
    scatter_values(d_values, d_batch_floats, d_batch_indices, DIM, BATCH_SIZE);
    timer.stop();

    if (run >= WARMUP) {
      // Verify: find-back to count actually successful inserts
      table.find(d_batch_keys, d_batch_keys + BATCH_SIZE, d_verify_indices);
      CUDA_CHECK(cudaDeviceSynchronize());
      size_t success = count_successful_indices(
          reinterpret_cast<uint64_t*>(d_verify_indices), BATCH_SIZE);

      double tp = throughput_bkvs(success, timer.elapsed_seconds());
      std::cout << "BGHT,insert," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  insert run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s (" << success << "/" << BATCH_SIZE << " ok)"
                << std::endl;
    }
  }

  /* ─── FIND Benchmark ─── */
  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create and fully populate table
    bght::bcht<key_type, index_type> table(table_capacity, SENTINEL_KEY,
                                           SENTINEL_VAL);
    {
      const size_t chunk = 4UL * 1024 * 1024;
      pair_type* d_chunk_pairs = nullptr;
      CUDA_CHECK(cudaMalloc(&d_chunk_pairs, chunk * sizeof(pair_type)));
      std::vector<pair_type> h_chunk_pairs(chunk);

      for (size_t off = 0; off < target_n; off += chunk) {
        size_t cur = std::min(chunk, target_n - off);
        for (size_t i = 0; i < cur; i++) {
          h_chunk_pairs[i] = {h_all_keys[off + i],
                              static_cast<index_type>(off + i)};
        }
        CUDA_CHECK(cudaMemcpy(d_chunk_pairs, h_chunk_pairs.data(),
                               cur * sizeof(pair_type),
                               cudaMemcpyHostToDevice));
        table.insert(d_chunk_pairs, d_chunk_pairs + cur);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      CUDA_CHECK(cudaFree(d_chunk_pairs));

      // Scatter all values
      CUDA_CHECK(cudaMemcpy(d_values, h_all_float_vals.data(),
                             target_n * DIM * sizeof(float),
                             cudaMemcpyHostToDevice));
    }

    // Timed: find indices + gather values
    CudaTimer timer;
    timer.start();
    table.find(d_query_keys, d_query_keys + BATCH_SIZE, d_result_indices);
    gather_values(d_out, d_values, d_result_indices, DIM, BATCH_SIZE);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "BGHT,find," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  find run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_prefill_pairs));
  CUDA_CHECK(cudaFree(d_prefill_floats));
  CUDA_CHECK(cudaFree(d_prefill_indices));
  CUDA_CHECK(cudaFree(d_batch_pairs));
  CUDA_CHECK(cudaFree(d_batch_floats));
  CUDA_CHECK(cudaFree(d_batch_indices));
  CUDA_CHECK(cudaFree(d_batch_keys));
  CUDA_CHECK(cudaFree(d_verify_indices));
  CUDA_CHECK(cudaFree(d_query_keys));
  CUDA_CHECK(cudaFree(d_result_indices));
  CUDA_CHECK(cudaFree(d_out));
}

/* ─── Main ─── */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E16: BGHT Baseline (indirection, key->index)" << std::endl;
  std::cerr << "DIM=" << DIM << " CAPACITY=" << CAPACITY
            << " BATCH=" << BATCH_SIZE << std::endl;

  std::cout << "library,operation,load_factor,run,throughput_bkvs" << std::endl;

  std::vector<float> load_factors;
  if (argc > 1) {
    load_factors.push_back(std::stof(argv[1]));
  } else {
    load_factors = {0.25f, 0.50f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f, 1.00f};
  }
  for (float lf : load_factors) {
    run_bght(lf);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
