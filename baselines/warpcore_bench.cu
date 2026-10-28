/*
 * E8: WarpCore Baseline Benchmark
 *
 * WarpCore stores native 128-byte values (struct Value128 = float[32]),
 * so no indirection is needed.  We use SingleValueHashTable.
 *
 * Sentinel keys: WarpCore reserves key=0 and key=~0ULL, so keys start from 1.
 *
 * Output: CSV  library,operation,load_factor,run,throughput_bkvs
 */

#include <warpcore/single_value_hash_table.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common.cuh"

/* ─── Value Type: 128 bytes = 32 x float ─── */

struct alignas(16) Value128 {
  float data[DIM];
};

/* ─── WarpCore Hash Table Type ─── */

using key_type = uint64_t;
using value_type = Value128;
using hash_table_t = warpcore::SingleValueHashTable<key_type, value_type>;

/* ─── Key Generation ─── */

static void generate_sequential_keys(std::vector<key_type>& keys, size_t n,
                                     key_type start = 1) {
  keys.resize(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = start + static_cast<key_type>(i);
  }
}

static void generate_random_values(std::vector<Value128>& vals, size_t n,
                                   unsigned seed = 42) {
  vals.resize(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; i++) {
    for (size_t d = 0; d < DIM; d++) {
      vals[i].data[d] = dist(rng);
    }
  }
}

/* ─── Benchmark ─── */

void run_warpcore(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  // Table capacity: enough slots so that target_n / table_cap ~ target_lf
  const size_t table_capacity =
      static_cast<size_t>(std::ceil(static_cast<double>(target_n) / target_lf));

  std::cerr << "WarpCore LF=" << std::fixed << std::setprecision(2)
            << target_lf << " target_n=" << target_n
            << " table_cap=" << table_capacity << std::endl;

  // Sentinel: WarpCore uses key=0 as empty sentinel by default.
  // Keys start from 1.

  // Generate all keys and values
  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, /*start=*/1);

  std::vector<Value128> h_all_vals;
  generate_random_values(h_all_vals, target_n);

  // Split: pre-fill keys [0, target_n - BATCH_SIZE), timed keys [target_n -
  // BATCH_SIZE, target_n)
  const size_t prefill_n = target_n - BATCH_SIZE;

  // Device arrays for the timed batch
  key_type* d_keys = nullptr;
  Value128* d_vals = nullptr;
  Value128* d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_vals, BATCH_SIZE * sizeof(Value128)));
  CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * sizeof(Value128)));

  // Device arrays for pre-fill
  key_type* d_prefill_keys = nullptr;
  Value128* d_prefill_vals = nullptr;

  CUDA_CHECK(cudaMalloc(&d_prefill_keys, prefill_n * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_prefill_vals, prefill_n * sizeof(Value128)));
  CUDA_CHECK(cudaMemcpy(d_prefill_keys, h_all_keys.data(),
                         prefill_n * sizeof(key_type),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prefill_vals, h_all_vals.data(),
                         prefill_n * sizeof(Value128),
                         cudaMemcpyHostToDevice));

  // Copy timed batch
  CUDA_CHECK(cudaMemcpy(d_keys, h_all_keys.data() + prefill_n,
                         BATCH_SIZE * sizeof(key_type),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, h_all_vals.data() + prefill_n,
                         BATCH_SIZE * sizeof(Value128),
                         cudaMemcpyHostToDevice));

  // Device array for retrieve-back verification
  Value128* d_verify = nullptr;
  CUDA_CHECK(cudaMalloc(&d_verify, BATCH_SIZE * sizeof(Value128)));

  /* ─── INSERT Benchmark ─── */
  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create a fresh table each run
    hash_table_t table(table_capacity);

    // Pre-fill
    table.insert(d_prefill_keys, d_prefill_vals, prefill_n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed insert of last BATCH_SIZE keys
    CudaTimer timer;
    timer.start();
    table.insert(d_keys, d_vals, BATCH_SIZE, 0);
    timer.stop();

    if (run >= WARMUP) {
      // Verify: retrieve-back to count actually successful inserts
      // Fill with NaN sentinel before retrieval
      CUDA_CHECK(cudaMemset(d_verify, 0xFF, BATCH_SIZE * sizeof(Value128)));
      table.retrieve(d_keys, BATCH_SIZE, d_verify, 0);
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<Value128> h_verify(BATCH_SIZE);
      CUDA_CHECK(cudaMemcpy(h_verify.data(), d_verify,
                             BATCH_SIZE * sizeof(Value128),
                             cudaMemcpyDeviceToHost));
      size_t success = 0;
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        if (!std::isnan(h_verify[i].data[0])) success++;
      }

      double tp = throughput_bkvs(success, timer.elapsed_seconds());
      std::cout << "WarpCore,insert," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  insert run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s (" << success << "/" << BATCH_SIZE << " ok)"
                << std::endl;
    }
  }

  /* ─── FIND Benchmark ─── */
  // Build a fully-populated table, then query BATCH_SIZE random existing keys.

  // Prepare random query keys (sample from inserted keys for 100% hit)
  std::vector<key_type> h_query_keys(BATCH_SIZE);
  {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_query_keys[i] = h_all_keys[dist(rng)];
    }
  }
  CUDA_CHECK(cudaMemcpy(d_keys, h_query_keys.data(),
                         BATCH_SIZE * sizeof(key_type),
                         cudaMemcpyHostToDevice));

  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create and fully populate table
    hash_table_t table(table_capacity);
    // Insert all keys in batches
    {
      const size_t chunk = 4UL * 1024 * 1024;  // 4M per batch
      for (size_t off = 0; off < target_n; off += chunk) {
        size_t cur = std::min(chunk, target_n - off);
        key_type* dk = nullptr;
        Value128* dv = nullptr;
        CUDA_CHECK(cudaMalloc(&dk, cur * sizeof(key_type)));
        CUDA_CHECK(cudaMalloc(&dv, cur * sizeof(Value128)));
        CUDA_CHECK(cudaMemcpy(dk, h_all_keys.data() + off,
                               cur * sizeof(key_type),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, h_all_vals.data() + off,
                               cur * sizeof(Value128),
                               cudaMemcpyHostToDevice));
        table.insert(dk, dv, cur, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dk));
        CUDA_CHECK(cudaFree(dv));
      }
    }

    // Timed find
    CudaTimer timer;
    timer.start();
    table.retrieve(d_keys, BATCH_SIZE, d_out, 0);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "WarpCore,find," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  find run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vals));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_verify));
  CUDA_CHECK(cudaFree(d_prefill_keys));
  CUDA_CHECK(cudaFree(d_prefill_vals));
}

/* ─── Main ─── */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E16: WarpCore Baseline (native 128B values)" << std::endl;
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
    run_warpcore(lf);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
