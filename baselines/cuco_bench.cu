/*
 * E8: cuCollections (cuco) Baseline Benchmark
 *
 * cuco stores (key -> index) pairs via cuco::static_map; actual float[32]
 * values live in a separate flat array.  Insert = map.insert + scatter;
 * Find = map.find + gather.  Both phases are timed end-to-end.
 *
 * Empty sentinels: key = -1LL, value = -1LL
 *
 * Output: CSV  library,operation,load_factor,run,throughput_bkvs
 */

#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common.cuh"

/* ─── Types ─── */

using key_type = int64_t;
using index_type = int64_t;

static constexpr key_type EMPTY_KEY = -1LL;
static constexpr index_type EMPTY_VAL = -1LL;

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

/* ─── Benchmark ─── */

void run_cuco(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  const size_t table_capacity = static_cast<size_t>(
      std::ceil(static_cast<double>(target_n) / target_lf));

  std::cerr << "cuCollections LF=" << std::fixed << std::setprecision(2)
            << target_lf << " target_n=" << target_n
            << " table_cap=" << table_capacity << std::endl;

  // Generate all keys and float values
  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, /*start=*/1);

  std::vector<float> h_all_float_vals;
  generate_random_float_values(h_all_float_vals, target_n);

  const size_t prefill_n = target_n - BATCH_SIZE;

  // Prepare query keys for find benchmark (100% hit)
  std::vector<key_type> h_query_keys(BATCH_SIZE);
  {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_query_keys[i] = h_all_keys[dist(rng)];
    }
  }

  // Flat value array on device: CAPACITY * DIM floats
  float* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, CAPACITY * DIM * sizeof(float)));

  // Device arrays for batch operations
  float* d_batch_floats = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_batch_floats, BATCH_SIZE * DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * DIM * sizeof(float)));

  // Copy batch float values
  CUDA_CHECK(cudaMemcpy(d_batch_floats,
                         h_all_float_vals.data() + prefill_n * DIM,
                         BATCH_SIZE * DIM * sizeof(float),
                         cudaMemcpyHostToDevice));

  // Batch indices for scatter
  uint64_t* d_batch_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_batch_indices, BATCH_SIZE * sizeof(uint64_t)));
  {
    std::vector<uint64_t> h_batch_idx(BATCH_SIZE);
    for (size_t i = 0; i < BATCH_SIZE; i++) h_batch_idx[i] = prefill_n + i;
    CUDA_CHECK(cudaMemcpy(d_batch_indices, h_batch_idx.data(),
                           BATCH_SIZE * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
  }

  // Result indices for find -> gather
  uint64_t* d_result_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_result_indices, BATCH_SIZE * sizeof(uint64_t)));

  /* ─── INSERT Benchmark ─── */
  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create fresh map
    auto map = cuco::static_map<key_type, index_type>(
        table_capacity, cuco::empty_key{EMPTY_KEY},
        cuco::empty_value{EMPTY_VAL});

    // Pre-fill: insert pairs in chunks
    {
      const size_t chunk = 4UL * 1024 * 1024;
      thrust::device_vector<cuco::pair<key_type, index_type>> d_pairs(chunk);

      for (size_t off = 0; off < prefill_n; off += chunk) {
        size_t cur = std::min(chunk, prefill_n - off);
        std::vector<cuco::pair<key_type, index_type>> h_pairs(cur);
        for (size_t i = 0; i < cur; i++) {
          h_pairs[i] = {h_all_keys[off + i],
                        static_cast<index_type>(off + i)};
        }
        d_pairs.resize(cur);
        thrust::copy(h_pairs.begin(), h_pairs.end(), d_pairs.begin());
        map.insert(d_pairs.begin(), d_pairs.end());
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      // Scatter prefill values into d_values
      float* d_prefill_floats = nullptr;
      uint64_t* d_prefill_indices = nullptr;
      CUDA_CHECK(
          cudaMalloc(&d_prefill_floats, prefill_n * DIM * sizeof(float)));
      CUDA_CHECK(
          cudaMalloc(&d_prefill_indices, prefill_n * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpy(d_prefill_floats, h_all_float_vals.data(),
                             prefill_n * DIM * sizeof(float),
                             cudaMemcpyHostToDevice));
      {
        std::vector<uint64_t> idx(prefill_n);
        for (size_t i = 0; i < prefill_n; i++) idx[i] = i;
        CUDA_CHECK(cudaMemcpy(d_prefill_indices, idx.data(),
                               prefill_n * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
      }
      scatter_values(d_values, d_prefill_floats, d_prefill_indices, DIM,
                     prefill_n);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(d_prefill_floats));
      CUDA_CHECK(cudaFree(d_prefill_indices));
    }

    // Timed: insert batch pairs + scatter batch values
    thrust::device_vector<cuco::pair<key_type, index_type>> d_insert_pairs(
        BATCH_SIZE);
    {
      std::vector<cuco::pair<key_type, index_type>> h_pairs(BATCH_SIZE);
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        h_pairs[i] = {h_all_keys[prefill_n + i],
                      static_cast<index_type>(prefill_n + i)};
      }
      thrust::copy(h_pairs.begin(), h_pairs.end(), d_insert_pairs.begin());
    }

    CudaTimer timer;
    timer.start();
    map.insert(d_insert_pairs.begin(), d_insert_pairs.end());
    scatter_values(d_values, d_batch_floats, d_batch_indices, DIM, BATCH_SIZE);
    timer.stop();

    if (run >= WARMUP) {
      // Verify: find-back to count actually successful inserts
      thrust::device_vector<key_type> d_verify_keys(
          h_all_keys.begin() + prefill_n, h_all_keys.end());
      thrust::device_vector<index_type> d_verify_vals(BATCH_SIZE);
      map.find(d_verify_keys.begin(), d_verify_keys.end(),
               d_verify_vals.begin());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<index_type> h_verify(BATCH_SIZE);
      thrust::copy(d_verify_vals.begin(), d_verify_vals.end(),
                   h_verify.begin());
      size_t success = 0;
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        if (h_verify[i] != EMPTY_VAL) success++;
      }

      double tp = throughput_bkvs(success, timer.elapsed_seconds());
      std::cout << "cuCollections,insert," << std::fixed
                << std::setprecision(2) << target_lf << ","
                << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                << std::endl;
      std::cerr << "  insert run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s (" << success << "/" << BATCH_SIZE << " ok)"
                << std::endl;
    }
  }

  /* ─── FIND Benchmark ─── */
  for (int run = 0; run < WARMUP + RUNS; run++) {
    // Create and fully populate map
    auto map = cuco::static_map<key_type, index_type>(
        table_capacity, cuco::empty_key{EMPTY_KEY},
        cuco::empty_value{EMPTY_VAL});

    {
      const size_t chunk = 4UL * 1024 * 1024;
      thrust::device_vector<cuco::pair<key_type, index_type>> d_pairs(chunk);

      for (size_t off = 0; off < target_n; off += chunk) {
        size_t cur = std::min(chunk, target_n - off);
        std::vector<cuco::pair<key_type, index_type>> h_pairs(cur);
        for (size_t i = 0; i < cur; i++) {
          h_pairs[i] = {h_all_keys[off + i],
                        static_cast<index_type>(off + i)};
        }
        d_pairs.resize(cur);
        thrust::copy(h_pairs.begin(), h_pairs.end(), d_pairs.begin());
        map.insert(d_pairs.begin(), d_pairs.end());
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      // Upload all values
      CUDA_CHECK(cudaMemcpy(d_values, h_all_float_vals.data(),
                             target_n * DIM * sizeof(float),
                             cudaMemcpyHostToDevice));
    }

    // Upload query keys
    thrust::device_vector<key_type> d_query(h_query_keys.begin(),
                                            h_query_keys.end());
    // Output: values (index_type) directly
    thrust::device_vector<index_type> d_found_values(BATCH_SIZE);

    // Timed: find + gather
    CudaTimer timer;
    timer.start();
    map.find(d_query.begin(), d_query.end(), d_found_values.begin());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert found index values to uint64_t for gather kernel
    {
      std::vector<index_type> h_found(BATCH_SIZE);
      thrust::copy(d_found_values.begin(), d_found_values.end(),
                   h_found.begin());
      std::vector<uint64_t> h_idx(BATCH_SIZE);
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        h_idx[i] = (h_found[i] == EMPTY_VAL)
                        ? ~0ULL
                        : static_cast<uint64_t>(h_found[i]);
      }
      CUDA_CHECK(cudaMemcpy(d_result_indices, h_idx.data(),
                             BATCH_SIZE * sizeof(uint64_t),
                             cudaMemcpyHostToDevice));
    }
    gather_values(d_out, d_values, d_result_indices, DIM, BATCH_SIZE);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "cuCollections,find," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - WARMUP + 1) << ","
                << std::setprecision(6) << tp << std::endl;
      std::cerr << "  find run " << (run - WARMUP + 1) << ": " << tp
                << " B-KV/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_batch_floats));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_batch_indices));
  CUDA_CHECK(cudaFree(d_result_indices));
}

/* ─── Main ─── */

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "E16: cuCollections Baseline (indirection, key->index)"
            << std::endl;
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
    run_cuco(lf);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
