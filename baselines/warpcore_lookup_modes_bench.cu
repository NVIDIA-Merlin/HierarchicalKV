/*
 * WarpCore lookup-mode audit.
 *
 * Measures two distinct lookup semantics:
 *   1. key_only: native HashSet membership lookup.
 *      key_only_proxy remains available as an audit fallback.
 *   2. value_returning: SingleValueHashTable retrieve of native 128B values.
 *
 * Output: CSV
 *   library,mode,operation,load_factor,run,throughput_bkvs
 */

#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <warpcore/hash_set.cuh>
#include <warpcore/single_value_hash_table.cuh>
#include "common.cuh"

struct alignas(16) Value128 {
  float data[DIM];
};

using key_type = uint64_t;
using value_type = Value128;
using tiny_value_type = uint8_t;
using hash_set_t = warpcore::HashSet<key_type>;
using value_table_t = warpcore::SingleValueHashTable<key_type, value_type>;
using tiny_table_t = warpcore::SingleValueHashTable<key_type, tiny_value_type>;

static void generate_sequential_keys(std::vector<key_type>& keys, size_t n,
                                     key_type start = 1) {
  keys.resize(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = start + static_cast<key_type>(i);
  }
}

static void generate_hit_queries(std::vector<key_type>& query_keys,
                                 const std::vector<key_type>& all_keys,
                                 size_t target_n) {
  query_keys.resize(BATCH_SIZE);
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<size_t> dist(0, target_n - 1);
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    query_keys[i] = all_keys[dist(rng)];
  }
}

static size_t table_capacity_for(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  return static_cast<size_t>(
      std::ceil(static_cast<double>(target_n) / target_lf));
}

static void insert_tiny_table_in_chunks(tiny_table_t& table,
                                        const std::vector<key_type>& keys,
                                        size_t target_n) {
  const size_t chunk = 4UL * 1024 * 1024;
  for (size_t off = 0; off < target_n; off += chunk) {
    size_t cur = std::min(chunk, target_n - off);
    key_type* dk = nullptr;
    tiny_value_type* dv = nullptr;
    CUDA_CHECK(cudaMalloc(&dk, cur * sizeof(key_type)));
    CUDA_CHECK(cudaMalloc(&dv, cur * sizeof(tiny_value_type)));
    CUDA_CHECK(cudaMemset(dv, 1, cur * sizeof(tiny_value_type)));
    CUDA_CHECK(cudaMemcpy(dk, keys.data() + off, cur * sizeof(key_type),
                          cudaMemcpyHostToDevice));
    table.insert(dk, dv, cur, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
  }
}

static void insert_hash_set_in_chunks(hash_set_t& table,
                                      const std::vector<key_type>& keys,
                                      size_t target_n) {
  const size_t chunk = 4UL * 1024 * 1024;
  for (size_t off = 0; off < target_n; off += chunk) {
    size_t cur = std::min(chunk, target_n - off);
    key_type* dk = nullptr;
    CUDA_CHECK(cudaMalloc(&dk, cur * sizeof(key_type)));
    CUDA_CHECK(cudaMemcpy(dk, keys.data() + off, cur * sizeof(key_type),
                          cudaMemcpyHostToDevice));
    table.insert(dk, cur, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dk));
  }
}

template <typename Set>
__global__ void hash_set_retrieve_kernel(
    const typename Set::key_type* keys_in, typename Set::index_type num_in,
    bool* flags_out, Set table, typename Set::index_type probing_length) {
  const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  const auto gid = tid / Set::cg_size();
  const auto group = cooperative_groups::tiled_partition<Set::cg_size()>(
      cooperative_groups::this_thread_block());

  if (gid < num_in) {
    bool found = false;
    table.retrieve(keys_in[gid], found, group, probing_length);
    if (group.thread_rank() == 0) {
      flags_out[gid] = found;
    }
  }
}

static void retrieve_hash_set(hash_set_t& table, const key_type* keys_in,
                              size_t num_in, bool* flags_out) {
  constexpr int block_size = 256;
  const auto groups = num_in * hash_set_t::cg_size();
  const auto blocks = (groups + block_size - 1) / block_size;
  hash_set_retrieve_kernel<hash_set_t><<<blocks, block_size>>>(
      keys_in, num_in, flags_out, table, warpcore::defaults::probing_length());
  CUDA_CHECK(cudaGetLastError());
}

static void insert_value_table_in_chunks(value_table_t& table,
                                         const std::vector<key_type>& keys,
                                         size_t target_n) {
  const size_t chunk = 4UL * 1024 * 1024;
  for (size_t off = 0; off < target_n; off += chunk) {
    size_t cur = std::min(chunk, target_n - off);
    key_type* dk = nullptr;
    Value128* dv = nullptr;
    CUDA_CHECK(cudaMalloc(&dk, cur * sizeof(key_type)));
    CUDA_CHECK(cudaMalloc(&dv, cur * sizeof(Value128)));
    CUDA_CHECK(cudaMemcpy(dk, keys.data() + off, cur * sizeof(key_type),
                          cudaMemcpyHostToDevice));
    // Lookup-mode audit only needs to move a native 128B value; value contents
    // are irrelevant, so avoid generating a 16GB host-side random array.
    CUDA_CHECK(cudaMemset(dv, 1, cur * sizeof(Value128)));
    table.insert(dk, dv, cur, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
  }
}

static void run_key_only_hashset(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  const size_t table_capacity = table_capacity_for(target_lf);

  std::cerr << "WarpCore key_only HashSet LF=" << std::fixed
            << std::setprecision(2) << target_lf << " target_n=" << target_n
            << " table_cap=" << table_capacity << std::endl;

  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, 1);

  std::vector<key_type> h_query_keys;
  generate_hit_queries(h_query_keys, h_all_keys, target_n);

  key_type* d_query_keys = nullptr;
  bool* d_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query_keys, BATCH_SIZE * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_flags, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_query_keys, h_query_keys.data(),
                        BATCH_SIZE * sizeof(key_type), cudaMemcpyHostToDevice));

  for (int run = 0; run < WARMUP + RUNS; run++) {
    hash_set_t table(table_capacity);
    insert_hash_set_in_chunks(table, h_all_keys, target_n);

    CudaTimer timer;
    timer.start();
    retrieve_hash_set(table, d_query_keys, BATCH_SIZE, d_flags);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "WarpCore,key_only,contains," << std::fixed
                << std::setprecision(2) << target_lf << ","
                << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_query_keys));
  CUDA_CHECK(cudaFree(d_flags));
}

static void run_key_only_proxy(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  const size_t table_capacity = table_capacity_for(target_lf);

  std::cerr << "WarpCore key_only_proxy LF=" << std::fixed
            << std::setprecision(2) << target_lf << " target_n=" << target_n
            << " table_cap=" << table_capacity << std::endl;

  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, 1);

  std::vector<key_type> h_query_keys;
  generate_hit_queries(h_query_keys, h_all_keys, target_n);

  key_type* d_query_keys = nullptr;
  tiny_value_type* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query_keys, BATCH_SIZE * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * sizeof(tiny_value_type)));
  CUDA_CHECK(cudaMemcpy(d_query_keys, h_query_keys.data(),
                        BATCH_SIZE * sizeof(key_type), cudaMemcpyHostToDevice));

  for (int run = 0; run < WARMUP + RUNS; run++) {
    tiny_table_t table(table_capacity);
    insert_tiny_table_in_chunks(table, h_all_keys, target_n);

    CudaTimer timer;
    timer.start();
    table.retrieve(d_query_keys, BATCH_SIZE, d_out, 0);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "WarpCore,key_only_proxy,find," << std::fixed
                << std::setprecision(2) << target_lf << ","
                << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_query_keys));
  CUDA_CHECK(cudaFree(d_out));
}

static void run_value_returning(float target_lf) {
  const size_t target_n = static_cast<size_t>(CAPACITY * target_lf);
  const size_t table_capacity = table_capacity_for(target_lf);

  std::cerr << "WarpCore value_returning LF=" << std::fixed
            << std::setprecision(2) << target_lf << " target_n=" << target_n
            << " table_cap=" << table_capacity << std::endl;

  std::vector<key_type> h_all_keys;
  generate_sequential_keys(h_all_keys, target_n, 1);

  std::vector<key_type> h_query_keys;
  generate_hit_queries(h_query_keys, h_all_keys, target_n);

  key_type* d_query_keys = nullptr;
  Value128* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query_keys, BATCH_SIZE * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(&d_out, BATCH_SIZE * sizeof(Value128)));
  CUDA_CHECK(cudaMemcpy(d_query_keys, h_query_keys.data(),
                        BATCH_SIZE * sizeof(key_type), cudaMemcpyHostToDevice));

  for (int run = 0; run < WARMUP + RUNS; run++) {
    value_table_t table(table_capacity);
    insert_value_table_in_chunks(table, h_all_keys, target_n);

    CudaTimer timer;
    timer.start();
    table.retrieve(d_query_keys, BATCH_SIZE, d_out, 0);
    timer.stop();

    if (run >= WARMUP) {
      double tp = throughput_bkvs(BATCH_SIZE, timer.elapsed_seconds());
      std::cout << "WarpCore,value_returning,find," << std::fixed
                << std::setprecision(2) << target_lf << ","
                << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_query_keys));
  CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review P0: WarpCore lookup-mode audit" << std::endl;
  std::cerr << "DIM=" << DIM << " CAPACITY=" << CAPACITY
            << " BATCH=" << BATCH_SIZE << std::endl;

  std::string mode = "both";
  std::vector<float> load_factors;

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else {
      load_factors.push_back(std::stof(arg));
    }
  }

  if (load_factors.empty()) {
    load_factors = {0.25f, 0.50f, 0.75f, 1.00f};
  }

  std::cout << "library,mode,operation,load_factor,run,throughput_bkvs"
            << std::endl;

  for (float lf : load_factors) {
    if (mode == "both" || mode == "key_only" || mode == "key_only_hashset" ||
        mode == "all") {
      run_key_only_hashset(lf);
    }
    if (mode == "key_only_proxy" || mode == "all") {
      run_key_only_proxy(lf);
    }
    if (mode == "both" || mode == "value_returning" || mode == "all") {
      run_value_returning(lf);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
