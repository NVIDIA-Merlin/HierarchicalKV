/*
 * WarpSpeed-style WarpCore value-size audit.
 *
 * This benchmark preserves the key parts of the WarpSpeed load-factor query
 * path for WarpCore:
 *   - SingleValueHashTable
 *   - cooperative group / tile size = 4
 *   - capacity=100M, LF=0.25 by default
 *   - successful retrieve queries over the inserted key prefix
 *
 * It sweeps the value payload through BASELINE_DIM fp32 values and reports
 * decimal billion operations/s. One operation is one queried key, independent
 * of value size.
 */

#include <cooperative_groups.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <warpcore/single_value_hash_table.cuh>
#include "common.cuh"

namespace cg = cooperative_groups;

using key_type = uint64_t;
static constexpr uint32_t kTileSize = 4;

template <int D>
struct Value {
  float data[D];
};

template <int D>
__host__ __device__ Value<D> make_value(key_type key) {
  Value<D> v;
#pragma unroll
  for (int i = 0; i < D; i++) {
    v.data[i] = static_cast<float>((key & 0xffffULL) + i);
  }
  return v;
}

template <int D>
__host__ __device__ bool value_equal(const Value<D>& a, const Value<D>& b) {
#pragma unroll
  for (int i = 0; i < D; i++) {
    if (a.data[i] != b.data[i]) return false;
  }
  return true;
}

__host__ __device__ inline key_type splitmix64_stateless(key_type x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}

__global__ void make_keys_kernel(key_type* keys, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
    key_type k = splitmix64_stateless(i + 1);
    if (k == warpcore::defaults::empty_key<key_type>() ||
        k == warpcore::defaults::tombstone_key<key_type>()) {
      k ^= 0x517cc1b727220a95ULL;
    }
    keys[i] = k;
  }
}

template <int D>
using table_t = warpcore::SingleValueHashTable<
    key_type, Value<D>, warpcore::defaults::empty_key<key_type>(),
    warpcore::defaults::tombstone_key<key_type>(),
    warpcore::defaults::probing_scheme_t<key_type, kTileSize>,
    warpcore::defaults::table_storage_t<key_type, Value<D>>,
    warpcore::defaults::temp_memory_bytes()>;

template <int D>
__global__ void insert_kernel(table_t<D> table, const key_type* keys,
                              uint64_t n, uint64_t* misses) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<kTileSize>(block);
  uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) / kTileSize;
  if (tid >= n) return;

  const key_type key = keys[tid];
  auto status = table.insert(key, make_value<D>(key), tile);
  if (tile.thread_rank() == 0 && status != warpcore::Status::none()) {
    atomicAdd(reinterpret_cast<unsigned long long*>(misses), 1ULL);
  }
}

template <int D, bool Materialize>
__global__ void retrieve_kernel(table_t<D> table, const key_type* keys,
                                uint64_t n, Value<D>* out, uint64_t* misses) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<kTileSize>(block);
  uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) / kTileSize;
  if (tid >= n) return;

  const key_type key = keys[tid];
  Value<D> value;
  auto status = table.retrieve(key, value, tile);
  bool ok = (status == warpcore::Status::none()) &&
            value_equal(value, make_value<D>(key));
  if constexpr (Materialize) {
    if (ok && tile.thread_rank() == 0) {
      out[tid] = value;
    }
  }
  if (tile.thread_rank() == 0 && !ok) {
    atomicAdd(reinterpret_cast<unsigned long long*>(misses), 1ULL);
  }
}

static double throughput_bops(uint64_t n, double seconds) {
  return static_cast<double>(n) / seconds / 1.0e9;
}

template <int D>
void run_one(float target_lf, bool materialize) {
  const uint64_t n = static_cast<uint64_t>(BASELINE_CAPACITY * target_lf);
  const uint64_t capacity =
      static_cast<uint64_t>(std::ceil(static_cast<double>(n) / target_lf));

  std::cerr << "WarpCore WarpSpeed-style value-size LF=" << std::fixed
            << std::setprecision(2) << target_lf << " dim=" << D
            << " value_bytes=" << (D * 4) << " n=" << n
            << " capacity=" << capacity
            << " materialize=" << (materialize ? 1 : 0) << std::endl;

  key_type* d_keys = nullptr;
  Value<D>* d_out = nullptr;
  uint64_t* misses = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(key_type)));
  if (materialize) {
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Value<D>)));
  }
  CUDA_CHECK(cudaMallocManaged(&misses, sizeof(uint64_t)));

  const int block = 256;
  const int grid =
      static_cast<int>(std::min<uint64_t>((n + block - 1) / block, 65535));
  make_keys_kernel<<<grid, block>>>(d_keys, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  table_t<D> table(capacity, 42);
  *misses = 0;
  CUDA_CHECK(cudaDeviceSynchronize());
  insert_kernel<D>
      <<<static_cast<int>((n * kTileSize + block - 1) / block), block>>>(
          table, d_keys, n, misses);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (*misses != 0) {
    std::cerr << "insert misses=" << *misses << std::endl;
    std::exit(2);
  }

  for (int run = 0; run < BASELINE_WARMUP + BASELINE_RUNS; run++) {
    *misses = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer timer;
    timer.start();
    if (materialize) {
      retrieve_kernel<D, true>
          <<<static_cast<int>((n * kTileSize + block - 1) / block), block>>>(
              table, d_keys, n, d_out, misses);
    } else {
      retrieve_kernel<D, false>
          <<<static_cast<int>((n * kTileSize + block - 1) / block), block>>>(
              table, d_keys, n, nullptr, misses);
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (*misses != 0) {
      std::cerr << "retrieve misses=" << *misses << std::endl;
      std::exit(3);
    }
    if (run >= BASELINE_WARMUP) {
      std::cout << "WarpCore,"
                << (materialize ? "warpspeed_materialized" : "warpspeed_local")
                << ",retrieve," << std::fixed << std::setprecision(2)
                << target_lf << "," << (run - BASELINE_WARMUP + 1) << ","
                << std::setprecision(6)
                << throughput_bops(n, timer.elapsed_seconds()) << std::endl;
    }
  }

  CUDA_CHECK(cudaFree(d_keys));
  if (d_out != nullptr) CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(misses));
}

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "WarpSpeed-style WarpCore value-size audit, dim=" << DIM
            << " capacity=" << BASELINE_CAPACITY
            << " batch ignored; queries=capacity*LF" << std::endl;

  std::string mode = "both";
  std::vector<float> load_factors;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg.rfind("--mode=", 0) == 0) {
      mode = arg.substr(7);
    } else {
      load_factors.push_back(std::stof(arg));
    }
  }
  if (load_factors.empty()) load_factors = {0.25f};

  std::cout << "library,mode,operation,load_factor,run,throughput_bops"
            << std::endl;
  for (float lf : load_factors) {
    if (mode == "both" || mode == "local") {
      run_one<DIM>(lf, false);
    }
    if (mode == "both" || mode == "materialized") {
      run_one<DIM>(lf, true);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
