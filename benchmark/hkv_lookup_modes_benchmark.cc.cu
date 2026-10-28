/*
 * Review-response P0: HKV lookup-mode audit.
 *
 * Measures key-only membership lookup (`contains`) against value-returning
 * lookup (`find`) under the same table and query setup as the E8 baseline.
 *
 * Output: CSV
 *   Decimal completed requests per host-wall second; lookup-only contracts.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using namespace nv::merlin;
using namespace benchmark;

#ifndef HKV_LOOKUP_DIM
#define HKV_LOOKUP_DIM 32
#endif
#ifndef HKV_LOOKUP_CAPACITY
#define HKV_LOOKUP_CAPACITY 134217728
#endif
#ifndef HKV_LOOKUP_BATCH_SIZE
#define HKV_LOOKUP_BATCH_SIZE 1048576
#endif
#ifndef HKV_LOOKUP_WARMUP
#define HKV_LOOKUP_WARMUP 3
#endif
#ifndef HKV_LOOKUP_RUNS
#define HKV_LOOKUP_RUNS 5
#endif
#ifndef HKV_LOOKUP_HBM_GB
#define HKV_LOOKUP_HBM_GB 16
#endif
#ifndef HKV_LOOKUP_BLOCK_SIZE
#define HKV_LOOKUP_BLOCK_SIZE 128
#endif
#ifndef HKV_LOOKUP_API_LOCK
#define HKV_LOOKUP_API_LOCK 1
#endif

static constexpr size_t DIM = HKV_LOOKUP_DIM;
static constexpr size_t INIT_CAPACITY = HKV_LOOKUP_CAPACITY;
static constexpr size_t HBM_GB = HKV_LOOKUP_HBM_GB;
static constexpr size_t BATCH_SIZE = HKV_LOOKUP_BATCH_SIZE;
static constexpr int WARMUP = HKV_LOOKUP_WARMUP;
static constexpr int RUNS = HKV_LOOKUP_RUNS;
static constexpr int BLOCK_SIZE = HKV_LOOKUP_BLOCK_SIZE;
static constexpr bool API_LOCK = (HKV_LOOKUP_API_LOCK != 0);

using HkvTable = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

static_assert(DIM > 0 && INIT_CAPACITY > 0 && BATCH_SIZE > 0 && RUNS > 0 &&
                  WARMUP >= 0,
              "Nonzero benchmark work is required");

__host__ __device__ static V payload_value(K key, size_t component) {
  if (component < 4)
    return static_cast<V>((key >> (16 * component)) & 0xffffULL);
  K mixed = key ^ static_cast<K>(component);
  mixed += 0x9e3779b97f4a7c15ULL;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<V>((mixed ^ (mixed >> 31)) & 0xffffffULL);
}

__global__ void init_payload_kernel(const K* keys, V* values, size_t count) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; index < count * DIM; index += stride)
    values[index] = payload_value(keys[index / DIM], index % DIM);
}

class CountingAllocator : public DefaultAllocator {
 public:
  std::array<size_t, 4> live{};
  std::array<size_t, 4> peak{};

  void alloc(MemoryType type, void** pointer, size_t bytes,
             unsigned int flags = cudaHostAllocDefault) override {
    DefaultAllocator::alloc(type, pointer, bytes, flags);
    record(type, *pointer, bytes);
  }
  void alloc_async(MemoryType type, void** pointer, size_t bytes,
                   cudaStream_t stream) override {
    DefaultAllocator::alloc_async(type, pointer, bytes, stream);
    record(type, *pointer, bytes);
  }
  void free(MemoryType type, void* pointer) override {
    forget(type, pointer);
    DefaultAllocator::free(type, pointer);
  }
  void free_async(MemoryType type, void* pointer,
                  cudaStream_t stream) override {
    forget(type, pointer);
    DefaultAllocator::free_async(type, pointer, stream);
  }

 private:
  std::unordered_map<void*, size_t> sizes;
  void record(MemoryType type, void* pointer, size_t bytes) {
    sizes[pointer] = bytes;
    live[type] += bytes;
    peak[type] = std::max(peak[type], live[type]);
  }
  void forget(MemoryType type, void* pointer) {
    if (!pointer) return;
    live[type] -= sizes.at(pointer);
    sizes.erase(pointer);
  }
};

class WallTimer {
 public:
  explicit WallTimer(cudaStream_t stream) : stream_(stream) {}
  void start() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    start_ = std::chrono::steady_clock::now();
  }
  void end() { end_ = std::chrono::steady_clock::now(); }
  double getResult() const {
    return std::chrono::duration<double>(end_ - start_).count();
  }

 private:
  cudaStream_t stream_;
  std::chrono::steady_clock::time_point start_, end_;
};

__global__ void touch_ptr_values_kernel(V** ptrs, const bool* founds,
                                        unsigned long long* errors, size_t dim,
                                        size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  float acc = 0.0f;
  for (size_t d = 0; d < dim; d++) {
    acc += ptrs[idx][d];
  }
  if (acc == -1.0f) {
    atomicAdd(errors, 1ULL);
  }
}

__global__ void gather_ptr_values_kernel(V** ptrs, const bool* founds, V* out,
                                         unsigned long long* errors, size_t dim,
                                         size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * dim;
  if (tid >= total) return;
  const size_t idx = tid / dim;
  const size_t d = tid - idx * dim;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  out[tid] = ptrs[idx][d];
}

__global__ void gather_ptr_values_by_key_kernel(V** ptrs, const bool* founds,
                                                V* out,
                                                unsigned long long* errors,
                                                size_t dim, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  V* src = ptrs[idx];
  V* dst = out + idx * dim;
  for (size_t d = 0; d < dim; d++) {
    dst[d] = src[d];
  }
}

__global__ void gather_ptr_values_warp_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t dim, size_t n) {
  constexpr int WARP_SIZE = 32;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  if (warp_id >= n) return;
  if (!founds[warp_id] || ptrs[warp_id] == nullptr) {
    if (lane == 0) atomicAdd(errors, 1ULL);
    return;
  }
  V* src = ptrs[warp_id];
  V* dst = out + warp_id * dim;
  for (size_t d = lane; d < dim; d += WARP_SIZE) {
    dst[d] = src[d];
  }
}

__global__ void gather_ptr_values_vec2_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float2* src = reinterpret_cast<const float2*>(ptrs[idx]);
  float2* dst = reinterpret_cast<float2*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_vec4_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float4* src = reinterpret_cast<const float4*>(ptrs[idx]);
  float4* dst = reinterpret_cast<float4*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_nofound_kernel(V** ptrs, V* out,
                                                 unsigned long long* errors,
                                                 size_t dim, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * dim;
  if (tid >= total) return;
  const size_t idx = tid / dim;
  const size_t d = tid - idx * dim;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  out[tid] = ptrs[idx][d];
}

__global__ void gather_ptr_values_vec2_nofound_kernel(
    V** ptrs, V* out, unsigned long long* errors, size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float2* src = reinterpret_cast<const float2*>(ptrs[idx]);
  float2* dst = reinterpret_cast<float2*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_vec4_nofound_kernel(
    V** ptrs, V* out, unsigned long long* errors, size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float4* src = reinterpret_cast<const float4*>(ptrs[idx]);
  float4* dst = reinterpret_cast<float4*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

static K prepopulate(std::shared_ptr<HkvTable>& table, size_t target_count,
                     cudaStream_t stream) {
  K* h_keys;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));

  K* d_keys;
  V* d_vectors;
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));

  K start = 0;
  while (start < target_count) {
    size_t cur = std::min(BATCH_SIZE, target_count - start);
    for (size_t i = 0; i < cur; i++) h_keys[i] = start + i;
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    init_payload_kernel<<<std::min<size_t>((cur * DIM + 255) / 256, 65535), 256,
                          0, stream>>>(d_keys, d_vectors, cur);
    CUDA_CHECK(cudaGetLastError());
    table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  return start;
}

static std::shared_ptr<HkvTable> build_table(float target_lf,
                                             cudaStream_t stream,
                                             CountingAllocator* allocator) {
  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
  options.block_size = BLOCK_SIZE;
  options.api_lock = API_LOCK;

  auto table = std::make_shared<HkvTable>();
  table->init(options, allocator);
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  prepopulate(table, target_n, stream);
  return table;
}

static void prepare_queries(K* h_keys, float target_lf,
                            const std::string& query_pattern, uint64_t seed) {
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  if (query_pattern == "sequential") {
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_keys[i] = static_cast<K>(i % target_n);
    }
  } else {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<K> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_keys[i] = dist(rng);
    }
  }
}

static void verify_all_found(const bool* d_found, const std::string& label) {
  std::vector<uint8_t> h_found(BATCH_SIZE);
  CUDA_CHECK(cudaMemcpy(h_found.data(), d_found, BATCH_SIZE * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  size_t found_count = 0;
  for (uint8_t found : h_found) {
    found_count += (found != 0);
  }
  if (found_count != BATCH_SIZE) {
    throw std::runtime_error(label + " failed successful-query check");
  }
}

static void verify_values(const K* keys, const V* device_values) {
  const size_t chunk = 2048;
  std::vector<V> values(std::min(chunk, BATCH_SIZE) * DIM);
  for (size_t begin = 0; begin < BATCH_SIZE; begin += chunk) {
    size_t count = std::min(chunk, BATCH_SIZE - begin);
    CUDA_CHECK(cudaMemcpy(values.data(), device_values + begin * DIM,
                          count * DIM * sizeof(V), cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < count; ++row)
      for (size_t component = 0; component < DIM; ++component)
        if (values[row * DIM + component] !=
            payload_value(keys[begin + row], component))
          throw std::runtime_error("HKV exact payload oracle failed row=" +
                                   std::to_string(begin + row));
  }
}

static void verify_result(const std::string& contract, bool has_flags,
                          const K* keys, V** pointers, bool* found, V* values,
                          unsigned long long* errors, cudaStream_t stream) {
  if (has_flags) verify_all_found(found, contract);
  if (contract == "pointer_only" || contract == "pointer_touch") {
    gather_ptr_values_nofound_kernel<<<(BATCH_SIZE * DIM + 255) / 256, 256, 0,
                                       stream>>>(pointers, values, errors, DIM,
                                                 BATCH_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  unsigned long long count = 0;
  CUDA_CHECK(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
  if (count)
    throw std::runtime_error("HKV gather/pointer errors=" +
                             std::to_string(count));
  if (contract != "membership_only") verify_values(keys, values);
}

static std::string row_suffix(double seconds, const std::string& contract,
                              const CountingAllocator& allocator,
                              size_t capacity, size_t resident, uint64_t seed) {
  if (!(seconds > 0) || !std::isfinite(seconds))
    throw std::runtime_error("Invalid wall duration");
  const size_t request_bytes =
      BATCH_SIZE * (sizeof(K) + DIM * sizeof(V) + sizeof(V*) + sizeof(bool)) +
      sizeof(unsigned long long);
  std::ostringstream result;
  result << std::setprecision(12) << ',' << seconds * 1000.0 << ','
         << BATCH_SIZE << ",lookup_only_host_wall," << contract << ",PASS,"
         << seed << ',' << capacity << ',' << resident << ','
         << capacity * DIM * sizeof(V) << ',' << allocator.live[Device] << ','
         << allocator.live[Pinned] << ',' << allocator.live[Host] << ','
         << allocator.peak[Device] << ',' << request_bytes << ','
         << BATCH_SIZE * sizeof(K)
         << ",allocator_requested_bytes_excludes_cuda_pool_reservation,"
            "unseparated,0,0,0,0,0,"
         << BATCH_SIZE << ",target_generator_fraction,per_batch_oracle";
  return result.str();
}

static void run_lookup_modes(float target_lf, const std::string& mode,
                             const std::string& query_pattern,
                             cudaStream_t stream, uint64_t seed) {
  std::cerr << "HKV lookup modes LF=" << std::fixed << std::setprecision(9)
            << target_lf << std::endl;

  CountingAllocator allocator;
  auto table = build_table(target_lf, stream, &allocator);
  const size_t resident = table->size(stream);

  K* h_keys;
  K* d_keys;
  V* d_vectors;
  V** d_value_ptrs;
  bool* d_found;
  unsigned long long* d_errors;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_value_ptrs, BATCH_SIZE * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_errors, sizeof(unsigned long long)));

  prepare_queries(h_keys, target_lf, query_pattern, seed);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));

  if (mode == "both" || mode == "all" || mode == "academic" ||
      mode == "key_only") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->contains(BATCH_SIZE, d_keys, d_found, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("membership_only", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "membership_only", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,key_only,contains," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    verify_all_found(d_found, "HKV key_only contains");
  }

  if (mode == "all" || mode == "academic" || mode == "ptr_only" ||
      mode == "ptr_touch" || mode == "ptr_gather" ||
      mode == "ptr_only_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs,
                  mode == "ptr_only_nofound" ? nullptr : d_found, nullptr,
                  stream, true);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("pointer_only", mode != "ptr_only_nofound", h_keys,
                    d_value_ptrs, d_found, d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "pointer_only", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,"
                  << (mode == "ptr_only_nofound" ? "academic_ptr_only_nofound"
                                                 : "academic_ptr_only")
                  << ","
                  << (mode == "ptr_only_nofound" ? "find_ptr_nofound"
                                                 : "find_ptr")
                  << "," << std::fixed << std::setprecision(9) << target_lf
                  << "," << (run - WARMUP + 1) << "," << std::setprecision(9)
                  << tp << suffix << std::endl;
      }
    }
    if (mode != "ptr_only_nofound") {
      verify_all_found(d_found, "HKV academic ptr-only find");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "ptr_touch") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      touch_ptr_values_kernel<<<(BATCH_SIZE + 255) / 256, 256, 0, stream>>>(
          d_value_ptrs, d_found, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("pointer_touch", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "pointer_touch", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_touch,find_ptr_touch," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-touch failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_kernel<<<(BATCH_SIZE * DIM + 255) / 256, 256, 0,
                                 stream>>>(d_value_ptrs, d_found, d_vectors,
                                           d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather,find_ptr_gather," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather_key") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_by_key_kernel<<<(BATCH_SIZE + 255) / 256, 256, 0,
                                        stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_key,find_ptr_gather_key,"
                  << std::fixed << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-key failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather_warp") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_warp_kernel<<<(BATCH_SIZE * 32 + 127) / 128, 128, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_warp,find_ptr_gather_warp,"
                  << std::fixed << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-warp failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "gather_sweep" ||
       mode == "ptr_gather_vec2") &&
      DIM % 2 == 0) {
    const size_t chunks = DIM / 2;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_vec2_kernel<<<(BATCH_SIZE * chunks + 255) / 256, 256, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_vec2,find_ptr_gather_vec2,"
                  << std::fixed << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec2 failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "gather_sweep" ||
       mode == "ptr_gather_vec4") &&
      DIM % 4 == 0) {
    const size_t chunks = DIM / 4;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_vec4_kernel<<<(BATCH_SIZE * chunks + 255) / 256, 256, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_vec4,find_ptr_gather_vec4,"
                  << std::fixed << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec4 failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
      mode == "ptr_gather_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_nofound_kernel<<<(BATCH_SIZE * DIM + 255) / 256, 256, 0,
                                         stream>>>(d_value_ptrs, d_vectors,
                                                   d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", false, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_nofound,"
                  << "find_ptr_gather_nofound," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-nofound failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
       mode == "ptr_gather_vec2_nofound") &&
      DIM % 2 == 0) {
    const size_t chunks = DIM / 2;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_vec2_nofound_kernel<<<(BATCH_SIZE * chunks + 255) / 256,
                                              256, 0, stream>>>(
          d_value_ptrs, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", false, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_vec2_nofound,"
                  << "find_ptr_gather_vec2_nofound," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec2-nofound failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
       mode == "ptr_gather_vec4_nofound") &&
      DIM % 4 == 0) {
    const size_t chunks = DIM / 4;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_vec4_nofound_kernel<<<(BATCH_SIZE * chunks + 255) / 256,
                                              256, 0, stream>>>(
          d_value_ptrs, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("full_row_gather", false, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "full_row_gather", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_ptr_gather_vec4_nofound,"
                  << "find_ptr_gather_vec4_nofound," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec4-nofound failed");
    }
  }

  if (mode == "both" || mode == "all" || mode == "value_returning") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("direct_full_row", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "direct_full_row", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,value_returning,find," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    verify_all_found(d_found, "HKV value_returning find");
  }

  if (mode == "all" || mode == "academic" || mode == "readonly_fast_sweep" ||
      mode == "readonly_fast") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find_readonly_fast(BATCH_SIZE, d_keys, d_vectors, d_found, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("readonly_full_row", true, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "readonly_full_row", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_readonly_fast,find_readonly_fast,"
                  << std::fixed << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
    verify_all_found(d_found, "HKV readonly_fast find");
  }

  if (mode == "all" || mode == "academic" || mode == "readonly_fast_sweep" ||
      mode == "readonly_fast_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaMemsetAsync(d_vectors, 0xff, BATCH_SIZE * DIM * sizeof(V),
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = WallTimer(stream);
      timer.start();
      table->find_readonly_fast(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      verify_result("readonly_full_row", false, h_keys, d_value_ptrs, d_found,
                    d_vectors, d_errors, stream);
      const auto suffix =
          row_suffix(timer.getResult(), "readonly_full_row", allocator,
                     table->capacity(), resident, seed);

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / 1e9;
        std::cout << "HKV,academic_readonly_fast_nofound,"
                  << "find_readonly_fast_nofound," << std::fixed
                  << std::setprecision(9) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(9) << tp
                  << suffix << std::endl;
      }
    }
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_value_ptrs));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_errors));
}

static uint64_t shard_clock_ns() {
  timespec stamp;
  if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0)
    throw std::runtime_error("CLOCK_MONOTONIC unavailable");
  return static_cast<uint64_t>(stamp.tv_sec) * 1000000000ULL + stamp.tv_nsec;
}

static size_t shard_argument(const char* text) {
  std::string value(text);
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Invalid unsigned shard argument");
  size_t consumed = 0;
  auto number = std::stoull(value, &consumed);
  if (consumed != value.size() || number > SIZE_MAX)
    throw std::runtime_error("Out-of-range shard argument");
  return static_cast<size_t>(number);
}

static int run_shard_worker(int argc, char** argv) {
  static_assert(sizeof(bool) == 1 && sizeof(K) == 8 && sizeof(V) == 4,
                "Unsupported shard wire types");
  if (argc != 8)
    throw std::runtime_error(
        "--shard-worker rank world capacity resident max_batch cache_bytes");
  const size_t rank = shard_argument(argv[2]);
  const size_t world = shard_argument(argv[3]);
  const size_t capacity = shard_argument(argv[4]);
  const size_t resident = shard_argument(argv[5]);
  const size_t max_batch = shard_argument(argv[6]);
  const size_t cache_budget = shard_argument(argv[7]);
  if (world == 0 || rank >= world || capacity < 128 ||
      (capacity & (capacity - 1)) || resident > capacity || max_batch == 0 ||
      max_batch > static_cast<size_t>(INT_MAX) / DIM ||
      capacity > static_cast<size_t>(INT_MAX) ||
      world > UINT64_MAX / capacity || cache_budget == 0)
    throw std::runtime_error("Unsupported shard shape/budget");
  CountingAllocator allocator;
  auto table = std::make_shared<HkvTable>();
  HashTableOptions options;
  options.init_capacity = options.max_capacity = capacity;
  options.dim = DIM;
  options.max_hbm_for_vectors = capacity * DIM * sizeof(V);
  options.block_size = BLOCK_SIZE;
  options.api_lock = API_LOCK;
  table->init(options, &allocator);
  if (table->capacity() != capacity)
    throw std::runtime_error("Shard capacity rounded unexpectedly");
  K *host_keys, *device_keys;
  V *host_values, *device_values;
  bool *host_found, *device_found;
  CUDA_CHECK(cudaMallocHost(&host_keys, max_batch * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&host_values, max_batch * DIM * sizeof(V)));
  CUDA_CHECK(cudaMallocHost(&host_found, max_batch * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&device_keys, max_batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&device_values, max_batch * DIM * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&device_found, max_batch * sizeof(bool)));
  auto reset = [&]() {
    CUDA_CHECK(cudaMemsetAsync(device_found, 0, max_batch * sizeof(bool)));
    CUDA_CHECK(
        cudaMemsetAsync(device_values, 0xff, max_batch * DIM * sizeof(V)));
  };
  for (size_t begin = 0; begin < resident; begin += max_batch) {
    size_t count = std::min(max_batch, resident - begin);
    for (size_t row = 0; row < count; ++row)
      host_keys[row] = (begin + row) * world + rank;
    CUDA_CHECK(cudaMemcpy(device_keys, host_keys, count * sizeof(K),
                          cudaMemcpyHostToDevice));
    init_payload_kernel<<<std::min<size_t>((count * DIM + 255) / 256, 65535),
                          256>>>(device_keys, device_values, count);
    CUDA_CHECK(cudaGetLastError());
    table->insert_or_assign(count, device_keys, device_values, nullptr, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
  }
  if (table->size(0) != resident)
    throw std::runtime_error("Prefill eviction changed initial resident set");
  std::fill(host_keys, host_keys + max_batch, static_cast<K>(rank));
  CUDA_CHECK(cudaMemcpy(device_keys, host_keys, max_batch * sizeof(K),
                        cudaMemcpyHostToDevice));
  reset();
  table->find(max_batch, device_keys, device_values, device_found, nullptr, 0);
  CUDA_CHECK(cudaStreamSynchronize(0));
  for (size_t begin = 0; begin < resident; begin += max_batch) {
    size_t count = std::min(max_batch, resident - begin);
    for (size_t row = 0; row < count; ++row)
      host_keys[row] = (begin + row) * world + rank;
    CUDA_CHECK(cudaMemcpy(device_keys, host_keys, count * sizeof(K),
                          cudaMemcpyHostToDevice));
    reset();
    table->find(count, device_keys, device_values, device_found, nullptr, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpy(host_found, device_found, count * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_values, device_values, count * DIM * sizeof(V),
                          cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < count; ++row) {
      if (!host_found[row])
        throw std::runtime_error("Missing initial resident key");
      for (size_t component = 0; component < DIM; ++component)
        if (host_values[row * DIM + component] !=
            payload_value(host_keys[row], component))
          throw std::runtime_error("Initial shard payload mismatch");
    }
  }
  const auto cache_bytes = allocator.live;
  const size_t requested_cache =
      cache_bytes[Device] + cache_bytes[Pinned] + cache_bytes[Host];
  if (requested_cache > cache_budget)
    throw std::runtime_error("Cache allocation exceeds fixed byte budget");
  const size_t padding_bytes = cache_budget - requested_cache;
  void* padding = nullptr;
  if (padding_bytes) CUDA_CHECK(cudaMalloc(&padding, padding_bytes));
  const size_t request_bytes =
      max_batch * (sizeof(K) + DIM * sizeof(V) + sizeof(bool));
  cudaEvent_t event_start, event_stop;
  CUDA_CHECK(cudaEventCreate(&event_start));
  CUDA_CHECK(cudaEventCreate(&event_stop));
  std::cout << "READY " << rank << ' ' << world << ' ' << capacity << ' '
            << resident << ' ' << cache_bytes[Device] << ' '
            << cache_bytes[Pinned] << ' ' << cache_bytes[Host] << ' '
            << padding_bytes << ' ' << cache_budget << ' ' << request_bytes
            << ' ' << request_bytes << ' ' << DIM << ' ' << BLOCK_SIZE << ' '
            << static_cast<int>(API_LOCK) << '\n'
            << std::flush;
  size_t sequence = 0;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "STOP") break;
    std::istringstream preparation(line);
    std::string command, extra;
    size_t received_sequence = SIZE_MAX;
    if (!(preparation >> command >> received_sequence) ||
        command != "PREPARE" || received_sequence != sequence ||
        preparation >> extra)
      throw std::runtime_error("Invalid PREPARE sequence");
    reset();
    CUDA_CHECK(cudaStreamSynchronize(0));
    std::cout << "PREPARED " << sequence << '\n' << std::flush;
    if (!std::getline(std::cin, line)) throw std::runtime_error("Missing GO");
    std::istringstream launch(line);
    size_t count = SIZE_MAX;
    if (!(launch >> command >> received_sequence >> count) || command != "GO" ||
        received_sequence != sequence || count > max_batch || launch >> extra)
      throw std::runtime_error("Invalid GO sequence/count");
    const uint64_t started = shard_clock_ns();
    if (!std::cin.read(reinterpret_cast<char*>(host_keys), count * sizeof(K)))
      throw std::runtime_error("Truncated shard request");
    for (size_t row = 0; row < count; ++row)
      if (host_keys[row] % world != rank)
        throw std::runtime_error("Request routed to wrong shard");
    float gpu_ms = 0;
    if (count) {
      CUDA_CHECK(cudaMemcpyAsync(device_keys, host_keys, count * sizeof(K),
                                 cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventRecord(event_start));
      table->find(count, device_keys, device_values, device_found, nullptr, 0);
      CUDA_CHECK(cudaEventRecord(event_stop));
      CUDA_CHECK(cudaMemcpyAsync(host_values, device_values,
                                 count * DIM * sizeof(V),
                                 cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpyAsync(host_found, device_found, count * sizeof(bool),
                                 cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaStreamSynchronize(0));
      CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, event_start, event_stop));
    }
    const uint64_t stopped = shard_clock_ns();
    if (allocator.live != cache_bytes)
      throw std::runtime_error("Cache allocation changed after READY");
    size_t hits = 0;
    for (size_t row = 0; row < count; ++row) hits += host_found[row];
    std::cout << "DONE " << sequence << ' ' << count << ' ' << hits << ' '
              << started << ' ' << stopped << ' ' << std::setprecision(12)
              << gpu_ms << '\n';
    std::cout.write(reinterpret_cast<char*>(host_found), count * sizeof(bool));
    std::cout.write(reinterpret_cast<char*>(host_values),
                    count * DIM * sizeof(V));
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("Shard output failed");
    ++sequence;
  }
  CUDA_CHECK(cudaEventDestroy(event_start));
  CUDA_CHECK(cudaEventDestroy(event_stop));
  CUDA_CHECK(cudaFree(padding));
  CUDA_CHECK(cudaFree(device_keys));
  CUDA_CHECK(cudaFree(device_values));
  CUDA_CHECK(cudaFree(device_found));
  CUDA_CHECK(cudaFreeHost(host_keys));
  CUDA_CHECK(cudaFreeHost(host_values));
  CUDA_CHECK(cudaFreeHost(host_found));
  return 0;
}

int main(int argc, char** argv) try {
  if (argc > 1 && std::string(argv[1]) == "--shard-worker")
    return run_shard_worker(argc, argv);
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review P0: HKV lookup-mode audit" << std::endl;
  std::cerr << "Config: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", batch=" << BATCH_SIZE << ", warmup=" << WARMUP
            << ", runs=" << RUNS << ", block=" << BLOCK_SIZE
            << ", HBM=" << HBM_GB
            << "GB, api_lock=" << (API_LOCK ? "on" : "off") << ", kLru"
            << std::endl;

  std::string mode = "both";
  std::string query_pattern = "random";
  uint64_t seed = 12345;
  std::vector<float> load_factors;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg.rfind("--mode=", 0) == 0) {
      mode = arg.substr(7);
    } else if (arg == "--query" && i + 1 < argc) {
      query_pattern = argv[++i];
    } else if (arg.rfind("--query=", 0) == 0) {
      query_pattern = arg.substr(8);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg.rfind("--seed=", 0) == 0) {
      seed = std::stoull(arg.substr(7));
    } else {
      load_factors.push_back(std::stof(arg));
    }
  }

  if (mode != "both" && mode != "all" && mode != "key_only" &&
      mode != "value_returning" && mode != "academic" && mode != "ptr_only" &&
      mode != "ptr_touch" && mode != "ptr_gather" && mode != "gather_sweep" &&
      mode != "ptr_gather_key" && mode != "ptr_gather_warp" &&
      mode != "ptr_gather_vec2" && mode != "ptr_gather_vec4" &&
      mode != "nofound_sweep" && mode != "ptr_only_nofound" &&
      mode != "ptr_gather_nofound" && mode != "ptr_gather_vec2_nofound" &&
      mode != "ptr_gather_vec4_nofound" && mode != "readonly_fast_sweep" &&
      mode != "readonly_fast" && mode != "readonly_fast_nofound") {
    std::cerr << "Unknown mode: " << mode
              << " (expected both, all, key_only, value_returning, academic, "
                 "ptr_only, ptr_touch, ptr_gather, gather_sweep, "
                 "ptr_gather_key, ptr_gather_warp, ptr_gather_vec2, "
                 "ptr_gather_vec4, nofound_sweep, ptr_only_nofound, "
                 "ptr_gather_nofound, ptr_gather_vec2_nofound, "
                 "ptr_gather_vec4_nofound, readonly_fast_sweep, "
                 "readonly_fast, readonly_fast_nofound)"
              << std::endl;
    return 1;
  }
  if (query_pattern != "random" && query_pattern != "sequential") {
    std::cerr << "Unknown query pattern: " << query_pattern
              << " (expected random or sequential)" << std::endl;
    return 1;
  }

  if (load_factors.empty()) {
    load_factors = {0.25f, 0.50f, 0.75f, 1.00f};
  }

  if ((mode.find("vec2") != std::string::npos && DIM % 2 != 0) ||
      (mode.find("vec4") != std::string::npos && DIM % 4 != 0))
    throw std::runtime_error("Requested vector mode is incompatible with DIM");
  for (float load_factor : load_factors)
    if (!std::isfinite(load_factor) || load_factor <= 0 || load_factor > 1 ||
        static_cast<size_t>(INIT_CAPACITY * load_factor) == 0)
      throw std::runtime_error(
          "Load factor must produce nonzero resident input in (0,1]");
  std::cerr << "Lookup-only, not cache service; no admission/writeback or "
               "speedup comparison. "
               "Wall timing excludes setup, resets, oracle/readback and CSV; "
               "metadata/payload allocation "
               "split is unseparated, logical payload capacity is not "
               "allocated bytes. "
               "Allocator-requested memory excludes CUDA pool "
               "reservation/runtime overhead and CPU oracle storage. "
               "Pointer/read-only modes are immutable-table diagnostics; "
               "in-process runs are not independent.\n";
  std::cout << "library,mode,operation,load_factor,run,throughput_grequests_s,"
               "elapsed_ms,completed_requests,"
               "timing_scope,output_contract,oracle_status,seed,capacity_slots,"
               "resident_slots,"
               "logical_payload_capacity_bytes,hkv_allocator_device_bytes,hkv_"
               "allocator_pinned_bytes,"
               "hkv_allocator_host_bytes,hkv_allocator_peak_device_bytes,"
               "request_device_bytes,request_pinned_bytes,"
               "memory_scope,cache_metadata_bytes,misses,admissions,rejected,"
               "evictions,failed,hits,load_factor_scope,measurement_regime"
            << std::endl;

  for (float lf : load_factors) {
    run_lookup_modes(lf, mode, query_pattern, 0, seed);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << std::endl;
  return 1;
}
