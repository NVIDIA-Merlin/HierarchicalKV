/*
 * E8: Baseline Benchmark — Common Utilities
 *
 * Shared CUDA helpers for WarpCore, BGHT, and cuCollections benchmarks.
 * Provides: error checking, CUDA-event timer, gather/scatter kernels,
 * and benchmark constants matching HKV's Config B setup.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

/* ─── CUDA Error Checking ─── */

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

/* ─── Benchmark Constants (Config B by default) ─── */

#ifndef BASELINE_DIM
#define BASELINE_DIM 32
#endif
#ifndef BASELINE_CAPACITY
#define BASELINE_CAPACITY 134217728
#endif
#ifndef BASELINE_BATCH_SIZE
#define BASELINE_BATCH_SIZE 1048576
#endif
#ifndef BASELINE_WARMUP
#define BASELINE_WARMUP 3
#endif
#ifndef BASELINE_RUNS
#define BASELINE_RUNS 5
#endif

static constexpr size_t DIM = BASELINE_DIM;
static constexpr size_t CAPACITY = BASELINE_CAPACITY;
static constexpr size_t BATCH_SIZE = BASELINE_BATCH_SIZE;
static constexpr int WARMUP = BASELINE_WARMUP;
static constexpr int RUNS = BASELINE_RUNS;

/* ─── CUDA Event Timer ─── */

struct CudaTimer {
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  void stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
  }

  // Returns elapsed time in seconds.
  double elapsed_seconds() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return static_cast<double>(ms) * 1e-3;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

/* ─── Gather Kernel ─── */
// out[i * dim .. (i+1)*dim) = values[indices[i] * dim .. ]
// Used after table.find returns indices into the flat value array.

__global__ void gather_values_kernel(float* __restrict__ out,
                                     const float* __restrict__ values,
                                     const uint64_t* __restrict__ indices,
                                     size_t dim, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = n * dim;
  for (size_t t = tid; t < total; t += blockDim.x * gridDim.x) {
    size_t row = t / dim;
    size_t col = t % dim;
    uint64_t idx = indices[row];
    if (idx != ~0ULL) {  // sentinel check
      out[row * dim + col] = values[idx * dim + col];
    }
  }
}

inline void gather_values(float* out, const float* values,
                          const uint64_t* indices, size_t dim, size_t n,
                          cudaStream_t stream = 0) {
  const size_t block = 256;
  const size_t total = n * dim;
  const size_t grid = (total + block - 1) / block;
  gather_values_kernel<<<grid, block, 0, stream>>>(out, values, indices, dim,
                                                   n);
}

/* ─── Scatter Kernel ─── */
// values[indices[i] * dim + col] = in[i * dim + col]
// Used during insert: after table.insert stores (key, index), scatter the
// actual value data into the flat value array.

__global__ void scatter_values_kernel(float* __restrict__ values,
                                      const float* __restrict__ in,
                                      const uint64_t* __restrict__ indices,
                                      size_t dim, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = n * dim;
  for (size_t t = tid; t < total; t += blockDim.x * gridDim.x) {
    size_t row = t / dim;
    size_t col = t % dim;
    uint64_t idx = indices[row];
    values[idx * dim + col] = in[row * dim + col];
  }
}

inline void scatter_values(float* values, const float* in,
                           const uint64_t* indices, size_t dim, size_t n,
                           cudaStream_t stream = 0) {
  const size_t block = 256;
  const size_t total = n * dim;
  const size_t grid = (total + block - 1) / block;
  scatter_values_kernel<<<grid, block, 0, stream>>>(values, in, indices, dim,
                                                    n);
}

/* ─── Count Successful Inserts (index-based) ─── */
// After insert + find-back, count how many keys got non-sentinel indices.
// Used by BGHT, P2BHT, cuCollections to compute actual insert success count.

inline size_t count_successful_indices(const uint64_t* d_indices, size_t n,
                                       uint64_t sentinel = ~0ULL) {
  std::vector<uint64_t> h_indices(n);
  CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, n * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    if (h_indices[i] != sentinel) count++;
  }
  return count;
}

/* ─── Throughput Helper ─── */
// Throughput in Billion-KV/s.

inline double throughput_bkvs(size_t n_keys, double seconds) {
  return static_cast<double>(n_keys) / seconds / (1024.0 * 1024.0 * 1024.0);
}
