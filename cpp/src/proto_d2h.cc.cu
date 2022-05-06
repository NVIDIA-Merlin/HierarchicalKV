// by test

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

typedef float V;

constexpr int DIM = 64;
struct Vector {
  V values[DIM];
};

__global__ void test(const Vector *__restrict src, Vector **__restrict dst,
                     int N, unsigned int *size) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    ((Vector *)(dst[vec_index]))->values[dim_index] =
        src[vec_index].values[dim_index];
  }
}

__global__ void d2h(const Vector *__restrict src, Vector **__restrict dst,
                    int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / (DIM / 2));
    int dim_index = tid % (DIM / 2);

    (*(dst[vec_index])).values[dim_index] = 0.1f;
    (*(dst[vec_index])).values[dim_index + 1] = 0.1f;
    //  (*(dst[vec_index])).values[dim_index] =
    //  src[vec_index].values[dim_index];
  }
}

__global__ void create_fake_ptr(const Vector *__restrict dst,
                                Vector **__restrict vectors, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    vectors[tid] = (Vector *)((Vector *)dst + tid * (tid % 31));
    //     vectors[tid] = (Vector *)((Vector *)dst + tid * 31);
  }
}

int main() {
  constexpr int KEY_NUM = 1024 * 1024;  // assume keys num = 1M
  constexpr int N = KEY_NUM * DIM / 2;  // problem size
  constexpr int TEST_TIMES = 10;

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  unsigned int h_size = 0;
  unsigned int *d_size;

  cudaMalloc(&d_size, sizeof(unsigned int));
  cudaMemset(&d_size, 0, sizeof(unsigned int));

  Vector *src;
  Vector *dst;
  Vector **dst_ptr;
  cudaMalloc(&src, KEY_NUM * sizeof(Vector));
  cudaMallocManaged(&dst_ptr, KEY_NUM * sizeof(Vector *));
  size_t vectors_size = KEY_NUM * 32 * sizeof(Vector);
  cudaMallocHost(&dst, vectors_size);
  std::cout << "vectors_size=" << vectors_size << std::endl;

  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, KEY_NUM);

  cudaDeviceSynchronize();
  for (int i = 0; i < TEST_TIMES; i++) {
    auto start_test = std::chrono::steady_clock::now();
    d2h<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
    cudaDeviceSynchronize();
    auto end_test = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_test = end_test - start_test;
    printf("[timing] id=%d, test=%.2fms\n", i, diff_test.count() * 1000);
  }

  cudaMemcpy(&h_size, d_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFreeHost(dst);
  cudaFree(dst_ptr);
  cudaFree(src);
  cudaFree(d_size);

  std::cout << "size=" << h_size;
  //   assert(h_size == NUM_BLOCKS * NUM_THREADS);
  std::cout << ", COMPLETED SUCCESSFULLY\n";

  return 0;
}
