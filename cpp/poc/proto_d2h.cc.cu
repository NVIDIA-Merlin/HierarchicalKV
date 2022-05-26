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

void create_random_offset(int *offset, int num, int range) {
  std::unordered_set<int> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<int> distr;
  int i = 0;

  while (numbers.size() < num) {
    numbers.insert(distr(eng) % range);
  }

  for (const int num : numbers) {
    offset[i++] = num;
  }
}

void create_continuous_offset(int *offset, int num) {
  for (int i = 0; i < num; i++) {
    offset[i] = i;
  }
}

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

__global__ void d2h_nothing(const Vector *__restrict src,
                            Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
  }
}

__global__ void d2h_immediate_number(const Vector *__restrict src,
                                     Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = 0.1f;
  }
}

__global__ void d2h_hbm_data(const Vector *__restrict src,
                             Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];
  }
}

__global__ void d2h_hbm_data_continuous(const Vector *__restrict src,
                                        Vector *__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    dst[vec_index].values[dim_index] = src[vec_index].values[dim_index];
  }
}

__global__ void create_fake_ptr(const Vector *__restrict dst,
                                Vector **__restrict vectors, int *offset,
                                int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    vectors[tid] = (Vector *)((Vector *)dst + offset[tid]);
  }
}

int main() {
  constexpr int KEY_NUM = 1024 * 1024;     // assume keys num = 1M
  constexpr int INIT_SIZE = KEY_NUM * 32;  // assume keys num = 1M
  constexpr int N = KEY_NUM * DIM;         // problem size
  constexpr int TEST_TIMES = 1;
  constexpr size_t vectors_size = INIT_SIZE * sizeof(Vector);

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  unsigned int h_size = 0;
  unsigned int *d_size;

  cudaMalloc(&d_size, sizeof(unsigned int));
  cudaMemset(&d_size, 0, sizeof(unsigned int));

  int *h_offset;
  int *d_offset;

  cudaMallocHost(&h_offset, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_offset, sizeof(int) * KEY_NUM);
  cudaMemset(&h_offset, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_offset, 0, sizeof(int) * KEY_NUM);

  Vector *src;
  Vector *dst;
  Vector **dst_ptr;
  cudaMalloc(&src, KEY_NUM * sizeof(Vector));
  cudaMalloc(&dst_ptr, KEY_NUM * sizeof(Vector *));
  cudaMallocHost(&dst, vectors_size);

  create_random_offset(h_offset, KEY_NUM, INIT_SIZE);
  //   create_continuous_offset(h_offset, KEY_NUM);
  cudaMemcpy(d_offset, h_offset, sizeof(int) * KEY_NUM, cudaMemcpyHostToDevice);
  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, d_offset, KEY_NUM);
  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  // 1.
  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_nothing<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] nothing d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  // 2.
  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_immediate_number<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] constant d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  // 3.
  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  // 4.
  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data_continuous<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data continous d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  cudaMemcpy(&h_size, d_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFreeHost(dst);
  cudaFreeHost(h_offset);
  cudaFree(dst_ptr);
  cudaFree(src);
  cudaFree(d_size);
  cudaFree(d_offset);

  std::cout << "size=" << h_size;
  std::cout << ", COMPLETED SUCCESSFULLY\n";

  return 0;
}
