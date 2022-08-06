#include <cooperative_groups.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <unordered_set>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

typedef float V;

constexpr int DIM = 64;
struct Vector {
  V values[DIM];
};
constexpr int VECTOR_SIZE = sizeof(Vector);

void create_random_offset(int *offset, int num, int range) {
  std::unordered_set<int> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned int> distr;
  int i = 0;

  while (numbers.size() < num) {
    numbers.insert(distr(eng) % range);
  }

  for (const int num : numbers) {
    offset[i++] = num;
  }
}

void create_random_offset_ordered(int *offset, int num, int range) {
  std::set<int> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned int> distr;
  int i = 0;

  while (numbers.size() < num) {
    numbers.insert(distr(eng) % range);
  }

  for (const int num : numbers) {
    offset[i++] = num;
  }
}

__global__ void d2h_const_data(const Vector *__restrict src,
                               Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    //     (*(dst[vec_index])).values[dim_index] = 0.1f;
    V *vector_addr = (V *)*(dst + vec_index);
    *(vector_addr + dim_index) = 0.1f;
  }
}

__global__ void d2h_hbm_data_all(
    Vector *__restrict src, Vector **__restrict dst,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];

    //     src[vec_index].values[dim_index] =
    //     (*(dst[vec_index])).values[dim_index];
  }
}

#define TILE_SIZE 8

__global__ void d2h_hbm_data(
    Vector *__restrict src, Vector **__restrict dst,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  if (tid < N) {
    uint32_t vec_index = int(tid / TILE_SIZE);
    uint32_t dim_index = 0;
#pragma unroll
    for (uint32_t tile_offset = 0; tile_offset < DIM;
         tile_offset += TILE_SIZE) {
      dim_index = tile_offset + rank;
      (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];
    }
  }
}

__global__ void d2h_hbm_data_with_random_src(
    const Vector *__restrict src, Vector **__restrict dst, int *src_idx,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] =
        src[src_idx[vec_index]].values[dim_index];
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

// void sort_keys(){
//   int A[N] = {1, 4, 2, 8, 5, 7};
//   thrust::sort(thrust::host, A, A + N);
// }

int main() {
  constexpr int KEY_NUM = 1024 * 1024;
  constexpr int INIT_SIZE = KEY_NUM * 32;
  constexpr int N = KEY_NUM * DIM;
  constexpr int TEST_TIMES = 1;
  constexpr size_t vectors_size = INIT_SIZE * sizeof(Vector);

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  int *h_offset;
  int *d_offset;

  int *h_src_idx;
  int *d_src_idx;

  cudaMallocHost(&h_offset, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_offset, sizeof(int) * KEY_NUM);
  cudaMemset(&h_offset, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_offset, 0, sizeof(int) * KEY_NUM);

  cudaMallocHost(&h_src_idx, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_src_idx, sizeof(int) * KEY_NUM);
  cudaMemset(&h_src_idx, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_src_idx, 0, sizeof(int) * KEY_NUM);

  Vector *src;
  Vector *dst;
  Vector **dst_ptr;
  cudaMalloc(&src, KEY_NUM * sizeof(Vector));
  cudaMalloc(&dst_ptr, KEY_NUM * sizeof(Vector *));
  cudaMallocHost(&dst, vectors_size,
                 cudaHostAllocMapped | cudaHostAllocWriteCombined);

  create_random_offset_ordered(h_offset, KEY_NUM, INIT_SIZE);
  cudaMemcpy(d_offset, h_offset, sizeof(int) * KEY_NUM, cudaMemcpyHostToDevice);
  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, d_offset, KEY_NUM);

  //   create_random_offset_ordered(h_src_idx, KEY_NUM, KEY_NUM);
  //   cudaMemcpy(d_src_idx, h_src_idx, sizeof(int) * KEY_NUM,
  //   cudaMemcpyHostToDevice);

  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;

  cudaDeviceSynchronize();
  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_const_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] Constant d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  {
    constexpr int N = KEY_NUM * DIM;
    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    d2h_hbm_data_all<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
    cudaDeviceSynchronize();
    start_test = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_TIMES; i++) {
      d2h_hbm_data_all<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
    }
    cudaDeviceSynchronize();
    diff_test = std::chrono::steady_clock::now() - start_test;
  }
  printf("[timing] HBM data d2d=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  {
    constexpr int N = KEY_NUM * TILE_SIZE;
    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
    cudaDeviceSynchronize();
    start_test = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_TIMES; i++) {
      d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
    }
    cudaDeviceSynchronize();
    diff_test = std::chrono::steady_clock::now() - start_test;
  }
  printf("[timing] HBM data d2d=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  cudaFreeHost(dst);
  cudaFreeHost(h_offset);
  cudaFreeHost(h_src_idx);
  cudaFree(dst_ptr);
  cudaFree(src);
  cudaFree(d_offset);
  cudaFree(d_src_idx);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
