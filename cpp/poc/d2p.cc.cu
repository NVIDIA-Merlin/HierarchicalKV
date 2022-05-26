//
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>

#include "merlin/utils.cuh"

using std::cout;
using std::endl;

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string &what) : runtime_error(what) {}
};
inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

#define CUDA_CHECK(val) \
  { cuda_check_((val), __FILE__, __LINE__); }

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

__global__ void d2h_const_data(const Vector *__restrict src,
                               Vector **__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = 0.1f;
  }
}

__global__ void d2h_hbm_data(
    const Vector *__restrict src, Vector **__restrict dst,
    int N) {  // dst is a set of Vector* in the pinned memory
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];
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

void ConnectMasterSlaveDevice(const int master_device_id,
                              const int slave_device_id) {
  int can_access;
  int target_device_id;
  target_device_id = slave_device_id;

  CUDA_CHECK(
      cudaDeviceCanAccessPeer(&can_access, master_device_id, slave_device_id));

  if (can_access == 0) {
    cout << "Device P2P access from GPU " << master_device_id << " to GPU "
         << slave_device_id
         << " can not be enabled. Data transfering may be slow." << endl;
  } else {
    CUDA_CHECK(cudaSetDevice(master_device_id));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(target_device_id, 0));
    cout << "Device P2P access from GPU " << master_device_id << " to GPU "
         << slave_device_id << "  enabled." << endl;
  }

  CUDA_CHECK(
      cudaDeviceCanAccessPeer(&can_access, slave_device_id, master_device_id));

  if (can_access == 0) {
    cout << "Device P2P access from GPU " << slave_device_id << " to GPU "
         << master_device_id
         << " can not be enabled. Data transfering may be slow." << endl;
  } else {
    CUDA_CHECK(cudaSetDevice(slave_device_id));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(master_device_id, 0));
    cout << "Device P2P access from GPU " << slave_device_id << " to GPU "
         << master_device_id << " enabled." << endl;
  }
  CUDA_CHECK(cudaSetDevice(master_device_id));
}

int main() {
  constexpr int KEY_NUM = 1024 * 1024;
  constexpr int INIT_SIZE = KEY_NUM * 32;
  constexpr int N = KEY_NUM * DIM;
  constexpr int TEST_TIMES = 1;
  constexpr size_t vectors_size = INIT_SIZE * sizeof(Vector);
  const int master_device_id = 0;
  const int slave_device_id = 1;

  int NUM_THREADS = 1024;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  int *h_offset;
  int *d_offset;

  ConnectMasterSlaveDevice(0, 1);
  ConnectMasterSlaveDevice(1, 0);

  CUDA_CHECK(cudaSetDevice(master_device_id));

  cudaMallocHost(&h_offset, sizeof(int) * KEY_NUM);
  cudaMalloc(&d_offset, sizeof(int) * KEY_NUM);
  cudaMemset(&h_offset, 0, sizeof(int) * KEY_NUM);
  cudaMemset(&d_offset, 0, sizeof(int) * KEY_NUM);

  Vector *src;
  Vector *dst;
  Vector **dst_ptr;
  cudaMalloc(&src, KEY_NUM * sizeof(Vector));
  cudaMalloc(&dst_ptr, KEY_NUM * sizeof(Vector *));

  CUDA_CHECK(cudaSetDevice(slave_device_id));
  cudaMalloc(&dst, vectors_size);
  CUDA_CHECK(cudaSetDevice(master_device_id));

  create_random_offset(h_offset, KEY_NUM, INIT_SIZE);
  cudaMemcpy(d_offset, h_offset, sizeof(int) * KEY_NUM, cudaMemcpyHostToDevice);
  create_fake_ptr<<<1024, 1024>>>(dst, dst_ptr, d_offset, KEY_NUM);
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

  start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TEST_TIMES; i++) {
    d2h_hbm_data<<<NUM_BLOCKS, NUM_THREADS>>>(src, dst_ptr, N);
  }
  cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] HBM data d2h=%.2fms\n",
         diff_test.count() * 1000 / TEST_TIMES);

  CUDA_CHECK(cudaSetDevice(slave_device_id));
  cudaFree(dst);
  CUDA_CHECK(cudaSetDevice(master_device_id));

  cudaFreeHost(h_offset);
  cudaFree(dst_ptr);
  cudaFree(src);
  cudaFree(d_offset);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
