// by test

#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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

using std::begin;
using std::cerr;
using std::copy;
using std::cout;
using std::endl;
using std::generate;
using std::max;
using std::min;

using namespace std::chrono;
namespace cg = cooperative_groups;

typedef uint64_t K;
typedef uint64_t M;
typedef float V;
typedef int P;

constexpr uint64_t DIM = 64;
constexpr uint64_t INIT_SIZE = 32 * 4 * 1024 * 1024;  // 134,217,728
constexpr uint64_t BUCKETS_SIZE = 128;
constexpr uint64_t CACHE_SIZE = 2;
constexpr uint64_t BUCKETS_NUM = INIT_SIZE / BUCKETS_SIZE;  // 1,048,576
constexpr K EMPTY_KEY = (K)(0xFFFFFFFFFFFFFFFF);
constexpr M MAX_META = (M)(0xFFFFFFFFFFFFFFFF);

__global__ void upsert_cg(int N, int *keys_pos) {
  extern __shared__ unsigned char s[];
  int *key_pos = (int *)s;
  int *min_meta_pos = (int *)key_pos + 32;
  uint64_t *min_meta = (uint64_t *)((int *)min_meta_pos + 32);
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  //   cg::thread_block wholeBlock = cg::this_thread_block();
  //   cg::thread_group bucket_tile =
  //       cg::tiled_partition(wholeBlock, BUCKETS_SIZE);  // 128 * 8 = 1024
  //   cg::thread_group warp_tile = cg::tiled_partition(bucket_tile, 32);
  //   cg::coalesced_group rank0_tile = cg::labeled_partition(bucket_tile,
  //   int(tid % 32 == 0));

  uint64_t target_key = 90;
  uint64_t my_key = tid % 1024;

  if (tid < N) {
    unsigned int pos =
        __reduce_max_sync(0xFFFFFFFF, int(my_key == target_key) * tid);
    if (tid % 32 == 0) {
      key_pos[tid / 32] = pos;
      //       printf("tid=%d, match=%d\n", tid, pos);
    }
    __syncthreads();
    int max_pos = -1;
    if (tid % BUCKETS_SIZE == 0) {
      for (int i = 0; i < BUCKETS_SIZE / 32; i++) {
        //         printf("i=%d, read=%d\n", i, key_pos[i]);
        max_pos = max(max_pos, key_pos[i]);
      }
      keys_pos[tid / BUCKETS_SIZE] = max_pos;
    }
  }
}

int main() {
  int key_num = 1024 * 1024;
  int NUM_THREADS = 1024;
  int N = key_num * BUCKETS_SIZE;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  int *keys_pos;
  int TIMES = 10;
  cudaMalloc(&keys_pos, N * sizeof(K));  // 8MB

  auto start_test = std::chrono::steady_clock::now();
  for (int i = 0; i < TIMES; i++) {
    upsert_cg<<<NUM_BLOCKS, NUM_THREADS, 16 * NUM_THREADS / 32, 0>>>(N * 1024,
                                                                     keys_pos);
    cudaDeviceSynchronize();
  }
  auto end_test = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff_upsert = end_test - start_test;
  cudaFree(keys_pos);

  std::cout << "timing=" << diff_upsert.count() * 1000 / TIMES << "ms"
            << std::endl;
  return 0;
}
