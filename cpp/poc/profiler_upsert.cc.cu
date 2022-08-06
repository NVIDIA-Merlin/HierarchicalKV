#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>

typedef uint64_t K;
typedef uint64_t M;
typedef float V;

template <class K>
void create_random_keys(K *h_keys, int KEY_NUM, K start = 0) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    i++;
  }
}

template <class K>
void create_continuous_keys(K *h_keys, int KEY_NUM, K start = 0) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
  }
}

template <class M>
struct Meta {
  M val;
};

constexpr uint64_t EMPTY_KEY = std::numeric_limits<uint64_t>::max();
constexpr uint64_t MAX_META = std::numeric_limits<uint64_t>::max();
constexpr uint64_t EMPTY_META = std::numeric_limits<uint64_t>::min();

template <class K>
struct Bucket {
  K *keys;         // HBM
  Meta<M> *metas;  // HBM
  V *cache;        // HBM(optional)
  V *vectors;      // Pinned memory or HBM

  /* For upsert_kernel without user specified metas
     recording the current meta, the cur_meta will
     increment by 1 when a new inserting happens. */
  M cur_meta;

  /* min_meta and min_pos is for or upsert_kernel
     with user specified meta. They record the minimum
     meta and its pos in the bucket. */
  M min_meta;
  int min_pos;
};

constexpr int KEY_NUM = 1024 * 1024;
constexpr int INIT_SIZE = KEY_NUM * 64;
constexpr int MAX_BUCKET_SIZE = 128;
constexpr const size_t BLOCK_SIZE = 128;
constexpr int TILE_SIZE = 4;
constexpr const size_t N = KEY_NUM * TILE_SIZE;
constexpr const size_t GRID_SIZE = ((N)-1) / BLOCK_SIZE + 1;
constexpr int BUCKETS_NUM = INIT_SIZE / MAX_BUCKET_SIZE;
constexpr int DIM = 4;

__inline__ __device__ uint64_t Murmur3HashDevice(uint64_t const &key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <class Key>
__global__ void upsert_kernel(const Key *__restrict keys,
                              const Bucket<K> *__restrict buckets,
                              int *__restrict d_sizes, V **__restrict vectors,
                              const V *__restrict values,
                              int *__restrict src_offset, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;
    Key insert_key = *(keys + key_idx);
    Key hashed_key = Murmur3HashDevice(insert_key);
    size_t bkt_idx = hashed_key & (BUCKETS_NUM - 1);
    size_t start_idx = hashed_key & (MAX_BUCKET_SIZE - 1);

    int src_lane;

    const Bucket<Key> *bucket = buckets + bkt_idx;

    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

#pragma unroll
    for (uint32_t tile_offset = 0; tile_offset < MAX_BUCKET_SIZE;
         tile_offset += TILE_SIZE) {
      size_t key_offset =
          (start_idx + tile_offset + rank) & (MAX_BUCKET_SIZE - 1);
      Key current_key = *(bucket->keys + key_offset);
      auto const found_or_empty_vote =
          g.ballot(current_key == EMPTY_KEY || insert_key == current_key);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & MAX_BUCKET_SIZE;
        if (rank == src_lane) {
          *(bucket->keys + key_pos) = insert_key;
          //          *(vectors + key_idx) = (bucket->vectors + key_pos);
          if (current_key == EMPTY_KEY) {
            d_sizes[bkt_idx]++;
          }
        }
        for (auto i = g.thread_rank(); i < DIM; i += g.size()) {
          *(bucket->vectors + key_pos * DIM + i) =
              *(values + key_idx * DIM + i);
        }
        return;
      }
    }
    if (rank == 0 && key_pos == -1) {
      key_pos = bucket->min_pos;
      *(bucket->keys + key_pos) = insert_key;
    }
    key_pos = g.shfl(key_pos, 0);
    //        *(vectors + key_idx) = (bucket->vectors + key_pos);
    for (auto i = g.thread_rank(); i < DIM; i += g.size()) {
      *(bucket->vectors + key_pos * DIM + i) = *(values + key_idx * DIM + i);
    }
    return;
  }
}
int main() {
  K *h_keys;
  K *d_keys;
  K *d_all_keys;
  int *d_sizes;
  V **vectors;

  V *values;

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));
  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));
  Bucket<K> *buckets;
  cudaMallocManaged(&buckets, sizeof(Bucket<K>) * BUCKETS_NUM);
  cudaMalloc(&(d_all_keys), sizeof(K) * MAX_BUCKET_SIZE * BUCKETS_NUM);
  cudaMemset(d_all_keys, 0xFF, sizeof(K) * MAX_BUCKET_SIZE * BUCKETS_NUM);

  cudaMalloc(&(values), sizeof(V) * KEY_NUM * DIM);

  for (int i = 0; i < BUCKETS_NUM; i++) {
    buckets[i].keys = d_all_keys + i * MAX_BUCKET_SIZE;
    cudaMalloc(&(buckets[i].vectors), sizeof(V) * MAX_BUCKET_SIZE * DIM);
  }
  cudaMalloc(&(d_sizes), sizeof(int) * BUCKETS_NUM);
  cudaMemset(d_sizes, 0, sizeof(int) * BUCKETS_NUM);

  cudaMalloc(&(vectors), sizeof(V *) * KEY_NUM);
  cudaMemset(vectors, 0, sizeof(V *) * KEY_NUM);

  create_random_keys<K>(h_keys, KEY_NUM, 0);
  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  upsert_kernel<K><<<GRID_SIZE, BLOCK_SIZE>>>(d_keys, buckets, d_sizes, vectors,
                                              values, nullptr, N);
  cudaDeviceSynchronize();

  create_random_keys<K>(h_keys, KEY_NUM, KEY_NUM);
  auto start_insert_or_assign = std::chrono::steady_clock::now();
  upsert_kernel<K><<<GRID_SIZE, BLOCK_SIZE>>>(d_keys, buckets, d_sizes, vectors,
                                              values, nullptr, N);
  cudaDeviceSynchronize();
  auto end_insert_or_assign = std::chrono::steady_clock::now();

  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed1  : %s\n",
            cudaGetErrorString(err));
    exit(-1);
  }
  for (int i = 0; i < BUCKETS_NUM; i++) {
    cudaFree(buckets[i].vectors);
  }
  cudaFree(d_all_keys);
  cudaFree(values);
  cudaFree(buckets);
  cudaFreeHost(h_keys);
  cudaFree(d_keys);
  cudaFree(d_sizes);
  cudaFree(vectors);
  std::chrono::duration<double> diff_insert_or_assign =
      end_insert_or_assign - start_insert_or_assign;

  printf("[prepare] insert_or_assign=%.2fms\n",
         diff_insert_or_assign.count() * 1000);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
