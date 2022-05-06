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

using std::begin;
using std::cerr;
using std::copy;
using std::cout;
using std::endl;
using std::generate;
using std::max;
using std::min;

using namespace std::chrono;

typedef uint64_t K;
typedef uint64_t M;
typedef float V;
typedef int P;

constexpr uint64_t DIM = 64;
constexpr uint64_t INIT_SIZE = 4 * 32 * 1024 * 1024;  // 134,217,728
constexpr uint64_t BUCKETS_SIZE = 128;
constexpr uint64_t CACHE_SIZE = 2;
constexpr uint64_t TABLE_SIZE = INIT_SIZE / BUCKETS_SIZE;  // 1,048,576
constexpr K EMPTY_KEY = (K)(0xFFFFFFFFFFFFFFFF);
constexpr M MAX_META = (M)(0xFFFFFFFFFFFFFFFF);

#define CUDA_CHECK(call)                                                 \
  if ((call) != cudaSuccess) {                                           \
    cudaError_t err = cudaGetLastError();                                \
    cerr << "CUDA error calling \"" #call "\", code is " << err << endl; \
  }

__inline__ __device__ uint64_t atomicCAS(uint64_t *address, uint64_t compare,
                                         uint64_t val) {
  return (uint64_t)atomicCAS((unsigned long long *)address,
                             (unsigned long long)compare,
                             (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicMax(uint64_t *address, uint64_t val) {
  return (uint64_t)atomicMax((unsigned long long *)address,
                             (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicMin(uint64_t *address, uint64_t val) {
  return (uint64_t)atomicMin((unsigned long long *)address,
                             (unsigned long long)val);
}

uint64_t getTimestamp() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

template <typename T>
class FlexMemory {
 public:
  FlexMemory(int size) : ptr_(nullptr) {
    if (!ptr_) {
      size_ = size;
      assert(size_ > 0);
      cudaMalloc(&ptr_, sizeof(T) * size_);
    }
  }
  ~FlexMemory() {
    if (!ptr_) cudaFree(ptr_);
  }
  V *get(size_t size = 0) {
    if (size > size_) {
      cudaFree(ptr_);
      size_ = size;
      assert(size_ > 0);
      cudaMalloc(&ptr_, sizeof(T) * size_);
    }
    return ptr_;
  }

 private:
  T *ptr_;
  size_t size_;
};

struct __align__(16) Vector {
  V values[DIM];
};

struct __align__(sizeof(M)) Meta {
  M val;
};

struct Bucket {
  K *keys;          // Device memory
  Meta *metas;      // Device memory
  Vector *cache;    // Device memory
  Vector *vectors;  // Pinned host memory
  M min_meta;
  int min_pos;
  int size;
};

struct __align__(32) Table {
  Bucket *buckets;
  unsigned int *locks;
};

inline uint64_t Murmur3Hash(const uint64_t &key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

void create_table(Table **table) {
  cudaMallocManaged((void **)table, sizeof(Table));
  cudaMallocManaged((void **)&((*table)->buckets), TABLE_SIZE * sizeof(Bucket));

  cudaMalloc((void **)&((*table)->locks), TABLE_SIZE * sizeof(int));
  cudaMemset((*table)->locks, 0, TABLE_SIZE * sizeof(unsigned int));

  for (int i = 0; i < TABLE_SIZE; i++) {
    cudaMalloc(&((*table)->buckets[i].keys), BUCKETS_SIZE * sizeof(K));
    cudaMemset((*table)->buckets[i].keys, 0xFF, BUCKETS_SIZE * sizeof(K));
    cudaMalloc(&((*table)->buckets[i].metas), BUCKETS_SIZE * sizeof(M));
    cudaMalloc(&((*table)->buckets[i].cache), CACHE_SIZE * sizeof(Vector));
    cudaMallocHost(&((*table)->buckets[i].vectors),
                   BUCKETS_SIZE * sizeof(Vector), cudaHostRegisterMapped);
  }
}

void destroy_table(Table **table) {
  for (int i = 0; i < TABLE_SIZE; i++) {
    cudaFree((*table)->buckets[i].keys);
    cudaFree((*table)->buckets[i].metas);
    cudaFree((*table)->buckets[i].cache);
    cudaFreeHost((*table)->buckets[i].vectors);
  }
  cudaFree((*table)->locks);
  cudaFree((*table)->buckets);
  cudaFree(*table);
}

__global__ void write(const Vector *__restrict src, Vector **__restrict dst,
                      int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;

    if (dst[vec_index] != nullptr) {
      (*(dst[vec_index])).values[dim_index] = src[vec_index].values[dim_index];
    }
  }
}

__global__ void read(Vector **__restrict src, Vector *__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    if (src[vec_index] != nullptr) {
      dst[vec_index].values[dim_index] = (*(src[vec_index])).values[dim_index];
    }
  }
}

__global__ void upsert(const Table *__restrict table, const K *__restrict keys,
                       const M *__restrict metas, Vector **__restrict vectors,
                       int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = 0;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % TABLE_SIZE;
    const K insert_key = keys[tid];
    Bucket *bucket = &(table->buckets[bkt_idx]);

    bool release_lock = false;
    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        for (int i = 0; i < BUCKETS_SIZE; i++) {
          if (bucket->keys[i] == insert_key) {
            found = true;
            key_pos = i;
            break;
          }
        }
        if (metas[key_idx] < bucket->min_meta && !found &&
            bucket->size >= BUCKETS_SIZE) {
          vectors[tid] = nullptr;
        } else {
          if (!found) {
            bucket->size += 1;
            key_pos = (bucket->size <= BUCKETS_SIZE) ? bucket->size + 1
                                                     : bucket->min_pos;
            if (bucket->size > BUCKETS_SIZE) {
              bucket->size = BUCKETS_SIZE;
            }
          }
          bucket->keys[key_pos] = insert_key;
          bucket->metas[key_pos].val = metas[key_idx];

          M tmp_min_val = MAX_META;
          int tmp_min_pos = 0;
          for (int i = 0; i < BUCKETS_SIZE; i++) {
            if (bucket->keys[i] == EMPTY_KEY) {
              break;
            }
            if (bucket->metas[i].val < tmp_min_val) {
              tmp_min_pos = i;
              tmp_min_val = bucket->metas[i].val;
            }
          }
          bucket->min_pos = tmp_min_pos;
          bucket->min_meta = tmp_min_val;
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }

    vectors[tid] = (Vector *)((Vector *)(bucket->vectors) + key_pos);
    /**
    while (key_pos < BUCKETS_SIZE) {
      const K old_key =
          atomicCAS((K *)&bucket->keys[key_pos], EMPTY_KEY, insert_key);
      if (EMPTY_KEY == old_key || insert_key == old_key) {
        break;
      }
      key_pos++;
    }
    key_pos = key_pos % BUCKETS_SIZE;
    bucket->metas[key_pos] = metas[key_idx];
    vectors[tid] = (Vector *)((Vector *)(bucket->vectors) + key_pos);
    */
  }
}

__global__ void upsert_(const Table *__restrict table, const K *__restrict keys,
                        const M *__restrict metas, Vector **__restrict vectors,
                        int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = 0;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % TABLE_SIZE;
    const K insert_key = keys[tid];
    Bucket *bucket = &(table->buckets[bkt_idx]);
    while (key_pos < BUCKETS_SIZE) {
      const K old_key =
          atomicCAS((K *)&bucket->keys[key_pos], EMPTY_KEY, insert_key);
      if (EMPTY_KEY == old_key || insert_key == old_key) {
        break;
      }
      key_pos++;
    }
    key_pos = key_pos % BUCKETS_SIZE;
    bucket->metas[key_pos].val = metas[key_idx];
    vectors[tid] = (Vector *)((Vector *)(bucket->vectors) + key_pos);
  }
}

__global__ void lookup(const Table *__restrict table, const K *__restrict keys,
                       Vector **__restrict vectors, bool *__restrict found,
                       int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    int key_idx = tid / BUCKETS_SIZE;
    int key_pos = tid % BUCKETS_SIZE;
    int bkt_idx = keys[key_idx] % TABLE_SIZE;
    K target_key = keys[key_idx];
    Bucket *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (Vector *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

__global__ void size(Table *table, size_t *size, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    for (int i = 0; i < BUCKETS_SIZE; i++) {
      if (table->buckets[tid].keys[i] != EMPTY_KEY) {
        atomicAdd((unsigned long long int *)&(size[tid]), 1);
      }
    }
  }
}

template <typename T>
void create_random_keys(T *h_keys, M *h_metas, int KEY_NUM) {
  std::unordered_set<T> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<T> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const T num : numbers) {
    h_keys[i] = Murmur3Hash(num);
    h_metas[i] = getTimestamp();
    i++;
  }
}

int main() {
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024;
  constexpr uint64_t TEST_TIMES = 1;

  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  size_t *h_size;
  bool *h_found;

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, KEY_NUM * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_size, TABLE_SIZE * sizeof(size_t));  // 8MB
  cudaMallocHost(&h_found, KEY_NUM * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector));

  create_random_keys<K>(h_keys, h_metas, KEY_NUM);

  Table *d_table;
  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector **d_vectors_ptr;
  size_t *d_size;
  bool *d_found;

  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_size, TABLE_SIZE * sizeof(size_t));        // 8MB
  cudaMalloc(&d_found, KEY_NUM * sizeof(bool));            // 4MB

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
  cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint64_t NUM_THREADS = 1024;
  uint64_t N = KEY_NUM;
  uint64_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  create_table(&d_table);
  for (int i = 0; i < TEST_TIMES; i++) {
    // upsert test
    N = KEY_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    auto start_upsert = std::chrono::steady_clock::now();
    upsert<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, d_metas, d_vectors_ptr,
                                        N);

    cudaDeviceSynchronize();
    auto end_upsert = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_upsert = end_upsert - start_upsert;

    N = KEY_NUM * DIM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    auto start_write = std::chrono::steady_clock::now();
    write<<<NUM_BLOCKS, NUM_THREADS>>>(d_vectors, d_vectors_ptr, N);
    cudaDeviceSynchronize();
    auto end_write = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_write = end_write - start_write;

    // size test
    N = TABLE_SIZE;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    size<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    cudaDeviceSynchronize();

    // lookup test
    N = BUCKETS_SIZE * KEY_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
    cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));
    cudaDeviceSynchronize();

    auto start_lookup = std::chrono::steady_clock::now();
    lookup<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, d_vectors_ptr, d_found,
                                        N);
    cudaDeviceSynchronize();
    auto end_lookup = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_lookup = end_lookup - start_lookup;

    N = KEY_NUM * DIM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // TODO: remove
    cudaMemset(d_vectors, 0, KEY_NUM * sizeof(Vector));

    auto start_read = std::chrono::steady_clock::now();
    read<<<NUM_BLOCKS, NUM_THREADS>>>(d_vectors_ptr, d_vectors, N);
    cudaDeviceSynchronize();
    auto end_read = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_read = end_read - start_read;
    printf("[timing] upsert=%.2fms, write=%.2fms\n", diff_upsert.count() * 1000,
           diff_write.count() * 1000);
    printf("[timing] lookup=%.2fms, read = % .2fms\n ",
           diff_lookup.count() * 1000, diff_read.count() * 1000);
  }
  destroy_table(&d_table);
  cudaStreamDestroy(stream);
  cudaMemcpy(h_size, d_size, TABLE_SIZE * sizeof(size_t),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
             cudaMemcpyDeviceToHost);

  int total_size = 0;
  size_t max_bucket_len = 0;
  size_t min_bucket_len = KEY_NUM;
  int found_num = 0;
  std::unordered_map<int, int> size2length;

  //   for (int i = 0; i < DIM; i++) {
  //     V tmp = h_vectors[1].values[i];
  //     int *tmp_int = reinterpret_cast<int *>((V *)(&tmp));
  //     cout << "vector = " << *tmp_int;
  //   }
  cout << endl;

  for (int i = 0; i < TABLE_SIZE; i++) {
    total_size += h_size[i];
    if (size2length.find(h_size[i]) != size2length.end()) {
      size2length[h_size[i]] += 1;
    } else {
      size2length[h_size[i]] = 1;
    }
    max_bucket_len = max(max_bucket_len, h_size[i]);
    min_bucket_len = min(min_bucket_len, h_size[i]);
  }

  //   for(auto n: size2length){
  //     cout << n.first << "    " << n.second << endl;
  //   }

  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) found_num++;
  }
  cout << "Capacity = " << INIT_SIZE << ", total_size = " << total_size
       << ", max_bucket_len = " << max_bucket_len
       << ", min_bucket_len = " << min_bucket_len
       << ", found_num = " << found_num << endl;

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_size);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_vectors_ptr);
  cudaFree(d_size);
  cudaFree(d_found);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
