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
constexpr uint64_t INIT_SIZE = 32 * 4 * 1024 * 1024;  // 134,217,728
constexpr uint64_t BUCKETS_SIZE = 128;
constexpr uint64_t CACHE_SIZE = 2;
constexpr uint64_t BUCKETS_NUM = INIT_SIZE / BUCKETS_SIZE;  // 1,048,576
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

__inline__ __device__ uint64_t atomicExch(uint64_t *address, uint64_t val) {
  return (uint64_t)atomicExch((unsigned long long *)address,
                              (unsigned long long)val);
}

__inline__ __device__ int64_t atomicExch(int64_t *address, int64_t val) {
  return (int64_t)atomicExch((unsigned long long *)address,
                             (unsigned long long)val);
}

__inline__ __device__ int64_t atomicAdd(int64_t *address, const int64_t val) {
  return (int64_t)atomicAdd((unsigned long long *)address,
                            (const unsigned long long)val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t *address,
                                         const uint64_t val) {
  return (uint64_t)atomicAdd((unsigned long long *)address,
                             (const unsigned long long)val);
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
  V value[DIM];
};

struct __align__(sizeof(M)) Meta {
  M val;
  //   int prev;
  //   int next;
};

struct Bucket {
  K *keys;          // Device memory
  Meta *metas;      // Device memory
  Vector *cache;    // Device memory
  Vector *vectors;  // Pinned host memory
  Vector *slots1;   // Pinned host memory
  Vector *slots2;   // Pinned host memory
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
  cudaMallocManaged((void **)&((*table)->buckets),
                    BUCKETS_NUM * sizeof(Bucket));
  cudaMemset(((*table)->buckets), 0, BUCKETS_NUM * sizeof(Bucket));

  cudaMalloc((void **)&((*table)->locks), BUCKETS_NUM * sizeof(int));
  cudaMemset((*table)->locks, 0, BUCKETS_NUM * sizeof(unsigned int));

  for (int i = 0; i < BUCKETS_NUM; i++) {
    cudaMalloc(&((*table)->buckets[i].keys), BUCKETS_SIZE * sizeof(K));
    cudaMemset((*table)->buckets[i].keys, 0xFF, BUCKETS_SIZE * sizeof(K));
    cudaMalloc(&((*table)->buckets[i].metas), BUCKETS_SIZE * sizeof(M));
    cudaMalloc(&((*table)->buckets[i].cache), CACHE_SIZE * sizeof(Vector));
    cudaMallocHost(&((*table)->buckets[i].vectors),
                   BUCKETS_SIZE * sizeof(Vector), cudaHostRegisterMapped);
  }
}

void destroy_table(Table **table) {
  for (int i = 0; i < BUCKETS_NUM; i++) {
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
      (*(dst[vec_index])).value[dim_index] = src[vec_index].value[dim_index];
    }
  }
}

__global__ void read(Vector **__restrict src, Vector *__restrict dst, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    int vec_index = int(tid / DIM);
    int dim_index = tid % DIM;
    if (src[vec_index] != nullptr) {
      dst[vec_index].value[dim_index] = (*(src[vec_index])).value[dim_index];
    }
  }
}

__global__ void upsert(const Table *__restrict table, const K *__restrict keys,
                       const M *__restrict metas, Vector **__restrict vectors,
                       int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = -1;
  bool found = false;

  if (tid < N) {
    int key_idx = tid;
    int bkt_idx = keys[tid] % BUCKETS_NUM;
    const K insert_key = keys[tid];
    bool release_lock = false;

    while (!release_lock) {
      if (atomicExch(&(table->locks[bkt_idx]), 1u) == 0u) {
        Bucket *bucket = &(table->buckets[bkt_idx]);
        for (int i = 0; i < BUCKETS_SIZE; i++) {
          if (bucket->keys[i] == insert_key) {
            found = true;
            key_pos = i;
            break;
          }
        }
        for (int i = 0; i < BUCKETS_SIZE; i++) {
          K old_key = atomicCAS(&(bucket->keys[i]), EMPTY_KEY, insert_key);
          if (old_key == EMPTY_KEY) {
            key_pos = i;
            break;
          }
        }
        if (metas[key_idx] >= bucket->min_meta || found ||
            bucket->size < BUCKETS_SIZE) {
          if (!found) {
            key_pos = key_pos == -1 ? bucket->min_pos : key_pos;
            atomicAdd(&(bucket->size), 1);
            atomicMin(&(bucket->size), BUCKETS_SIZE);
          }
          atomicExch(&(bucket->keys[key_pos]), insert_key);
          atomicExch(&(bucket->metas[key_pos].val), metas[key_idx]);

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
          atomicExch(&(bucket->min_pos), tmp_min_pos);
          atomicExch(&(bucket->min_meta), tmp_min_val);
          atomicCAS((uint64_t *)&(vectors[tid]), (uint64_t)(nullptr),
                    (uint64_t)((Vector *)(bucket->vectors) + key_pos));
        }
        release_lock = true;
        atomicExch(&(table->locks[bkt_idx]), 0u);
      }
    }
  }
}

__global__ void upsert(const Table *__restrict table, const K *__restrict keys,
                       Vector **__restrict vectors, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int key_pos = 0;

  if (tid < N) {
    int bkt_idx = keys[tid] % BUCKETS_NUM;
    const K insert_key = keys[tid];
    Bucket *bucket = &(table->buckets[bkt_idx]);
    key_pos = atomicInc((unsigned int *)&(bucket->min_pos), BUCKETS_SIZE - 1);
    bucket->keys[key_pos] = insert_key;
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
    int bkt_idx = keys[key_idx] % BUCKETS_NUM;
    K target_key = keys[key_idx];
    Bucket *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      vectors[key_idx] = (Vector *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

__global__ void lookup(const Table *__restrict table, const K *__restrict keys,
                       M *__restrict metas, Vector **__restrict vectors,
                       bool *__restrict found, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    int key_idx = tid / BUCKETS_SIZE;
    int key_pos = tid % BUCKETS_SIZE;
    int bkt_idx = keys[key_idx] % BUCKETS_NUM;
    K target_key = keys[key_idx];
    Bucket *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_pos] == target_key) {
      metas[key_idx] = bucket->metas[key_pos].val;
      vectors[key_idx] = (Vector *)&(bucket->vectors[key_pos]);
      found[key_idx] = true;
    }
  }
}

__global__ void remove(const Table *__restrict table, const K *__restrict keys,
                       int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    int key_idx = tid / BUCKETS_SIZE;
    int key_pos = tid % BUCKETS_SIZE;
    int bkt_idx = keys[key_idx] % BUCKETS_NUM;
    K target_key = keys[key_idx];
    Bucket *bucket = &(table->buckets[bkt_idx]);

    K old_key = atomicCAS((K *)&bucket->keys[key_pos], target_key, EMPTY_KEY);
    if (old_key == target_key) {
      atomicExch((K *)&(bucket->metas[key_pos].val), MAX_META);
      atomicDec((unsigned int *)&(bucket->size), BUCKETS_SIZE);
    }
  }
}

__global__ void clear(Table *table, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    int key_idx = tid % BUCKETS_SIZE;
    int bkt_idx = tid / BUCKETS_SIZE;
    Bucket *bucket = &(table->buckets[bkt_idx]);
    atomicExch((K *)&(bucket->keys[key_idx]), EMPTY_KEY);
    atomicExch((K *)&(bucket->metas[key_idx].val), MAX_META);
    if (key_idx == 0) atomicExch(&(bucket->size), 0);
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

__global__ void size_new(Table *table, size_t *size, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    atomicAdd((unsigned long long int *)(size), table->buckets[tid].size);
  }
}

__global__ void dump(const Table *table, K *d_key, Vector *d_val,
                     const size_t offset, const size_t search_length,
                     size_t *d_dump_counter) {
  extern __shared__ unsigned char s[];
  K *smem = (K *)s;
  K *block_result_key = smem;
  Vector *block_result_val = (Vector *)&(smem[blockDim.x]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / BUCKETS_SIZE;
    int key_idx = (tid + offset) % BUCKETS_SIZE;
    Bucket *bucket = &(table->buckets[bkt_idx]);

    if (bucket->keys[key_idx] != EMPTY_KEY) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = bucket->keys[key_idx];
      for (int i = 0; i < DIM; i++) {
        block_result_val[local_index].value[i] =
            bucket->vectors[key_idx].value[i];
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    //     printf("block_acc=%llu \n", block_acc);
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    for (int i = 0; i < DIM; i++) {
      d_val[global_acc + threadIdx.x].value[i] =
          block_result_val[threadIdx.x].value[i];
    }
  }
}

template <typename T>
void create_random_keys_test(T *h_keys, M *h_metas, int KEY_NUM) {
  std::unordered_set<T> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<T> distr;
  T max_key = 0;
  T min_key = 0xFFFFFFFFFFFFFFFF;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    T tmp = distr(eng);
    if (Murmur3Hash(tmp) % BUCKETS_NUM == 0) numbers.insert(tmp);
  }
  for (const T num : numbers) {
    h_keys[i] = Murmur3Hash(num);
    max_key = max_key < h_keys[i] ? h_keys[i] : max_key;
    min_key = min_key > h_keys[i] ? h_keys[i] : min_key;
    h_metas[i] = getTimestamp() + i;
    i++;
  }
  std::cout << "create_random_keys: " << max_key << " " << min_key << std::endl;
}

std::unordered_set<K> numbers;
template <typename T>
void create_random_keys(T *h_keys, M *h_metas, int KEY_NUM) {
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<T> distr;
  T max_key = 0;
  T min_key = 0xFFFFFFFFFFFFFFFF;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    T tmp = distr(eng);
    numbers.insert(Murmur3Hash(tmp));
  }
  for (const T num : numbers) {
    h_keys[i] = num;
    max_key = max_key < h_keys[i] ? h_keys[i] : max_key;
    min_key = min_key > h_keys[i] ? h_keys[i] : min_key;
    h_metas[i] = (M)(h_keys[i]);
    i++;
  }
  //   std::cout << "create_random_keys: " << max_key << " " << min_key <<
  //   std::endl;
}

int main() {
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024;
  constexpr uint64_t TEST_TIMES = 1;

  int total_size = 0;
  size_t max_bucket_len = 0;
  size_t min_bucket_len = KEY_NUM;
  int found_num = 0;
  std::unordered_map<int, int> size2length;

  K *h_keys;
  K *h_dump_keys;
  M *h_metas;
  Vector *h_vectors;
  Vector *h_dump_vectors;
  size_t *h_size;
  size_t h_counter;
  bool *h_found;

  cudaMallocHost(&h_keys, KEY_NUM * sizeof(K));               // 8MB
  cudaMallocHost(&h_dump_keys, KEY_NUM * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, KEY_NUM * sizeof(M));              // 8MB
  cudaMallocHost(&h_vectors, KEY_NUM * sizeof(Vector));       // 256MB
  cudaMallocHost(&h_dump_vectors, KEY_NUM * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_size, BUCKETS_NUM * sizeof(size_t));      // 8MB
  cudaMallocHost(&h_found, KEY_NUM * sizeof(bool));           // 4MB

  cudaMemset(h_vectors, 0, KEY_NUM * sizeof(Vector));

  create_random_keys<K>(h_keys, h_metas, KEY_NUM);

  Table *d_table;
  K *d_keys;
  K *d_dump_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector *d_dump_vectors;
  Vector **d_vectors_ptr;
  size_t *d_size;
  size_t *d_counter;
  bool *d_found;

  cudaMalloc(&d_dump_keys, KEY_NUM * sizeof(K));           // 8MB
  cudaMalloc(&d_keys, KEY_NUM * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, KEY_NUM * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, KEY_NUM * sizeof(Vector));        // 256MB
  cudaMalloc(&d_dump_vectors, KEY_NUM * sizeof(Vector));   // 256MB
  cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_size, BUCKETS_NUM * sizeof(size_t));       // 8MB
  cudaMalloc(&d_found, KEY_NUM * sizeof(bool));            // 4MB
  cudaMalloc(&d_counter, sizeof(size_t));                  // 4MB

  cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M), cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, KEY_NUM * sizeof(Vector));
  cudaMemset(d_dump_vectors, 0, KEY_NUM * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
  cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));
  cudaMemset(d_counter, 0, sizeof(size_t));
  cudaMemset(d_size, 0, BUCKETS_NUM * sizeof(size_t));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint64_t NUM_THREADS = 1024;
  uint64_t N = KEY_NUM;
  uint64_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  create_table(&d_table);
  for (int i = 0; i < TEST_TIMES; i++) {
    found_num = 0;
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

    //     // size test
    //     N = BUCKETS_NUM;
    //     NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    //     size<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    //     cudaDeviceSynchronize();

    // size test
    cudaMemset(d_size, 0, BUCKETS_NUM * sizeof(size_t));
    N = BUCKETS_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    size_new<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_size, d_size, BUCKETS_NUM * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cout << "after upsert, size=" << h_size[0] << endl;

    // dump:
    cudaMemset(d_counter, 0, sizeof(d_counter));
    cudaMemset(d_vectors, 0, KEY_NUM * sizeof(Vector));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t shared_mem_size = deviceProp.sharedMemPerBlock;

    size_t search_length = INIT_SIZE;
    size_t block_size = shared_mem_size * 0.5 / (sizeof(K) + sizeof(Vector));
    cout << "dump block_Size=" << block_size << endl;
    block_size = block_size <= 1024 ? block_size : 1024;
    assert(block_size > 0 &&
           "nv::merlinhash: block_size <= 0, the KV size may be too large!");
    size_t shared_size = sizeof(K) * block_size + sizeof(Vector) * block_size;
    const int grid_size = (search_length - 1) / (block_size) + 1;

    dump<<<grid_size, block_size, shared_size, stream>>>(
        d_table, d_dump_keys, d_dump_vectors, 0, search_length, d_counter);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_counter, d_counter, sizeof(d_counter),
               cudaMemcpyDeviceToHost);
    cout << "dump, h_counter=" << h_counter << endl;

    cudaMemcpy(h_dump_keys, d_dump_keys, KEY_NUM * sizeof(K),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dump_vectors, d_dump_vectors, KEY_NUM * sizeof(Vector),
               cudaMemcpyDeviceToHost);
    int error_dump_count = 0;
    for (int i = 0; i < KEY_NUM; i++) {
      for (int j = 0; j < DIM; j++) {
        V tmp = h_dump_vectors[i].value[j];
        int *tmp_int = reinterpret_cast<int *>((V *)(&tmp));
        if (*tmp_int != 16843009) error_dump_count++;
      }
    }
    cout << "check1: dump error_count=" << error_dump_count << endl;
    error_dump_count = 0;
    for (int i = 0; i < KEY_NUM; i++) {
      if (numbers.end() == numbers.find(h_dump_keys[i])) error_dump_count++;
    }
    cout << "check2: dump error_count=" << error_dump_count << endl;

    // lookup test
    N = BUCKETS_SIZE * KEY_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(Vector *));
    cudaMemset(d_found, 0, KEY_NUM * sizeof(bool));
    cudaMemset(d_metas, 0, KEY_NUM * sizeof(M));
    cudaDeviceSynchronize();

    auto start_lookup = std::chrono::steady_clock::now();
    cudaDeviceSynchronize();
    lookup<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, d_metas, d_vectors_ptr,
                                        d_found, N);
    cudaDeviceSynchronize();
    auto end_lookup = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_lookup = end_lookup - start_lookup;

    N = KEY_NUM * DIM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    cudaMemset(d_vectors, 0, KEY_NUM * sizeof(Vector));
    auto start_read = std::chrono::steady_clock::now();
    read<<<NUM_BLOCKS, NUM_THREADS>>>(d_vectors_ptr, d_vectors, N);
    cudaDeviceSynchronize();
    auto end_read = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_read = end_read - start_read;

    // remove:
    cudaMemset(d_size, 0, BUCKETS_NUM * sizeof(size_t));
    int remove_key_num = KEY_NUM / 2;
    N = remove_key_num * BUCKETS_SIZE;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    remove<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_keys, N);
    cudaDeviceSynchronize();
    N = BUCKETS_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    size_new<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_size, d_size, BUCKETS_NUM * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cout << "after remove, size=" << h_size[0] << endl;

    // clear:
    cudaMemset(d_size, 0, BUCKETS_NUM * sizeof(size_t));
    N = BUCKETS_NUM * BUCKETS_SIZE;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    clear<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, N);
    cudaDeviceSynchronize();

    N = BUCKETS_NUM;
    NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    size_new<<<NUM_BLOCKS, NUM_THREADS>>>(d_table, d_size, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_size, d_size, BUCKETS_NUM * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cout << "after clear, size=" << h_size[0] << endl;

    printf("[timing] upsert=%.2fms, write=%.2fms\n", diff_upsert.count() * 1000,
           diff_write.count() * 1000);
    printf("[timing] lookup=%.2fms, read = % .2fms\n ",
           diff_lookup.count() * 1000, diff_read.count() * 1000);
    cudaMemcpy(h_size, d_size, BUCKETS_NUM * sizeof(size_t),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(Vector),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counter, d_counter, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M), cudaMemcpyDeviceToHost);
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
    }

    for (int i = 0; i < BUCKETS_NUM; i++) {
      total_size += h_size[i];
      if (size2length.find(h_size[i]) != size2length.end()) {
        size2length[h_size[i]] += 1;
      } else {
        size2length[h_size[i]] = 1;
      }
      max_bucket_len = max(max_bucket_len, h_size[i]);
      min_bucket_len = min(min_bucket_len, h_size[i]);
    }

    cout << "Capacity = " << INIT_SIZE << ", total_size = " << total_size
         << ", h_size[0] = " << h_size[0] << ", h_counter = " << h_counter
         << ", max_bucket_len = " << max_bucket_len
         << ", min_bucket_len = " << min_bucket_len
         << ", found_num = " << found_num << endl;
    //     assert(found_num == 128);
  }
  destroy_table(&d_table);
  cudaStreamDestroy(stream);

  int error_count = 0;
  for (int i = 0; i < KEY_NUM; i++) {
    for (int j = 0; j < DIM; j++) {
      V tmp = h_vectors[i].value[j];
      int *tmp_int = reinterpret_cast<int *>((V *)(&tmp));
      if (*tmp_int != 16843009) error_count++;
    }
  }
  cout << "check1: error_count=" << error_count << endl;

  error_count = 0;
  for (int i = 0; i < KEY_NUM; i++) {
    if (h_keys[i] != h_metas[i]) error_count++;
  }
  cout << "check2:error_count=" << error_count << endl;

  uint64_t min_meta = 0xFFFFFFFFFFFFFFFF;
  for (int i = 0; i < KEY_NUM; i++) {
    if (!h_found[i]) continue;
    min_meta = h_metas[i] < min_meta ? h_metas[i] : min_meta;
  }
  int bigger = 0;
  int smaller = 0;
  for (int i = 0; i < KEY_NUM; i++) {
    if (h_keys[i] > min_meta) bigger++;
    if (h_keys[i] < min_meta) smaller++;
  }
  cout << "check3:bigger=" << bigger << endl;
  cout << "check3:smaller=" << smaller << endl;
  //   for(auto n: size2length){
  //     cout << n.first << "    " << n.second << endl;
  //   }

  cudaFreeHost(h_keys);
  cudaFreeHost(h_dump_keys);
  cudaFreeHost(h_vectors);
  cudaFreeHost(h_dump_vectors);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_size);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_dump_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_dump_vectors);
  cudaFree(d_vectors_ptr);
  cudaFree(d_size);
  cudaFree(d_found);
  cudaFree(d_counter);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
