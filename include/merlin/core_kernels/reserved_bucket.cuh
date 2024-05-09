/*
* Copyright (c) 2024, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http:///www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once
#include "../allocator.cuh"
#include "../types.cuh"

namespace nv {
namespace merlin {

#define RESERVED_BUCKET_SIZE 4
#define RESERVED_BUCKET_MASK 3

template <class K, class V> struct ReservedBucket;

template <class K, class V>
__global__ static void rb_size_kernel(ReservedBucket<K, V>* reserved_bucket, size_t* size);

template <class K, class V>
struct ReservedBucket {
  cuda::atomic<bool, cuda::thread_scope_device> locks[RESERVED_BUCKET_SIZE];
  bool keys[RESERVED_BUCKET_SIZE];
  static void initialize(ReservedBucket<K, V>** reserved_bucket,
                         BaseAllocator* allocator, size_t dim) {
    size_t total_size = sizeof (ReservedBucket<K, V>);
    total_size += sizeof(V) * RESERVED_BUCKET_SIZE * dim;
    void* memory_block;
    allocator->alloc(MemoryType::Device, &memory_block, total_size);
    CUDA_CHECK(cudaMemset(memory_block, 0, total_size));
    *reserved_bucket = static_cast<ReservedBucket<K, V>*>(memory_block);
  }

  __forceinline__ __device__ V* get_vector(K key, size_t dim) {
    V* vector = reinterpret_cast<V*>(keys + RESERVED_BUCKET_SIZE);
    size_t index = key & RESERVED_BUCKET_MASK;
    return vector + index * dim;
  }

  __forceinline__ __device__ bool contains(K key) {
    size_t index = key & RESERVED_BUCKET_MASK;
    return keys[index];
  }

  __forceinline__ __device__ void set_key(K key, bool value = true) {
    size_t index = key & RESERVED_BUCKET_MASK;
    keys[index] = value;
  }

  // since reserved bucket key should always exist
  // insert_or_assign insert_and_evict assign all equal to write_vector
  __forceinline__ __device__ void write_vector(
      K key, size_t dim, const V* data) {
    V* vectors = get_vector(key, dim);
    set_key(key);
    for (int i = 0; i < dim; i++) {
      vectors[i] = data[i];
      printf("vectors[%d] = %f  %f \n", i, vectors[i], data[i]);
    }
  }

  __forceinline__ __device__ void read_vector(
      K key, size_t dim, V* out_data) {
    V* vectors = get_vector(key, dim);
    for (int i = 0; i < dim; i++) {
      out_data[i] = vectors[i];
      printf("out_data[%d] = %f  %f \n", i, out_data[i], vectors[i]);
    }
  }

  __forceinline__ __device__ void erase(K key, size_t dim) {
    V* vectors = get_vector(key, dim);
    set_key(key, false);
    for (int i = 0; i < dim; i++) {
      vectors[i] = 0;
    }
  }

  // Search for the specified keys and return the pointers of values.
  __forceinline__ __device__ bool find(K key, size_t dim, V** values) {
    if (contains(key)) {
      V* vectors = get_vector(key, dim);
      *values = vectors;
      return true;
    } else {
      return false;
    }
  }

  // Search for the specified keys and Insert them firstly when missing.
  __forceinline__ __device__ bool find_or_insert(K key, size_t dim, V* values) {
    if (contains(key)) {
      return true;
    } else {
      write_vector(key, dim, values);
      set_key(key);
      return false;
    }
  }

  // Search for the specified keys and return the pointers of values.
  // Insert them firstly when missing.
  __forceinline__ __device__ bool find_or_insert(
      K key, size_t dim, V** values) {
    if (contains(key)) {
      V* vectors = get_vector(key, dim);
      *values = vectors;
      return true;
    } else {
      write_vector(key, dim, *values);
      set_key(key);
      return false;
    }
  }
  __forceinline__ __device__ void accum_or_assign(
      K key, bool is_accum, size_t dim, const V* values) {
    if (is_accum) {
      V* vectors = get_vector(key, dim);
      for (int i = 0; i < dim; i++) {
        vectors[i] += values[i];
      }
    } else {
      write_vector(key, dim, values);
    }
    set_key(key);
  }

  /*
    * @brief Exports reserved bucket to key-value tuples
    * @param n The maximum number of exported pairs.
    * @param offset The position of the key to search.
    * @param keys The keys to dump from GPU-accessible memory with shape (n).
    * @param values The values to dump from GPU-accessible memory with shape
    * (n, DIM).
   * @return The number of elements dumped.
   */
  __forceinline__ __device__ size_t export_batch(
      size_t n, const size_t offset,
      K* keys, size_t dim, V* values, size_t batch_size) {
    if (offset >= size()) {
      return 0;
    }

    size_t count = 0;
    V* vector = reinterpret_cast<V*>(keys + RESERVED_BUCKET_SIZE);
    for (int i = offset; i < RESERVED_BUCKET_SIZE && offset < n; i++) {
      vector += i * dim;
      offset++;
      if (keys[i]) {
        for (int j = 0; j < dim; j++) {
          values[i * dim + j] = vector[j];
        }
      }
    }
    return count;
  }

  /**
   * @brief Returns the reserved bucket size.
   */
  __forceinline__ __device__ size_t size() {
    size_t count = 0;
    for (int i = 0; i < RESERVED_BUCKET_SIZE; i++) {
      if (keys[i]) {
        count++;
      }
    }
    return count;
  }

  size_t size_host() {
    size_t * d_size;
    cudaMalloc(&d_size, sizeof(int));
    rb_size_kernel<<<1, 1>>>(this, d_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_size;
    cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(d_size));
    return h_size;
  }
  /**
   * @brief Removes all of the elements in the reserved bucket with no release
   * object.
   */
  __forceinline__ __device__ void clear(size_t dim) {
    size_t total_size = sizeof (ReservedBucket<K, V>);
    total_size += sizeof(V) * RESERVED_BUCKET_SIZE * dim;
    CUDA_CHECK(cudaMemset(this, 0, total_size));
  }
};

template <class K, class V>
__global__ static void rb_size_kernel(ReservedBucket<K, V>* reserved_bucket, size_t* size) {
  *size = reserved_bucket->size();
}

template <class K, class V>
__global__ void rb_write_vector_kernel(ReservedBucket<K, V>* reserved_bucket,
                                       K key, size_t dim, const V* data) {
  reserved_bucket->write_vector(key, dim, data);
}

template <class K, class V>
__global__ void rb_read_vector_kernel(ReservedBucket<K, V>* reserved_bucket,
                                      K key, size_t dim, V* out_data) {
  reserved_bucket->read_vector(key, dim, out_data);
}

template <class K, class V>
__global__ void rb_erase_kernel(ReservedBucket<K, V>* reserved_bucket,
                                K key, size_t dim) {
  reserved_bucket->erase(key, dim);
}

template <class K, class V>
__global__ void rb_clear_kernel(ReservedBucket<K, V>* reserved_bucket, size_t dim) {
  reserved_bucket->clear(dim);
}

template <class K, class V>
__global__ void rb_find_or_insert_kernel(
    ReservedBucket<K, V>* reserved_bucket,
    K key, size_t dim, const V* data, bool* is_found) {
  *is_found = reserved_bucket->find_or_insert(key, dim, data);
}

template <class K, class V> __global__ void rb_find_or_insert_kernel(
    ReservedBucket<K, V>* reserved_bucket, K key, size_t dim, bool* is_found, V** values) {
  *is_found = reserved_bucket->find_or_insert(key, dim, values);
}

template <class K, class V>
__global__ void rb_accum_or_assign_kernel(
    ReservedBucket<K, V>* reserved_bucket,
    K key, bool is_accum,
    size_t dim, const V* data) {
  printf("rb_accum_or_assign_kernel\n");
  reserved_bucket->accum_or_assign(key, is_accum, dim, data);
}

template <class K, class V> __global__ void rb_find_kernel(
    ReservedBucket<K, V>* reserved_bucket, K key, size_t dim,
    bool* found, V** values) {
  *found = reserved_bucket->find(key, dim, values);
}

template <class K, class V> __global__ void rb_export_batch_kernel(
    ReservedBucket<K, V>* reserved_bucket, size_t n, size_t offset, K* keys,
    size_t dim, V* values, size_t batch_size) {
  reserved_bucket->export_batch(n, offset, keys, dim, values, batch_size);
}


}  // namespace merlin
}  // namespace nv