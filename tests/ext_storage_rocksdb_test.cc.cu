/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dirent.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include "merlin/debug.hpp"
#include "merlin/rocksdb_storage.hpp"

using namespace nv::merlin;

using rocks_db_storage = RocksDBStorage<int64_t, float>;

const size_t value_dims{3};

template <class T>
using device_ptr = std::unique_ptr<T[], std::function<void(T*)>>;

template <class T>
device_ptr<T> device_alloc(const size_t n, cudaStream_t stream) {
  T* ptr;
  CUDA_CHECK(cudaMallocAsync(&ptr, sizeof(T) * n, stream));
  return {ptr,
          [stream](T* const ptr) { CUDA_CHECK(cudaFreeAsync(ptr, stream)); }};
}

template <class T>
device_ptr<T> to_device(const std::vector<T>& vec, cudaStream_t stream) {
  auto ptr{device_alloc<T>(vec.size(), stream)};
  CUDA_CHECK(cudaMemcpyAsync(ptr.get(), vec.data(), sizeof(T) * vec.size(),
                             cudaMemcpyHostToDevice, stream));
  return ptr;
}

template <class T>
void zero_fill(device_ptr<T>& ptr, const size_t n, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(ptr.get(), 0, sizeof(T) * n, stream));
}

template <class T,
          class U = std::conditional_t<std::is_same<T, bool>::value, char, T>>
std::vector<U> to_host(const device_ptr<T>& ptr, const size_t n,
                       cudaStream_t stream) {
  static_assert(sizeof(bool) == sizeof(char));

  std::vector<U> vec(n);
  CUDA_CHECK(cudaMemcpyAsync(vec.data(), ptr.get(), sizeof(T) * n,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return vec;
}

void test_rocksdb_create() {
  RocksDBStorageOptions opts;
  opts.path = "/tmp/rocksdb_fantasy_path_that_does_not_exist";

  // Ensure rocksdb database doesn't exist.
  std::system(("rm -rf " + opts.path).c_str());
  {
    struct stat st;
    ASSERT_NE(stat(opts.path.c_str(), &st), 0);
  }

  // Create rocksdb database.
  rocks_db_storage store(opts);

  std::ifstream log(opts.path + "/LOG");
  ASSERT_TRUE(log.is_open());
}

void test_rocksdb_open_write_and_read() {
  using key_type = typename rocks_db_storage::key_type;
  using value_type = typename rocks_db_storage::value_type;

  CUDA_CHECK(cudaSetDevice(0));

  // Create rocksdb database.
  RocksDBStorageOptions opts;
  opts.path = "/tmp/rocksdb_fantasy_path_that_does_not_exist";
  rocks_db_storage store(opts);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    std::vector<key_type> keys{1, 2, 3, 4};
    auto d_keys{to_device(keys, stream)};

    std::vector<value_type> values;
    values.resize(value_dims * keys.size());
    for (size_t i{}; i != values.size(); ++i)
      values[i] = static_cast<value_type>(i + 1);
    auto d_values{to_device(values, stream)};

    store.insert_or_assign(keys.size(), d_keys.get(), d_values.get(),
                           value_dims, stream);
  }

  {
    std::vector<key_type> keys{2, 3, 5, 1, 99, 1};
    auto d_keys{to_device(keys, stream)};
    const size_t n{keys.size()};

    device_ptr<value_type> d_values{
        device_alloc<value_type>(n * value_dims, stream)};
    zero_fill(d_values, n * value_dims, stream);

    device_ptr<bool> d_founds{device_alloc<bool>(n, stream)};
    zero_fill(d_founds, n, stream);

    const size_t hit_count{store.find(n, d_keys.get(), d_values.get(),
                                      value_dims, d_founds.get(), stream)};
    ASSERT_EQ(hit_count, 4);

    const std::vector<char> founds{to_host(d_founds, n, stream)};
    ASSERT_EQ(founds, (std::vector<char>{1, 1, 0, 1, 0, 1}));

    const std::vector<float> values{to_host(d_values, n * value_dims, stream)};
    ASSERT_EQ(values, (std::vector<float>{4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2, 3, 0,
                                          0, 0, 1, 2, 3}));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_rocksdb_open_erase_and_read() {
  using key_type = typename rocks_db_storage::key_type;
  using value_type = typename rocks_db_storage::value_type;

  CUDA_CHECK(cudaSetDevice(0));

  // Create rocksdb database.
  RocksDBStorageOptions opts;
  opts.path = "/tmp/rocksdb_fantasy_path_that_does_not_exist";
  rocks_db_storage store(opts);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    std::vector<key_type> keys{2, 3};
    auto d_keys{to_device(keys, stream)};

    store.erase(keys.size(), d_keys.get(), stream);
  }

  {
    std::vector<key_type> keys{1, 2, 3, 4, 5};
    auto d_keys{to_device(keys, stream)};
    const size_t n{keys.size()};

    device_ptr<value_type> d_values{
        device_alloc<value_type>(n * value_dims, stream)};
    zero_fill(d_values, n * value_dims, stream);

    device_ptr<bool> d_founds{device_alloc<bool>(n, stream)};
    zero_fill(d_founds, n, stream);

    const size_t hit_count{store.find(n, d_keys.get(), d_values.get(),
                                      value_dims, d_founds.get(), stream)};
    ASSERT_EQ(hit_count, 2);

    const std::vector<char> founds{to_host(d_founds, n, stream)};
    ASSERT_EQ(founds, (std::vector<char>{1, 0, 0, 1, 0}));

    const std::vector<float> values{to_host(d_values, n * value_dims, stream)};
    ASSERT_EQ(values, (std::vector<float>{1, 2, 3, 0, 0, 0, 0, 0, 0, 10, 11, 12,
                                          0, 0, 0}));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(RocksDBTest, create) { test_rocksdb_create(); }
TEST(RocksDBTest, open_write_and_read) { test_rocksdb_open_write_and_read(); }
TEST(RocksDBTest, open_erase_and_read) { test_rocksdb_open_erase_and_read(); }
