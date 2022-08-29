/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <stddef.h>
#include <stdio.h>
#include <string>
#include "merlin/types.cuh"

namespace nv {
namespace merlin {

/**
 * The KV file on local file system.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam M The data type for `meta`.
 *           The currently supported data type is only `uint64_t`.
 * @tparam D The dimension of the vectors.
 *
 */
template <class K, class V, class M, size_t D>
class LocalKVFile : public BaseKVFile<K, V, M, D> {
 public:
  /**
   * @brief Default constructor for the LocalKVFile class.
   */
  LocalKVFile() : keys_fp_(nullptr), values_fp_(nullptr) {}

  /**
   * @brief Destructor of LocalKVFile frees the resources.
   */
  ~LocalKVFile() { close(); }

  /**
   * @brief Open file.
   */
  bool open(const std::string& keyfile, const std::string& valuefile,
            const char* mode) {
    close();
    keys_fp_ = fopen(keyfile.c_str(), mode);
    if (!keys_fp_) {
      return false;
    }
    values_fp_ = fopen(valuefile.c_str(), mode);
    if (!values_fp_) {
      close();
      return false;
    }
    return 0;
  }

  /**
   * @brief Close file.
   */
  void close() noexcept {
    if (keys_fp_) {
      fclose(keys_fp_);
      keys_fp_ = nullptr;
    }
    if (values_fp_) {
      fclose(values_fp_);
      values_fp_ = nullptr;
    }
  }

  /**
   * @brief Read data from the opened file.
   */
  int64_t read(size_t n, K* keys, V* vectors, M* metas) override {
    size_t nread_keys = fread(keys, sizeof(K), n, keys_fp_);
    size_t nread_vevs = fread(vectors, sizeof(V) * D, n, values_fp_);
    if (nread_keys != nread_vevs) {
      return -1;
    }
    return static_cast<int64_t>(nread_keys);
  }

  /**
   * @brief Write data to the opened file.
   */
  int64_t write(size_t n, const K* keys, const V* vectors,
                const M* metas) override {
    size_t nwritten_keys = fwrite(keys, sizeof(K), n, keys_fp_);
    size_t nwritten_vecs = fwrite(vectors, sizeof(V) * D, n, values_fp_);
    if (nwritten_keys != nwritten_vecs) {
      return -1;
    }
    return static_cast<int64_t>(nwritten_keys);
  }

 private:
  FILE* keys_fp_;
  FILE* values_fp_;
};

}  // namespace merlin
}  // namespace nv
