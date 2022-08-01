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

#include <string>
#include <stdio.h>

namespace nv {
namespace merlin {

template <typename Key, typename V, size_t DIM>
class KVFile {
 public:
  virtual size_t Read(Key* h_keys, V* h_vectors, size_t nkeys) = 0;
  virtual size_t Write(const Key* h_keys, const V* h_vectors, size_t nkeys) = 0;
};

template <typename Key, typename V, size_t DIM>
class PosixKVFile : public KVFile<Key, V, DIM> {
 public:
  PosixKVFile(std::string prefix, const char* mode) {
    keyfile_ = prefix + ".keys";
    valuefile_ = prefix + ".values";
    kfp_ = fopen(keyfile_.c_str(), mode);
    if (!kfp_) {
      throw std::runtime_error("Failed to open key file" + keyfile_);
    }
    vfp_ = fopen(valuefile_.c_str(), mode);
    if (!vfp_) {
      throw std::runtime_error("Failed to open value file" + valuefile_);
    }
  }

  ~PosixKVFile() {
    if (kfp_) {
      fclose(kfp_);
    }
    if (vfp_) {
      fclose(vfp_);
    }
  }

  size_t Read(Key* h_keys, V* h_vectors, size_t nkeys) {
    size_t nread_keys = fread(h_keys, sizeof(Key), nkeys, kfp_);
    size_t nread_vevs = fread(h_vectors, sizeof(V) * DIM, nkeys, vfp_);
    if (nread_keys != nread_vevs) {
      throw std::runtime_error("Get different keys and vectors number when read from " + keyfile_ + " and " + valuefile_);
    }
    return nread_keys;
  }

  size_t Write(const Key* h_keys, const V* h_vectors, size_t nkeys) {
    size_t nwritten_keys = fwrite(h_keys, sizeof(Key), nkeys, kfp_);
    size_t nwritten_vecs = fwrite(h_vectors, sizeof(V) * DIM, nkeys, vfp_);
    if (nwritten_keys  != nwritten_vecs) {
      throw std::runtime_error("Get different keys and vectors number when write to " + keyfile_ + " and " + valuefile_);
    }
    return nwritten_keys;
  }

 private:
  std::string keyfile_;
  std::string valuefile_;
  FILE* kfp_;
  FILE* vfp_;
  size_t size_;
};

}  // namespace merlin
}  // namespace nv
