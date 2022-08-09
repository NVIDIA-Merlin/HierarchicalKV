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

#include <stdio.h>
#include <cuda/std/semaphore>
#include <string>

namespace nv {
namespace merlin {

template <class M>
struct Meta {
  M val;
};

constexpr uint64_t EMPTY_KEY = std::numeric_limits<uint64_t>::max();
constexpr uint64_t MAX_META = std::numeric_limits<uint64_t>::max();
constexpr uint64_t EMPTY_META = std::numeric_limits<uint64_t>::min();

template <class K, class V, class M, size_t DIM>
struct Bucket {
  K* keys;         // HBM
  Meta<M>* metas;  // HBM
  V* cache;        // HBM(optional)
  V* vectors;      // Pinned memory or HBM

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

using Mutex = cuda::binary_semaphore<cuda::thread_scope_device>;

template <class K, class V, class M, size_t DIM>
struct Table {
  Bucket<K, V, M, DIM>* buckets;
  Mutex* locks;                 // mutex for write buckets
  int* buckets_size;            // size of each buckets.
  V** slices;                   // Handles of the HBM/ HMEM slices.
  size_t bytes_per_slice;       // Size by byte of one slice.
  size_t num_of_memory_slices;  // Number of vectors memory slices.
  size_t capacity = 134217728;  // Initial capacity.
  size_t max_size =
      std::numeric_limits<uint64_t>::max();  // Up limit of the table capacity.
  size_t buckets_num;                        // Number of the buckets.
  size_t bucket_max_size = 128;              // Volume of each buckets.
  size_t max_hbm_for_vectors = 0;            // Max HBM allocated for vectors
  size_t remaining_hbm_for_vectors = 0;  // Remaining HBM allocated for vectors
  bool is_pure_hbm = true;               // unused
  bool primary = true;                   // unused
  int slots_offset = 0;                  // unused
  int slots_number = 0;                  // unused
  int device_id = 0;                     // Device id
  int tile_size;
};

template <class K, class M>
using EraseIfPredictInternal =
    bool (*)(const K& key,       ///< iterated key in table
             const M& meta,      ///< iterated meta in table
             const K& pattern,   ///< input key from caller
             const M& threshold  ///< input meta from caller
    );

template <typename Key, typename V, typename M, size_t DIM>
class KVFile {
 public:
  virtual size_t Read(Key* keys, V* vectors, M* metas, size_t nkeys) = 0;
  virtual size_t Write(const Key* keys, const V* vectors, const M* metas,
                       size_t nkeys) = 0;
};

template <typename Key, typename V, typename M, size_t DIM>
class PosixKVFile : public KVFile<Key, V, M, DIM> {
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

  size_t Read(Key* keys, V* vectors, M* metas, size_t nkeys) {
    size_t nread_keys = fread(keys, sizeof(Key), nkeys, kfp_);
    size_t nread_vevs = fread(vectors, sizeof(V) * DIM, nkeys, vfp_);
    if (nread_keys != nread_vevs) {
      throw std::runtime_error(
          "Get different keys and vectors number when read from " + keyfile_ +
          " and " + valuefile_);
    }
    return nread_keys;
  }

  size_t Write(const Key* keys, const V* vectors, const M* metas,
               size_t nkeys) {
    size_t nwritten_keys = fwrite(keys, sizeof(Key), nkeys, kfp_);
    size_t nwritten_vecs = fwrite(vectors, sizeof(V) * DIM, nkeys, vfp_);
    if (nwritten_keys != nwritten_vecs) {
      throw std::runtime_error(
          "Get different keys and vectors number when write to " + keyfile_ +
          " and " + valuefile_);
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
