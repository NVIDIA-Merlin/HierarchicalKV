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

  /* The number of saved key-value in this buckets */
  int size;
};

template <class K, class V, class M, size_t DIM>
struct Table {
  Bucket<K, V, M, DIM> *buckets;
  unsigned int *locks;            // Write lock for each bucket.
  V **slices;                     // Handles of the HBM/ HMEM slices.
  uint64_t bytes_per_slice;       // Size by byte of one slice.
  uint64_t num_of_memory_slices;  // Number of vectors memory slices.
  uint64_t capacity = 134217728;  // Initial capacity.
  uint64_t max_size =
      std::numeric_limits<uint64_t>::max();  // Up limit of the table capacity.
  uint64_t buckets_num;                      // Number of the buckets.
  uint64_t buckets_size = 128;               // Volume of each buckets.
  uint64_t cache_size = 0;                   // unused
  bool vector_on_gpu = false;                // If the Values are stored on HBM.
  bool primary_table = true;                 // unused
  int slots_number = 0;                      // unused
  int slots_offset = 0;                      // unused
};

}  // namespace merlin
}  // namespace nv