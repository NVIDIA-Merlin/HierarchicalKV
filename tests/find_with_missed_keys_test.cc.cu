/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <array>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;

void test_find(size_t max_hbm_for_vectors, size_t max_bucket_size,
               double load_factor, bool pipeline_lookup, int key_start = 0) {
  MERLIN_CHECK(load_factor >= 0.0 && load_factor <= 1.0,
               "Invalid `load_factor`");

  constexpr uint64_t INIT_CAPACITY = 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  K* h_missed_keys;
  int* h_missed_indices;

  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::MB(max_hbm_for_vectors);
  if (pipeline_lookup) {
    options.max_bucket_size = 128;
  } else {
    options.max_bucket_size = 256;
  }
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_missed_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_missed_indices, KEY_NUM * sizeof(int)));

  K* d_keys;
  S* d_scores;
  V* d_vectors;
  K* d_missed_keys;
  int* d_missed_indices;
  int* d_missed_size;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, KEY_NUM * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_missed_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_missed_indices, KEY_NUM * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_missed_size, sizeof(int)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int missed_size;
  for (int i = 0; i < TEST_TIMES; ++i) {
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, h_vectors,
                                                KEY_NUM);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, KEY_NUM * sizeof(S),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));

    Table table;
    table.init(options);
    size_t size = table.size(stream);
    ASSERT_EQ(size, 0);

    size_t insert_num = (double)KEY_NUM * load_factor;
    table.insert_or_assign(insert_num, d_keys, d_vectors, d_scores, stream);
    table.find(KEY_NUM, d_keys, d_vectors, d_missed_keys, d_missed_indices,
               d_missed_size, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&missed_size, d_missed_size, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_missed_keys, d_missed_keys, missed_size * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_missed_indices, d_missed_indices,
                          missed_size * sizeof(int), cudaMemcpyDeviceToHost));

    if (insert_num == 0) {
      ASSERT_EQ(missed_size, KEY_NUM);
    } else {
      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));

      ASSERT_TRUE(missed_size > 0 && missed_size < KEY_NUM);
      std::vector<bool> founds(KEY_NUM, true);
      // Check missed
      for (int j = 0; j < missed_size; ++j) {
        int idx = h_missed_indices[i];
        ASSERT_TRUE(idx >= 0 && idx < KEY_NUM);
        ASSERT_EQ(h_keys[idx], h_missed_keys[i]);
        founds[idx] = false;
      }
      // Check hit
      for (uint64_t j = 0; j < KEY_NUM; ++j) {
        if (founds[j]) {
          for (int k = 0; k < options.dim; ++k) {
            ASSERT_EQ(h_vectors[j * options.dim + k],
                      static_cast<float>(h_keys[j] * 0.00001));
          }
        }
      }
    }
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_vectors));
  CUDA_CHECK(cudaFreeHost(h_missed_keys));
  CUDA_CHECK(cudaFreeHost(h_missed_indices));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_missed_keys));
  CUDA_CHECK(cudaFree(d_missed_indices));
  CUDA_CHECK(cudaFree(d_missed_size));

  CudaCheckError();
}

TEST(FindTest, test_find_when_empty) {
  // pure HMEM
  test_find(0, 128, 0.0, true, 12);
  test_find(0, 256, 0.0, false);
  // hybrid
  test_find(32, 128, 0.0, true, 58);
  test_find(32, 256, 0.0, false);
  // pure HBM
  test_find(1024, 128, 0.0, true);
  test_find(1024, 256, 0.0, false, 12);
}

TEST(FindTest, test_find_when_full) {
  // pure HMEM
  test_find(0, 128, 1.0, true);
  test_find(0, 256, 1.0, false);
  // hybrid
  test_find(32, 128, 1.0, true);
  test_find(32, 256, 1.0, false, 60);
  // pure HBM
  test_find(1024, 128, 1.0, true);
  test_find(1024, 256, 1.0, false);
}

TEST(FindTest, test_find_load_factor) {
  // pure HMEM
  test_find(0, 128, 0.2, true, 45);
  test_find(0, 256, 0.2, false, 12);
  // hybrid
  test_find(32, 128, 0.2, true, 27);
  test_find(32, 256, 0.2, false, 53);
  // pure HBM
  test_find(1024, 128, 0.2, true, 9);
  test_find(1024, 256, 0.2, false, 38);

  // pure HMEM
  test_find(0, 128, 0.5, true, 21);
  test_find(0, 256, 0.5, false, 46);
  // hybrid
  test_find(32, 128, 0.5, true, 31);
  test_find(32, 256, 0.5, false, 59);
  // pure HBM
  test_find(1024, 128, 0.5, true, 4);
  test_find(1024, 256, 0.5, false, 22);

  // pure HMEM
  test_find(0, 128, 0.75, true, 11);
  test_find(0, 256, 0.75, false, 34);
  // hybrid
  test_find(32, 128, 0.75, true, 18);
  test_find(32, 256, 0.75, false, 47);
  // pure HBM
  test_find(1024, 128, 0.75, true, 7);
  test_find(1024, 256, 0.75, false, 29);
}

namespace {

__global__ void copy_r_role_rows(V* const* pointers, V* rows,
                                 size_t element_count) {
  const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < element_count) {
    const V* row = pointers[element / DIM];
    rows[element] = row == nullptr ? -1.0f : row[element % DIM];
  }
}

template <int Strategy>
S r_role_score_after(S before, S increment) {
  if constexpr (Strategy == EvictStrategy::kLfu) {
    constexpr S cap = nv::merlin::MAX_SCORE - 1;
    return increment > cap - before ? cap : before + increment;
  } else if constexpr (Strategy == EvictStrategy::kEpochLfu) {
    constexpr S cap = UINT64_C(0xffffffff);
    const S count = before & cap;
    return (S{3} << 32) | (increment > cap - count ? cap : count + increment);
  } else {
    return increment;
  }
}

template <int Strategy>
void test_r_role_find_and_update() {
#ifdef CUDA_ERROR_CHECK
  FAIL() << "R-role overlap requires CUDA_ERROR_CHECK disabled.";
#endif
  constexpr size_t key_count = 4096;
  constexpr size_t rounds = 4;
  constexpr size_t stream_count = 4;
  constexpr size_t element_count = key_count * DIM;
  constexpr S score_cap = Strategy == EvictStrategy::kEpochLfu
                              ? UINT64_C(0xffffffff)
                              : nv::merlin::MAX_SCORE - 1;
  constexpr S epoch = Strategy == EvictStrategy::kEpochLfu ? S{3} << 32 : 0;
  using Table = nv::merlin::HashTable<K, V, S, Strategy>;

  TableOptions options;
  options.init_capacity = 32768;
  options.max_capacity = 32768;
  options.max_bucket_size = 128;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::MB(16);
  options.api_lock = true;
  Table table;
  table.init(options);
  if constexpr (Strategy == EvictStrategy::kEpochLfu) {
    table.set_global_epoch(3);
  }

  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaStream_t control;
  std::array<cudaStream_t, stream_count> streams;
  CUDA_CHECK(cudaStreamCreateWithFlags(&control, cudaStreamNonBlocking));
  for (auto& stream : streams) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  test_util::HostAndDeviceBuffer<K> keys;
  test_util::HostAndDeviceBuffer<V> initial_rows;
  test_util::HostAndDeviceBuffer<S> initial_scores;
  test_util::HostAndDeviceBuffer<V*> resident_pointers;
  test_util::HostAndDeviceBuffer<bool> final_found;
  test_util::HostAndDeviceBuffer<V> final_rows;
  test_util::HostAndDeviceBuffer<S> final_scores;
  std::array<test_util::HostAndDeviceBuffer<S>, 2> increments;
  std::array<test_util::HostAndDeviceBuffer<V*>, 3> pointers;
  std::array<test_util::HostAndDeviceBuffer<bool>, stream_count> found;
  std::array<test_util::HostAndDeviceBuffer<V>, stream_count> rows;
  std::array<test_util::HostAndDeviceBuffer<S>, stream_count> scores;
  keys.Alloc(key_count, control);
  initial_rows.Alloc(element_count, control);
  initial_scores.Alloc(key_count, control);
  resident_pointers.Alloc(key_count, control);
  final_found.Alloc(key_count, control);
  final_rows.Alloc(element_count, control);
  final_scores.Alloc(key_count, control);
  for (auto& buffer : increments) buffer.Alloc(key_count, control);
  for (auto& buffer : pointers) buffer.Alloc(key_count, control);
  for (auto& buffer : found) buffer.Alloc(key_count, control);
  for (auto& buffer : rows) buffer.Alloc(element_count, control);
  for (auto& buffer : scores) buffer.Alloc(key_count, control);
  for (size_t key_index = 0; key_index < key_count; ++key_index) {
    keys.h_data[key_index] = key_index + 1;
    initial_scores.h_data[key_index] =
        epoch | (key_index % 2 == 0 ? S{17} : score_cap - 3);
    increments[0].h_data[key_index] = 2;
    increments[1].h_data[key_index] = 5;
    for (size_t component = 0; component < DIM; ++component) {
      initial_rows.h_data[key_index * DIM + component] =
          static_cast<V>(key_index * DIM + component + 1);
    }
  }
  keys.SyncData(true, control);
  initial_rows.SyncData(true, control);
  initial_scores.SyncData(true, control);
  for (auto& buffer : increments) buffer.SyncData(true, control);
  table.insert_or_assign(
      key_count, keys.d_data, initial_rows.d_data,
      Strategy == EvictStrategy::kLru ? nullptr : initial_scores.d_data,
      control);
  CUDA_CHECK(cudaStreamSynchronize(control));

  if constexpr (Strategy == EvictStrategy::kCustomized) {
    CUDA_CHECK(cudaMemcpyAsync(scores[0].d_data, initial_scores.d_data,
                               key_count * sizeof(S), cudaMemcpyDeviceToDevice,
                               control));
  } else {
    CUDA_CHECK(
        cudaMemsetAsync(scores[0].d_data, 0, key_count * sizeof(S), control));
  }
  table.find_and_update(
      key_count, keys.d_data, pointers[0].d_data, found[0].d_data,
      Strategy == EvictStrategy::kLru ? nullptr : scores[0].d_data, control,
      true);
  table.find(key_count, keys.d_data, resident_pointers.d_data,
             final_found.d_data, final_scores.d_data, control, true);
  copy_r_role_rows<<<(element_count + 127) / 128, 128, 0, control>>>(
      resident_pointers.d_data, rows[0].d_data, element_count);
  table.find(key_count, keys.d_data, final_rows.d_data, found[3].d_data,
             scores[3].d_data, control);
  resident_pointers.SyncData(false, control);
  final_found.SyncData(false, control);
  final_scores.SyncData(false, control);
  final_rows.SyncData(false, control);
  CUDA_CHECK(cudaStreamSynchronize(control));
  for (size_t key_index = 0; key_index < key_count; ++key_index) {
    ASSERT_TRUE(final_found.h_data[key_index]) << key_index;
    ASSERT_NE(resident_pointers.h_data[key_index], nullptr) << key_index;
    if constexpr (Strategy != EvictStrategy::kLru) {
      ASSERT_EQ(final_scores.h_data[key_index],
                initial_scores.h_data[key_index])
          << key_index;
    }
  }
  for (size_t element = 0; element < element_count; ++element) {
    ASSERT_EQ(final_rows.h_data[element], initial_rows.h_data[element])
        << element;
  }

  for (size_t round = 0; round < rounds; ++round) {
    SCOPED_TRACE(round);
    std::vector<S> previous(final_scores.h_data,
                            final_scores.h_data + key_count);
    for (size_t stream_index = 0; stream_index < stream_count; ++stream_index) {
      CUDA_CHECK(cudaMemsetAsync(found[stream_index].d_data, 0,
                                 key_count * sizeof(bool),
                                 streams[stream_index]));
      CUDA_CHECK(cudaMemsetAsync(rows[stream_index].d_data, 0xff,
                                 element_count * sizeof(V),
                                 streams[stream_index]));
      CUDA_CHECK(cudaMemsetAsync(scores[stream_index].d_data, 0xff,
                                 key_count * sizeof(S), streams[stream_index]));
      if (stream_index < 3) {
        CUDA_CHECK(cudaMemsetAsync(pointers[stream_index].d_data, 0,
                                   key_count * sizeof(V*),
                                   streams[stream_index]));
      }
      if (stream_index < 2) {
        CUDA_CHECK(cudaMemcpyAsync(
            scores[stream_index].d_data, increments[stream_index].d_data,
            key_count * sizeof(S), cudaMemcpyDeviceToDevice,
            streams[stream_index]));
      }
    }
    for (auto stream : streams) CUDA_CHECK(cudaStreamSynchronize(stream));
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::array<std::exception_ptr, stream_count> failures{};
    std::array<std::thread, stream_count> workers;
    for (size_t stream_index = 0; stream_index < stream_count; ++stream_index) {
      workers[stream_index] = std::thread([&, stream_index] {
        try {
          CUDA_CHECK(cudaSetDevice(device_id));
          if (stream_index == 2) {
            table.find(key_count, keys.d_data, pointers[2].d_data,
                       found[2].d_data, scores[2].d_data, streams[2], true);
            CUDA_CHECK(cudaMemsetAsync(pointers[2].d_data, 0,
                                       key_count * sizeof(V*), streams[2]));
            CUDA_CHECK(cudaMemsetAsync(found[2].d_data, 0,
                                       key_count * sizeof(bool), streams[2]));
            CUDA_CHECK(cudaMemsetAsync(scores[2].d_data, 0xff,
                                       key_count * sizeof(S), streams[2]));
            CUDA_CHECK(cudaStreamSynchronize(streams[2]));
          }
        } catch (...) {
          failures[stream_index] = std::current_exception();
        }
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        if (failures[stream_index]) return;
        try {
          if (stream_index < 2) {
            table.find_and_update(
                key_count, keys.d_data, pointers[stream_index].d_data,
                found[stream_index].d_data,
                Strategy == EvictStrategy::kLru ? nullptr
                                                : scores[stream_index].d_data,
                streams[stream_index], true);
          } else if (stream_index == 2) {
            table.find(key_count, keys.d_data, pointers[2].d_data,
                       found[2].d_data, scores[2].d_data, streams[2], true);
          } else {
            table.find(key_count, keys.d_data, rows[3].d_data, found[3].d_data,
                       scores[3].d_data, streams[3]);
          }
        } catch (...) {
          failures[stream_index] = std::current_exception();
        }
      });
    }
    while (ready.load(std::memory_order_acquire) != stream_count) {
      std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (const auto& failure : failures) {
      if (failure) std::rethrow_exception(failure);
    }
    for (size_t stream_index = 0; stream_index < stream_count; ++stream_index) {
      if (stream_index < 3) {
        copy_r_role_rows<<<(element_count + 127) / 128, 128, 0,
                           streams[stream_index]>>>(
            pointers[stream_index].d_data, rows[stream_index].d_data,
            element_count);
        pointers[stream_index].SyncData(false, streams[stream_index]);
      }
      found[stream_index].SyncData(false, streams[stream_index]);
      rows[stream_index].SyncData(false, streams[stream_index]);
      scores[stream_index].SyncData(false, streams[stream_index]);
    }
    for (auto stream : streams) CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(control));
    table.find(key_count, keys.d_data, final_rows.d_data, final_found.d_data,
               final_scores.d_data, control);
    final_found.SyncData(false, control);
    final_rows.SyncData(false, control);
    final_scores.SyncData(false, control);
    CUDA_CHECK(cudaStreamSynchronize(control));

    for (size_t key_index = 0; key_index < key_count; ++key_index) {
      SCOPED_TRACE(key_index);
      const S first = r_role_score_after<Strategy>(previous[key_index], 2);
      const S second = r_role_score_after<Strategy>(previous[key_index], 5);
      const S completed = r_role_score_after<Strategy>(first, 5);
      ASSERT_TRUE(final_found.h_data[key_index]);
      if constexpr (Strategy == EvictStrategy::kCustomized) {
        ASSERT_TRUE(final_scores.h_data[key_index] == first ||
                    final_scores.h_data[key_index] == second);
      } else if constexpr (Strategy != EvictStrategy::kLru) {
        ASSERT_EQ(final_scores.h_data[key_index], completed);
      }
      for (size_t stream_index = 0; stream_index < stream_count;
           ++stream_index) {
        SCOPED_TRACE(stream_index);
        ASSERT_TRUE(found[stream_index].h_data[key_index]);
        if (stream_index < 3) {
          ASSERT_EQ(pointers[stream_index].h_data[key_index],
                    resident_pointers.h_data[key_index]);
        }
        if constexpr (Strategy != EvictStrategy::kLru) {
          const S observed = scores[stream_index].h_data[key_index];
          if (stream_index < 2) {
            const S own = stream_index == 0 ? first : second;
            const S other = stream_index == 0 ? second : first;
            ASSERT_TRUE(observed == own ||
                        observed == (Strategy == EvictStrategy::kCustomized
                                         ? other
                                         : completed))
                << observed;
          } else {
            ASSERT_TRUE(observed == previous[key_index] || observed == first ||
                        observed == second || observed == completed)
                << observed;
          }
        }
      }
    }
    for (size_t element = 0; element < element_count; ++element) {
      ASSERT_EQ(final_rows.h_data[element], initial_rows.h_data[element])
          << element;
      for (size_t stream_index = 0; stream_index < stream_count;
           ++stream_index) {
        ASSERT_EQ(rows[stream_index].h_data[element],
                  initial_rows.h_data[element])
            << "stream=" << stream_index << " element=" << element;
      }
    }
  }
  for (auto stream : streams) CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaStreamDestroy(control));
}

}  // namespace

TEST(FindTest, find_and_update_rr_lfu_exactly_once) {
  test_r_role_find_and_update<EvictStrategy::kLfu>();
}

TEST(FindTest, find_and_update_rr_epoch_lfu_exactly_once) {
  test_r_role_find_and_update<EvictStrategy::kEpochLfu>();
}

TEST(FindTest, find_and_update_rr_customized) {
  test_r_role_find_and_update<EvictStrategy::kCustomized>();
}

TEST(FindTest, find_and_update_rr_lru_null_scores) {
  test_r_role_find_and_update<EvictStrategy::kLru>();
}
