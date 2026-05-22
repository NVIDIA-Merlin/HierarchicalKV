/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>
#include <gtest/gtest.h>
#include <array>
#include "merlin_hashtable.cuh"
#include "merlin_hashtable_device.cuh"
#include "test_util.cuh"

namespace cg = cooperative_groups;

constexpr uint32_t kTileSize = 4;
constexpr size_t kDim = 4;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

template <uint32_t TILE_SIZE>
__global__ void ReadonlyLookupKernel(
    nv::merlin::device::HashTableDeviceView<K, V, S> view, const K* keys,
    int key_count, bool* founds, int* positions, bool* key_matches) {
  cg::thread_block_tile<TILE_SIZE> tile =
      cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  const int tile_global =
      blockIdx.x * (blockDim.x / TILE_SIZE) + (threadIdx.x / TILE_SIZE);
  if (tile_global >= key_count) {
    return;
  }
  const K key = keys[tile_global];
  const int pos = nv::merlin::device::find_readonly_no_sync<K, V, S, TILE_SIZE>(
      view, key, tile.thread_rank());
  const unsigned int vote = tile.ballot(pos >= 0);
  int found_pos = -1;
  if (vote) {
    const int src_lane = __ffs(vote) - 1;
    found_pos = tile.shfl(pos, src_lane);
  }
  if (tile.thread_rank() == 0) {
    positions[tile_global] = found_pos;
    founds[tile_global] = (found_pos >= 0);
    key_matches[tile_global] =
        test_util::key_matches<K, V, S, TILE_SIZE>(view, key, found_pos);
  }
}

template <uint32_t TILE_SIZE>
__global__ void ReadonlyLookupInBucketKernel(
    nv::merlin::device::HashTableDeviceView<K, V, S> view, const K* keys,
    int key_count, bool* founds, int* positions, bool* key_matches) {
  cg::thread_block_tile<TILE_SIZE> tile =
      cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  const int tile_global =
      blockIdx.x * (blockDim.x / TILE_SIZE) + (threadIdx.x / TILE_SIZE);
  if (tile_global >= key_count) {
    return;
  }
  const K key = keys[tile_global];
  uint32_t bucket_idx = 0;
  uint32_t aligned_start = 0;
  if (!nv::merlin::device::compute_bucket_index_and_aligned_start<K, V, S,
                                                                  TILE_SIZE>(
          view, key, &bucket_idx, &aligned_start)) {
    if (tile.thread_rank() == 0) {
      positions[tile_global] = -1;
      founds[tile_global] = false;
      key_matches[tile_global] = false;
    }
    return;
  }
  auto* bucket = view.buckets + bucket_idx;
  const int pos =
      nv::merlin::device::find_readonly_no_sync_in_bucket<K, V, S, TILE_SIZE>(
          bucket, key, aligned_start,
          static_cast<uint32_t>(view.bucket_max_size), tile.thread_rank());
  const unsigned int vote = tile.ballot(pos >= 0);
  int found_pos = -1;
  if (vote) {
    const int src_lane = __ffs(vote) - 1;
    found_pos = tile.shfl(pos, src_lane);
  }
  if (tile.thread_rank() == 0) {
    positions[tile_global] = found_pos;
    founds[tile_global] = (found_pos >= 0);
    key_matches[tile_global] =
        test_util::key_matches<K, V, S, TILE_SIZE>(view, key, found_pos);
  }
}

namespace {

void RunReadonlyLookupTest(bool use_in_bucket) {
  TableOptions options;
  options.init_capacity = 1024;
  options.max_capacity = 1024;
  options.max_bucket_size = 128;
  options.dim = kDim;
  options.max_hbm_for_vectors = nv::merlin::GB(1);
  options.reserved_key_start_bit = 2;
  options.num_of_buckets_per_alloc = 1;

  Table table;
  table.init(options);

  constexpr int insert_count = 4;
  std::array<K, insert_count> h_keys = {1, 2, 3, 4};
  std::array<V, insert_count * kDim> h_values{};
  std::array<S, insert_count> h_scores{};
  for (int i = 0; i < insert_count; ++i) {
    h_scores[i] = static_cast<S>(i + 1);
    for (size_t j = 0; j < kDim; ++j) {
      h_values[i * kDim + j] = static_cast<V>(h_keys[i] * 0.1f + j);
    }
  }

  K* d_keys = nullptr;
  V* d_values = nullptr;
  S* d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K) * insert_count));
  CUDA_CHECK(cudaMalloc(&d_values, sizeof(V) * insert_count * kDim));
  CUDA_CHECK(cudaMalloc(&d_scores, sizeof(S) * insert_count));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), sizeof(K) * insert_count,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(),
                        sizeof(V) * insert_count * kDim,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), sizeof(S) * insert_count,
                        cudaMemcpyHostToDevice));

  table.insert_or_assign(insert_count, d_keys, d_values, d_scores);
  CUDA_CHECK(cudaDeviceSynchronize());

  constexpr int query_count = 6;
  std::array<K, query_count> h_query = {1, 4, 5, 2, 42, 3};
  K* d_query = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, sizeof(K) * query_count));
  CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), sizeof(K) * query_count,
                        cudaMemcpyHostToDevice));

  bool* d_founds = nullptr;
  int* d_positions = nullptr;
  bool* d_key_matches = nullptr;
  CUDA_CHECK(cudaMalloc(&d_founds, sizeof(bool) * query_count));
  CUDA_CHECK(cudaMalloc(&d_positions, sizeof(int) * query_count));
  CUDA_CHECK(cudaMalloc(&d_key_matches, sizeof(bool) * query_count));
  CUDA_CHECK(cudaMemset(d_founds, 0, sizeof(bool) * query_count));
  CUDA_CHECK(cudaMemset(d_positions, 0xff, sizeof(int) * query_count));
  CUDA_CHECK(cudaMemset(d_key_matches, 0, sizeof(bool) * query_count));

  auto view = nv::merlin::device::make_device_view(table);
  constexpr int threads = 128;
  constexpr int tiles_per_block = threads / kTileSize;
  const int blocks = (query_count + tiles_per_block - 1) / tiles_per_block;
  if (use_in_bucket) {
    ReadonlyLookupInBucketKernel<kTileSize><<<blocks, threads>>>(
        view, d_query, query_count, d_founds, d_positions, d_key_matches);
  } else {
    ReadonlyLookupKernel<kTileSize><<<blocks, threads>>>(
        view, d_query, query_count, d_founds, d_positions, d_key_matches);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::array<bool, query_count> h_founds{};
  std::array<int, query_count> h_positions{};
  std::array<bool, query_count> h_key_matches{};
  CUDA_CHECK(cudaMemcpy(h_founds.data(), d_founds, sizeof(bool) * query_count,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_positions.data(), d_positions,
                        sizeof(int) * query_count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_key_matches.data(), d_key_matches,
                        sizeof(bool) * query_count, cudaMemcpyDeviceToHost));

  EXPECT_TRUE(h_founds[0]);
  EXPECT_TRUE(h_founds[1]);
  EXPECT_FALSE(h_founds[2]);
  EXPECT_TRUE(h_founds[3]);
  EXPECT_FALSE(h_founds[4]);
  EXPECT_TRUE(h_founds[5]);

  for (int i = 0; i < query_count; ++i) {
    if (h_founds[i]) {
      EXPECT_GE(h_positions[i], 0);
      EXPECT_TRUE(h_key_matches[i]);
    } else {
      EXPECT_EQ(h_positions[i], -1);
      EXPECT_FALSE(h_key_matches[i]);
    }
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_founds));
  CUDA_CHECK(cudaFree(d_positions));
  CUDA_CHECK(cudaFree(d_key_matches));
}

}  // namespace

TEST(HashTableDeviceApiTest, ReadonlyLookup) { RunReadonlyLookupTest(false); }

TEST(HashTableDeviceApiTest, ReadonlyLookupInBucket) {
  RunReadonlyLookupTest(true);
}
