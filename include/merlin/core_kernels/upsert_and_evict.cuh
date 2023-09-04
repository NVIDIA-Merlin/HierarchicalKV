/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "kernel_utils.cuh"

namespace nv {
namespace merlin {

// Use 1 thread to deal with a KV-pair, including copying value.
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128, int Strategy = -1>
__global__ void tlp_v1_upsert_and_evict_kernel_unique(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, const VecV* __restrict__ values,
    const S* __restrict__ scores, K* __restrict__ evicted_keys,
    VecV* __restrict__ evicted_values, S* __restrict__ evicted_scores,
    uint64_t n, uint64_t* __restrict__ evicted_counter, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, 1>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ __align__(sizeof(byte16))
      S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  VecV* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  int32_t* bucket_size_ptr{nullptr};
  uint32_t key_pos = {0};
  uint32_t evict_idx{0};
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = global_idx & (bucket_capacity - 1);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      bucket_size_ptr = buckets_size + bkt_idx;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_size = *bucket_size_ptr;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = reinterpret_cast<VecV*>(bucket->vectors);
    } else {
      return;
    }
  } else {
    return;
  }

  // Load `STRIDE` digests every time.
  constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);
  // One more loop to handle empty keys.
  for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
    if (occupy_result != OccupyResult::INITIAL) break;

    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
    VecD_Load digests_vec = *(reinterpret_cast<VecD_Load*>(digests_ptr));
    VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                digests_vec.w};

    for (int i = 0; i < 4; i++) {
      VecD_Comp probe_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      bool result = false;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      int cmp_result = __vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        // CUDA uses little endian,
        // and the lowest byte in register stores in the lowest address.
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = key;
        // Modifications to the bucket will not before this instruction.
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::DUPLICATE;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(
            bucket_keys_ptr, key_pos, scores, kv_idx, score, bucket_capacity,
            get_digest<K>(key), (occupy_result != OccupyResult::DUPLICATE));
        break;
      } else if (bucket_size == bucket_capacity) {
        continue;
      }
      VecD_Comp empty_digests_ = empty_digests<K>();
      cmp_result = __vcmpeq4(probe_digests, empty_digests_);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = static_cast<K>(EMPTY_KEY);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::OCCUPIED_EMPTY;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(
            bucket_keys_ptr, key_pos, scores, kv_idx, score, bucket_capacity,
            get_digest<K>(key), (occupy_result != OccupyResult::DUPLICATE));
        atomicAdd(bucket_size_ptr, 1);
        break;
      }
    }
  }
  if (occupy_result == OccupyResult::INITIAL) {
    evict_idx = atomicAdd(evicted_counter, 1);
  }
  while (occupy_result == OccupyResult::INITIAL) {
    S* bucket_scores_ptr = BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
    S min_score = MAX_SCORE;
    int min_pos = -1;
#pragma unroll
    for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
      __pipeline_memcpy_async(sm_bucket_scores[tx] + j, bucket_scores_ptr + j,
                              sizeof(S) * Load_LEN_S);
    }
    __pipeline_commit();
    for (int i = 0; i < bucket_capacity; i += STRIDE_S) {
      if (i < bucket_capacity - STRIDE_S) {
#pragma unroll
        for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
          __pipeline_memcpy_async(
              sm_bucket_scores[tx] + diff_buf(i / STRIDE_S) * STRIDE_S + j,
              bucket_scores_ptr + i + STRIDE_S + j, sizeof(S) * Load_LEN_S);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      S temp_scores[Load_LEN_S];
      S* src = sm_bucket_scores[tx] + same_buf(i / STRIDE_S) * STRIDE_S;
#pragma unroll
      for (int k = 0; k < STRIDE_S; k += Load_LEN_S) {
        *reinterpret_cast<byte16*>(temp_scores) =
            *reinterpret_cast<byte16*>(src + k);
#pragma unroll
        for (int j = 0; j < Load_LEN_S; j += 1) {
          S temp_score = temp_scores[j];
          if (temp_score < min_score) {
            min_score = temp_score;
            min_pos = i + k + j;
          }
        }
      }
    }
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (score < min_score) {
      occupy_result = OccupyResult::REFUSED;
      evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx, key,
                            score);
      break;
    }
    auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos);
    auto expected_key = min_score_key->load(cuda::std::memory_order_relaxed);
    if (expected_key != static_cast<K>(LOCKED_KEY) &&
        expected_key != static_cast<K>(EMPTY_KEY)) {
      bool result = min_score_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      if (result) {
        S* min_score_ptr =
            BUCKET::scores(bucket_keys_ptr, bucket_capacity, min_pos);
        auto verify_score_ptr =
            reinterpret_cast<AtomicScore<S>*>(min_score_ptr);
        auto verify_score =
            verify_score_ptr->load(cuda::std::memory_order_relaxed);
        if (verify_score <= min_score) {
          key_pos = min_pos;
          ScoreFunctor::update_with_digest(
              bucket_keys_ptr, key_pos, scores, kv_idx, score, bucket_capacity,
              get_digest<K>(key), (occupy_result != OccupyResult::DUPLICATE));

          if (expected_key == static_cast<K>(RECLAIM_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
            atomicAdd(bucket_size_ptr, 1);
          } else {
            occupy_result = OccupyResult::EVICT;
            evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx,
                                  expected_key, min_score);
          }
        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }
  VecV* bucket_value_ptr = bucket_values_ptr + key_pos * dim;
  const VecV* param_value_ptr = values + kv_idx * dim;
  VecV* evicted_value_ptr = evicted_values + evict_idx * dim;

  if (occupy_result != OccupyResult::REFUSED) {
    if (occupy_result == OccupyResult::EVICT) {
      CopyValue::ldg_stg(0, evicted_value_ptr, bucket_value_ptr, dim);
    }
    CopyValue::ldg_stg(0, bucket_value_ptr, param_value_ptr, dim);
    auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
    // memory_order_release:
    // Modifications to the bucket will not after this instruction.
    key_address->store(key, cuda::std::memory_order_release);
  } else {
    CopyValue::ldg_stg(0, evicted_value_ptr, param_value_ptr, dim);
  }
}

// Use 1 thread to deal with a KV-pair, but use a threads group cto copy value.
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128,
          uint32_t GROUP_SIZE = 16, int Strategy = -1>
__global__ void tlp_v2_upsert_and_evict_kernel_unique(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, const VecV* __restrict__ values,
    const S* __restrict__ scores, K* __restrict__ evicted_keys,
    VecV* __restrict__ evicted_values, S* __restrict__ evicted_scores,
    uint64_t n, uint64_t* __restrict__ evicted_counter, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ __align__(sizeof(byte16))
      S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  VecV* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  int32_t* bucket_size_ptr{nullptr};
  uint32_t key_pos = {0};
  uint32_t evict_idx{0};
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = global_idx & (bucket_capacity - 1);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      bucket_size_ptr = buckets_size + bkt_idx;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_size = *bucket_size_ptr;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = reinterpret_cast<VecV*>(bucket->vectors);
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  // Load `STRIDE` digests every time.
  constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);
  // One more loop to handle empty keys.
  for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
    if (occupy_result != OccupyResult::INITIAL) break;

    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
    VecD_Load digests_vec = *(reinterpret_cast<VecD_Load*>(digests_ptr));
    VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                digests_vec.w};

    for (int i = 0; i < 4; i++) {
      VecD_Comp probe_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      bool result = false;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      int cmp_result = __vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        // CUDA uses little endian,
        // and the lowest byte in register stores in the lowest address.
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = key;
        // Modifications to the bucket will not before this instruction.
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::DUPLICATE;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                         kv_idx, score, bucket_capacity,
                                         get_digest<K>(key), false);
        break;
      } else if (bucket_size == bucket_capacity) {
        continue;
      }
      VecD_Comp empty_digests_ = empty_digests<K>();
      cmp_result = __vcmpeq4(probe_digests, empty_digests_);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) break;
        uint32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = static_cast<K>(EMPTY_KEY);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      if (result) {
        occupy_result = OccupyResult::OCCUPIED_EMPTY;
        key_pos = possible_pos;
        ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                         kv_idx, score, bucket_capacity,
                                         get_digest<K>(key), true);
        atomicAdd(bucket_size_ptr, 1);
        break;
      }
    }
  }
  if (occupy_result == OccupyResult::INITIAL) {
    evict_idx = atomicAdd(evicted_counter, 1);
  }
  while (occupy_result == OccupyResult::INITIAL) {
    S* bucket_scores_ptr = BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
    S min_score = static_cast<S>(MAX_SCORE);
    int min_pos = -1;
#pragma unroll
    for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
      __pipeline_memcpy_async(sm_bucket_scores[tx] + j, bucket_scores_ptr + j,
                              sizeof(S) * Load_LEN_S);
    }
    __pipeline_commit();
    for (int i = 0; i < bucket_capacity; i += STRIDE_S) {
      if (i < bucket_capacity - STRIDE_S) {
#pragma unroll
        for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
          __pipeline_memcpy_async(
              sm_bucket_scores[tx] + diff_buf(i / STRIDE_S) * STRIDE_S + j,
              bucket_scores_ptr + i + STRIDE_S + j, sizeof(S) * Load_LEN_S);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      S temp_scores[Load_LEN_S];
      S* src = sm_bucket_scores[tx] + same_buf(i / STRIDE_S) * STRIDE_S;
#pragma unroll
      for (int k = 0; k < STRIDE_S; k += Load_LEN_S) {
        *reinterpret_cast<byte16*>(temp_scores) =
            *reinterpret_cast<byte16*>(src + k);
#pragma unroll
        for (int j = 0; j < Load_LEN_S; j += 1) {
          S temp_score = temp_scores[j];
          if (temp_score < min_score) {
            auto verify_key_ptr = BUCKET::keys(bucket_keys_ptr, i + k + j);
            auto verify_key =
                verify_key_ptr->load(cuda::std::memory_order_relaxed);
            if (verify_key != static_cast<K>(LOCKED_KEY) &&
                verify_key != static_cast<K>(EMPTY_KEY)) {
              min_score = temp_score;
              min_pos = i + k + j;
            }
          }
        }
      }
    }
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (score < min_score) {
      occupy_result = OccupyResult::REFUSED;
      evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx, key,
                            score);
      break;
    }
    auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos);
    auto expected_key = min_score_key->load(cuda::std::memory_order_relaxed);
    if (expected_key != static_cast<K>(LOCKED_KEY) &&
        expected_key != static_cast<K>(EMPTY_KEY)) {
      bool result = min_score_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      if (result) {
        S* min_score_ptr =
            BUCKET::scores(bucket_keys_ptr, bucket_capacity, min_pos);
        auto verify_score_ptr =
            reinterpret_cast<AtomicScore<S>*>(min_score_ptr);
        auto verify_score =
            verify_score_ptr->load(cuda::std::memory_order_relaxed);
        if (verify_score <= min_score) {
          key_pos = min_pos;
          ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                           kv_idx, score, bucket_capacity,
                                           get_digest<K>(key), true);
          if (expected_key == static_cast<K>(RECLAIM_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
            atomicAdd(bucket_size_ptr, 1);
          } else {
            occupy_result = OccupyResult::EVICT;
            evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx,
                                  expected_key, min_score);
          }
        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }
  VecV* bucket_value_ptr{nullptr};
  if (occupy_result != OccupyResult::ILLEGAL) {
    bucket_value_ptr = bucket_values_ptr + key_pos * dim;
  }
  uint32_t rank = g.thread_rank();
  uint32_t groupID = threadIdx.x / GROUP_SIZE;

  // Shared memory reuse:
  // __shared__ S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];
  // __shared__ VecV sm_values_buffer[GROUP_NUM][2][GROUP_BUF];
  // assert(GROUP_BUF >= 2 * dim);
  constexpr uint32_t GROUP_BUFs =
      GROUP_SIZE * 2 * STRIDE_S * sizeof(S) / sizeof(VecV);
  constexpr uint32_t GROUP_BUF = GROUP_BUFs / 2;
  auto sm_values_buffer =
      reinterpret_cast<VecV*>(&(sm_bucket_scores[0][0])) + groupID * GROUP_BUFs;

  auto occupy_result_next = g.shfl(occupy_result, 0);
  if (occupy_result_next != OccupyResult::ILLEGAL) {
    auto kv_idx_next = g.shfl(kv_idx, 0);
    const VecV* src = values + kv_idx_next * dim;
    VecV* dst = sm_values_buffer;
    CopyValue::ldg_sts(rank, dst, src, dim);

    if (occupy_result_next == OccupyResult::EVICT) {
      const VecV* src = g.shfl(bucket_value_ptr, 0);
      dst = dst + dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    }
  }
  __pipeline_commit();

  for (int i = 0; i < GROUP_SIZE; i++) {
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      if (occupy_result_next != OccupyResult::ILLEGAL) {
        auto kv_idx_next = g.shfl(kv_idx, i + 1);
        const VecV* src = values + kv_idx_next * dim;
        VecV* dst = sm_values_buffer + diff_buf(i) * GROUP_BUF;
        CopyValue::ldg_sts(rank, dst, src, dim);

        if (occupy_result_next == OccupyResult::EVICT) {
          const VecV* src = g.shfl(bucket_value_ptr, i + 1);
          dst = dst + dim;
          CopyValue::ldg_sts(rank, dst, src, dim);
        }
      }
    }
    __pipeline_commit();
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur != OccupyResult::ILLEGAL) {
      auto evict_idx_cur = g.shfl(evict_idx, i);

      VecV* src = sm_values_buffer + same_buf(i) * GROUP_BUF;
      if (occupy_result_cur != OccupyResult::REFUSED) {
        VecV* dst = g.shfl(bucket_value_ptr, i);
        __pipeline_wait_prior(1);
        CopyValue::lds_stg(rank, dst, src, dim);
        if (rank == i) {
          auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
          // memory_order_release:
          // Modifications to the bucket will not after this instruction.
          key_address->store(key, cuda::std::memory_order_release);
        }
        if (occupy_result_cur == OccupyResult::EVICT) {
          src = src + dim;
          VecV* dst = evicted_values + evict_idx_cur * dim;
          CopyValue::lds_stg(rank, dst, src, dim);
        }
      } else {
        VecV* dst = evicted_values + evict_idx_cur * dim;
        __pipeline_wait_prior(1);
        CopyValue::lds_stg(rank, dst, src, dim);
      }
    }
  }
}

template <typename K, typename V, typename S, typename VecV,
          uint32_t BLOCK_SIZE, uint32_t GROUP_SIZE, uint32_t BUCKET_SIZE,
          uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE,
          uint32_t offset_param_scores = 0,
          uint32_t offset_bucket_values_ptr =
              offset_param_scores + sizeof(S) * BLOCK_SIZE,
          uint32_t offset_buckets_size_ptr =
              offset_bucket_values_ptr + sizeof(VecV*) * BLOCK_SIZE,
          uint32_t offset_bucket_digests =
              offset_buckets_size_ptr + sizeof(int*) * BLOCK_SIZE,
          uint32_t offset_bucket_scores =
              offset_bucket_digests + sizeof(D) * GROUP_NUM * 2 * BUCKET_SIZE,
          uint32_t offset_values_buffer =
              offset_bucket_scores + sizeof(S) * GROUP_NUM * 2 * BUCKET_SIZE>
struct SharedMemoryManager_Pipeline_UpsertAndEvict {
  // __shared__ S sm_param_scores[BLOCK_SIZE];
  // __shared__ VecV* sm_bucket_values_ptr[BLOCK_SIZE];
  // __shared__ int* sm_buckets_size_ptr[BLOCK_SIZE];
  // __shared__ D sm_bucket_digests[GROUP_NUM][2][BUCKET_SIZE];
  // __shared__ S sm_bucket_scores[GROUP_NUM][2][BUCKET_SIZE];
  // __shared__ VecV sm_values_buffer[GROUP_NUM][2][dim * 2];

  static inline uint32_t total_size(uint32_t dim) {
    return BLOCK_SIZE * (sizeof(S) + sizeof(VecV*) + sizeof(int*)) +
           GROUP_NUM * 2 *
               (BUCKET_SIZE * (sizeof(D) + sizeof(S)) + 2 * dim * sizeof(VecV));
  }
  static __forceinline__ __device__ S* param_scores(byte* smem) {
    return reinterpret_cast<S*>(smem + offset_param_scores);
  }
  static __forceinline__ __device__ VecV** bucket_values_ptr(byte* smem) {
    return reinterpret_cast<VecV**>(smem + offset_bucket_values_ptr);
  }
  static __forceinline__ __device__ int** buckets_size_ptr(byte* smem) {
    return reinterpret_cast<int**>(smem + offset_buckets_size_ptr);
  }
  static __forceinline__ __device__ D* bucket_digests(byte* smem,
                                                      uint32_t groupID,
                                                      uint32_t buf) {
    return reinterpret_cast<D*>(smem + offset_bucket_digests) +
           BUCKET_SIZE * (groupID * 2 + buf);
  }
  static __forceinline__ __device__ S* bucket_scores(byte* smem,
                                                     uint32_t groupID,
                                                     uint32_t buf) {
    return reinterpret_cast<S*>(smem + offset_bucket_scores) +
           BUCKET_SIZE * (groupID * 2 + buf);
  }
  static __forceinline__ __device__ VecV* values_buffer(byte* smem,
                                                        uint32_t groupID,
                                                        uint32_t buf,
                                                        uint32_t dim) {
    return reinterpret_cast<VecV*>(smem + offset_values_buffer) +
           2 * dim * (groupID * 2 + buf);
  }
};

template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128, int Strategy = -1>
__global__ void pipeline_upsert_and_evict_kernel_unique(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, const uint32_t dim, const K* __restrict__ keys,
    const VecV* __restrict__ values, const S* __restrict__ scores,
    K* __restrict__ evicted_keys, VecV* __restrict__ evicted_values,
    S* __restrict__ evicted_scores, uint64_t n,
    uint64_t* __restrict__ evicted_counter, const S global_epoch) {
  // Here, GROUP_SIZE * Comp_LEN = BUCKET_SIZE.
  constexpr uint32_t BUCKET_SIZE = 128;
  constexpr uint32_t GROUP_SIZE = 32;
  constexpr uint32_t Comp_LEN = sizeof(VecD_Comp) / sizeof(D);
  constexpr uint32_t Load_LEN = sizeof(VecD_Load) / sizeof(D);
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);

  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
  using SMM =
      SharedMemoryManager_Pipeline_UpsertAndEvict<K, V, S, VecV, BLOCK_SIZE,
                                                  GROUP_SIZE, BUCKET_SIZE>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  extern __shared__ __align__(sizeof(byte16)) byte smem[];

  // Initialization.
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  VecD_Comp target_digests;
  K* bucket_keys_ptr{nullptr};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  uint32_t key_pos = 0;
  uint32_t evict_idx = 0;
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (scores != nullptr) {
      S* sm_param_scores = SMM::param_scores(smem);
      __pipeline_memcpy_async(sm_param_scores + tx, scores + kv_idx, sizeof(S));
    }
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * BUCKET_SIZE));
      uint64_t bkt_idx = global_idx / BUCKET_SIZE;
      key_pos = global_idx & (BUCKET_SIZE - 1);
      int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
      sm_buckets_size_ptr[tx] = buckets_size + bkt_idx;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
      __pipeline_memcpy_async(sm_bucket_values_ptr + tx, &(bucket->vectors),
                              sizeof(VecV*));
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  uint32_t rank = g.thread_rank();
  uint32_t groupID = threadIdx.x / GROUP_SIZE;

  // Pipeline loading.
  auto occupy_result_next = g.shfl(occupy_result, 0);
  auto keys_ptr_next = g.shfl(bucket_keys_ptr, 0);
  if (occupy_result_next == OccupyResult::INITIAL) {
    D* sm_bucket_digests = SMM::bucket_digests(smem, groupID, 0);
    D* dst = sm_bucket_digests + rank * Load_LEN;
    D* src = BUCKET::digests(keys_ptr_next, BUCKET_SIZE, rank * Load_LEN);
    if (rank * Load_LEN < BUCKET_SIZE) {
      __pipeline_memcpy_async(dst, src, sizeof(VecD_Load));
    }
  }
  __pipeline_commit();
  // Padding, meet the param of the first `__pipeline_wait_prior`
  // in the first loop.
  __pipeline_commit();
  __pipeline_commit();
  for (int32_t i = 0; i < GROUP_SIZE; i++) {
    // Step1: load digests from global memory to shared memory.
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      auto keys_ptr_next = g.shfl(bucket_keys_ptr, i + 1);
      if (occupy_result_next == OccupyResult::INITIAL) {
        D* sm_bucket_digests = SMM::bucket_digests(smem, groupID, diff_buf(i));
        D* dst = sm_bucket_digests + rank * Load_LEN;
        D* src = BUCKET::digests(keys_ptr_next, BUCKET_SIZE, rank * Load_LEN);
        if (rank * Load_LEN < BUCKET_SIZE) {
          __pipeline_memcpy_async(dst, src, sizeof(VecD_Load));
        }
      }
    }
    __pipeline_commit();
    // Step2: to lock the target_key or empty_key by querying digests.
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur == OccupyResult::INITIAL) {
      uint32_t tx_cur = groupID * GROUP_SIZE + i;
      int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
      auto bucket_size_ptr = sm_buckets_size_ptr[tx_cur];
      K key_cur = g.shfl(key, i);
      auto target_digests_cur = g.shfl(target_digests, i);
      auto start_pos_cur = g.shfl(key_pos, i);
      auto keys_ptr_cur = g.shfl(bucket_keys_ptr, i);
      auto bucket_size_cur = bucket_size_ptr[0];
      __pipeline_wait_prior(3);
      D* src = SMM::bucket_digests(smem, groupID, same_buf(i));
      uint32_t start_offset = start_pos_cur / Comp_LEN;
      uint32_t probe_offset =
          Comp_LEN * ((start_offset + rank) & (GROUP_SIZE - 1));
      VecD_Comp probe_digests =
          *reinterpret_cast<VecD_Comp*>(src + probe_offset);
      uint32_t cmp_result = __vcmpeq4(probe_digests, target_digests_cur);
      cmp_result &= 0x01010101;
      uint32_t possible_pos = 0;
      bool result = false;
      do {
        if (cmp_result == 0) break;
        int32_t index = (__ffs(cmp_result) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = probe_offset + index;
        auto current_key = BUCKET::keys(keys_ptr_cur, possible_pos);
        K expected_key = key_cur;
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
      } while (!result);
      uint32_t found_vote = g.ballot(result);
      if (found_vote) {
        int32_t src_lane = __ffs(found_vote) - 1;
        possible_pos = g.shfl(possible_pos, src_lane);
        if (rank == i) {
          occupy_result = OccupyResult::DUPLICATE;
          S* sm_param_scores = SMM::param_scores(smem);
          key_pos = possible_pos;
          S score = ScoreFunctor::desired_when_missed(sm_param_scores, tx,
                                                      global_epoch);
          ScoreFunctor::update_with_digest(
              bucket_keys_ptr, key_pos, sm_param_scores, tx, score, BUCKET_SIZE,
              get_digest<K>(key), false);
        }
      } else if (bucket_size_cur < BUCKET_SIZE) {
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = __vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        // One more loop to deal with the empty key which was ignored.
        for (int32_t offset = 0; offset < GROUP_SIZE + 1; offset += 1) {
          if (rank == offset) {
            do {
              if (cmp_result == 0) break;
              int32_t index = (__ffs(cmp_result) - 1) >> 3;
              cmp_result &= (cmp_result - 1);
              possible_pos = probe_offset + index;
              if (offset == 0 && possible_pos < start_pos_cur) continue;
              auto current_key = BUCKET::keys(keys_ptr_cur, possible_pos);
              K expected_key = static_cast<K>(EMPTY_KEY);
              result = current_key->compare_exchange_strong(
                  expected_key, static_cast<K>(LOCKED_KEY),
                  cuda::std::memory_order_acquire,
                  cuda::std::memory_order_relaxed);
            } while (!result);
          }
          uint32_t found_vote = g.ballot(result);
          if (found_vote) {
            int32_t src_lane = __ffs(found_vote) - 1;
            possible_pos = g.shfl(possible_pos, src_lane);
            if (rank == i) {
              occupy_result = OccupyResult::OCCUPIED_EMPTY;
              S* sm_param_scores = SMM::param_scores(smem);
              S score = ScoreFunctor::desired_when_missed(sm_param_scores, tx,
                                                          global_epoch);
              int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
              int* bucket_size_ptr = sm_buckets_size_ptr[tx];
              key_pos = possible_pos;
              ScoreFunctor::update_with_digest(
                  bucket_keys_ptr, key_pos, sm_param_scores, tx, score,
                  BUCKET_SIZE, get_digest<K>(key), true);
              atomicAdd(bucket_size_ptr, 1);
            }
            break;
          }
        }
      }
      occupy_result_cur = g.shfl(occupy_result, i);
      if (occupy_result_cur == OccupyResult::INITIAL) {
        if (rank == i) {
          evict_idx = atomicAdd(evicted_counter, 1);
        }
        S* sm_bucket_scores = SMM::bucket_scores(smem, groupID, same_buf(i));
        S* dst = sm_bucket_scores + rank * Load_LEN_S;
        S* src = BUCKET::scores(keys_ptr_cur, BUCKET_SIZE, rank * Load_LEN_S);
#pragma unroll
        for (int32_t k = 0; k < BUCKET_SIZE; k += GROUP_SIZE * Load_LEN_S) {
          __pipeline_memcpy_async(dst + k, src + k, sizeof(S) * Load_LEN_S);
        }
      }
    }
    __pipeline_commit();
    // Step 3: reduce to get the key with the minimum score.
    if (i > 0) {
      occupy_result_cur = g.shfl(occupy_result, i - 1);
      uint32_t tx_cur = groupID * GROUP_SIZE + i - 1;
      S* sm_param_scores = SMM::param_scores(smem);
      S score_cur = ScoreFunctor::desired_when_missed(sm_param_scores, tx_cur,
                                                      global_epoch);
      int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
      auto bucket_size_ptr = sm_buckets_size_ptr[tx_cur];
      __pipeline_wait_prior(3);
      S* src = SMM::bucket_scores(smem, groupID, diff_buf(i));
      while (occupy_result_cur == OccupyResult::INITIAL) {
        int min_pos_local = -1;
        S min_score_local = MAX_SCORE;
#pragma unroll
        for (int j = 0; j < BUCKET_SIZE; j += GROUP_SIZE * Load_LEN_S) {
          S temp_scores[Load_LEN_S];
          *reinterpret_cast<byte16*>(temp_scores) =
              *reinterpret_cast<byte16*>(src + rank * Load_LEN_S + j);
#pragma unroll
          for (int k = 0; k < Load_LEN_S; k++) {
            S temp_score = temp_scores[k];
            if (temp_score < min_score_local) {
              min_score_local = temp_score;
              min_pos_local = rank * Load_LEN_S + j + k;
            }
          }
        }
        const S min_score_global =
            cg::reduce(g, min_score_local, cg::less<S>());
        if (score_cur < min_score_global) {
          if (rank == i - 1) {
            occupy_result = OccupyResult::REFUSED;
            evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx, key,
                                  score_cur);
          }
          occupy_result_cur = g.shfl(occupy_result, i - 1);
          break;
        }
        uint32_t vote = g.ballot(min_score_local <= min_score_global);
        if (vote) {
          int src_lane = __ffs(vote) - 1;
          int min_pos_global = g.shfl(min_pos_local, src_lane);
          if (rank == i - 1) {
            src[min_pos_global] = static_cast<S>(MAX_SCORE);  // Mark visited.
            auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos_global);
            auto expected_key =
                min_score_key->load(cuda::std::memory_order_relaxed);
            if (expected_key != static_cast<K>(LOCKED_KEY) &&
                expected_key != static_cast<K>(EMPTY_KEY)) {
              bool result = min_score_key->compare_exchange_strong(
                  expected_key, static_cast<K>(LOCKED_KEY),
                  cuda::std::memory_order_acquire,
                  cuda::std::memory_order_relaxed);
              if (result) {
                S* score_ptr = BUCKET::scores(bucket_keys_ptr, BUCKET_SIZE,
                                              min_pos_global);
                auto verify_score_ptr =
                    reinterpret_cast<AtomicScore<S>*>(score_ptr);
                auto verify_score =
                    verify_score_ptr->load(cuda::std::memory_order_relaxed);
                if (verify_score <= min_score_global) {
                  if (expected_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    atomicAdd(bucket_size_ptr, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                    evict_key_score<K, S>(evicted_keys, evicted_scores,
                                          evict_idx, expected_key,
                                          min_score_global);
                  }
                  key_pos = min_pos_global;
                  ScoreFunctor::update_with_digest(
                      bucket_keys_ptr, key_pos, sm_param_scores, tx_cur,
                      score_cur, BUCKET_SIZE, get_digest<K>(key), true);

                } else {
                  min_score_key->store(expected_key,
                                       cuda::std::memory_order_release);
                }
              }
            }
          }
          occupy_result_cur = g.shfl(occupy_result, i - 1);
        }
      }
      // Prefetch values to shared memory.
      if (occupy_result_cur != OccupyResult::ILLEGAL) {
        auto kv_idx_cur = g.shfl(kv_idx, i - 1);
        const VecV* src = values + kv_idx_cur * dim;
        VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(i), dim);
        CopyValue::ldg_sts(rank, dst, src, dim);

        if (occupy_result_cur == OccupyResult::EVICT) {
          VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
          auto bucket_values_ptr =
              sm_bucket_values_ptr[groupID * GROUP_SIZE + i - 1];
          auto key_pos_cur = g.shfl(key_pos, i - 1);
          const VecV* src = bucket_values_ptr + key_pos_cur * dim;
          dst = dst + dim;
          CopyValue::ldg_sts(rank, dst, src, dim);
        }
      }
    }
    __pipeline_commit();

    // Step 4: write values to bucket and evicted buffer.
    if (i > 1) {
      occupy_result_cur = g.shfl(occupy_result, i - 2);
      if (occupy_result_cur != OccupyResult::ILLEGAL) {
        VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
        auto bucket_values_ptr =
            sm_bucket_values_ptr[groupID * GROUP_SIZE + i - 2];
        auto key_pos_cur = g.shfl(key_pos, i - 2);
        auto evict_idx_cur = g.shfl(evict_idx, i - 2);

        VecV* src = SMM::values_buffer(smem, groupID, same_buf(i), dim);
        if (occupy_result_cur == OccupyResult::REFUSED) {
          VecV* dst = evicted_values + evict_idx_cur * dim;
          __pipeline_wait_prior(3);
          CopyValue::lds_stg(rank, dst, src, dim);
        } else {
          VecV* dst = bucket_values_ptr + key_pos_cur * dim;
          __pipeline_wait_prior(3);
          CopyValue::lds_stg(rank, dst, src, dim);
          if (rank == i - 2) {
            auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
            key_address->store(key, cuda::std::memory_order_release);
          }
          if (occupy_result_cur == OccupyResult::EVICT) {
            src = src + dim;
            VecV* dst = evicted_values + evict_idx_cur * dim;
            __pipeline_wait_prior(3);
            CopyValue::lds_stg(rank, dst, src, dim);
          }
        }
      }
    }
  }
  auto occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
  uint32_t tx_cur = groupID * GROUP_SIZE + GROUP_SIZE - 1;
  S* sm_param_scores = SMM::param_scores(smem);
  S score_cur =
      ScoreFunctor::desired_when_missed(sm_param_scores, tx_cur, global_epoch);
  int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
  auto bucket_size_ptr = sm_buckets_size_ptr[tx_cur];
  __pipeline_wait_prior(1);
  S* src = SMM::bucket_scores(smem, groupID, diff_buf(GROUP_SIZE));
  while (occupy_result_cur == OccupyResult::INITIAL) {
    int min_pos_local = -1;
    S min_score_local = MAX_SCORE;
#pragma unroll
    for (int j = 0; j < BUCKET_SIZE; j += GROUP_SIZE * Load_LEN_S) {
      S temp_scores[Load_LEN_S];
      *reinterpret_cast<byte16*>(temp_scores) =
          *reinterpret_cast<byte16*>(src + rank * Load_LEN_S + j);
#pragma unroll
      for (int k = 0; k < Load_LEN_S; k++) {
        S temp_score = temp_scores[k];
        if (temp_score < min_score_local) {
          min_score_local = temp_score;
          min_pos_local = rank * Load_LEN_S + j + k;
        }
      }
    }
    const S min_score_global = cg::reduce(g, min_score_local, cg::less<S>());
    if (score_cur < min_score_global) {
      if (rank == GROUP_SIZE - 1) {
        occupy_result = OccupyResult::REFUSED;
        evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx, key,
                              score_cur);
      }
      occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
      break;
    }
    uint32_t vote = g.ballot(min_score_local <= min_score_global);
    if (vote) {
      int src_lane = __ffs(vote) - 1;
      int min_pos_global = g.shfl(min_pos_local, src_lane);
      if (rank == GROUP_SIZE - 1) {
        src[min_pos_global] = MAX_SCORE;  // Mark visited.
        auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos_global);
        auto expected_key =
            min_score_key->load(cuda::std::memory_order_acquire);
        if (expected_key != static_cast<K>(LOCKED_KEY) &&
            expected_key != static_cast<K>(EMPTY_KEY)) {
          auto min_score_ptr =
              BUCKET::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos_global);
          bool result = min_score_key->compare_exchange_strong(
              expected_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
          if (result) {
            S* score_ptr =
                BUCKET::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos_global);
            auto verify_score_ptr =
                reinterpret_cast<AtomicScore<S>*>(score_ptr);
            auto verify_score =
                verify_score_ptr->load(cuda::std::memory_order_relaxed);
            if (verify_score <= min_score_global) {
              if (expected_key == static_cast<K>(RECLAIM_KEY)) {
                atomicAdd(bucket_size_ptr, 1);
                occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
              } else {
                occupy_result = OccupyResult::EVICT;
                evict_key_score<K, S>(evicted_keys, evicted_scores, evict_idx,
                                      expected_key, min_score_global);
              }
              key_pos = min_pos_global;
              ScoreFunctor::update_with_digest(
                  bucket_keys_ptr, key_pos, sm_param_scores, tx_cur, score_cur,
                  BUCKET_SIZE, get_digest<K>(key), true);
            } else {
              min_score_key->store(expected_key,
                                   cuda::std::memory_order_release);
            }
          }
        }
      }
      occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
    }
  }
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    auto kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 1);
    VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
    auto bucket_values_ptr =
        sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 1];
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);

    const VecV* src = values + kv_idx_cur * dim;
    VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(GROUP_SIZE), dim);
    CopyValue::ldg_sts(rank, dst, src, dim);

    if (occupy_result_cur == OccupyResult::EVICT) {
      const VecV* src = bucket_values_ptr + key_pos_cur * dim;
      dst = dst + dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    }
  }
  __pipeline_commit();

  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 2);
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
    auto bucket_values_ptr =
        sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 2];
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 2);
    auto evict_idx_cur = g.shfl(evict_idx, GROUP_SIZE - 2);

    VecV* src = SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE), dim);
    if (occupy_result_cur == OccupyResult::REFUSED) {
      VecV* dst = evicted_values + evict_idx_cur * dim;
      __pipeline_wait_prior(1);
      CopyValue::lds_stg(rank, dst, src, dim);
    } else {
      VecV* dst = bucket_values_ptr + key_pos_cur * dim;
      __pipeline_wait_prior(1);
      CopyValue::lds_stg(rank, dst, src, dim);
      if (rank == GROUP_SIZE - 2) {
        auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
      if (occupy_result_cur == OccupyResult::EVICT) {
        src = src + dim;
        VecV* dst = evicted_values + evict_idx_cur * dim;
        __pipeline_wait_prior(1);
        CopyValue::lds_stg(rank, dst, src, dim);
      }
    }
  }

  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
    auto bucket_values_ptr =
        sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 1];
    auto evict_idx_cur = g.shfl(evict_idx, GROUP_SIZE - 1);
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);

    VecV* src =
        SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE + 1), dim);
    if (occupy_result_cur == OccupyResult::REFUSED) {
      VecV* dst = evicted_values + evict_idx_cur * dim;
      __pipeline_wait_prior(0);
      CopyValue::lds_stg(rank, dst, src, dim);
    } else {
      VecV* dst = bucket_values_ptr + key_pos_cur * dim;
      __pipeline_wait_prior(0);
      CopyValue::lds_stg(rank, dst, src, dim);
      if (rank == GROUP_SIZE - 1) {
        auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
      if (occupy_result_cur == OccupyResult::EVICT) {
        src = src + dim;
        VecV* dst = evicted_values + evict_idx_cur * dim;
        __pipeline_wait_prior(0);
        CopyValue::lds_stg(rank, dst, src, dim);
      }
    }
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
struct Params_UpsertAndEvict {
  Params_UpsertAndEvict(
      float load_factor_, Bucket<K, V, S>* __restrict__ buckets_,
      int* buckets_size_, size_t buckets_num_, uint32_t bucket_capacity_,
      uint32_t dim_, const K* __restrict__ keys_, const V* __restrict__ values_,
      const S* __restrict__ scores_, K* __restrict__ evicted_keys_,
      V* __restrict__ evicted_values_, S* __restrict__ evicted_scores_,
      size_t n_, size_t* evicted_counter_, const S global_epoch_)
      : load_factor(load_factor_),
        buckets(buckets_),
        buckets_size(buckets_size_),
        buckets_num(buckets_num_),
        bucket_capacity(bucket_capacity_),
        dim(dim_),
        keys(keys_),
        values(values_),
        scores(scores_),
        evicted_keys(evicted_keys_),
        evicted_values(evicted_values_),
        evicted_scores(evicted_scores_),
        n(n_),
        evicted_counter(evicted_counter_),
        global_epoch(global_epoch_) {}
  float load_factor;
  Bucket<K, V, S>* __restrict__ buckets;
  int* buckets_size;
  size_t buckets_num;
  uint32_t bucket_capacity;
  uint32_t dim;
  const K* __restrict__ keys;
  const V* __restrict__ values;
  const S* __restrict__ scores;
  K* __restrict__ evicted_keys;
  V* __restrict__ evicted_values;
  S* __restrict__ evicted_scores;
  uint64_t n;
  uint64_t* evicted_counter;
  const S global_epoch;
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_TLPv1_UpsertAndEvict {
  using Params = Params_UpsertAndEvict<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    tlp_v1_upsert_and_evict_kernel_unique<K, V, S, VecV, BLOCK_SIZE, Strategy>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            params.buckets, params.buckets_size, params.buckets_num,
            params.bucket_capacity, params.dim, params.keys,
            reinterpret_cast<const VecV*>(params.values), params.scores,
            params.evicted_keys, reinterpret_cast<VecV*>(params.evicted_values),
            params.evicted_scores, params.n, params.evicted_counter,
            params.global_epoch);
  }
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_TLPv2_UpsertAndEvict {
  using Params = Params_UpsertAndEvict<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    if (params.dim <= 8) {
      constexpr int GROUP_SIZE = 8;
      tlp_v2_upsert_and_evict_kernel_unique<K, V, S, VecV, BLOCK_SIZE,
                                            GROUP_SIZE, Strategy>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_size, params.buckets_num,
              params.bucket_capacity, params.dim, params.keys,
              reinterpret_cast<const VecV*>(params.values), params.scores,
              params.evicted_keys,
              reinterpret_cast<VecV*>(params.evicted_values),
              params.evicted_scores, params.n, params.evicted_counter,
              params.global_epoch);
    } else {
      constexpr int GROUP_SIZE = 16;
      tlp_v2_upsert_and_evict_kernel_unique<K, V, S, VecV, BLOCK_SIZE,
                                            GROUP_SIZE, Strategy>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_size, params.buckets_num,
              params.bucket_capacity, params.dim, params.keys,
              reinterpret_cast<const VecV*>(params.values), params.scores,
              params.evicted_keys,
              reinterpret_cast<VecV*>(params.evicted_values),
              params.evicted_scores, params.n, params.evicted_counter,
              params.global_epoch);
    }
  }
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_Pipeline_UpsertAndEvict {
  using Params = Params_UpsertAndEvict<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr uint32_t GROUP_SIZE = 32;
    constexpr uint32_t BUCKET_SIZE = 128;
    using SMM =
        SharedMemoryManager_Pipeline_UpsertAndEvict<K, V, S, VecV, BLOCK_SIZE,
                                                    GROUP_SIZE, BUCKET_SIZE>;

    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    uint32_t shared_mem = SMM::total_size(params.dim);
    shared_mem =
        (shared_mem + sizeof(byte16) - 1) / sizeof(byte16) * sizeof(byte16);
    pipeline_upsert_and_evict_kernel_unique<K, V, S, VecV, BLOCK_SIZE, Strategy>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_mem,
           stream>>>(params.buckets, params.buckets_size, params.buckets_num,
                     params.dim, params.keys,
                     reinterpret_cast<const VecV*>(params.values),
                     params.scores, params.evicted_keys,
                     reinterpret_cast<VecV*>(params.evicted_values),
                     params.evicted_scores, params.n, params.evicted_counter,
                     params.global_epoch);
  }
};

template <typename ArchTag>
struct ValueConfig_UpsertAndEvict;

/// TODO: support more arch
template <>
struct ValueConfig_UpsertAndEvict<Sm80> {
  // Value size greater than it will bring poor performance for TLPv1.
  static constexpr uint32_t size_tlp_v1 = 16 * sizeof(byte4);
  // Value size greater than it will bring wrong result for TLPv2.
  static constexpr uint32_t size_tlp_v2 = 64 * sizeof(byte4);
  // Value size greater than it will reduce the occupancy for Pipeline.
  // When the value is very high, the kernel will fail to launch.
  static constexpr uint32_t size_pipeline = 128 * sizeof(byte4);
};

template <typename K, typename V, typename S, int Strategy, typename ArchTag>
struct KernelSelector_UpsertAndEvict;

template <typename K, typename V, typename S, int Strategy>
struct KernelSelector_UpsertAndEvict<K, V, S, Strategy, Sm80> {
  using ArchTag = Sm80;
  using ValueConfig = ValueConfig_UpsertAndEvict<ArchTag>;
  using Params = Params_UpsertAndEvict<K, V, S>;

  static bool callable(bool unique_key, uint32_t bucket_size, uint32_t dim) {
    return true;
    if (!unique_key) return false;
    uint32_t value_size = dim * sizeof(V);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
    if (value_size <= ValueConfig::size_tlp_v2) return true;
#else
    if (value_size <= ValueConfig::size_tlp_v1) return true;
#endif
    if (bucket_size == 128 && value_size <= ValueConfig::size_pipeline) {
      return true;
    }
    return false;
  }

  static void select_kernel(Params& params, cudaStream_t& stream) {
    const uint32_t total_value_size =
        static_cast<uint32_t>(params.dim * sizeof(V));

    auto launch_TLPv1 = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_TLPv1_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_TLPv1_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_TLPv1_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_TLPv1_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_TLPv1_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      }
    };

    auto launch_TLPv2 = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_TLPv2_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_TLPv2_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_TLPv2_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_TLPv2_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_TLPv2_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      }
    };

    auto launch_Pipeline = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_Pipeline_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_Pipeline_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_Pipeline_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_Pipeline_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_Pipeline_UpsertAndEvict<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      }
    };

    // This part is according to the test on A100.
    if (params.bucket_capacity != 128) {
      if (total_value_size <= ValueConfig::size_tlp_v1) {
        launch_TLPv1();
      } else {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
        launch_TLPv2();
#else
        launch_TLPv1();
#endif
      }
    } else {
      if (total_value_size <= ValueConfig::size_tlp_v1) {
        if (params.load_factor <= 0.90f) {
          launch_TLPv1();
        } else {
          launch_Pipeline();
        }
      } else if (total_value_size <= ValueConfig::size_tlp_v2) {
        if (params.load_factor <= 0.85f) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
          launch_TLPv2();
#else
          launch_Pipeline();
#endif
        } else {
          launch_Pipeline();
        }
      } else {
        launch_Pipeline();
      }
    }
  }  // End function
};

template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
    const K* __restrict keys, const V* __restrict values,
    const S* __restrict scores, K* __restrict evicted_keys,
    V* __restrict evicted_values, S* __restrict evicted_scores,
    const S global_epoch, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const S insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);
    const V* insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, insert_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      }
      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      if (g.thread_rank() == 0) {
        evicted_keys[key_idx] = insert_key;
        evicted_scores[key_idx] = insert_score;
      }
      copy_vector<V, TILE_SIZE>(g, insert_value, evicted_values + key_idx * dim,
                                dim);
      continue;
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == src_lane) {
        evicted_keys[key_idx] = evicted_key;
        if (scores != nullptr) {
          evicted_scores[key_idx] = scores[key_idx];
        }
      }
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                evicted_values + key_idx * dim, dim);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      ScoreFunctor::update(bucket, key_pos, scores, key_idx, insert_score,
                           (occupy_result != OccupyResult::DUPLICATE));
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
    }
  }
}

template <typename K, typename V, typename S, int Strategy>
struct SelectUpsertAndEvictKernelWithIO {
  static void execute_kernel(
      const float& load_factor, const int& block_size,
      const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
      cudaStream_t& stream, const size_t& n,
      const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
      const K* __restrict keys, const V* __restrict values,
      const S* __restrict scores, K* __restrict evicted_keys,
      V* __restrict evicted_values, S* __restrict evicted_scores,
      const S global_epoch) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, evicted_keys, evicted_values, evicted_scores,
              global_epoch, N);

    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      upsert_and_evict_kernel_with_io_core<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, evicted_keys, evicted_values, evicted_scores,
              global_epoch, N);

    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, evicted_keys, evicted_values, evicted_scores,
              global_epoch, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv