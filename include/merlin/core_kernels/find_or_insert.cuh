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
__global__ void tlp_v1_find_or_insert_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, VecV* __restrict__ values,
    S* __restrict__ scores, uint64_t n, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, 1>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

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
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
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
            min_score = temp_score;
            min_pos = i + k + j;
          }
        }
      }
    }
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
    if (score < min_score) {
      occupy_result = OccupyResult::REFUSED;
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
          }

        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }
  VecV* bucket_value_ptr = bucket_values_ptr + key_pos * dim;
  VecV* param_value_ptr = values + kv_idx * dim;

  if (occupy_result != OccupyResult::REFUSED) {
    if (occupy_result == OccupyResult::DUPLICATE) {
      CopyValue::ldg_stg(0, param_value_ptr, bucket_value_ptr, dim);
    } else {
      CopyValue::ldg_stg(0, bucket_value_ptr, param_value_ptr, dim);
    }
    auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
    // memory_order_release:
    // Modifications to the bucket will not after this instruction.
    key_address->store(key, cuda::std::memory_order_release);
  }
}

// Use 1 thread to deal with a KV-pair, including copying value.
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128,
          uint32_t GROUP_SIZE = 16, int Strategy = -1>
__global__ void tlp_v2_find_or_insert_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, VecV* __restrict__ values,
    S* __restrict__ scores, uint64_t n, const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

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
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
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
          }
        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }

  VecV* bucket_value_ptr{nullptr};
  if ((occupy_result != OccupyResult::ILLEGAL) &&
      (occupy_result != OccupyResult::REFUSED)) {
    bucket_value_ptr = bucket_values_ptr + key_pos * dim;
  }
  __syncthreads();
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
  if ((occupy_result_next != OccupyResult::ILLEGAL) &&
      (occupy_result_next != OccupyResult::REFUSED)) {
    VecV* dst = sm_values_buffer;
    if (occupy_result_next == OccupyResult::DUPLICATE) {
      const VecV* src = g.shfl(bucket_value_ptr, 0);
      CopyValue::ldg_sts(rank, dst, src, dim);
    } else {
      auto kv_idx_next = g.shfl(kv_idx, 0);
      const VecV* src = values + kv_idx_next * dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    }
  }
  __pipeline_commit();

  for (int i = 0; i < GROUP_SIZE; i++) {
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      if ((occupy_result_next != OccupyResult::ILLEGAL) &&
          (occupy_result_next != OccupyResult::REFUSED)) {
        VecV* dst = sm_values_buffer + diff_buf(i) * GROUP_BUF;
        if (occupy_result_next == OccupyResult::DUPLICATE) {
          const VecV* src = g.shfl(bucket_value_ptr, i + 1);
          CopyValue::ldg_sts(rank, dst, src, dim);
        } else {
          auto kv_idx_next = g.shfl(kv_idx, i + 1);
          const VecV* src = values + kv_idx_next * dim;
          CopyValue::ldg_sts(rank, dst, src, dim);
        }
      }
    }
    __pipeline_commit();
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if ((occupy_result_cur != OccupyResult::ILLEGAL) &&
        (occupy_result_cur != OccupyResult::REFUSED)) {
      VecV* src = sm_values_buffer + same_buf(i) * GROUP_BUF;
      __pipeline_wait_prior(0);
      if (occupy_result_cur == OccupyResult::DUPLICATE) {
        auto kv_idx_cur = g.shfl(kv_idx, i);
        VecV* dst = values + kv_idx_cur * dim;
        __pipeline_wait_prior(1);
        CopyValue::lds_stg(rank, dst, src, dim);
      } else {
        VecV* dst = g.shfl(bucket_value_ptr, i);
        __pipeline_wait_prior(1);
        CopyValue::lds_stg(rank, dst, src, dim);
      }
    }
  }

  if ((occupy_result != OccupyResult::ILLEGAL) &&
      (occupy_result != OccupyResult::REFUSED)) {
    auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
    // memory_order_release:
    // Modifications to the bucket will not after this instruction.
    key_address->store(key, cuda::std::memory_order_release);
  }
}

template <
    typename K, typename V, typename S, typename VecV, uint32_t BLOCK_SIZE,
    uint32_t GROUP_SIZE, uint32_t BUCKET_SIZE,
    uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE, uint32_t OFST_ParamScores = 0,
    uint32_t OFST_BucketValuesPtr = OFST_ParamScores + sizeof(S) * BLOCK_SIZE,
    uint32_t OFST_BucketsSizePtr =
        OFST_BucketValuesPtr + sizeof(VecV*) * BLOCK_SIZE,
    uint32_t OFST_BucketDigests =
        OFST_BucketsSizePtr + sizeof(int*) * BLOCK_SIZE,
    uint32_t OFST_BucketScores =
        OFST_BucketDigests + sizeof(D) * GROUP_NUM * 2 * BUCKET_SIZE,
    uint32_t OFST_BucketValues =
        OFST_BucketScores + sizeof(S) * GROUP_NUM * 2 * BUCKET_SIZE>
struct SharedMemoryManager_Pipeline_FindOrInsert {
  /*
    __shared__ S sm_param_scores[BLOCK_SIZE];
    __shared__ VecV* sm_bucket_values_ptr[BLOCK_SIZE];
    __shared__ int* sm_buckets_size_ptr[BLOCK_SIZE];
    __shared__ D sm_bucket_digests[GROUP_NUM][2][BUCKET_SIZE];
    __shared__ S sm_bucket_scores[GROUP_NUM][2][BUCKET_SIZE];
    __shared__ VecV sm_values_buffer[GROUP_NUM][2][dim];
  */
  static inline uint32_t total_size(uint32_t dim) {
    return BLOCK_SIZE * (sizeof(S) + sizeof(VecV*) + sizeof(int*)) +
           GROUP_NUM * 2 *
               (BUCKET_SIZE * (sizeof(D) + sizeof(S)) + dim * sizeof(VecV));
  }
  static __forceinline__ __device__ S* param_scores(byte* smem) {
    return reinterpret_cast<S*>(smem + OFST_ParamScores);
  }
  static __forceinline__ __device__ VecV** bucket_values_ptr(byte* smem) {
    return reinterpret_cast<VecV**>(smem + OFST_BucketValuesPtr);
  }
  static __forceinline__ __device__ int** buckets_size_ptr(byte* smem) {
    return reinterpret_cast<int**>(smem + OFST_BucketsSizePtr);
  }
  static __forceinline__ __device__ D* bucket_digests(byte* smem,
                                                      uint32_t groupID,
                                                      uint32_t buf) {
    return reinterpret_cast<D*>(smem + OFST_BucketDigests) +
           BUCKET_SIZE * (groupID * 2 + buf);
  }
  static __forceinline__ __device__ S* bucket_scores(byte* smem,
                                                     uint32_t groupID,
                                                     uint32_t buf) {
    return reinterpret_cast<S*>(smem + OFST_BucketScores) +
           BUCKET_SIZE * (groupID * 2 + buf);
  }
  static __forceinline__ __device__ VecV* values_buffer(byte* smem,
                                                        uint32_t groupID,
                                                        uint32_t buf,
                                                        uint32_t dim) {
    return reinterpret_cast<VecV*>(smem + OFST_BucketValues) +
           dim * (groupID * 2 + buf);
  }
};

template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128, int Strategy = -1>
__global__ void pipeline_find_or_insert_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, const uint32_t dim, const K* __restrict__ keys,
    VecV* __restrict__ values, S* __restrict__ scores, uint64_t n,
    const S global_epoch) {
  // Here, GROUP_SIZE * Comp_LEN = BUCKET_SIZE.
  constexpr uint32_t BUCKET_SIZE = 128;
  constexpr uint32_t GROUP_SIZE = 32;
  constexpr uint32_t Comp_LEN = sizeof(VecD_Comp) / sizeof(D);
  constexpr uint32_t Load_LEN = sizeof(VecD_Load) / sizeof(D);
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);

  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
  using SMM =
      SharedMemoryManager_Pipeline_FindOrInsert<K, V, S, VecV, BLOCK_SIZE,
                                                GROUP_SIZE, BUCKET_SIZE>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __shared__ extern __align__(alignof(byte16)) byte smem[];

  // Initialization.
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  VecD_Comp target_digests;
  K* bucket_keys_ptr{nullptr};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  uint32_t key_pos = 0;
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (scores != nullptr) {
      S* sm_param_scores = SMM::param_scores(smem);
      __pipeline_memcpy_async(sm_param_scores + tx, scores + kv_idx, sizeof(S));
    }
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * BUCKET_SIZE));
      uint64_t bkt_idx = global_idx / BUCKET_SIZE;
      key_pos = get_start_position(global_idx, BUCKET_SIZE);
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
      auto bucket_size_cur = *bucket_size_ptr;
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
          key_pos = possible_pos;
          S* sm_param_scores = SMM::param_scores(smem);
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
        for (int32_t offset = 0; offset < GROUP_SIZE; offset += 1) {
          if (rank == offset) {
            do {
              if (cmp_result == 0) break;
              int32_t index = (__ffs(cmp_result) - 1) >> 3;
              cmp_result &= (cmp_result - 1);
              possible_pos = probe_offset + index;
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
        S min_score_local = static_cast<S>(MAX_SCORE);
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
      if (occupy_result_cur != OccupyResult::ILLEGAL &&
          occupy_result_cur != OccupyResult::REFUSED) {
        VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(i), dim);
        if (occupy_result_cur == OccupyResult::DUPLICATE) {
          VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
          auto bucket_values_ptr =
              sm_bucket_values_ptr[groupID * GROUP_SIZE + i - 1];
          auto key_pos_cur = g.shfl(key_pos, i - 1);
          const VecV* src = bucket_values_ptr + key_pos_cur * dim;
          CopyValue::ldg_sts(rank, dst, src, dim);
        } else {
          auto kv_idx_cur = g.shfl(kv_idx, i - 1);
          const VecV* src = values + kv_idx_cur * dim;
          CopyValue::ldg_sts(rank, dst, src, dim);
        }
      }
    }
    __pipeline_commit();

    // Step 4: write values to bucket or param buffer.
    if (i > 1) {
      occupy_result_cur = g.shfl(occupy_result, i - 2);
      if (occupy_result_cur != OccupyResult::ILLEGAL &&
          occupy_result_cur != OccupyResult::REFUSED) {
        VecV* src = SMM::values_buffer(smem, groupID, same_buf(i), dim);
        if (occupy_result_cur == OccupyResult::DUPLICATE) {
          uint32_t kv_idx_cur = g.shfl(kv_idx, i - 2);
          VecV* dst = values + kv_idx_cur * dim;
          __pipeline_wait_prior(3);
          CopyValue::lds_stg(rank, dst, src, dim);
        } else {
          VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
          auto bucket_values_ptr =
              sm_bucket_values_ptr[groupID * GROUP_SIZE + i - 2];
          auto key_pos_cur = g.shfl(key_pos, i - 2);
          VecV* dst = bucket_values_ptr + key_pos_cur * dim;
          __pipeline_wait_prior(3);
          CopyValue::lds_stg(rank, dst, src, dim);
        }
        if (rank == i - 2) {
          auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
          key_address->store(key, cuda::std::memory_order_release);
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
      }
      occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
      break;
    }
    uint32_t vote = g.ballot(min_score_local <= min_score_global);
    if (vote) {
      int src_lane = __ffs(vote) - 1;
      int min_pos_global = g.shfl(min_pos_local, src_lane);
      if (rank == GROUP_SIZE - 1) {
        src[min_pos_global] = static_cast<S>(MAX_SCORE);  // Mark visited.
        auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos_global);
        auto expected_key =
            min_score_key->load(cuda::std::memory_order_relaxed);
        if (expected_key != static_cast<K>(LOCKED_KEY) &&
            expected_key != static_cast<K>(EMPTY_KEY)) {
          auto min_score_ptr =
              BUCKET::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos_global);
          bool result = min_score_key->compare_exchange_strong(
              expected_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
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
  // Prefetch values to shared memory.
  if (occupy_result_cur != OccupyResult::ILLEGAL &&
      occupy_result_cur != OccupyResult::REFUSED) {
    VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(GROUP_SIZE), dim);
    if (occupy_result_cur == OccupyResult::DUPLICATE) {
      VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
      auto bucket_values_ptr =
          sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 1];
      auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);
      const VecV* src = bucket_values_ptr + key_pos_cur * dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    } else {
      auto kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 1);
      const VecV* src = values + kv_idx_cur * dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    }
  }
  __pipeline_commit();

  // Step 4: write values to bucket or param buffer.
  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 2);
  if (occupy_result_cur != OccupyResult::ILLEGAL &&
      occupy_result_cur != OccupyResult::REFUSED) {
    VecV* src = SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE), dim);
    if (occupy_result_cur == OccupyResult::DUPLICATE) {
      uint32_t kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 2);
      VecV* dst = values + kv_idx_cur * dim;
      __pipeline_wait_prior(1);
      CopyValue::lds_stg(rank, dst, src, dim);
    } else {
      VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
      auto bucket_values_ptr =
          sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 2];
      auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 2);
      VecV* dst = bucket_values_ptr + key_pos_cur * dim;
      __pipeline_wait_prior(1);
      CopyValue::lds_stg(rank, dst, src, dim);
    }
    if (rank == GROUP_SIZE - 2) {
      auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
      key_address->store(key, cuda::std::memory_order_release);
    }
  }

  // Step 4: write values to bucket or param buffer.
  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
  if (occupy_result_cur != OccupyResult::ILLEGAL &&
      occupy_result_cur != OccupyResult::REFUSED) {
    VecV* src =
        SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE + 1), dim);
    if (occupy_result_cur == OccupyResult::DUPLICATE) {
      uint32_t kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 1);
      VecV* dst = values + kv_idx_cur * dim;
      __pipeline_wait_prior(0);
      CopyValue::lds_stg(rank, dst, src, dim);
    } else {
      VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
      auto bucket_values_ptr =
          sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 1];
      auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);
      VecV* dst = bucket_values_ptr + key_pos_cur * dim;
      __pipeline_wait_prior(0);
      CopyValue::lds_stg(rank, dst, src, dim);
    }
    if (rank == GROUP_SIZE - 1) {
      auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
      key_address->store(key, cuda::std::memory_order_release);
    }
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
struct Params_FindOrInsert {
  Params_FindOrInsert(float load_factor_,
                      Bucket<K, V, S>* __restrict__ buckets_,
                      int* buckets_size_, size_t buckets_num_,
                      uint32_t bucket_capacity_, uint32_t dim_,
                      const K* __restrict__ keys_, V* __restrict__ values_,
                      S* __restrict__ scores_, size_t n_, const S global_epoch_)
      : load_factor(load_factor_),
        buckets(buckets_),
        buckets_size(buckets_size_),
        buckets_num(buckets_num_),
        bucket_capacity(bucket_capacity_),
        dim(dim_),
        keys(keys_),
        values(values_),
        scores(scores_),
        n(n_),
        global_epoch(global_epoch_) {}
  float load_factor;
  Bucket<K, V, S>* __restrict__ buckets;
  int* buckets_size;
  size_t buckets_num;
  uint32_t bucket_capacity;
  uint32_t dim;
  const K* __restrict__ keys;
  V* __restrict__ values;
  S* __restrict__ scores;
  uint64_t n;
  const S global_epoch;
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_TLPv1_FindOrInsert {
  using Params = Params_FindOrInsert<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    tlp_v1_find_or_insert_kernel_with_io<K, V, S, VecV, BLOCK_SIZE, Strategy>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            params.buckets, params.buckets_size, params.buckets_num,
            params.bucket_capacity, params.dim, params.keys,
            reinterpret_cast<VecV*>(params.values), params.scores, params.n,
            params.global_epoch);
  }
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_TLPv2_FindOrInsert {
  using Params = Params_FindOrInsert<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    const uint32_t value_size = params.dim * sizeof(V);
    params.dim = value_size / sizeof(VecV);

    if (value_size <= 256) {
      constexpr int GROUP_SIZE = 8;
      tlp_v2_find_or_insert_kernel_with_io<K, V, S, VecV, BLOCK_SIZE,
                                           GROUP_SIZE, Strategy>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_size, params.buckets_num,
              params.bucket_capacity, params.dim, params.keys,
              reinterpret_cast<VecV*>(params.values), params.scores, params.n,
              params.global_epoch);
    } else {
      constexpr int GROUP_SIZE = 16;
      tlp_v2_find_or_insert_kernel_with_io<K, V, S, VecV, BLOCK_SIZE,
                                           GROUP_SIZE, Strategy>
          <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              params.buckets, params.buckets_size, params.buckets_num,
              params.bucket_capacity, params.dim, params.keys,
              reinterpret_cast<VecV*>(params.values), params.scores, params.n,
              params.global_epoch);
    }
  }
};

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_Pipeline_FindOrInsert {
  using Params = Params_FindOrInsert<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr uint32_t GROUP_SIZE = 32;
    constexpr uint32_t BUCKET_SIZE = 128;
    using SMM =
        SharedMemoryManager_Pipeline_FindOrInsert<K, V, S, VecV, BLOCK_SIZE,
                                                  GROUP_SIZE, BUCKET_SIZE>;

    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    uint32_t shared_mem = SMM::total_size(params.dim);
    shared_mem =
        (shared_mem + sizeof(byte16) - 1) / sizeof(byte16) * sizeof(byte16);
    pipeline_find_or_insert_kernel_with_io<K, V, S, VecV, BLOCK_SIZE, Strategy>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_mem,
           stream>>>(params.buckets, params.buckets_size, params.buckets_num,
                     params.dim, params.keys,
                     reinterpret_cast<VecV*>(params.values), params.scores,
                     params.n, params.global_epoch);
  }
};

template <typename ArchTag>
struct ValueConfig_FindOrInsert;

template <>
struct ValueConfig_FindOrInsert<Sm80> {
  // Value size greater than it will bring poor performance for TLPv1.
  static constexpr uint32_t size_tlp_v1 = 16 * sizeof(byte4);
  static constexpr uint32_t size_tlp_v2 = 128 * sizeof(byte4);
};

template <>
struct ValueConfig_FindOrInsert<Sm70> {
  // Value size greater than it will bring poor performance for TLPv1.
  static constexpr uint32_t size_tlp_v1 = 16 * sizeof(byte4);
  static constexpr uint32_t size_tlp_v2 = 128 * sizeof(byte4);
};

template <typename K, typename V, typename S, int Strategy, typename ArchTag>
struct KernelSelector_FindOrInsert {
  using ValueConfig = ValueConfig_FindOrInsert<ArchTag>;
  using Params = Params_FindOrInsert<K, V, S>;

  static bool callable(bool unique_key, uint32_t bucket_size, uint32_t dim) {
    constexpr uint32_t MinBucketCap = sizeof(VecD_Load) / sizeof(D);
    if (!unique_key || bucket_size < MinBucketCap) return false;
    uint32_t value_size = dim * sizeof(V);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
    if (value_size <= ValueConfig::size_tlp_v2) return true;
#else
    if (value_size <= ValueConfig::size_tlp_v1) return true;
#endif
    return false;
  }

  static void select_kernel(Params& params, cudaStream_t& stream) {
    const uint32_t total_value_size =
        static_cast<uint32_t>(params.dim * sizeof(V));

    auto launch_TLPv1 = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_TLPv1_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_TLPv1_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_TLPv1_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_TLPv1_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_TLPv1_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      }
    };

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
    auto launch_TLPv2 = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_TLPv2_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_TLPv2_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_TLPv2_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_TLPv2_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_TLPv2_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      }
    };
#endif

    auto launch_Pipeline = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_Pipeline_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_Pipeline_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_Pipeline_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_Pipeline_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
            params, stream);
      } else {
        using VecV = byte;
        Launch_Pipeline_FindOrInsert<K, V, S, VecV, Strategy>::launch_kernel(
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
        if (params.load_factor <= 0.98f) {
          launch_TLPv1();
        } else {
          launch_Pipeline();
        }
      } else {
        if (params.load_factor <= 0.95f) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
          launch_TLPv2();
#else
          launch_Pipeline();
#endif
        } else {
          launch_Pipeline();
        }
      }
    }
  }  // End function
};

/*
 * find or insert with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel_with_io(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
    const S global_epoch, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY<K>(find_or_insert_key)) continue;

    const S find_or_insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);
    V* find_or_insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(buckets, find_or_insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                find_or_insert_value, dim);
    } else {
      copy_vector<V, TILE_SIZE>(g, find_or_insert_value,
                                bucket->vectors + key_pos * dim, dim);
    }
    if (g.thread_rank() == src_lane) {
      ScoreFunctor::update(bucket, key_pos, scores, key_idx,
                           find_or_insert_score,
                           (occupy_result != OccupyResult::DUPLICATE));
    }

    if (g.thread_rank() == src_lane) {
      bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
    }
  }
}

template <typename K, typename V, typename S, int Strategy>
struct SelectFindOrInsertKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, const K* __restrict keys,
                             V* __restrict values, S* __restrict scores,
                             const S global_epoch) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, global_epoch, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, S, Strategy, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, buckets, bucket_max_size, buckets_num, dim, keys, values,
              scores, global_epoch, N);
    }
    return;
  }
};

// Use 1 thread to deal with a KV-pair.
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          uint32_t BLOCK_SIZE = 128, int Strategy = -1>
__global__ void find_or_insert_kernel_lock_key_hybrid(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, uint32_t bucket_capacity, const uint32_t dim,
    const K* __restrict__ keys, V** __restrict__ value_ptrs,
    S* __restrict__ scores, K** __restrict__ key_ptrs,
    int* __restrict keys_index, bool* __restrict__ founds, uint64_t n,
    const S global_epoch) {
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // bucket_capacity is a multiple of 4.
  constexpr uint32_t STRIDE_S = 4;
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);
  __shared__ S sm_bucket_scores[BLOCK_SIZE][2 * STRIDE_S];

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  V* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  int32_t* bucket_size_ptr{nullptr};
  uint32_t key_pos = {0};
  uint32_t bucket_size{0};
  if (kv_idx < n) {
    key = keys[kv_idx];

    // help to address the original key after sorting value pointers.
    if (keys_index) {
      keys_index[kv_idx] = kv_idx;
    }

    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      bucket_size_ptr = buckets_size + bkt_idx;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_size = *bucket_size_ptr;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = reinterpret_cast<V*>(bucket->vectors);
    } else {
      key_ptrs[kv_idx] = nullptr;
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
    if (score <= min_score) {
      occupy_result = OccupyResult::REFUSED;
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
          }
        } else {
          min_score_key->store(expected_key, cuda::std::memory_order_release);
        }
      }
    }
  }

  if (kv_idx < n) {
    if (occupy_result == OccupyResult::REFUSED) {
      value_ptrs[kv_idx] = nullptr;
      key_ptrs[kv_idx] = nullptr;
    } else {
      value_ptrs[kv_idx] = bucket_values_ptr + key_pos * dim;
      founds[kv_idx] = occupy_result == OccupyResult::DUPLICATE;
      auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
      key_ptrs[kv_idx] = reinterpret_cast<K*>(key_address);
    }
  }
}

template <class K, class V, class S, class VecV = byte16>
__global__ void read_or_write_kernel_unlock_key(
    VecV** __restrict table_value_addrs, VecV* __restrict param_values,
    const bool* mask, const int* __restrict param_key_index,
    K** __restrict__ key_ptrs, const K* __restrict__ keys, const size_t dim,
    const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_key_index =
        param_key_index != nullptr ? param_key_index[vec_index] : vec_index;

    K* key_ptr = key_ptrs[real_key_index];
    K key = keys[real_key_index];

    /// if found, read the value form table, otherwise write it
    if (table_value_addrs[vec_index] != nullptr) {
      // unlock the key.
      if (key_ptr && dim_index == 0) *key_ptr = key;

      /// find
      if (mask[real_key_index]) {
        param_values[real_key_index * dim + dim_index] =
            table_value_addrs[vec_index][dim_index];
      }
      /// insert
      else {
        table_value_addrs[vec_index][dim_index] =
            param_values[real_key_index * dim + dim_index];
      }
    }
  }
}

/* find or insert with the end-user specified score.
 */
template <class K, class V, class S, int Strategy, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
    const K* __restrict keys, V** __restrict vectors, S* __restrict scores,
    bool* __restrict found, int* __restrict keys_index, const S global_epoch,
    const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY<K>(find_or_insert_key)) continue;

    const S find_or_insert_score =
        ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(buckets, find_or_insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    if (g.thread_rank() == 0) {
      *(keys_index + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE,
                                                ScoreFunctor::LOCK_MEM_ORDER,
                                                ScoreFunctor::UNLOCK_MEM_ORDER>(
            g, bucket, find_or_insert_key, find_or_insert_score, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (g.thread_rank() == src_lane) {
      *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
      ScoreFunctor::update(bucket, key_pos, scores, key_idx,
                           find_or_insert_score,
                           occupy_result != OccupyResult::DUPLICATE);
      if (occupy_result == OccupyResult::DUPLICATE) {
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
      bucket->digests(key_pos)[0] = get_digest<K>(find_or_insert_key);
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, ScoreFunctor::UNLOCK_MEM_ORDER);
    }
  }
}

/* Read the data from address of table_value_addrs to corresponding position
  in param_value if mask[i] is true, otherwise write data to table_value_addrs
  form param_value,
  usually called by find_or_insert kernel.

  `table_value_addrs`: A pointer of pointer of V which should be on HBM,
        but each value (a pointer of V) could point to a
        memory on HBM or HMEM.
  `param_value`: A continue memory pointer with Vector
        which should be HBM.
  `mask`: One for each `param_value`. If true, reading from table_value_addrs,
          or false writing table_value_addrs from  param_value.
  `param_key_index`: N values from address of table_value_addrs are mapped to
        param_values according to param_key_index.
  `dim`: the dim of value.
  `N`: The number of vectors needed to be read.
*/
template <class K, class V, class S>
__global__ void read_or_write_kernel(V** __restrict table_value_addrs,
                                     V* __restrict param_values,
                                     const bool* mask,
                                     const int* __restrict param_key_index,
                                     const size_t dim, const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_key_index =
        param_key_index != nullptr ? param_key_index[vec_index] : vec_index;

    /// if found, read the value form table, otherwise write it
    if (table_value_addrs[vec_index] != nullptr) {
      /// find
      if (mask[real_key_index]) {
        param_values[real_key_index * dim + dim_index] =
            table_value_addrs[vec_index][dim_index];
      }
      /// insert
      else {
        table_value_addrs[vec_index][dim_index] =
            param_values[real_key_index * dim + dim_index];
      }
    }
  }
}

/* If founds[i] = true, read data from corresponding address of
 * table_value_addrs and write to param_values; if founds[i] = false, write data
 * from param_values to corresponding address of table_value_addrs. usually
 * called by find_or_insert kernel.
 */
template <class V>
void read_or_write_by_cpu(V** __restrict table_value_addrs,
                          V* __restrict param_values,
                          const int* __restrict offset, const bool* founds,
                          size_t dim, int N, int n_worker = 16) {
  std::vector<std::thread> thds;
  if (n_worker < 1) n_worker = 1;

  auto functor = [founds, dim](V** __restrict table_value_addrs,
                               V* __restrict param_values,
                               const int* __restrict offset, int handled_size,
                               int trunk_size) -> void {
    for (int i = handled_size; i < handled_size + trunk_size; i++) {
      if (table_value_addrs[i] != nullptr) {
        if (founds[offset[i]]) {
          memcpy(param_values + offset[i] * dim, table_value_addrs[i],
                 sizeof(V) * dim);
        } else {
          memcpy(table_value_addrs[i], param_values + offset[i] * dim,
                 sizeof(V) * dim);
        }
      }
    }
  };

  int32_t trunk_size_floor = N / n_worker;
  int32_t trunk_size_remain = N % n_worker;
  int32_t n_worker_used = trunk_size_floor == 0 ? trunk_size_remain : n_worker;

  size_t handled_size = 0;
  for (int i = 0; i < n_worker_used; i++) {
    int32_t cur_trunk_size = trunk_size_floor;
    if (trunk_size_remain != 0) {
      cur_trunk_size += 1;
      trunk_size_remain--;
    }
    thds.push_back(std::thread(functor, table_value_addrs, param_values, offset,
                               handled_size, cur_trunk_size));
    handled_size += cur_trunk_size;
  }

  for (int i = 0; i < n_worker_used; i++) {
    thds[i].join();
  }
}

}  // namespace merlin
}  // namespace nv