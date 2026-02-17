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

#pragma once

#include "dual_bucket_utils.cuh"
#include "kernel_utils.cuh"

namespace nv {
namespace merlin {

/**
 * Dual-bucket pipeline upsert kernel — True Two-Choice.
 *
 * Implements dual-bucket insert_or_assign with three distinct phases:
 *   Phase 0: DUPLICATE detection in BOTH buckets (no empty-slot occupation)
 *   Phase 1: D1 Two-Choice load-balance — compare bucket sizes, insert into
 *            the emptier bucket first, fallback to the other
 *   Phase 2: D2 score-eviction — when both buckets full, evict the entry
 *            with the global minimum score across both buckets
 *
 * Key invariant: DUPLICATE search completes in BOTH buckets before any
 * empty-slot insertion attempt. This ensures correct insert_or_assign
 * semantics (no spurious duplicates across buckets).
 *
 * Concurrent model: pure slot-level CAS (no per-bucket Mutex).
 * Constraint: unique_key=true (caller guarantees no duplicate keys in batch).
 *
 * Based on pipeline_upsert_kernel_with_io architecture:
 * - 32 threads per key (GROUP_SIZE)
 * - 128-thread blocks
 * - 128-slot buckets
 * - 4-stage software pipeline
 */
template <class K, class V, class S, class VecV, int BLOCK_SIZE = 128,
          int Strategy = 0>
__global__ void dual_bucket_pipeline_upsert_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, int32_t* __restrict__ buckets_size,
    const uint64_t buckets_num, const uint32_t dim, const K* __restrict__ keys,
    const VecV* __restrict__ values, const S* __restrict__ scores, uint64_t n,
    const S global_epoch) {
  constexpr uint32_t BUCKET_SIZE = 128;
  constexpr uint32_t GROUP_SIZE = 32;
  constexpr uint32_t Comp_LEN = sizeof(VecD_Comp) / sizeof(D);
  constexpr uint32_t Load_LEN = sizeof(VecD_Load) / sizeof(D);
  constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);

  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;
  using SMM = SharedMemoryManager_Pipeline_Upsert<K, V, S, VecV, BLOCK_SIZE,
                                                  GROUP_SIZE, BUCKET_SIZE>;
  using ScoreFunctor_ = ScoreFunctor<K, V, S, Strategy>;

  __shared__ extern __align__(alignof(byte16)) byte smem[];

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  VecD_Comp target_digests;
  K* bucket_keys_ptr1{nullptr};
  K* bucket_keys_ptr2{nullptr};
  VecV* bucket_values_ptr2{nullptr};
  int* bucket_size_ptr2{nullptr};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  uint32_t key_pos = 0;
  uint32_t key_pos2 = 0;  // b2 start position (independent from b1)
  int target_bucket = 1;  // 1 = b1, 2 = b2

  if (kv_idx < n) {
    key = keys[kv_idx];
    if (scores != nullptr) {
      S* sm_param_scores = SMM::param_scores(smem);
      __pipeline_memcpy_async(sm_param_scores + tx, scores + kv_idx, sizeof(S));
    }
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      // Dual-bucket digest from bit[56:63].
      target_digests = dual_bucket_digests_from_hashed<K>(hashed_key);

      // Dual-bucket indices (centralized in dual_bucket_utils.cuh).
      size_t bkt_idx1, bkt_idx2;
      get_dual_bucket_indices<K>(hashed_key, buckets_num, bkt_idx1, bkt_idx2);

      // b1 setup (stored in SMM shared memory).
      const uint32_t lo = static_cast<uint32_t>(hashed_key);
      uint64_t global_idx1 =
          static_cast<uint64_t>(lo % (buckets_num * BUCKET_SIZE));
      key_pos = get_start_position(global_idx1, BUCKET_SIZE);

      // b2 start position from high 32 bits (independent from b1).
      const uint32_t hi =
          static_cast<uint32_t>(static_cast<uint64_t>(hashed_key) >> 32);
      uint64_t global_idx2 =
          static_cast<uint64_t>(hi % (buckets_num * BUCKET_SIZE));
      key_pos2 = get_start_position(global_idx2, BUCKET_SIZE);

      int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
      sm_buckets_size_ptr[tx] = buckets_size + bkt_idx1;

      BUCKET* bucket1 = buckets + bkt_idx1;
      bucket_keys_ptr1 = reinterpret_cast<K*>(bucket1->keys(0));
      VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
      __pipeline_memcpy_async(sm_bucket_values_ptr + tx, &(bucket1->vectors),
                              sizeof(VecV*));

      // b2 setup (stored in registers, broadcast via warp shuffle).
      BUCKET* bucket2 = buckets + bkt_idx2;
      bucket_keys_ptr2 = reinterpret_cast<K*>(bucket2->keys(0));
      bucket_values_ptr2 = reinterpret_cast<VecV*>(bucket2->vectors);
      bucket_size_ptr2 = buckets_size + bkt_idx2;
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  uint32_t rank = g.thread_rank();
  uint32_t groupID = threadIdx.x / GROUP_SIZE;

  // =========== Main pipeline loop (processes one key per iteration)
  // =========== True Two-Choice algorithm for each key i in the warp:
  //   Phase 0: DUPLICATE detection in BOTH b1 and b2 (no empty occupation)
  //   Phase 1: D1 Two-Choice — compare bucket sizes, try emptier bucket first
  //   Phase 2: D2 score-eviction when both buckets are full

  auto occupy_result_next = g.shfl(occupy_result, 0);
  auto keys_ptr_next = g.shfl(bucket_keys_ptr1, 0);

  // Prefetch b1 digests for first key.
  if (occupy_result_next == OccupyResult::INITIAL) {
    D* sm_bucket_digests = SMM::bucket_digests(smem, groupID, 0);
    D* dst = sm_bucket_digests + rank * Load_LEN;
    D* src = BUCKET::digests(keys_ptr_next, BUCKET_SIZE, rank * Load_LEN);
    if (rank * Load_LEN < BUCKET_SIZE) {
      __pipeline_memcpy_async(dst, src, sizeof(VecD_Load));
    }
  }
  __pipeline_commit();
  __pipeline_commit();
  __pipeline_commit();

  for (int32_t i = 0; i < GROUP_SIZE; i++) {
    // === Step 1: Prefetch b1 digests for next key ===
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      auto keys_ptr_next = g.shfl(bucket_keys_ptr1, i + 1);
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

    // === Step 2: Three-phase True Two-Choice probe ===
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur == OccupyResult::INITIAL) {
      uint32_t tx_cur = groupID * GROUP_SIZE + i;
      int** sm_buckets_size_ptr = SMM::buckets_size_ptr(smem);
      auto bucket_size_ptr1 = sm_buckets_size_ptr[tx_cur];
      K key_cur = g.shfl(key, i);
      auto target_digests_cur = g.shfl(target_digests, i);
      auto start_pos_cur = g.shfl(key_pos, i);
      auto keys_ptr_cur = g.shfl(bucket_keys_ptr1, i);

      // b2 info for key i (shuffled from owning thread).
      auto keys_ptr2_cur = g.shfl(bucket_keys_ptr2, i);
      auto bsize_ptr2_cur = reinterpret_cast<int*>(static_cast<uintptr_t>(
          g.shfl(static_cast<unsigned long long>(
                     reinterpret_cast<uintptr_t>(bucket_size_ptr2)),
                 i)));
      auto start_pos2_cur = g.shfl(key_pos2, i);

      __pipeline_wait_prior(3);
      D* digest_src = SMM::bucket_digests(smem, groupID, same_buf(i));

      // b1 probe offset (from b1's hash).
      uint32_t start_offset = start_pos_cur / Comp_LEN;
      uint32_t probe_offset =
          Comp_LEN * ((start_offset + rank) & (GROUP_SIZE - 1));
      VecD_Comp probe_digests =
          *reinterpret_cast<VecD_Comp*>(digest_src + probe_offset);
      uint32_t cmp_result = __vcmpeq4(probe_digests, target_digests_cur);
      cmp_result &= 0x01010101;

      // b2 probe offset (from b2's independent hash).
      uint32_t start_offset2 = start_pos2_cur / Comp_LEN;
      uint32_t b2_probe_offset =
          Comp_LEN * ((start_offset2 + rank) & (GROUP_SIZE - 1));
      // Load b2 digests (synchronous read).
      D* b2_digests_ptr = BUCKET::digests(keys_ptr2_cur, BUCKET_SIZE, 0);
      VecD_Comp b2_probe_digests =
          *reinterpret_cast<VecD_Comp*>(b2_digests_ptr + b2_probe_offset);
      uint32_t b2_cmp = __vcmpeq4(b2_probe_digests, target_digests_cur);
      b2_cmp &= 0x01010101;

      // ============================================================
      // Phase 0: DUPLICATE detection in BOTH buckets
      // ============================================================

      // --- Phase 0a: DUPLICATE scan in b1 ---
      uint32_t possible_pos = 0;
      bool result = false;
      {
        uint32_t cmp_copy = cmp_result;
        do {
          if (cmp_copy == 0) break;
          int32_t index = (__ffs(cmp_copy) - 1) >> 3;
          cmp_copy &= (cmp_copy - 1);
          possible_pos = probe_offset + index;
          auto current_key = BUCKET::keys(keys_ptr_cur, possible_pos);
          K expected_key = key_cur;
          result = current_key->compare_exchange_strong(
              expected_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
        } while (!result);
      }

      uint32_t found_vote = g.ballot(result);
      if (found_vote) {
        // DUPLICATE found in b1 -> update in place.
        int32_t src_lane = __ffs(found_vote) - 1;
        possible_pos = g.shfl(possible_pos, src_lane);
        if (rank == i) {
          occupy_result = OccupyResult::DUPLICATE;
          key_pos = possible_pos;
          target_bucket = 1;
          S* sm_param_scores = SMM::param_scores(smem);
          // Note: desired_when_missed is intentionally used here for
          // DUPLICATE keys.  For kCustomized strategy the actual score
          // semantics are determined by update_with_digest, which
          // overwrites the score unconditionally.  The naming is
          // inherited from the single-bucket API and does not imply
          // "key was absent".
          S score = ScoreFunctor_::desired_when_missed(sm_param_scores, tx,
                                                       global_epoch);
          D digest = get_dual_bucket_digest<K>(key);
          ScoreFunctor_::update_with_digest(bucket_keys_ptr1, key_pos,
                                            sm_param_scores, tx, score,
                                            BUCKET_SIZE, digest, false);
        }
      }

      // --- Phase 0b: DUPLICATE scan in b2 (only if not found in b1) ---
      occupy_result_cur = g.shfl(occupy_result, i);
      if (occupy_result_cur == OccupyResult::INITIAL) {
        result = false;
        possible_pos = 0;
        {
          uint32_t cmp_copy = b2_cmp;
          do {
            if (cmp_copy == 0) break;
            int32_t index = (__ffs(cmp_copy) - 1) >> 3;
            cmp_copy &= (cmp_copy - 1);
            possible_pos = b2_probe_offset + index;
            auto current_key = BUCKET::keys(keys_ptr2_cur, possible_pos);
            K expected_key = key_cur;
            result = current_key->compare_exchange_strong(
                expected_key, static_cast<K>(LOCKED_KEY),
                cuda::std::memory_order_acquire,
                cuda::std::memory_order_relaxed);
          } while (!result);
        }

        found_vote = g.ballot(result);
        if (found_vote) {
          // DUPLICATE found in b2.
          int32_t src_lane = __ffs(found_vote) - 1;
          possible_pos = g.shfl(possible_pos, src_lane);
          if (rank == i) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            target_bucket = 2;
            S* sm_param_scores = SMM::param_scores(smem);
            // See Phase 0a comment: desired_when_missed is used for
            // DUPLICATE keys; actual semantics governed by
            // update_with_digest.
            S score = ScoreFunctor_::desired_when_missed(sm_param_scores, tx,
                                                         global_epoch);
            D digest = get_dual_bucket_digest<K>(key);
            ScoreFunctor_::update_with_digest(bucket_keys_ptr2, key_pos,
                                              sm_param_scores, tx, score,
                                              BUCKET_SIZE, digest, false);
          }
        }
      }

      // ============================================================
      // Phase 1: D1 Two-Choice load-balanced EMPTY insertion
      // ============================================================
      occupy_result_cur = g.shfl(occupy_result, i);
      if (occupy_result_cur == OccupyResult::INITIAL) {
        auto bucket_size1 = *bucket_size_ptr1;
        auto bucket_size2 = *bsize_ptr2_cur;

        // True Two-Choice: prefer the emptier bucket.
        bool prefer_b1 = (bucket_size1 <= bucket_size2);

        // First bucket (emptier one).
        K* first_keys_ptr = prefer_b1 ? keys_ptr_cur : keys_ptr2_cur;
        int* first_bsize_ptr = prefer_b1 ? bucket_size_ptr1 : bsize_ptr2_cur;
        int first_size = prefer_b1 ? bucket_size1 : bucket_size2;
        VecD_Comp first_probe_digests =
            prefer_b1 ? probe_digests : b2_probe_digests;
        uint32_t first_probe_offset =
            prefer_b1 ? probe_offset : b2_probe_offset;
        int first_bucket_id = prefer_b1 ? 1 : 2;

        // Second bucket (fuller one).
        K* second_keys_ptr = prefer_b1 ? keys_ptr2_cur : keys_ptr_cur;
        int* second_bsize_ptr = prefer_b1 ? bsize_ptr2_cur : bucket_size_ptr1;
        int second_size = prefer_b1 ? bucket_size2 : bucket_size1;
        VecD_Comp second_probe_digests =
            prefer_b1 ? b2_probe_digests : probe_digests;
        uint32_t second_probe_offset =
            prefer_b1 ? b2_probe_offset : probe_offset;
        int second_bucket_id = prefer_b1 ? 2 : 1;

        // --- Try EMPTY in first (emptier) bucket ---
        if (first_size < BUCKET_SIZE) {
          VecD_Comp empty_digests_ = dual_bucket_empty_digests<K>();
          uint32_t empty_result =
              __vcmpeq4(first_probe_digests, empty_digests_);
          empty_result &= 0x01010101;
          result = false;
          possible_pos = 0;
          for (int32_t offset = 0; offset < GROUP_SIZE; offset += 1) {
            if (rank == offset) {
              do {
                if (empty_result == 0) break;
                int32_t index = (__ffs(empty_result) - 1) >> 3;
                empty_result &= (empty_result - 1);
                possible_pos = first_probe_offset + index;
                auto current_key = BUCKET::keys(first_keys_ptr, possible_pos);
                K expected_key = static_cast<K>(EMPTY_KEY);
                result = current_key->compare_exchange_strong(
                    expected_key, static_cast<K>(LOCKED_KEY),
                    cuda::std::memory_order_acquire,
                    cuda::std::memory_order_relaxed);
              } while (!result);
            }
            found_vote = g.ballot(result);
            if (found_vote) {
              int32_t src_lane = __ffs(found_vote) - 1;
              possible_pos = g.shfl(possible_pos, src_lane);
              if (rank == i) {
                occupy_result = OccupyResult::OCCUPIED_EMPTY;
                key_pos = possible_pos;
                target_bucket = first_bucket_id;
                S* sm_param_scores = SMM::param_scores(smem);
                S score = ScoreFunctor_::desired_when_missed(sm_param_scores,
                                                             tx, global_epoch);
                D digest = get_dual_bucket_digest<K>(key);
                K* target_keys = (first_bucket_id == 1) ? bucket_keys_ptr1
                                                        : bucket_keys_ptr2;
                ScoreFunctor_::update_with_digest(target_keys, key_pos,
                                                  sm_param_scores, tx, score,
                                                  BUCKET_SIZE, digest, true);
                atomicAdd(first_bsize_ptr, 1);
              }
              break;
            }
          }
        }

        // --- Try EMPTY in second (fuller) bucket (fallback) ---
        occupy_result_cur = g.shfl(occupy_result, i);
        if (occupy_result_cur == OccupyResult::INITIAL &&
            second_size < BUCKET_SIZE) {
          VecD_Comp empty_digests_ = dual_bucket_empty_digests<K>();
          uint32_t empty_result =
              __vcmpeq4(second_probe_digests, empty_digests_);
          empty_result &= 0x01010101;
          result = false;
          possible_pos = 0;
          for (int32_t offset = 0; offset < GROUP_SIZE; offset += 1) {
            if (rank == offset) {
              do {
                if (empty_result == 0) break;
                int32_t index = (__ffs(empty_result) - 1) >> 3;
                empty_result &= (empty_result - 1);
                possible_pos = second_probe_offset + index;
                auto current_key = BUCKET::keys(second_keys_ptr, possible_pos);
                K expected_key = static_cast<K>(EMPTY_KEY);
                result = current_key->compare_exchange_strong(
                    expected_key, static_cast<K>(LOCKED_KEY),
                    cuda::std::memory_order_acquire,
                    cuda::std::memory_order_relaxed);
              } while (!result);
            }
            found_vote = g.ballot(result);
            if (found_vote) {
              int32_t src_lane = __ffs(found_vote) - 1;
              possible_pos = g.shfl(possible_pos, src_lane);
              if (rank == i) {
                occupy_result = OccupyResult::OCCUPIED_EMPTY;
                key_pos = possible_pos;
                target_bucket = second_bucket_id;
                S* sm_param_scores = SMM::param_scores(smem);
                S score = ScoreFunctor_::desired_when_missed(sm_param_scores,
                                                             tx, global_epoch);
                D digest = get_dual_bucket_digest<K>(key);
                K* target_keys = (second_bucket_id == 1) ? bucket_keys_ptr1
                                                         : bucket_keys_ptr2;
                ScoreFunctor_::update_with_digest(target_keys, key_pos,
                                                  sm_param_scores, tx, score,
                                                  BUCKET_SIZE, digest, true);
                atomicAdd(second_bsize_ptr, 1);
              }
              break;
            }
          }
        }
      }

      // ============================================================
      // Phase 2: D2 Score Eviction (both buckets full)
      // ============================================================
      occupy_result_cur = g.shfl(occupy_result, i);
      if (occupy_result_cur == OccupyResult::INITIAL) {
        S* sm_param_scores = SMM::param_scores(smem);
        S score_cur = ScoreFunctor_::desired_when_missed(sm_param_scores,
                                                         tx_cur, global_epoch);

        S* b1_scores = BUCKET::scores(keys_ptr_cur, BUCKET_SIZE, 0);
        S* b2_scores = BUCKET::scores(keys_ptr2_cur, BUCKET_SIZE, 0);

        // Cache scores in per-thread registers for eviction retry.
        constexpr int SCORES_PER_THREAD =
            BUCKET_SIZE / (GROUP_SIZE * Load_LEN_S) * Load_LEN_S;
        S b1_cached[SCORES_PER_THREAD];
        int b1_pos_cached[SCORES_PER_THREAD];
        S b2_cached[SCORES_PER_THREAD];
        int b2_pos_cached[SCORES_PER_THREAD];
        {
          int idx = 0;
          for (int j = 0; j < BUCKET_SIZE; j += GROUP_SIZE * Load_LEN_S) {
            S tmp[Load_LEN_S];
            *reinterpret_cast<byte16*>(tmp) =
                *reinterpret_cast<byte16*>(b1_scores + rank * Load_LEN_S + j);
            for (int k = 0; k < Load_LEN_S; k++) {
              b1_cached[idx] = tmp[k];
              b1_pos_cached[idx] = rank * Load_LEN_S + j + k;
              idx++;
            }
          }
        }
        {
          int idx = 0;
          for (int j = 0; j < BUCKET_SIZE; j += GROUP_SIZE * Load_LEN_S) {
            S tmp[Load_LEN_S];
            *reinterpret_cast<byte16*>(tmp) =
                *reinterpret_cast<byte16*>(b2_scores + rank * Load_LEN_S + j);
            for (int k = 0; k < Load_LEN_S; k++) {
              b2_cached[idx] = tmp[k];
              b2_pos_cached[idx] = rank * Load_LEN_S + j + k;
              idx++;
            }
          }
        }

        // Eviction retry loop.
        while (true) {
          occupy_result_cur = g.shfl(occupy_result, i);
          if (occupy_result_cur != OccupyResult::INITIAL) break;

          // Find per-thread min for b1 and b2 from cached scores.
          S min_b1_local = static_cast<S>(MAX_SCORE);
          int min_b1_idx = -1;
          for (int s = 0; s < SCORES_PER_THREAD; s++) {
            if (b1_cached[s] < min_b1_local) {
              min_b1_local = b1_cached[s];
              min_b1_idx = s;
            }
          }
          S min_b2_local = static_cast<S>(MAX_SCORE);
          int min_b2_idx = -1;
          for (int s = 0; s < SCORES_PER_THREAD; s++) {
            if (b2_cached[s] < min_b2_local) {
              min_b2_local = b2_cached[s];
              min_b2_idx = s;
            }
          }

          S min_b1_global = cg::reduce(g, min_b1_local, cg::less<S>());
          S min_b2_global = cg::reduce(g, min_b2_local, cg::less<S>());
          S overall_min =
              (min_b1_global <= min_b2_global) ? min_b1_global : min_b2_global;

          // REFUSED: new score too low to evict anything.
          if (score_cur < overall_min) {
            if (rank == i) {
              occupy_result = OccupyResult::REFUSED;
            }
            break;
          }

          // Pick the bucket with lower min_score (Two-Choice eviction).
          bool use_b1 = (min_b1_global <= min_b2_global);
          S min_score_local = use_b1 ? min_b1_local : min_b2_local;
          int min_local_idx = use_b1 ? min_b1_idx : min_b2_idx;
          int min_pos_local = (min_local_idx >= 0)
                                  ? (use_b1 ? b1_pos_cached[min_local_idx]
                                            : b2_pos_cached[min_local_idx])
                                  : -1;
          S min_score_global = use_b1 ? min_b1_global : min_b2_global;
          K* evict_keys_ptr = use_b1 ? keys_ptr_cur : keys_ptr2_cur;
          int* evict_bsize_ptr = use_b1 ? bucket_size_ptr1 : bsize_ptr2_cur;

          uint32_t vote = g.ballot(min_score_local <= min_score_global);
          if (vote) {
            int src_lane = __ffs(vote) - 1;
            int min_pos_evict = g.shfl(min_pos_local, src_lane);

            // Mark this position as visited for the winning thread.
            if (use_b1) {
              int visited_idx = g.shfl(min_local_idx, src_lane);
              if (rank == src_lane && visited_idx >= 0)
                b1_cached[visited_idx] = static_cast<S>(MAX_SCORE);
            } else {
              int visited_idx = g.shfl(min_local_idx, src_lane);
              if (rank == src_lane && visited_idx >= 0)
                b2_cached[visited_idx] = static_cast<S>(MAX_SCORE);
            }

            if (rank == i) {
              auto min_score_key = BUCKET::keys(evict_keys_ptr, min_pos_evict);
              auto expected_key =
                  min_score_key->load(cuda::std::memory_order_relaxed);
              if (expected_key != static_cast<K>(LOCKED_KEY) &&
                  expected_key != static_cast<K>(EMPTY_KEY)) {
                bool cas_ok = min_score_key->compare_exchange_strong(
                    expected_key, static_cast<K>(LOCKED_KEY),
                    cuda::std::memory_order_acquire,
                    cuda::std::memory_order_relaxed);
                if (cas_ok) {
                  S* score_ptr = BUCKET::scores(evict_keys_ptr, BUCKET_SIZE,
                                                min_pos_evict);
                  auto verify_score_ptr =
                      reinterpret_cast<AtomicScore<S>*>(score_ptr);
                  auto verify_score =
                      verify_score_ptr->load(cuda::std::memory_order_relaxed);
                  if (verify_score <= min_score_global) {
                    if (expected_key == static_cast<K>(RECLAIM_KEY)) {
                      occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                      atomicAdd(evict_bsize_ptr, 1);
                    } else {
                      occupy_result = OccupyResult::EVICT;
                    }
                    key_pos = min_pos_evict;
                    target_bucket = use_b1 ? 1 : 2;
                    K* target_keys_ptr =
                        use_b1 ? bucket_keys_ptr1 : bucket_keys_ptr2;
                    D digest = get_dual_bucket_digest<K>(key);
                    ScoreFunctor_::update_with_digest(
                        target_keys_ptr, key_pos, sm_param_scores, tx,
                        score_cur, BUCKET_SIZE, digest, true);
                  } else {
                    min_score_key->store(expected_key,
                                         cuda::std::memory_order_release);
                  }
                }
              }
            }
          } else {
            // No thread holds the minimum — all positions exhausted.
            if (rank == i) {
              occupy_result = OccupyResult::REFUSED;
            }
            break;
          }
        }  // while eviction retry
      }
    }  // end of INITIAL check

    // === Step 3: Prefetch values to shared memory for previous key ===
    if (i > 0) {
      auto occupy_result_prev = g.shfl(occupy_result, i - 1);
      if (occupy_result_prev != OccupyResult::ILLEGAL &&
          occupy_result_prev != OccupyResult::REFUSED) {
        VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(i), dim);
        auto kv_idx_cur = g.shfl(kv_idx, i - 1);
        const VecV* src = values + kv_idx_cur * dim;
        CopyValue::ldg_sts(rank, dst, src, dim);
      }
    }
    __pipeline_commit();

    // === Step 4: Write values for key (i-2) ===
    if (i > 1) {
      auto occupy_result_wb = g.shfl(occupy_result, i - 2);
      if (occupy_result_wb != OccupyResult::ILLEGAL &&
          occupy_result_wb != OccupyResult::REFUSED) {
        VecV* src = SMM::values_buffer(smem, groupID, same_buf(i), dim);
        auto key_pos_wb = g.shfl(key_pos, i - 2);
        auto target_bucket_wb = g.shfl(target_bucket, i - 2);

        // Get the correct values pointer for the target bucket.
        VecV* dst;
        if (target_bucket_wb == 1) {
          VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
          dst = sm_bucket_values_ptr[groupID * GROUP_SIZE + i - 2] +
                key_pos_wb * dim;
        } else {
          auto bv2 = g.shfl(bucket_values_ptr2, i - 2);
          dst = bv2 + key_pos_wb * dim;
        }
        __pipeline_wait_prior(3);
        CopyValue::lds_stg(rank, dst, src, dim);

        // Unlock key.
        if (rank == i - 2) {
          K* target_keys_ptr =
              (target_bucket == 1) ? bucket_keys_ptr1 : bucket_keys_ptr2;
          auto key_address = BUCKET::keys(target_keys_ptr, key_pos);
          key_address->store(key, cuda::std::memory_order_release);
        }
      }
    }
  }  // end main loop

  // =========== Pipeline draining ===========

  // Step 3 for last key (i = GROUP_SIZE - 1).
  {
    auto occupy_result_prev = g.shfl(occupy_result, GROUP_SIZE - 1);
    if (occupy_result_prev != OccupyResult::ILLEGAL &&
        occupy_result_prev != OccupyResult::REFUSED) {
      VecV* dst = SMM::values_buffer(smem, groupID, diff_buf(GROUP_SIZE), dim);
      auto kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 1);
      const VecV* src = values + kv_idx_cur * dim;
      CopyValue::ldg_sts(rank, dst, src, dim);
    }
  }
  __pipeline_commit();

  // Step 4 for key (GROUP_SIZE - 2).
  {
    auto occupy_result_wb = g.shfl(occupy_result, GROUP_SIZE - 2);
    if (occupy_result_wb != OccupyResult::ILLEGAL &&
        occupy_result_wb != OccupyResult::REFUSED) {
      VecV* src = SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE), dim);
      auto key_pos_wb = g.shfl(key_pos, GROUP_SIZE - 2);
      auto target_bucket_wb = g.shfl(target_bucket, GROUP_SIZE - 2);
      VecV* dst;
      if (target_bucket_wb == 1) {
        VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
        dst = sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 2] +
              key_pos_wb * dim;
      } else {
        auto bv2 = g.shfl(bucket_values_ptr2, GROUP_SIZE - 2);
        dst = bv2 + key_pos_wb * dim;
      }
      __pipeline_wait_prior(1);
      CopyValue::lds_stg(rank, dst, src, dim);
      if (rank == GROUP_SIZE - 2) {
        K* target_keys_ptr =
            (target_bucket == 1) ? bucket_keys_ptr1 : bucket_keys_ptr2;
        auto key_address = BUCKET::keys(target_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
    }
  }

  // Step 4 for last key (GROUP_SIZE - 1).
  {
    auto occupy_result_wb = g.shfl(occupy_result, GROUP_SIZE - 1);
    if (occupy_result_wb != OccupyResult::ILLEGAL &&
        occupy_result_wb != OccupyResult::REFUSED) {
      VecV* src =
          SMM::values_buffer(smem, groupID, same_buf(GROUP_SIZE + 1), dim);
      auto key_pos_wb = g.shfl(key_pos, GROUP_SIZE - 1);
      auto target_bucket_wb = g.shfl(target_bucket, GROUP_SIZE - 1);
      VecV* dst;
      if (target_bucket_wb == 1) {
        VecV** sm_bucket_values_ptr = SMM::bucket_values_ptr(smem);
        dst = sm_bucket_values_ptr[groupID * GROUP_SIZE + GROUP_SIZE - 1] +
              key_pos_wb * dim;
      } else {
        auto bv2 = g.shfl(bucket_values_ptr2, GROUP_SIZE - 1);
        dst = bv2 + key_pos_wb * dim;
      }
      __pipeline_wait_prior(0);
      CopyValue::lds_stg(rank, dst, src, dim);
      if (rank == GROUP_SIZE - 1) {
        K* target_keys_ptr =
            (target_bucket == 1) ? bucket_keys_ptr1 : bucket_keys_ptr2;
        auto key_address = BUCKET::keys(target_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
    }
  }
}

// --- Kernel Launcher ---

template <typename K, typename V, typename S, typename VecV, int Strategy>
struct Launch_DualBucket_Pipeline_Upsert {
  using Params = Params_Upsert<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr uint32_t GROUP_SIZE = 32;
    constexpr uint32_t BUCKET_SIZE = 128;
    using SMM = SharedMemoryManager_Pipeline_Upsert<K, V, S, VecV, BLOCK_SIZE,
                                                    GROUP_SIZE, BUCKET_SIZE>;

    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    uint32_t shared_mem = SMM::total_size(params.dim);
    shared_mem =
        (shared_mem + sizeof(byte16) - 1) / sizeof(byte16) * sizeof(byte16);
    dual_bucket_pipeline_upsert_kernel_with_io<K, V, S, VecV, BLOCK_SIZE,
                                               Strategy>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_mem,
           stream>>>(params.buckets, params.buckets_size, params.buckets_num,
                     params.dim, params.keys,
                     reinterpret_cast<const VecV*>(params.values),
                     params.scores, params.n, params.global_epoch);
  }
};

// --- Kernel Selector ---

template <typename K, typename V, typename S, int Strategy, typename ArchTag>
struct KernelSelector_DualBucketUpsert {
  using Params = Params_Upsert<K, V, S>;

  static void select_kernel(Params& params, cudaStream_t& stream) {
    const uint32_t total_value_size =
        static_cast<uint32_t>(params.dim * sizeof(V));

    // Dual-bucket always uses pipeline kernel (optimized for bucket_size=128).
    if (total_value_size % sizeof(byte16) == 0) {
      using VecV = byte16;
      Launch_DualBucket_Pipeline_Upsert<K, V, S, VecV, Strategy>::launch_kernel(
          params, stream);
    } else if (total_value_size % sizeof(byte8) == 0) {
      using VecV = byte8;
      Launch_DualBucket_Pipeline_Upsert<K, V, S, VecV, Strategy>::launch_kernel(
          params, stream);
    } else if (total_value_size % sizeof(byte4) == 0) {
      using VecV = byte4;
      Launch_DualBucket_Pipeline_Upsert<K, V, S, VecV, Strategy>::launch_kernel(
          params, stream);
    } else if (total_value_size % sizeof(byte2) == 0) {
      using VecV = byte2;
      Launch_DualBucket_Pipeline_Upsert<K, V, S, VecV, Strategy>::launch_kernel(
          params, stream);
    } else {
      using VecV = byte;
      Launch_DualBucket_Pipeline_Upsert<K, V, S, VecV, Strategy>::launch_kernel(
          params, stream);
    }
  }
};

}  // namespace merlin
}  // namespace nv
