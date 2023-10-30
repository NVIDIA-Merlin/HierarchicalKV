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
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128>
__global__ void tlp_update_values_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, const uint64_t buckets_num,
    uint32_t bucket_capacity, const uint32_t dim, const K* __restrict__ keys,
    const VecV* __restrict__ values, uint64_t n) {
  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, 1>;

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  VecV* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  uint32_t key_pos = {0};
  if (kv_idx < n) {
    key = keys[kv_idx];

    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      BUCKET* bucket = buckets + bkt_idx;
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
        VecV* bucket_value_ptr = bucket_values_ptr + key_pos * dim;
        const VecV* param_value_ptr = values + kv_idx * dim;
        CopyValue::ldg_stg(0, bucket_value_ptr, param_value_ptr, dim);
        auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
        // memory_order_release:
        // Modifications to the bucket will not after this instruction.
        key_address->store(key, cuda::std::memory_order_release);
        return;
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
        auto probe_key = current_key->load(cuda::std::memory_order_relaxed);
        if (probe_key == static_cast<K>(EMPTY_KEY)) {
          return;
        }
      } while (true);
    }
  }
}
template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          typename VecV = byte16, uint32_t BLOCK_SIZE = 128,
          uint32_t GROUP_SIZE = 16>
__global__ void pipeline_update_values_kernel_with_io(
    Bucket<K, V, S>* __restrict__ buckets, const uint64_t buckets_num,
    const uint32_t dim, const K* __restrict__ keys,
    const VecV* __restrict__ values, uint64_t n) {
  constexpr uint32_t BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr uint32_t Comp_LEN = sizeof(VecD_Comp) / sizeof(D);
  // Here, GROUP_SIZE * Load_LEN = BUCKET_SIZE.
  using VecD_Load = byte8;
  constexpr uint32_t Load_LEN = sizeof(VecD_Load) / sizeof(D);
  constexpr int RESERVE = 8;

  using BUCKET = Bucket<K, V, S>;
  using CopyValue = CopyValueMultipleGroup<VecV, GROUP_SIZE>;

  __shared__ VecD_Comp sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr[BLOCK_SIZE];
  __shared__ VecV* sm_values_ptr[BLOCK_SIZE];
  // Reuse
  int* sm_counts = reinterpret_cast<int*>(sm_target_digests);
  int* sm_position = sm_counts;
  // Double buffer
  __shared__ D sm_digests[GROUP_NUM][2 * BUCKET_SIZE];
  __shared__ K sm_possible_keys[GROUP_NUM][2 * RESERVE];
  __shared__ int sm_possible_pos[GROUP_NUM][2 * RESERVE];
  __shared__ int sm_ranks[GROUP_NUM][2];
  // __shared__ VecV sm_values_buffer[GROUP_NUM][2 * dim];

  extern __shared__ __align__(alignof(byte16)) byte sm_values_buffer[];

  bool CAS_res[2]{false};

  // Initialization
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int groupID = threadIdx.x / GROUP_SIZE;
  int rank = g.thread_rank();
  uint64_t key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;
  if (key_idx_base >= n) return;
  int loop_num =
      (n - key_idx_base) < GROUP_SIZE ? (n - key_idx_base) : GROUP_SIZE;
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    K key = keys[key_idx_base + rank];
    sm_target_keys[idx_block] = key;
    const K hashed_key = Murmur3HashDevice(key);
    sm_target_digests[idx_block] = digests_from_hashed<K>(hashed_key);
    uint64_t global_idx = hashed_key % (buckets_num * BUCKET_SIZE);
    uint64_t bkt_idx = global_idx / BUCKET_SIZE;
    Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __pipeline_memcpy_async(sm_keys_ptr + idx_block, bucket->keys_addr(),
                            sizeof(K*));
    __pipeline_commit();
    __pipeline_memcpy_async(sm_values_ptr + idx_block, &(bucket->vectors),
                            sizeof(VecV*));
  }
  __pipeline_wait_prior(0);

  // Pipeline loading
  K* keys_ptr = sm_keys_ptr[groupID * GROUP_SIZE];
  D* digests_ptr = BUCKET::digests(keys_ptr, BUCKET_SIZE, rank * Load_LEN);
  __pipeline_memcpy_async(sm_digests[groupID] + rank * Load_LEN, digests_ptr,
                          sizeof(VecD_Load));
  __pipeline_commit();
  // Padding, meet the param of the first `__pipeline_wait_prior`
  // in the first loop.
  __pipeline_commit();
  __pipeline_commit();

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    /* Step1: prefetch all digests in one bucket */
    if ((i + 1) < loop_num) {
      K* keys_ptr = sm_keys_ptr[key_idx_block + 1];
      D* digests_ptr = BUCKET::digests(keys_ptr, BUCKET_SIZE, rank * Load_LEN);
      __pipeline_memcpy_async(
          sm_digests[groupID] + diff_buf(i) * BUCKET_SIZE + rank * Load_LEN,
          digests_ptr, sizeof(VecD_Load));
    }
    __pipeline_commit();

    /* Step2: check digests and load possible keys */
    VecD_Comp target_digests = sm_target_digests[key_idx_block];
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(3);
    VecD_Comp probing_digests = *reinterpret_cast<VecD_Comp*>(
        &sm_digests[groupID][same_buf(i) * BUCKET_SIZE + rank * Comp_LEN]);
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests);
    uint32_t find_result = 0;
    if ((find_result_ & 0x01) != 0) find_result |= 0x01;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x02;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x04;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x08;
    probing_digests = *reinterpret_cast<VecD_Comp*>(
        &sm_digests[groupID][same_buf(i) * BUCKET_SIZE +
                             (GROUP_SIZE + rank) * Comp_LEN]);
    find_result_ = __vcmpeq4(probing_digests, target_digests);
    if ((find_result_ & 0x01) != 0) find_result |= 0x10;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x20;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x40;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x80;
    int find_number = __popc(find_result);
    int group_base = 0;
    if (find_number > 0) {
      group_base = atomicAdd(sm_counts + key_idx_block, find_number);
    }
    bool gt_reserve = (group_base + find_number) > RESERVE;
    int gt_vote = g.ballot(gt_reserve);
    K* key_ptr = sm_keys_ptr[key_idx_block];
    if (gt_vote == 0) {
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
          sm_possible_pos[groupID][same_buf(i) * RESERVE + group_base] =
              key_pos;
          __pipeline_memcpy_async(
              sm_possible_keys[groupID] + same_buf(i) * RESERVE + group_base,
              key_ptr + key_pos, sizeof(K));
          group_base += 1;
        } else {
          break;
        }
      } while (true);
    } else {
      K target_key = sm_target_keys[key_idx_block];
      sm_counts[key_idx_block] = 0;
      int found_vote = 0;
      bool found = false;
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
          K possible_key = key_ptr[key_pos];
          if (possible_key == target_key) {
            found = true;
            sm_counts[key_idx_block] = 1;
            sm_possible_pos[groupID][same_buf(i) * RESERVE] = key_pos;
            sm_possible_keys[groupID][same_buf(i) * RESERVE] = possible_key;
          }
        }
        found_vote = g.ballot(found);
        if (found_vote) {
          break;
        }
        found_vote = digest_idx >= 0;
      } while (g.any(found_vote));
    }
    __pipeline_commit();

    /* Step3: check possible keys, and prefecth the value */
    if (i > 0) {
      key_idx_block -= 1;
      K target_key = sm_target_keys[key_idx_block];
      K* keys_ptr = sm_keys_ptr[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      sm_position[key_idx_block] = -1;
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[groupID][diff_buf(i) * RESERVE + rank];
        key_pos = sm_possible_pos[groupID][diff_buf(i) * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
          auto key_ptr = BUCKET::keys(keys_ptr, key_pos);
          sm_ranks[groupID][diff_buf(i)] = rank;
          if (diff_buf(i) == 0) {
            CAS_res[0] = key_ptr->compare_exchange_strong(
                possible_key, static_cast<K>(LOCKED_KEY),
                cuda::std::memory_order_acquire,
                cuda::std::memory_order_relaxed);
          } else {
            CAS_res[1] = key_ptr->compare_exchange_strong(
                possible_key, static_cast<K>(LOCKED_KEY),
                cuda::std::memory_order_acquire,
                cuda::std::memory_order_relaxed);
          }
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        sm_position[key_idx_block] = target_pos;
        int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
        const VecV* v_src = values + key_idx_grid * dim;
        auto tmp = reinterpret_cast<VecV*>(sm_values_buffer);
        VecV* v_dst = tmp + (groupID * 2 + diff_buf(i)) * dim;
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    /* Step4: write back value */
    if (i > 1) {
      key_idx_block -= 1;
      VecV* value_ptr = sm_values_ptr[key_idx_block];
      int target_pos = sm_position[key_idx_block];
      K target_key = sm_target_keys[key_idx_block];
      K* keys_ptr = sm_keys_ptr[key_idx_block];
      int src_lane = sm_ranks[groupID][same_buf(i)];
      __pipeline_wait_prior(3);
      int succ = 0;
      if (rank == src_lane) {
        bool CAS_res_cur = same_buf(i) == 0 ? CAS_res[0] : CAS_res[1];
        succ = CAS_res_cur ? 1 : 0;
      }
      succ = g.shfl(succ, src_lane);
      if (target_pos >= 0 && succ == 1) {
        auto tmp = reinterpret_cast<VecV*>(sm_values_buffer);
        VecV* v_src = tmp + (groupID * 2 + same_buf(i)) * dim;
        VecV* v_dst = value_ptr + target_pos * dim;
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        if (rank == 0) {
          auto key_address = BUCKET::keys(keys_ptr, target_pos);
          key_address->store(target_key, cuda::std::memory_order_release);
        }
      }
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    K* keys_ptr = sm_keys_ptr[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_position[key_idx_block] = -1;
    __pipeline_wait_prior(1);
    int key_pos;
    bool found_flag = false;
    if (rank < possible_num) {
      K possible_key =
          sm_possible_keys[groupID][diff_buf(loop_num) * RESERVE + rank];
      key_pos = sm_possible_pos[groupID][diff_buf(loop_num) * RESERVE + rank];
      if (possible_key == target_key) {
        found_flag = true;
        auto key_ptr = BUCKET::keys(keys_ptr, key_pos);
        sm_ranks[groupID][diff_buf(loop_num)] = rank;
        if (diff_buf(loop_num) == 0) {
          CAS_res[0] = key_ptr->compare_exchange_strong(
              possible_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
        } else {
          CAS_res[1] = key_ptr->compare_exchange_strong(
              possible_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
        }
      }
    }
    int found_vote = g.ballot(found_flag);
    if (found_vote) {
      int src_lane = __ffs(found_vote) - 1;
      int target_pos = g.shfl(key_pos, src_lane);
      sm_position[key_idx_block] = target_pos;
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      const VecV* v_src = values + key_idx_grid * dim;
      auto tmp = reinterpret_cast<VecV*>(sm_values_buffer);
      VecV* v_dst = tmp + (groupID * 2 + diff_buf(loop_num)) * dim;
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  /* Pipeline emptying: step4, i = loop_num */
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    VecV* value_ptr = sm_values_ptr[key_idx_block];
    int target_pos = sm_position[key_idx_block];
    K target_key = sm_target_keys[key_idx_block];
    K* keys_ptr = sm_keys_ptr[key_idx_block];
    int src_lane = sm_ranks[groupID][same_buf(loop_num)];
    __pipeline_wait_prior(1);
    int succ = 0;
    if (rank == src_lane) {
      bool CAS_res_cur = same_buf(loop_num) == 0 ? CAS_res[0] : CAS_res[1];
      succ = CAS_res_cur ? 1 : 0;
    }
    succ = g.shfl(succ, src_lane);
    if (target_pos >= 0 && succ == 1) {
      auto tmp = reinterpret_cast<VecV*>(sm_values_buffer);
      VecV* v_src = tmp + (groupID * 2 + same_buf(loop_num)) * dim;
      VecV* v_dst = value_ptr + target_pos * dim;
      CopyValue::lds_stg(rank, v_dst, v_src, dim);

      auto key_ptr = BUCKET::keys(keys_ptr, target_pos);
      if (rank == 0) {
        auto key_address = BUCKET::keys(keys_ptr, target_pos);
        key_address->store(target_key, cuda::std::memory_order_release);
      }
    }
  }

  /* Pipeline emptying: step4, i = loop_num + 1 */
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    VecV* value_ptr = sm_values_ptr[key_idx_block];
    int target_pos = sm_position[key_idx_block];
    K target_key = sm_target_keys[key_idx_block];
    K* keys_ptr = sm_keys_ptr[key_idx_block];
    int src_lane = sm_ranks[groupID][same_buf(loop_num + 1)];
    __pipeline_wait_prior(0);
    int succ = 0;
    if (rank == src_lane) {
      bool CAS_res_cur = same_buf(loop_num + 1) == 0 ? CAS_res[0] : CAS_res[1];
      succ = CAS_res_cur ? 1 : 0;
    }
    succ = g.shfl(succ, src_lane);
    if (target_pos >= 0 && succ == 1) {
      auto tmp = reinterpret_cast<VecV*>(sm_values_buffer);
      VecV* v_src = tmp + (groupID * 2 + same_buf(loop_num + 1)) * dim;
      VecV* v_dst = value_ptr + target_pos * dim;
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      if (rank == 0) {
        auto key_address = BUCKET::keys(keys_ptr, target_pos);
        key_address->store(target_key, cuda::std::memory_order_release);
      }
    }
  }
}  // End function

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
struct Params_UpdateValues {
  Params_UpdateValues(float load_factor_,
                      Bucket<K, V, S>* __restrict__ buckets_,
                      size_t buckets_num_, uint32_t bucket_capacity_,
                      uint32_t dim_, const K* __restrict__ keys_,
                      const V* __restrict__ values_, size_t n_)
      : load_factor(load_factor_),
        buckets(buckets_),
        buckets_num(buckets_num_),
        bucket_capacity(bucket_capacity_),
        dim(dim_),
        keys(keys_),
        values(values_),
        n(n_) {}
  float load_factor;
  Bucket<K, V, S>* __restrict__ buckets;
  size_t buckets_num;
  uint32_t bucket_capacity;
  uint32_t dim;
  const K* __restrict__ keys;
  const V* __restrict__ values;
  uint64_t n;
};

template <typename K, typename V, typename S, typename VecV>
struct Launch_TLP_UpdateValues {
  using Params = Params_UpdateValues<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    tlp_update_values_kernel_with_io<K, V, S, VecV, BLOCK_SIZE>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            params.buckets, params.buckets_num, params.bucket_capacity,
            params.dim, params.keys,
            reinterpret_cast<const VecV*>(params.values), params.n);
  }
};

template <typename K, typename V, typename S, typename VecV>
struct Launch_Pipeline_UpdateValues {
  using Params = Params_UpdateValues<K, V, S>;
  inline static void launch_kernel(Params& params, cudaStream_t& stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr uint32_t GROUP_SIZE = 16;
    constexpr uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;

    params.dim = params.dim * sizeof(V) / sizeof(VecV);
    uint32_t shared_mem = GROUP_NUM * 2 * params.dim * sizeof(VecV);
    shared_mem =
        (shared_mem + sizeof(byte16) - 1) / sizeof(byte16) * sizeof(byte16);
    pipeline_update_values_kernel_with_io<K, V, S, VecV, BLOCK_SIZE, GROUP_SIZE>
        <<<(params.n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_mem,
           stream>>>(params.buckets, params.buckets_num, params.dim,
                     params.keys, reinterpret_cast<const VecV*>(params.values),
                     params.n);
  }
};

template <typename ArchTag>
struct ValueConfig_UpdateValues;

/// TODO: support more arch.
template <>
struct ValueConfig_UpdateValues<Sm80> {
  // Value size greater than it will bring poor performance for TLP.
  static constexpr uint32_t size_tlp = 8 * sizeof(byte4);
  // Value size greater than it will reduce the occupancy for Pipeline.
  // When the value is very high, the kernel will fail to launch.
  static constexpr uint32_t size_pipeline = 128 * sizeof(byte4);
};

template <>
struct ValueConfig_UpdateValues<Sm70> {
  // Value size greater than it will bring poor performance for TLP.
  static constexpr uint32_t size_tlp = 8 * sizeof(byte4);
  // Value size greater than it will reduce the occupancy for Pipeline.
  // When the value is very high, the kernel will fail to launch.
  static constexpr uint32_t size_pipeline = 64 * sizeof(byte4);
};

template <typename K, typename V, typename S, typename ArchTag>
struct KernelSelector_UpdateValues {
  using ValueConfig = ValueConfig_UpdateValues<ArchTag>;
  using Params = Params_UpdateValues<K, V, S>;

  static bool callable(bool unique_key, uint32_t bucket_size, uint32_t dim) {
    constexpr uint32_t MinBucketCap = sizeof(VecD_Load) / sizeof(D);
    if (!unique_key || bucket_size < MinBucketCap) return false;
    uint32_t value_size = dim * sizeof(V);
    if (value_size <= ValueConfig::size_tlp) return true;
    if (bucket_size == 128 && value_size <= ValueConfig::size_pipeline) {
      return true;
    }
    return false;
  }

  static void select_kernel(Params& params, cudaStream_t& stream) {
    const uint32_t total_value_size =
        static_cast<uint32_t>(params.dim * sizeof(V));

    auto launch_TLP = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_TLP_UpdateValues<K, V, S, VecV>::launch_kernel(params, stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_TLP_UpdateValues<K, V, S, VecV>::launch_kernel(params, stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_TLP_UpdateValues<K, V, S, VecV>::launch_kernel(params, stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_TLP_UpdateValues<K, V, S, VecV>::launch_kernel(params, stream);
      } else {
        using VecV = byte;
        Launch_TLP_UpdateValues<K, V, S, VecV>::launch_kernel(params, stream);
      }
    };

    auto launch_Pipeline = [&]() {
      if (total_value_size % sizeof(byte16) == 0) {
        using VecV = byte16;
        Launch_Pipeline_UpdateValues<K, V, S, VecV>::launch_kernel(params,
                                                                   stream);
      } else if (total_value_size % sizeof(byte8) == 0) {
        using VecV = byte8;
        Launch_Pipeline_UpdateValues<K, V, S, VecV>::launch_kernel(params,
                                                                   stream);
      } else if (total_value_size % sizeof(byte4) == 0) {
        using VecV = byte4;
        Launch_Pipeline_UpdateValues<K, V, S, VecV>::launch_kernel(params,
                                                                   stream);
      } else if (total_value_size % sizeof(byte2) == 0) {
        using VecV = byte2;
        Launch_Pipeline_UpdateValues<K, V, S, VecV>::launch_kernel(params,
                                                                   stream);
      } else {
        using VecV = byte;
        Launch_Pipeline_UpdateValues<K, V, S, VecV>::launch_kernel(params,
                                                                   stream);
      }
    };
    // This part is according to the test on A100.
    if (params.bucket_capacity != 128) {
      launch_TLP();
    } else {
      if (total_value_size <= ValueConfig::size_tlp) {
        if (params.load_factor <= 0.60f) {
          launch_TLP();
        } else {
          launch_Pipeline();
        }
      } else {
        launch_Pipeline();
      }
    }
  }  // End function
};

/*
 * update with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void update_values_kernel_with_io(
    const Table<K, V, S>* __restrict table, Bucket<K, V, S>* buckets,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
    const K* __restrict keys, const V* __restrict values, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    const V* update_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, update_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, S, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, update_value,
                                bucket->vectors + key_pos * dim, dim);
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectUpdateValuesKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, S>* __restrict table,
                             Bucket<K, V, S>* buckets, const K* __restrict keys,
                             const V* __restrict values) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_values_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, buckets,
                                                 bucket_max_size, buckets_num,
                                                 dim, keys, values, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_values_kernel_with_io<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, buckets,
                                                 bucket_max_size, buckets_num,
                                                 dim, keys, values, N);
    }
    return;
  }
};

// Use 1 thread to deal with a KV-pair, including copying value.
template <typename K, typename V, typename S>
__global__ void tlp_update_values_kernel_hybrid(
    Bucket<K, V, S>* __restrict__ buckets, const uint64_t buckets_num,
    uint32_t bucket_capacity, const uint32_t dim, const K* __restrict__ keys,
    V** __restrict__ values, K** __restrict__ key_ptrs,
    int* __restrict src_offset, uint64_t n) {
  using BUCKET = Bucket<K, V, S>;

  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key{static_cast<K>(EMPTY_KEY)};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  VecD_Comp target_digests{0};
  V* bucket_values_ptr{nullptr};
  K* bucket_keys_ptr{nullptr};
  uint32_t key_pos = {0};
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (src_offset) src_offset[kv_idx] = kv_idx;
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx =
          static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
      key_pos = get_start_position(global_idx, bucket_capacity);
      uint64_t bkt_idx = global_idx / bucket_capacity;
      BUCKET* bucket = buckets + bkt_idx;
      bucket_keys_ptr = reinterpret_cast<K*>(bucket->keys(0));
      bucket_values_ptr = bucket->vectors;
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
        key_pos = possible_pos;
        V* bucket_value_ptr = bucket_values_ptr + key_pos * dim;
        values[kv_idx] = bucket_value_ptr;
        key_ptrs[kv_idx] = bucket_keys_ptr + key_pos;
        return;
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
        auto probe_key = current_key->load(cuda::std::memory_order_relaxed);
        if (probe_key == static_cast<K>(EMPTY_KEY)) {
          return;
        }
      } while (true);
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void update_values_kernel(const Table<K, V, S>* __restrict table,
                                     Bucket<K, V, S>* buckets,
                                     const size_t bucket_max_size,
                                     const size_t buckets_num, const size_t dim,
                                     const K* __restrict keys,
                                     V** __restrict vectors,
                                     int* __restrict src_offset, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, update_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    *(src_offset + key_idx) = key_idx;

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, S, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (g.thread_rank() == src_lane) {
      if (occupy_result == OccupyResult::DUPLICATE) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
      } else {
        *(vectors + key_idx) = nullptr;
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

}  // namespace merlin
}  // namespace nv