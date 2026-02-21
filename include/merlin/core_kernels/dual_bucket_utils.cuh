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

#include "kernel_utils.cuh"

namespace nv {
namespace merlin {

/**
 * Core dual-bucket index computation from a pre-computed hash.
 * b1 = low 32 bits mod buckets_num, b2 = high 32 bits mod buckets_num.
 * Guarantees b2 != b1 by advancing b2 on collision.
 *
 * This is the single source of truth for dual-bucket addressing.
 * All kernels (upsert, lookup, etc.) must use this function.
 */
template <class K>
__device__ __forceinline__ void get_dual_bucket_indices(
    const K hashed_key, const size_t buckets_num, size_t& bkt_idx1,
    size_t& bkt_idx2) {
  const uint32_t lo = static_cast<uint32_t>(hashed_key);
  const uint32_t hi =
      static_cast<uint32_t>(static_cast<uint64_t>(hashed_key) >> 32);

  bkt_idx1 = lo % buckets_num;
  bkt_idx2 = hi % buckets_num;
  if (bkt_idx2 == bkt_idx1) {
    bkt_idx2 = (bkt_idx2 + 1) % buckets_num;
  }
}

/**
 * Digest functions for dual-bucket mode.
 *
 * Dual-bucket digests use bits [56:63] (highest 8 bits) of the Murmur3 hash,
 * whereas single-bucket digests use bits [32:39].  The different bit range
 * avoids collision with the b2 bucket address, which is derived from the high
 * 32 bits (bits [32:63]).  Using [56:63] ensures that two keys mapping to the
 * same b2 bucket can still have distinct digests.
 *
 * INVARIANT: `dual_bucket_empty_digest()` must ALWAYS return the true
 * hash-derived value for EMPTY_KEY.  Kernels rely on this sentinel to
 * distinguish empty slots from occupied ones during the SIMD scan pass.
 * Returning a constant would cause every occupied slot to match the empty
 * digest, breaking the probing logic.
 */

// Target digest for a given key (bits [56:63] of Murmur3 hash).
template <class K>
__device__ __forceinline__ D get_dual_bucket_digest(const K& key) {
  const K hashed_key = Murmur3HashDevice(key);
  return static_cast<D>(static_cast<uint64_t>(hashed_key) >> 56);
}

// Target digest from a pre-computed hash.
template <class K>
__device__ __forceinline__ D
get_dual_bucket_digest_from_hash(const K& hashed_key) {
  return static_cast<D>(static_cast<uint64_t>(hashed_key) >> 56);
}

// Pack dual-bucket digest into all 4 bytes for SIMD `__vcmpeq4` comparison.
template <class K>
__device__ __forceinline__ VecD_Comp
dual_bucket_digests_from_hashed(const K& hashed_key) {
  D digest = static_cast<D>(static_cast<uint64_t>(hashed_key) >> 56);
  return static_cast<VecD_Comp>(__byte_perm(digest, digest, 0x0000));
}

// Sentinel digest for EMPTY_KEY in dual-bucket mode — must always use real
// hash value (bits [56:63]).
template <class K>
__device__ __forceinline__ D dual_bucket_empty_digest() {
  const K hashed_key = Murmur3HashDevice(static_cast<K>(EMPTY_KEY));
  return static_cast<D>(static_cast<uint64_t>(hashed_key) >> 56);
}

// Pack empty-key digest into all 4 bytes for SIMD comparison.
template <class K>
__device__ __forceinline__ VecD_Comp dual_bucket_empty_digests() {
  D digest = dual_bucket_empty_digest<K>();
  return static_cast<VecD_Comp>(__byte_perm(digest, digest, 0x0000));
}

}  // namespace merlin
}  // namespace nv
