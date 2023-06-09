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

#include <cstdint>
#include <type_traits>

namespace nv {
namespace merlin {

template <class Key, class Value>
class ExternalStorage {
 public:
  using size_type = size_t;
  using key_type = Key;
  using value_type = Value;

  /**
   * @brief Inserts key/value pairs into the external storage. If a key/value
   * pair already exists, overwrites the current value.
   *
   * @param n Number of key/value slots provided in other arguments.
   * @param d_masked_keys Device pointer to an (n)-sized array of keys.
   * Key-Value slots that should be ignored have the key set to `EMPTY_KEY`.
   * @param d_values Device pointer to an (n)-sized array containing pointers to
   * respectively a memory location where the current values for a key are
   * stored. Each pointer points to a vector of length `value_dim`. Pointers
   * *can* be set to `nullptr` for slots where the corresponding key equated to
   * the `EMPTY_KEY`. The memory locations can be device or host memory (see
   * also `hkvs_is_pure_hbm`).
   * @param stream Stream that MUST be used for queuing asynchronous CUDA
   * operations. If only the input arguments or resources obtained from
   * respectively `dev_mem_pool` and `host_mem_pool` are used for such
   * operations, it is not necessary to synchronize the stream prior to
   * returning from the function.
   */
  virtual void insert_or_assign(size_type n,
                                const key_type* d_masked_keys,  // (n)
                                const value_type* d_values,     // (n)
                                size_type value_dims, cudaStream_t stream) = 0;

  /**
   * @brief Attempts to find the supplied `d_keys` if the corresponding
   * `d_founds`-flag is `false` and fills the stored into the supplied memory
   * locations (i.e.  in `d_values`).
   *
   * @param n Number of key/value slots provided in other arguments.
   * @param d_keys Device pointer to an (n)-sized array of keys.
   * @param d_values Device pointer to an (n * value_dim)-sized array to store
   * the retrieved `d_values`. For slots where the corresponding `d_founds`-flag
   * is not `false`, the value may already have been assigned and, thus, MUST
   * not be altered.
   * @param d_founds Device pointer to an (n)-sized array which indicates
   * whether the corresponding `d_values` slot is already filled or not. So, if
   * and only if `d_founds` is still false, the implementation shall attempt to
   * retrieve and fill in the value for the corresponding key. If a key/value
   * was retrieved successfully from external storage, the implementation MUST
   * also set `d_founds` to `true`.
   * @param stream Stream that MUST be used for queuing asynchronous CUDA
   * operations. If only the input arguments or resources obtained from
   * respectively `dev_mem_pool` and `host_mem_pool` are used for such
   * operations, it is not necessary to synchronize the stream prior to
   * returning from the function.
   */
  virtual size_type find(size_type n,
                         const key_type* d_keys,  // (n)
                         value_type* d_values,    // (n * value_dim)
                         size_type value_dims,
                         bool* d_founds,  // (n)
                         cudaStream_t stream) const = 0;

  /**
   * @brief Attempts to erase the entries associated with the supplied `d_keys`.
   * For keys do not exist nothing happens. It is permissible for this function
   * to be implemented asynchronously (i.e., to return before the actual
   * deletion has happened).
   *
   * @param n Number of keys provided in `d_keys` arguments.
   * @param d_keys Device pointer to an (n)-sized array of keys. This pointer is
   * only guarnteed to be valid for the duration of the call. If easure is
   * implemented asynchronously, you must make a copy and manage its lifetime
   * yourself.
   */
  virtual void erase(size_type n, const key_type* d_keys,
                     cudaStream_t stream) = 0;
};

}  // namespace merlin
}  // namespace nv
