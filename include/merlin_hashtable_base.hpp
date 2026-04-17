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

#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace nv {
namespace merlin {

struct HashTableOptions;
class BaseAllocator;

template <class K, class V, class S>
struct Bucket;

template <class K, class V, class S>
class BaseKVFile;

template <typename K, typename V, typename S = uint64_t>
class HashTableBase {
 public:
  using size_type = size_t;
  using key_type = K;
  using value_type = V;
  using score_type = S;
  using allocator_type = BaseAllocator;
  using bucket_type = nv::merlin::Bucket<K, V, S>;

 public:
  virtual ~HashTableBase() {}

  /**
   * @brief Initialize a merlin::HashTable.
   *
   * @param options The configuration options.
   */
  virtual void init(const HashTableOptions& options,
                    allocator_type* allocator = nullptr) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores conforms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void insert_or_assign(const size_type n,
                                const key_type* keys,                // (n)
                                const value_type* values,            // (n, DIM)
                                const score_type* scores = nullptr,  // (n)
                                cudaStream_t stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param d_evicted_counter The number of elements evicted on GPU-accessible
   * memory. @notice The caller should guarantee it is set to `0` before
   * calling.
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void insert_and_evict(const size_type n,
                                const key_type* keys,          // (n)
                                const value_type* values,      // (n, DIM)
                                const score_type* scores,      // (n)
                                key_type* evicted_keys,        // (n)
                                value_type* evicted_values,    // (n, DIM)
                                score_type* evicted_scores,    // (n)
                                size_type* d_evicted_counter,  // (1)
                                cudaStream_t stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   *
   * @return The number of elements evicted.
   */
  virtual size_type insert_and_evict(const size_type n,
                                     const key_type* keys,        // (n)
                                     const value_type* values,    // (n, DIM)
                                     const score_type* scores,    // (n)
                                     key_type* evicted_keys,      // (n)
                                     value_type* evicted_values,  // (n, DIM)
                                     score_type* evicted_scores,  // (n)
                                     cudaStream_t stream = 0,
                                     bool unique_key = true,
                                     bool ignore_evict_strategy = false) = 0;

  /**
   * Searches for each key in @p keys in the hash table.
   * If the key is found and the corresponding value in @p accum_or_assigns is
   * `true`, the @p vectors_or_deltas is treated as a delta to the old
   * value, and the delta is added to the old value of the key.
   *
   * If the key is not found and the corresponding value in @p accum_or_assigns
   * is `false`, the @p vectors_or_deltas is treated as a new value and the
   * key-value pair is updated in the table directly.
   *
   * @note When the key is found and the value of @p accum_or_assigns is
   * `false`, or when the key is not found and the value of @p accum_or_assigns
   * is `true`, nothing is changed and this operation is ignored.
   * The algorithm assumes these situations occur while the key was modified or
   * removed by other processes just now.
   *
   * @param n The number of key-value-score tuples to process.
   * @param keys The keys to insert on GPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on GPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param scores The scores to insert on GPU-accessible memory with shape (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void accum_or_assign(const size_type n,
                               const key_type* keys,                // (n)
                               const value_type* value_or_deltas,   // (n, DIM)
                               const bool* accum_or_assigns,        // (n)
                               const score_type* scores = nullptr,  // (n)
                               cudaStream_t stream = 0,
                               bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Searches the hash table for the specified keys.
   * When a key is missing, the value in @p values and @p scores will be
   * inserted.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find_or_insert(const size_type n, const key_type* keys,  // (n)
                              value_type* values,            // (n * DIM)
                              score_type* scores = nullptr,  // (n)
                              cudaStream_t stream = 0, bool unique_key = true,
                              bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values. When a key is missing, the value in @p values and @p scores
   * will be inserted.
   *
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values  The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   * @param locked_key_ptrs If it isn't nullptr then the keys in the table will
   * be locked, and key's address will write to locked_key_ptrs. Using
   * unlock_keys to unlock these keys.
   *
   */
  virtual void find_or_insert(const size_type n, const key_type* keys,  // (n)
                              value_type** values,                      // (n)
                              bool* founds,                             // (n)
                              score_type* scores = nullptr,             // (n)
                              cudaStream_t stream = 0, bool unique_key = true,
                              bool ignore_evict_strategy = false,
                              key_type** locked_key_ptrs = nullptr) = 0;

  /**
   * @brief
   * This function will lock the keys in the table and unexisted keys will be
   * ignored.
   *
   * @param n The number of keys in the table to be locked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param succeededs The status that indicates if the lock operation is
   * succeed.
   * @param scores The scores of the input keys will set to scores if provided.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  virtual void lock_keys(const size_type n,
                         key_type const* keys,        // (n)
                         key_type** locked_key_ptrs,  // (n)
                         bool* succeededs = nullptr,  // (n)
                         cudaStream_t stream = 0,
                         score_type const* scores = nullptr) = 0;

  /**
   * @brief Using pointers to address the keys in the hash table and set them
   * to target keys.
   * This function will unlock the keys in the table which are locked by
   * the previous call to find_or_insert.
   *
   * @param n The number of keys in the table to be unlocked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param succeededs The status that indicates if the unlock operation is
   * succeed.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  virtual void unlock_keys(const size_type n,
                           key_type** locked_key_ptrs,  // (n)
                           const key_type* keys,        // (n)
                           bool* succeededs = nullptr,  // (n)
                           cudaStream_t stream = 0) = 0;

  /**
   * @brief Assign new key-value-score tuples into the hash table.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign(const size_type n,
                      const key_type* keys,                // (n)
                      const value_type* values,            // (n, DIM)
                      const score_type* scores = nullptr,  // (n)
                      cudaStream_t stream = 0, bool unique_key = true) = 0;

  /**
   * @brief Assign new scores for keys.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-score pairs to assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign_scores(const size_type n,
                             const key_type* keys,                // (n)
                             const score_type* scores = nullptr,  // (n)
                             cudaStream_t stream = 0,
                             bool unique_key = true) = 0;

  /**
   * @brief Alias of `assign_scores`.
   */
  virtual void assign(const size_type n,
                      const key_type* keys,                // (n)
                      const score_type* scores = nullptr,  // (n)
                      cudaStream_t stream = 0, bool unique_key = true) = 0;

  /**
   * @brief Assign new values for each keys .
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value pairs to assign.
   * @param keys The keys need to be operated, which must be on GPU-accessible
   * memory with shape (n).
   * @param values The values need to be updated, which must be on
   * GPU-accessible memory with shape (n, DIM).
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign_values(const size_type n,
                             const key_type* keys,      // (n)
                             const value_type* values,  // (n, DIM)
                             cudaStream_t stream = 0,
                             bool unique_key = true) = 0;
  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type* values,                       // (n, DIM)
                    bool* founds,                             // (n)
                    score_type* scores = nullptr,             // (n)
                    cudaStream_t stream = 0) const = 0;

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When the searched keys are not hit, missed keys/indices/size can be
   * obtained.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param missed_keys The missed keys to search on GPU-accessible memory with
   * shape (n).
   * @param missed_indices The missed indices to search on GPU-accessible memory
   * with shape (n).
   * @param missed_size The size of `missed_keys` and `missed_indices`.
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type* values,                       // (n, DIM)
                    key_type* missed_keys,                    // (n)
                    int* missed_indices,                      // (n)
                    int* missed_size,                         // scalar
                    score_type* scores = nullptr,             // (n)
                    cudaStream_t stream = 0) const = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type** values,                      // (n)
                    bool* founds,                             // (n)
                    score_type* scores = nullptr,             // (n)
                    cudaStream_t stream = 0, bool unique_key = true) const = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values, and will update the scores.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find_and_update(const size_type n, const key_type* keys,  // (n)
                               value_type** values,                      // (n)
                               bool* founds,                             // (n)
                               score_type* scores = nullptr,             // (n)
                               cudaStream_t stream = 0,
                               bool unique_key = true) = 0;

  /**
   * @brief Checks if there are elements with key equivalent to `keys` in the
   * table.
   *
   * @param n The number of `keys` to check.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param founds The result that indicates if the keys are found, and should
   * be allocated by caller on GPU-accessible memory with shape (n).
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  virtual void contains(const size_type n, const key_type* keys,  // (n)
                        bool* founds,                             // (n)
                        cudaStream_t stream = 0) const = 0;

  /**
   * @brief Removes specified elements from the hash table.
   *
   * @param n The number of keys to remove.
   * @param keys The keys to remove on GPU-accessible memory.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  virtual void erase(const size_type n, const key_type* keys,
                     cudaStream_t stream = 0) = 0;

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  virtual void clear(cudaStream_t stream = 0) = 0;

  /**
   * @brief Exports a certain number of the key-value-score tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter Accumulates amount of successfully exported values.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw CudaException If the key-value size is too large for GPU shared
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  virtual void export_batch(size_type n, const size_type offset,
                            size_type* d_counter,          // (1)
                            key_type* keys,                // (n)
                            value_type* values,            // (n, DIM)
                            score_type* scores = nullptr,  // (n)
                            cudaStream_t stream = 0) const = 0;

  virtual size_type export_batch(const size_type n, const size_type offset,
                                 key_type* keys,                // (n)
                                 value_type* values,            // (n, DIM)
                                 score_type* scores = nullptr,  // (n)
                                 cudaStream_t stream = 0) const = 0;

  /**
   * @brief Indicates if the hash table has no elements.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return `true` if the table is empty and `false` otherwise.
   */
  virtual bool empty(cudaStream_t stream = 0) const = 0;

  /**
   * @brief Returns the hash table size.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return The table size.
   */
  virtual size_type size(cudaStream_t stream = 0) const = 0;

  /**
   * @brief Returns the hash table capacity.
   *
   * @note The value that is returned might be less than the actual capacity of
   * the hash table because the hash table currently keeps the capacity to be
   * a power of 2 for performance considerations.
   *
   * @return The table capacity.
   */
  virtual size_type capacity() const = 0;

  /**
   * @brief Sets the number of buckets to the number that is needed to
   * accommodate at least @p new_capacity elements without exceeding the maximum
   * load factor. This method rehashes the hash table. Rehashing puts the
   * elements into the appropriate buckets considering that total number of
   * buckets has changed.
   *
   * @note If the value of @p new_capacity or double of @p new_capacity is
   * greater or equal than `options_.max_capacity`, the reserve does not perform
   * any change to the hash table.
   *
   * @param new_capacity The requested capacity for the hash table.
   * @param stream The CUDA stream that is used to execute the operation.
   */
  virtual void reserve(const size_type new_capacity,
                       cudaStream_t stream = 0) = 0;

  /**
   * @brief Returns the average number of elements per slot, that is, size()
   * divided by capacity().
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The load factor
   */
  virtual float load_factor(cudaStream_t stream = 0) const = 0;

  /**
   * @brief Set max_capacity of the table.
   *
   * @param new_max_capacity The new expecting max_capacity. It must be power
   * of 2. Otherwise it will raise an error.
   */
  virtual void set_max_capacity(size_type new_max_capacity) = 0;

  /**
   * @brief Returns the dimension of the vectors.
   *
   * @return The dimension of the vectors.
   */
  virtual size_type dim() const noexcept = 0;

  /**
   * @brief Returns The length of each bucket.
   *
   * @return The length of each bucket.
   */
  virtual size_type max_bucket_size() const noexcept = 0;

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the table.
   */
  virtual size_type bucket_count() const noexcept = 0;

  /**
   * @brief Save keys, vectors, scores in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param max_workspace_size Saving is conducted in chunks. This value denotes
   * the maximum amount of temporary memory to use when dumping the table.
   * Larger values *can* lead to higher performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  virtual size_type save(BaseKVFile<K, V, S>* file,
                         const size_t max_workspace_size = 1L * 1024 * 1024,
                         cudaStream_t stream = 0) const = 0;

  /**
   * @brief Load keys, vectors, scores from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param max_workspace_size Loading is conducted in chunks. This value
   * denotes the maximum size of such chunks. Larger values *can* lead to higher
   * performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  virtual size_type load(BaseKVFile<K, V, S>* file,
                         const size_t max_workspace_size = 1L * 1024 * 1024,
                         cudaStream_t stream = 0) = 0;

  virtual void set_global_epoch(const uint64_t epoch) = 0;
};

}  // namespace merlin
}  // namespace nv
