#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __device__ bool operator()(const K& key, S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score < threshold;
  }
};

void test_export_batch_if_with_limited_size() {
  constexpr uint64_t CAP = 1llu << 24;
  size_t n0 = (1llu << 23) - 163;
  size_t n1 = (1llu << 23) + 221;
  size_t n2 = (1llu << 23) - 17;
  size_t dim = 64;
  size_t table_size = 0;
  i64 pattern = 0;
  u64 threshold = 40;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  TableOptions options;
  options.init_capacity = CAP;
  options.max_capacity = CAP;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(100);
  using Table =
      nv::merlin::HashTable<i64, f32, u64, EvictStrategy::kCustomized>;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  size_t* d_cnt = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_cnt, sizeof(size_t), stream));
  CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, sizeof(size_t), stream));

  test_util::KVMSBuffer<i64, f32, u64> buffer0;
  buffer0.Reserve(n0, dim, stream);
  buffer0.ToRange(0, 1, stream);
  buffer0.Setscore((u64)15, stream);
  {
    test_util::KVMSBuffer<i64, f32, u64> buffer0_ev;
    buffer0_ev.Reserve(n0, dim, stream);
    buffer0_ev.ToZeros(stream);
    // table->insert_or_assign(n0, buffer0.keys_ptr(), buffer0.values_ptr(),
    //                         buffer0.scores_ptr(), stream, true, false);
    table->insert_and_evict(n0, buffer0.keys_ptr(), buffer0.values_ptr(),
                            buffer0.scores_ptr(), buffer0_ev.keys_ptr(),
                            buffer0_ev.values_ptr(), buffer0_ev.scores_ptr(),
                            d_cnt, stream, true, false);
    table_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MERLIN_EXPECT_TRUE(table_size == n0, "Invalid table size.");
  }

  test_util::KVMSBuffer<i64, f32, u64> buffer1;
  buffer1.Reserve(n1, dim, stream);
  buffer1.ToRange(n0, 1, stream);
  buffer1.Setscore((u64)30, stream);
  {
    test_util::KVMSBuffer<i64, f32, u64> buffer1_ev;
    buffer1_ev.Reserve(n0, dim, stream);
    buffer1_ev.ToZeros(stream);
    // table->insert_or_assign(n1, buffer1.keys_ptr(), buffer1.values_ptr(),
    //                         buffer1.scores_ptr(), stream, true, false);
    table->insert_and_evict(n0, buffer1.keys_ptr(), buffer1.values_ptr(),
                            buffer1.scores_ptr(), buffer1_ev.keys_ptr(),
                            buffer1_ev.values_ptr(), buffer1_ev.scores_ptr(),
                            d_cnt, stream, true, false);
    table_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  test_util::KVMSBuffer<i64, f32, u64> buffer2;
  buffer2.Reserve(n2, dim, stream);
  buffer2.ToRange(n0 + n1, 1, stream);
  buffer2.Setscore((u64)45, stream);
  {
    test_util::KVMSBuffer<i64, f32, u64> buffer2_ev;
    buffer2_ev.Reserve(n0, dim, stream);
    buffer2_ev.ToZeros(stream);
    // table->insert_or_assign(n2, buffer2.keys_ptr(), buffer2.values_ptr(),
    //                         buffer2.scores_ptr(), stream, true, false);
    table->insert_and_evict(n0, buffer2.keys_ptr(), buffer2.values_ptr(),
                            buffer2.scores_ptr(), buffer2_ev.keys_ptr(),
                            buffer2_ev.values_ptr(), buffer2_ev.scores_ptr(),
                            d_cnt, stream, true, false);
    table_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("final size: %zu, capacity: %zu\n", table_size, table->capacity());
  }

  size_t h_cnt = 0;
  size_t h_cnt2 = 0;

  table->size_if<ExportIfPredFunctor>(pattern, threshold, d_cnt, stream);
  CUDA_CHECK(cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(size_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("---> check h_cnt from size_if kernel: %zu\n", h_cnt);

  test_util::KVMSBuffer<i64, f32, u64> buffer_out;
  buffer_out.Reserve(h_cnt, dim, stream);
  buffer_out.ToZeros(stream);

  CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, sizeof(size_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  bool use_pin = true;

  uint64_t t0 = test_util::getTimestamp();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  table->export_batch_if<ExportIfPredFunctor>(
      pattern, threshold, static_cast<size_t>(CAP), 0, d_cnt,
      buffer_out.keys_ptr(!use_pin), buffer_out.values_ptr(!use_pin),
      buffer_out.scores_ptr(!use_pin), stream);
  cudaEventRecord(stop);
  CUDA_CHECK(cudaMemcpyAsync(&h_cnt2, d_cnt, sizeof(size_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("final h_cnt2: %zu\n", h_cnt2);

  MERLIN_EXPECT_TRUE(
      h_cnt == h_cnt2,
      "size_if and export_batch_if get different matching count.");
  float cu_cost = 0;
  cudaEventElapsedTime(&cu_cost, start, stop);
  uint64_t t1 = test_util::getTimestamp();
  printf("final h_cnt2: %zu, cost: %zu, cu_cost: %f\n", h_cnt2, t1 - t0,
         cu_cost);

  if (!use_pin) {
    buffer_out.SyncData(false, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  uint64_t t2 = test_util::getTimestamp();
  printf("use_pin: %d. After sycn data of len: %zu, total cost: %zu\n", use_pin,
         h_cnt2, t2 - t0);

  std::unordered_map<i64, u64> record;
  for (size_t i = 0; i < h_cnt; i++) {
    i64 key = buffer_out.keys_ptr(false)[i];
    u64 score = buffer_out.scores_ptr(false)[i];
    MERLIN_EXPECT_TRUE(score < threshold, "");
    record[key] = score;
    for (int j = 0; j < dim; j++) {
      f32 value = buffer_out.values_ptr(false)[i * dim + j];
      MERLIN_EXPECT_TRUE(key == static_cast<i64>(value), "");
    }
  }
  MERLIN_EXPECT_TRUE(record.size() == h_cnt2, "");
  printf("record: %zu\n", record.size());
  printf("n0+n1: %zu\n", n0 + n1);
  printf("n0+n1+n2: %zu\n", n0 + n1 + n2);
  printf("done\n");
}

int main() {
  test_export_batch_if_with_limited_size();
  return 0;
}
