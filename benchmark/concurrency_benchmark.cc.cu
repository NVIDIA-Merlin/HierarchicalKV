/*
 * Concurrency Mode Comparison Benchmark (E7v2)
 *
 * Compares triple-group concurrency (R/U/I) vs. conventional R/W lock
 * under mixed workloads of find, assign, and insert_or_assign.
 *
 * E7v2 changes from E7:
 *   - dim: 32 -> 128         (heavier kernels)
 *   - batch_size: 64K -> 64K (unchanged - lock overhead dominates at smaller)
 *   - Added assign scalability sweep: 1/2/5/10/20 threads
 *   - Per-batch sync retained (lock internals already sync per call)
 *
 * Build with -DUSE_RW_LOCK=ON to simulate R/W lock (assign takes
 * exclusive lock), or -DUSE_RW_LOCK=OFF for triple-group (default).
 *
 * Usage: ./concurrency_benchmark <triple_group|rw_lock>
 * Output: CSV to stdout, progress to stderr.
 */

#include <assert.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;
using HKVTable =
    nv::merlin::HashTable<K, V, S, EvictStrategy::kLru, nv::merlin::Sm80>;

/* ---- experiment knobs (Exp #3e, paper §5.4) ----
 * Paper specifies dim=16, 64K keys/batch, 200 batches/thread,
 * lambda=0.75 (see s5-evaluation.tex line 282).
 */
static constexpr int BATCHES_PER_THREAD = 200;
static constexpr size_t BATCH_SIZE = 64UL * 1024;  // 64K keys per batch
static constexpr size_t DIM = 16;                  // paper-spec dim
static constexpr size_t INIT_CAPACITY = 128UL * 1024 * 1024;  // 128M
static constexpr size_t HBM_GB = 16;
static constexpr float LOAD_FACTOR = 0.75f;
static constexpr float EPSILON = 0.001f;

/* ---- operation types ---- */
enum OpType { OP_FIND = 0, OP_ASSIGN = 1, OP_INSERT = 2 };

struct WorkloadMix {
  const char* name;
  int find_threads;
  int assign_threads;
  int insert_threads;
};

/* ---- per-thread result ---- */
struct ThreadResult {
  OpType op;
  size_t total_keys;
  double elapsed_s;
};

/* ---- barrier for synchronized start ---- */
struct Barrier {
  std::mutex mtx;
  std::condition_variable cv;
  int count;
  int target;
  bool released;

  explicit Barrier(int n) : count(0), target(n), released(false) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lk(mtx);
    count++;
    if (count >= target) {
      released = true;
      cv.notify_all();
    } else {
      cv.wait(lk, [this] { return released; });
    }
  }
};

/* ---- populate table to target load factor ---- */
void populate_table(std::shared_ptr<HKVTable>& table, cudaStream_t stream) {
  const size_t pop_batch = 1024 * 1024;
  K* h_keys;
  S* h_scores;

  CUDA_CHECK(cudaMallocHost(&h_keys, pop_batch * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, pop_batch * sizeof(S)));

  K* d_keys;
  V* d_vectors;
  CUDA_CHECK(cudaMalloc(&d_keys, pop_batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, pop_batch * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, pop_batch * sizeof(V) * DIM));

  uint64_t target = static_cast<uint64_t>(INIT_CAPACITY * LOAD_FACTOR);
  uint64_t inserted = 0;
  K start = 0;
  int epoch = 0;

  while (inserted < target) {
    uint64_t batch = std::min(pop_batch, target - inserted);
    benchmark::create_continuous_keys<K, S>(h_keys, h_scores, batch, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, batch * sizeof(K), cudaMemcpyHostToDevice));
    table->set_global_epoch(epoch++);
    table->insert_or_assign(batch, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += batch;
    inserted += batch;
  }

  /* fine-tune load factor */
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (LOAD_FACTOR - real_lf > EPSILON) {
    int64_t append =
        static_cast<int64_t>((LOAD_FACTOR - real_lf) * INIT_CAPACITY);
    if (append <= 0) break;
    append = std::min(static_cast<int64_t>(pop_batch), append);
    benchmark::create_continuous_keys<K, S>(h_keys, h_scores, append, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, append * sizeof(K), cudaMemcpyHostToDevice));
    table->insert_or_assign(append, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += append;
    real_lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  fprintf(stderr, "  Populated: size=%zu, load_factor=%.4f\n",
          table->size(stream), real_lf);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
}

/* ---- worker thread function ----
 *
 * Design: Each thread pre-allocates a small pool of GPU buffers
 * (BUFFER_POOL_SIZE slots) and pre-copies random keys into them.
 * During the timed region, threads rotate through the pool via
 * modular indexing. Since keys are random, reuse across pool slots
 * is fine for throughput measurement — this avoids OOM when
 * batches * batch_size * dim is large.
 */
static constexpr int BUFFER_POOL_SIZE = 4;

void worker_thread(HKVTable* table, OpType op, int batches, size_t batch_size,
                   size_t dim, uint64_t key_range, uint64_t insert_start,
                   int thread_id, Barrier* barrier, ThreadResult* result) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  /* Allocate a small pool of device buffers */
  const int pool = std::min(BUFFER_POOL_SIZE, batches);
  std::vector<K*> d_keys_vec(pool);
  std::vector<V*> d_vectors_vec(pool);
  std::vector<bool*> d_found_vec(pool);

  for (int b = 0; b < pool; b++) {
    CUDA_CHECK(cudaMalloc(&d_keys_vec[b], batch_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors_vec[b], batch_size * sizeof(V) * dim));
    CUDA_CHECK(cudaMemset(d_vectors_vec[b], 1, batch_size * sizeof(V) * dim));
    if (op == OP_FIND) {
      CUDA_CHECK(cudaMalloc(&d_found_vec[b], batch_size * sizeof(bool)));
    }
  }

  /* Pre-generate and upload keys for pool slots */
  K* h_keys;
  CUDA_CHECK(cudaMallocHost(&h_keys, batch_size * sizeof(K)));
  std::mt19937_64 rng(42 + thread_id);
  std::uniform_int_distribution<K> exist_dist(0, key_range - 1);
  K insert_key_base =
      insert_start + static_cast<K>(thread_id) * batches * batch_size;

  for (int b = 0; b < pool; b++) {
    switch (op) {
      case OP_FIND:
      case OP_ASSIGN: {
        for (size_t i = 0; i < batch_size; i++) {
          h_keys[i] = exist_dist(rng);
        }
        break;
      }
      case OP_INSERT: {
        for (size_t i = 0; i < batch_size / 2; i++) {
          h_keys[i] = exist_dist(rng);
        }
        for (size_t i = batch_size / 2; i < batch_size; i++) {
          h_keys[i] = insert_key_base++;
        }
        break;
      }
    }
    CUDA_CHECK(cudaMemcpy(d_keys_vec[b], h_keys, batch_size * sizeof(K),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaDeviceSynchronize());

  /* Wait for all threads to be ready */
  barrier->arrive_and_wait();

  auto t_start = std::chrono::high_resolution_clock::now();

  size_t total_keys = 0;
  for (int b = 0; b < batches; b++) {
    int slot = b % pool;
    switch (op) {
      case OP_FIND: {
        table->find(batch_size, d_keys_vec[slot], d_vectors_vec[slot],
                    d_found_vec[slot], nullptr, stream);
        break;
      }
      case OP_ASSIGN: {
        table->assign(batch_size, d_keys_vec[slot], d_vectors_vec[slot],
                      nullptr, stream);
        break;
      }
      case OP_INSERT: {
        table->insert_or_assign(batch_size, d_keys_vec[slot],
                                d_vectors_vec[slot], nullptr, stream);
        break;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    total_keys += batch_size;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  result->op = op;
  result->total_keys = total_keys;
  result->elapsed_s = elapsed;

  /* Cleanup */
  for (int b = 0; b < pool; b++) {
    CUDA_CHECK(cudaFree(d_keys_vec[b]));
    CUDA_CHECK(cudaFree(d_vectors_vec[b]));
    if (op == OP_FIND) {
      CUDA_CHECK(cudaFree(d_found_vec[b]));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/* ---- run one workload mix ---- */
void run_workload(const char* mode_name, const WorkloadMix& mix) {
  fprintf(stderr, "\n=== Workload: %s (%s) ===\n", mix.name, mode_name);
  fprintf(stderr, "  Threads: %d find, %d assign, %d insert\n",
          mix.find_threads, mix.assign_threads, mix.insert_threads);

  int total_threads =
      mix.find_threads + mix.assign_threads + mix.insert_threads;

  /* Create and populate table */
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
  options.api_lock = true; /* CRITICAL: enable API-level locking */

  auto table = std::make_shared<HKVTable>();
  table->init(options);

  cudaStream_t init_stream;
  CUDA_CHECK(cudaStreamCreate(&init_stream));
  populate_table(table, init_stream);

  uint64_t key_range = static_cast<uint64_t>(INIT_CAPACITY * LOAD_FACTOR);
  uint64_t insert_start = key_range + 1;

  /* Warmup: trigger kernel selection */
  {
    K* d_keys_w;
    V* d_vectors_w;
    bool* d_found_w;
    size_t warmup_n = 1;
    CUDA_CHECK(cudaMalloc(&d_keys_w, warmup_n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors_w, warmup_n * sizeof(V) * DIM));
    CUDA_CHECK(cudaMalloc(&d_found_w, warmup_n * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_keys_w, 0, warmup_n * sizeof(K)));
    CUDA_CHECK(cudaMemset(d_vectors_w, 0, warmup_n * sizeof(V) * DIM));

    for (int i = 0; i < 5; i++) {
      table->find(warmup_n, d_keys_w, d_vectors_w, d_found_w, nullptr,
                  init_stream);
      CUDA_CHECK(cudaStreamSynchronize(init_stream));
      table->assign(warmup_n, d_keys_w, d_vectors_w, nullptr, init_stream);
      CUDA_CHECK(cudaStreamSynchronize(init_stream));
      table->insert_or_assign(warmup_n, d_keys_w, d_vectors_w, nullptr,
                              init_stream);
      CUDA_CHECK(cudaStreamSynchronize(init_stream));
    }
    CUDA_CHECK(cudaFree(d_keys_w));
    CUDA_CHECK(cudaFree(d_vectors_w));
    CUDA_CHECK(cudaFree(d_found_w));
  }
  CUDA_CHECK(cudaStreamDestroy(init_stream));

  /* Prepare threads */
  std::vector<std::thread> threads;
  std::vector<ThreadResult> results(total_threads);
  Barrier barrier(total_threads);

  int tid = 0;
  for (int i = 0; i < mix.find_threads; i++, tid++) {
    threads.emplace_back(worker_thread, table.get(), OP_FIND,
                         BATCHES_PER_THREAD, BATCH_SIZE, DIM, key_range,
                         insert_start, tid, &barrier, &results[tid]);
  }
  for (int i = 0; i < mix.assign_threads; i++, tid++) {
    threads.emplace_back(worker_thread, table.get(), OP_ASSIGN,
                         BATCHES_PER_THREAD, BATCH_SIZE, DIM, key_range,
                         insert_start, tid, &barrier, &results[tid]);
  }
  for (int i = 0; i < mix.insert_threads; i++, tid++) {
    threads.emplace_back(worker_thread, table.get(), OP_INSERT,
                         BATCHES_PER_THREAD, BATCH_SIZE, DIM, key_range,
                         insert_start, tid, &barrier, &results[tid]);
  }

  /* Wall clock from main thread perspective */
  auto wall_start = std::chrono::high_resolution_clock::now();

  for (auto& t : threads) {
    t.join();
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_time =
      std::chrono::duration<double>(wall_end - wall_start).count();

  /* Compute aggregate results */
  size_t total_ops = 0;
  size_t find_ops = 0, assign_ops = 0, insert_ops = 0;

  for (int i = 0; i < total_threads; i++) {
    total_ops += results[i].total_keys;
    switch (results[i].op) {
      case OP_FIND:
        find_ops += results[i].total_keys;
        break;
      case OP_ASSIGN:
        assign_ops += results[i].total_keys;
        break;
      case OP_INSERT:
        insert_ops += results[i].total_keys;
        break;
    }
  }

  double max_thread_time = 0;
  for (int i = 0; i < total_threads; i++) {
    max_thread_time = std::max(max_thread_time, results[i].elapsed_s);
  }

  double agg_throughput = total_ops / max_thread_time / 1e9;
  double find_tput = (find_ops > 0) ? find_ops / max_thread_time / 1e9 : 0.0;
  double assign_tput =
      (assign_ops > 0) ? assign_ops / max_thread_time / 1e9 : 0.0;
  double insert_tput =
      (insert_ops > 0) ? insert_ops / max_thread_time / 1e9 : 0.0;

  fprintf(stderr,
          "  Wall time: %.3f s, Total ops: %zu, Throughput: %.4f B-KV/s\n",
          max_thread_time, total_ops, agg_throughput);
  fprintf(stderr, "  Per-role: find=%.4f, assign=%.4f, insert=%.4f B-KV/s\n",
          find_tput, assign_tput, insert_tput);

  /* CSV output to stdout */
  printf("%s,%s,%d,%zu,%.6f,%.6f,%zu,%.6f,%zu,%.6f,%zu,%.6f\n", mode_name,
         mix.name, total_threads, total_ops, max_thread_time, agg_throughput,
         find_ops, find_tput, assign_ops, assign_tput, insert_ops, insert_tput);
  fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <triple_group|rw_lock>\n", argv[0]);
    return 1;
  }

  const char* mode_name = argv[1];

  CUDA_CHECK(cudaSetDevice(0));

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  fprintf(stderr, "GPU: %s (SM %d.%d)\n", props.name, props.major, props.minor);
  fprintf(stderr, "Mode: %s\n", mode_name);
  fprintf(stderr, "Config: dim=%zu, capacity=%zuM, HBM=%zuGB, lambda=%.2f\n",
          DIM, INIT_CAPACITY / (1024 * 1024), HBM_GB, LOAD_FACTOR);
  fprintf(stderr, "Threads=variable, Batches/thread=%d, Batch size=%zuK\n",
          BATCHES_PER_THREAD, BATCH_SIZE / 1024);

  /* CSV header */
  printf(
      "mode,workload,threads,total_ops,wall_time_s,throughput_bkvs,"
      "find_ops,find_bkvs,assign_ops,assign_bkvs,"
      "insert_ops,insert_bkvs\n");

  WorkloadMix mixes[] = {
      /* Original E7 mixed workloads (10 threads) */
      {"read_heavy", 8, 1, 1},
      {"update_heavy", 4, 5, 1},
      {"insert_heavy", 4, 2, 4},
      /* Assign-only scalability sweep: key experiment */
      {"assign_only_1t", 0, 1, 0},
      {"assign_only_2t", 0, 2, 0},
      {"assign_only_5t", 0, 5, 0},
      {"assign_only_10t", 0, 10, 0},
      /* Mixed: assign + insert contention */
      {"assign5_insert5", 0, 5, 5},
  };

  for (const auto& mix : mixes) {
    try {
      run_workload(mode_name, mix);
    } catch (const nv::merlin::CudaException& e) {
      fprintf(stderr, "CUDA error in workload %s: %s\n", mix.name, e.what());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  fprintf(stderr, "\nALL_DONE\n");
  return 0;
}
