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

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

using namespace nv::merlin;
using namespace benchmark;

enum class Test_Mode {
  pure_hbm = 0,
  hybrid = 1,
};

const float EPSILON = 0.001f;

std::string rep(int n) { return std::string(n, ' '); }

using K = uint64_t;
using S = uint64_t;
using V = float;
using MerlinHashTable = nv::merlin::HashTable<K, V, S>;
using TableOptions = nv::merlin::HashTableOptions;

float test_one_api(std::shared_ptr<MerlinHashTable>& table,
                   const API_Select api, const size_t dim,
                   const size_t init_capacity, const size_t key_num_per_op,
                   const float load_factor, const float hitrate = 0.6f) {
  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * dim));
  CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, key_num_per_op * sizeof(V) * dim));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  K* d_keys_out;

  K* d_evict_keys;
  S* d_evict_scores;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, key_num_per_op * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, key_num_per_op * sizeof(K)));

  CUDA_CHECK(cudaMalloc(&d_evict_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_scores, key_num_per_op * sizeof(S)));

  CUDA_CHECK(cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, key_num_per_op * sizeof(V) * dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // initialize insert
  // step 1, no need to load load_factor
  uint64_t key_num_init = static_cast<uint64_t>(init_capacity * load_factor);
  const float target_load_factor = key_num_init * 1.0f / init_capacity;
  uint64_t key_num_remain = key_num_init % key_num_per_op == 0
                                ? key_num_per_op
                                : key_num_init % key_num_per_op;
  int32_t loop_num_init = (key_num_init + key_num_per_op - 1) / key_num_per_op;

  K start = 0UL;

  S threshold = benchmark::host_nano<S>();
  for (int i = 0; i < loop_num_init; i++) {
    uint64_t key_num_cur_insert =
        i == loop_num_init - 1 ? key_num_remain : key_num_per_op;
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_cur_insert, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_cur_insert * sizeof(K),
                          cudaMemcpyHostToDevice));
    table->find_or_insert(key_num_cur_insert, d_keys, d_vectors_ptr, d_found,
                          d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    start += key_num_cur_insert;
  }

  // step 2
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * init_capacity);
    if (key_num_append <= 0) break;
    key_num_append =
        std::min(static_cast<int64_t>(key_num_per_op), key_num_append);
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_append, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_append * sizeof(K),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(key_num_append, d_keys, d_vectors, d_scores,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_append;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // For trigger the kernel selection in advance.
  int key_num_per_op_warmup = 1;
  for (int i = 0; i < 9; i++) {
    switch (api) {
      case API_Select::find: {
        table->find(key_num_per_op_warmup, d_keys, d_vectors, d_found, d_scores,
                    stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::insert_or_assign: {
        table->insert_or_assign(key_num_per_op_warmup, d_keys, d_vectors,
                                d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::find_or_insert: {
        table->find_or_insert(key_num_per_op_warmup, d_keys, d_vectors,
                              d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::assign: {
        table->assign(key_num_per_op_warmup, d_keys, d_def_val, d_scores,
                      stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::insert_and_evict: {
        table->insert_and_evict(key_num_per_op_warmup, d_keys, d_vectors,
                                d_scores, d_evict_keys, d_def_val,
                                d_evict_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::find_ptr: {
        V** d_vectors_ptr = nullptr;
        CUDA_CHECK(
            cudaMalloc(&d_vectors_ptr, key_num_per_op_warmup * sizeof(V*)));
        benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
                             key_num_per_op_warmup, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        table->find(1, d_keys, d_vectors_ptr, d_found, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim,
                                 key_num_per_op_warmup, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_vectors_ptr));
        break;
      }
      case API_Select::find_or_insert_ptr: {
        V** d_vectors_ptr = nullptr;
        bool* d_found;
        CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op_warmup * sizeof(bool)));
        CUDA_CHECK(
            cudaMalloc(&d_vectors_ptr, key_num_per_op_warmup * sizeof(V*)));
        benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
                             key_num_per_op_warmup, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        table->find_or_insert(key_num_per_op_warmup, d_keys, d_vectors_ptr,
                              d_found, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_vectors_ptr));
        CUDA_CHECK(cudaFree(d_found));
        break;
      }
      case API_Select::export_batch: {
        size_t* d_dump_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

        table->export_batch(key_num_per_op_warmup, 0, d_dump_counter, d_keys,
                            d_vectors, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_dump_counter));
        break;
      }
      case API_Select::export_batch_if: {
        size_t* d_dump_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
        K pattern = 0;
        table->template export_batch_if<ExportIfPredFunctor>(
            pattern, threshold, key_num_per_op_warmup, 0, d_dump_counter,
            d_keys, d_vectors, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_dump_counter));
        break;
      }
      default: {
        std::cout << "[Unsupport API]\n";
      }
    }
  }
  create_keys_for_hitrate<K, S>(h_keys, h_scores, key_num_per_op, hitrate,
                                Hit_Mode::last_insert, start, true /*reset*/);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                        cudaMemcpyHostToDevice));
  auto timer = benchmark::Timer<double>();
  switch (api) {
    case API_Select::find: {
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors, d_found, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(key_num_per_op, d_keys, d_vectors, d_scores,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors, d_scores,
                            stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::assign: {
      timer.start();
      table->assign(key_num_per_op, d_keys, d_def_val, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_scores,
                              d_evict_keys, d_def_val, d_evict_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::find_ptr: {
      V** d_vectors_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(V*)));
      benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
                           stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors_ptr, d_found, d_scores,
                  stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
                               stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_vectors_ptr));
      break;
    }
    case API_Select::find_or_insert_ptr: {
      V** d_vectors_ptr = nullptr;
      bool* d_found;
      CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
      CUDA_CHECK(cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(V*)));
      benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
                           stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors_ptr, d_found,
                            d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      CUDA_CHECK(cudaFree(d_vectors_ptr));
      CUDA_CHECK(cudaFree(d_found));
      break;
    }
    case API_Select::export_batch: {
      size_t* d_dump_counter;

      // Try to export close to but less than `key_num_per_op` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
      timer.start();
      table->export_batch(key_num_per_op / target_load_factor * safe_ratio, 0,
                          d_dump_counter, d_keys, d_vectors, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      CUDA_CHECK(cudaFree(d_dump_counter));
      break;
    }
    case API_Select::export_batch_if: {
      size_t* d_dump_counter;

      // Try to export close to but less than `key_num_per_op` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
      timer.start();
      K pattern = 0;
      table->template export_batch_if<ExportIfPredFunctor>(
          pattern, threshold, key_num_per_op / target_load_factor * safe_ratio,
          0, d_dump_counter, d_keys, d_vectors, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      CUDA_CHECK(cudaFree(d_dump_counter));
      break;
    }
    default: {
      std::cout << "[Unsupport API]\n";
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_scores));

  CUDA_CHECK(cudaDeviceSynchronize());
  CudaCheckError();

  float througput =
      key_num_per_op / timer.getResult() / (1024 * 1024 * 1024.0f);
  return througput;
}

static Test_Mode test_mode = Test_Mode::pure_hbm;

void print_title_a() {
  cout << endl
       << "|    \u03BB "
       << "| insert_or_assign "
       << "|   find "
       << "| find_or_insert "
       << "| assign "
       << "|  find* "
       << "| find_or_insert* ";
  if (Test_Mode::pure_hbm == test_mode) {
    cout << "| insert_and_evict ";
  }
  cout << "|\n";

  //<< "| load_factor "
  cout << "|-----:"
       //<< "| insert_or_assign "
       << "|-----------------:"
       //<< "|   find "
       << "|-------:"
       //<< "| find_or_insert "
       << "|---------------:"
       //<< "| assign "
       << "|-------:"
       //<< "|   find* "
       << "|-------:"
       //<< "| find_or_insert* "
       << "|----------------:";
  if (Test_Mode::pure_hbm == test_mode) {
    //<< "| insert_and_evict "
    cout << "|-----------------:";
  }
  cout << "|\n";
}

void print_title_b() {
  cout << endl
       << "|    \u03BB "
       << "| export_batch "
       << "| export_batch_if ";
  cout << "|\n";

  //<< "| load_factor "
  cout << "|-----:"
       //<< "| export_batch "
       << "|-------------:"
       //<< "| export_batch_if "
       << "|----------------:";
  cout << "|\n";
}

void test_main(std::vector<API_Select>& apis, const size_t dim,
               const size_t init_capacity = 64 * 1024 * 1024UL,
               const size_t key_num_per_op = 1 * 1024 * 1024UL,
               const size_t hbm4values = 16, const float load_factor = 1.0f,
               const bool io_by_cpu = false,
               const std::vector<float> load_factors = {0.50f, 0.75f, 1.00f}) {
  size_t free, total;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  if (free / (1 << 30) < hbm4values) {
    std::cout << "free HBM is not enough, ignore current benchmark!"
              << std::endl;
    return;
  }
  TableOptions options;

  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(hbm4values);
  options.io_by_cpu = io_by_cpu;
  options.evict_strategy = EvictStrategy::kLru;

  std::shared_ptr<MerlinHashTable> table = std::make_shared<MerlinHashTable>();
  table->init(options);

  for (float load_factor : load_factors) {
    std::cout << "|" << rep(1) << fixed << setprecision(2) << load_factor
              << " ";

    for (auto api : apis) {
      table->clear();
      CUDA_CHECK(cudaDeviceSynchronize());
      // There is a sampling of load_factor after several times call to target
      // API. Two consecutive calls can avoid the impact of sampling.
      auto res1 = test_one_api(table, api, dim, init_capacity, key_num_per_op,
                               load_factor);
      auto res2 = test_one_api(table, api, dim, init_capacity, key_num_per_op,
                               load_factor);
      auto res = std::max(res1, res2);
      std::cout << "|";
      switch (api) {
        case API_Select::find: {
          std::cout << rep(1);
          break;
        }
        case API_Select::insert_or_assign: {
          std::cout << rep(11);
          break;
        }
        case API_Select::find_or_insert: {
          std::cout << rep(9);
          break;
        }
        case API_Select::assign: {
          std::cout << rep(1);
          break;
        }
        case API_Select::insert_and_evict: {
          std::cout << rep(11);
          break;
        }
        case API_Select::find_ptr: {
          std::cout << rep(1);
          break;
        }
        case API_Select::find_or_insert_ptr: {
          std::cout << rep(10);
          break;
        }
        case API_Select::export_batch: {
          std::cout << rep(7);
          break;
        }
        case API_Select::export_batch_if: {
          std::cout << rep(10);
          break;
        }
        default: {
          std::cout << "[Unsupport API]";
        }
      }
      std::cout << fixed << setprecision(3) << setw(6) << setfill(' ') << res
                << " ";
    }
    std::cout << "|\n";
  }
}

int main() {
  size_t key_num_per_op = 1 * 1024 * 1024UL;
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  cout << endl
       << "## Benchmark" << endl
       << endl
       << "* GPU: 1 x " << props.name << ": " << props.major << "."
       << props.minor << endl
       << "* Key Type = uint64_t" << endl
       << "* Value Type = float32 * {dim}" << endl
       << "* Key-Values per OP = " << key_num_per_op << endl
       << "* Evict strategy: LRU" << endl
       << "* `\u03BB`"
       << ": load factor" << endl
       << "* `find*` means the `find` API that directly returns the addresses "
          "of values."
       << endl
       << "* `find_or_insert*` means the `find_or_insert` API that directly "
          "returns the addresses of values."
       << endl
       << "* ***Throughput Unit: Billion-KV/second***" << endl
       << endl;
  auto print_configuration = [](const size_t dim, const size_t init_capacity,
                                const size_t hbm4values) {
    using V = float;
    int32_t capacity = static_cast<int32_t>(init_capacity / (1024 * 1024));
    size_t hmem4values = init_capacity * dim * sizeof(V) / (1024 * 1024 * 1024);
    hmem4values = hmem4values < hbm4values ? 0 : (hmem4values - hbm4values);
    cout << "\n* dim = " << dim << ", "
         << "capacity = " << capacity << " Million-KV, "
         << "HBM = " << hbm4values << " GB, "
         << "HMEM = " << hmem4values << " GB\n";
  };

  try {
    {
      std::vector<API_Select> apis_a{
          API_Select::insert_or_assign, API_Select::find,
          API_Select::find_or_insert,   API_Select::assign,
          API_Select::find_ptr,         API_Select::find_or_insert_ptr,
          API_Select::insert_and_evict};

      std::vector<API_Select> apis_b{API_Select::export_batch,
                                     API_Select::export_batch_if};
      test_mode = Test_Mode::pure_hbm;

      cout << "### On pure HBM mode: " << endl;
      print_configuration(8, 128 * 1024 * 1024UL, 4);
      print_title_a();
      test_main(apis_a, 8, 128 * 1024 * 1024UL, key_num_per_op, 4);

      print_title_b();
      test_main(apis_b, 8, 128 * 1024 * 1024UL, key_num_per_op, 4);

      print_configuration(32, 128 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 32, 128 * 1024 * 1024UL, key_num_per_op, 16);

      print_title_b();
      test_main(apis_b, 32, 128 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(64, 64 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 64, 64 * 1024 * 1024UL, key_num_per_op, 16);

      print_title_b();
      test_main(apis_b, 64, 64 * 1024 * 1024UL, key_num_per_op, 16);

      cout << endl;
    }

    {
      std::vector<API_Select> apis_a{
          API_Select::insert_or_assign, API_Select::find,
          API_Select::find_or_insert,   API_Select::assign,
          API_Select::find_ptr,         API_Select::find_or_insert_ptr};

      std::vector<API_Select> apis_b{API_Select::export_batch,
                                     API_Select::export_batch_if};

      cout << "### On HBM+HMEM hybrid mode: " << endl;
      test_mode = Test_Mode::hybrid;
      print_configuration(64, 128 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 64, 128 * 1024 * 1024UL, key_num_per_op, 16);

      print_title_b();
      test_main(apis_b, 64, 128 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(64, 512 * 1024 * 1024UL, 32);
      print_title_a();
      test_main(apis_a, 64, 512 * 1024 * 1024UL, key_num_per_op, 32);

      print_title_b();
      test_main(apis_b, 64, 512 * 1024 * 1024UL, key_num_per_op, 32);
      cout << endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
