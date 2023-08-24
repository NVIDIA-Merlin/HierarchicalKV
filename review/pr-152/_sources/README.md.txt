# [NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV)

[![Version](https://img.shields.io/github/v/release/NVIDIA-Merlin/HierarchicalKV?color=orange&include_prereleases)](https://github.com/NVIDIA-Merlin/HierarchicalKV/releases)
[![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/HierarchicalKV)](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/HierarchicalKV/master/README.html)

## About HierarchicalKV

HierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements.

The key capability of HierarchicalKV is to store key-value (feature-embedding) on high-bandwidth memory (HBM) of GPUs and in host memory.

You can also use the library for generic key-value storage.

## Benefits

When building large recommender systems, machine learning (ML) engineers face the following challenges:

- GPUs are needed, but HBM on a single GPU is too small for the large DLRMs that scale to several terabytes.
- Improving communication performance is getting more difficult in larger and larger CPU clusters.
- It is difficult to efficiently control consumption growth of limited HBM with customized strategies.
- Most generic key-value libraries provide low HBM and host memory utilization.

HierarchicalKV alleviates these challenges and helps the machine learning engineers in RecSys with the following benefits:

- Supports training large RecSys models on **HBM and host memory** at the same time.
- Provides better performance by **full bypassing CPUs** and reducing the communication workload.
- Implements table-size restraint strategies that are based on **LRU or customized strategies**.
  The strategies are implemented by CUDA kernels.
- Operates at a high working-status load factor that is close to 1.0.


## Key ideas

- Buckets are locally ordered
- Store keys and values separately
- Store all the keys in HBM
- Build-in and customizable eviction strategy

HierarchicalKV makes NVIDIA GPUs more suitable for training large and super-large models of ***search, recommendations, and advertising***.
The library simplifies the common challenges to building, evaluating, and serving sophisticated recommenders models.

## API Documentation

The main classes and structs are below, but reading the comments in the source code is recommended:

- [`class HashTable`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L151)
- [`class EvictStrategy`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L52)
- [`struct HashTableOptions`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L60)

For regular API doc, please refer to [API Docs](https://nvidia-merlin.github.io/HierarchicalKV/master/api/index.html)

### API Maturity Matrix

`industry-validated` means the API has been well-tested and verified in at least one real-world scenario.

| Name                 | Description                                                                                                              | Function           |
|:---------------------|:-------------------------------------------------------------------------------------------------------------------------|:-------------------|
| __insert_or_assign__ | Insert or assign for the specified keys. <br>Overwrite one key with minimum score when bucket is full.                   | industry-validated |
| __insert_and_evict__ | Insert new keys, and evict keys with minimum score when bucket is full.                                                  | industry-validated |
| __find_or_insert__   | Search for the specified keys, and insert them when missed.                                                              | well-tested        |
| __assign__           | Update for each key and bypass when missed.                                                                              | well-tested        |
| __accum_or_assign__  | Search and update for each key. If found, add value as a delta to the original value. <br>If missed, update it directly. | well-tested        |
| __find_or_insert\*__ | Search for the specified keys and return the pointers of values. Insert them firstly when missing.                       | well-tested        |
| __find__             | Search for the specified keys.                                                                                           | industry-validated |
| __find\*__           | Search and return the pointers of values, thread-unsafe but with high performance.                                       | well-tested        |
| __export_batch__     | Exports a certain number of the key-value-score tuples.                                                                  | industry-validated |
| __export_batch_if__  | Exports a certain number of the key-value-score tuples which match specific conditions.                                  | industry-validated |
| __warmup__           | Move the hot key-values from HMEM to HBM                                                                                 | June 15, 2023      |


### Evict Strategy

The `score` is introduced to define the importance of each key, the larger, the more important, the less likely they will be evicted. Eviction only happens when a bucket is full.
The `score_type` must be `uint64_t`. For more detail, please refer to [`class EvictStrategy`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L52).

| Name           | Definition of `Score`                                                                                                                                                                                           |
|:---------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __Lru__        | Device clock, in nano second (could be a little difference from host clock).                                                                                                                                    |
| __Lfu__        | Frequency provided by caller via the input parameter of `scores` of `insert-like` APIs as the increment of frequency.                                                                                           |
| __EpochLru__   | The high 32bits is the global epoch provided via the input parameter of `global_epoch`, <br>the low 32bits is equal to `(device_clock >> 20) & 0xffffffff` with granularity close to 1 ms.                      |
| __EpochLfu__   | The high 32bits is the global epoch provided via the input parameter of `global_epoch`, <br>the low 32bits is the frequency, <br>the frequency will keep constant after reaching the max value of `0xffffffff`. |
| __Customized__ | Fully provided by the caller via the input parameter of `scores` of `insert-like` APIs.                                                                                                                         |


* __Note__:
  - The `insert-like` APIs mean the APIs of `insert_or_assign`, `insert_and_evict`, `find_or_insert`, `accum_or_assign`, and `find_or_insert`. 
  - The `global_epoch` should be maintained by the caller and input as the input parameter of `insert-like` APIs.

### Configuration Options

It's recommended to keep the default configuration for the options ending with `*`.

| Name                    | Type            | Default | Description                                           |
|:------------------------|:----------------|:--------|:------------------------------------------------------|
| __init_capacity__       | size_t          | 0       | The initial capacity of the hash table.               |
| __max_capacity__        | size_t          | 0       | The maximum capacity of the hash table.               |
| __max_hbm_for_vectors__ | size_t          | 0       | The maximum HBM for vectors, in bytes.                |
| __dim__                 | size_t          | 64      | The dimension of the value vectors.                   |
| __max_bucket_size*__    | size_t          | 128     | The length of each bucket.                            |
| __max_load_factor*__    | float           | 0.5f    | The max load factor before rehashing.                 |
| __block_size*__         | int             | 128     | The default block size for CUDA kernels.              |
| __io_block_size*__      | int             | 1024    | The block size for IO CUDA kernels.                   |
| __device_id*__          | int             | -1      | The ID of device. Managed internally when set to `-1` |
| __io_by_cpu*__          | bool            | false   | The flag indicating if the CPU handles IO.            |

For more detail, please refer to [`struct HashTableOptions`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L60).

### How to use:

```cpp
#include "merlin_hashtable.cuh"


using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;

int main(int argc, char *argv[])
{
  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  
  // 1. Define the table and use LRU eviction strategy.
  using HKVTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
  std::unique_ptr<HKVTable> table = std::make_unique<HKVTable>();
  
  // 2. Define the configuration options.
  TableOptions options;
  options.init_capacity = 16 * 1024 * 1024;
  options.max_capacity = options.init_capacity;
  options.dim = 16;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  
  
  // 3. Initialize the table memory resource.
  table->init(options);
  
  // 4. Use table to do something.
  
  return 0;
}

```

### Usage restrictions

- The `key_type` must be `int64_t` or `uint64_t`.
- The `score_type` must be `uint64_t`.
- The keys of `0xFFFFFFFFFFFFFFFC`, `0xFFFFFFFFFFFFFFFD`, `0xFFFFFFFFFFFFFFFE`, and `0xFFFFFFFFFFFFFFFF` are reserved for internal using.

## Contributors

HierarchicalKV is co-maintianed by [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin) and NVIDIA product end-users,
and also open for public contributions, bug fixes, and documentation. [[Contribute](CONTRIBUTING.md)]

## How to build

Basically, HierarchicalKV is a headers only library, the commands below only create binaries for benchmark and unit testing.

### with cmake
```shell
git clone --recursive https://github.com/NVIDIA-Merlin/HierarchicalKV.git
cd HierarchicalKV && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -Dsm=80 .. && make -j
```

For Benchmark:
```shell
./merlin_hashtable_benchmark
```

For Unit Test:
```shell
./merlin_hashtable_test
```

### with bazel
```shell
git clone --recursive https://github.com/NVIDIA-Merlin/HierarchicalKV.git
cd HierarchicalKV && bazel build --config=cuda //...
```

For Benchmark:
```shell
./benchmark_util
```

Your environment must meet the following requirements:

- CUDA version >= 11.2
- NVIDIA GPU with compute capability 8.0, 8.6, 8.7 or 9.0

## Benchmark & Performance(W.I.P)

* GPU: 1 x NVIDIA A100 80GB PCIe: 8.0
* Key Type = uint64_t
* Value Type = float32 * {dim}
* Key-Values per OP = 1048576
* Evict strategy: LRU
* `λ`: load factor
* `find*` means the `find` API that directly returns the addresses of values.
* `find_or_insert*` means the `find_or_insert` API that directly returns the addresses of values.
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode: 

* dim = 8, capacity = 128 Million-KV, HBM = 4 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            1.153 |  2.580 |          1.706 |  1.798 |  4.111 |           1.801 |            1.083 |
| 0.75 |            1.010 |  2.561 |          0.683 |  1.863 |  1.939 |           1.305 |            0.994 |
| 1.00 |            0.366 |  2.580 |          0.371 |  1.894 |  0.932 |           0.393 |            0.517 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        2.180 |          18.766 |
| 0.75 |        2.179 |          18.873 |
| 1.00 |        2.081 |           2.813 |

* dim = 32, capacity = 128 Million-KV, HBM = 16 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            1.068 |  2.325 |          1.277 |  1.759 |  4.113 |           1.807 |            0.932 |
| 0.75 |            0.862 |  2.309 |          0.651 |  1.754 |  1.934 |           1.304 |            0.874 |
| 1.00 |            0.360 |  2.349 |          0.346 |  1.825 |  0.926 |           0.373 |            0.470 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.701 |          15.185 |
| 0.75 |        0.578 |          13.840 |
| 1.00 |        0.567 |           0.777 |

* dim = 64, capacity = 64 Million-KV, HBM = 16 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            0.862 |  2.041 |          0.930 |  1.543 |  4.399 |           1.854 |            0.807 |
| 0.75 |            0.662 |  2.022 |          0.579 |  1.541 |  1.983 |           1.299 |            0.771 |
| 1.00 |            0.334 |  2.052 |          0.335 |  1.590 |  0.939 |           0.393 |            0.473 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.321 |          10.791 |
| 0.75 |        0.298 |          10.864 |
| 1.00 |        0.293 |           0.393 |

### On HBM+HMEM hybrid mode: 

* dim = 64, capacity = 128 Million-KV, HBM = 16 GB, HMEM = 16 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|
| 0.50 |            0.083 |  0.123 |          0.115 |  0.131 |  4.015 |           1.804 |
| 0.75 |            0.082 |  0.123 |          0.111 |  0.129 |  1.911 |           1.136 |
| 1.00 |            0.069 |  0.110 |          0.085 |  0.106 |  0.929 |           0.393 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.322 |          10.720 |
| 0.75 |        0.299 |          10.587 |
| 1.00 |        0.293 |           0.394 |

* dim = 64, capacity = 512 Million-KV, HBM = 32 GB, HMEM = 96 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|
| 0.50 |            0.049 |  0.073 |          0.048 |  0.070 |  3.530 |           1.730 |
| 0.75 |            0.049 |  0.072 |          0.048 |  0.069 |  1.849 |           1.261 |
| 1.00 |            0.044 |  0.068 |          0.044 |  0.062 |  0.911 |           0.393 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.321 |          11.148 |
| 0.75 |        0.299 |          11.269 |
| 1.00 |        0.293 |           0.396 |

### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0
