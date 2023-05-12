# [NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV)

[![Version](https://img.shields.io/github/v/release/NVIDIA-Merlin/HierarchicalKV?color=orange&include_prereleases)](https://github.com/NVIDIA-Merlin/HierarchicalKV/releases)
[![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/HierarchicalKV)](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/HierarchicalKV/master/README.html)

## About HierarchicalKV

HierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements.

The key capability of HierarchicalKV is to store key-value (feature-embedding) on high-bandwidth memory (HBM) of GPUs and in host memory.

You can also use the library for generic key-value storage.

## Benefits of HierarchicalKV

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

HierarchicalKV makes NVIDIA GPUs more suitable for training large and super-large models of ***search, recommendations, and advertising***.
The library simplifies the common challenges to building, evaluating, and serving sophisticated recommenders models.

## API Documentation

The main classes and structs are below, and it's recommended to read the comments in the source code directly:

- [`class HashTable`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L101)
- [`class EvictStrategy`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L106)
- [`struct HashTableOptions`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L34)
- [`Struct HashTable::Vector`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L106)

For regular API doc, please refer to [API Docs](https://nvidia-merlin.github.io/HierarchicalKV/master/api/index.html)

## Usage restrictions

- The `key_type` and `meta_type` must be `uint64_t`.
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
* Hit rate = 0.60
* `find*` means the `find` API that directly returns the addresses of values.
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode: 

* dim = 4, capacity = 64 Million-KV, HBM = 32 GB, HMEM = 0 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |  find* | insert_and_evict |
|------------:|-----------------:|-------:|---------------:|-------:|-------:|-----------------:|
|        0.50 |            1.402 |  2.958 |          1.743 |  1.954 |  3.632 |            1.178 |
|        0.75 |            1.072 |  1.629 |          0.617 |  0.914 |  1.851 |            0.906 |
|        1.00 |            0.352 |  0.826 |          0.342 |  0.552 |  0.895 |            0.303 |

* dim = 64, capacity = 64 Million-KV, HBM = 16 GB, HMEM = 0 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |  find* | insert_and_evict |
|------------:|-----------------:|-------:|---------------:|-------:|-------:|-----------------:|
|        0.50 |            0.925 |  1.584 |          0.890 |  1.128 |  3.645 |            0.795 |
|        0.75 |            0.665 |  1.115 |          0.541 |  0.834 |  1.849 |            0.569 |
|        1.00 |            0.323 |  0.640 |          0.314 |  0.512 |  0.896 |            0.179 |

### On HBM+HMEM hybrid mode:

* dim = 64, capacity = 128 Million-KV, HBM = 16 GB, HMEM = 16 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |  find* |
|------------:|-----------------:|-------:|---------------:|-------:|-------:|
|        0.50 |            0.121 |  0.149 |          0.120 |  0.147 |  3.397 |
|        0.75 |            0.116 |  0.145 |          0.115 |  0.143 |  1.800 |
|        1.00 |            0.087 |  0.125 |          0.087 |  0.114 |  0.883 |

* dim = 64, capacity = 1024 Million-KV, HBM = 56 GB, HMEM = 200 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |  find* |
|------------:|-----------------:|-------:|---------------:|-------:|-------:|
|        0.50 |            0.036 |  0.054 |          0.035 |  0.045 |  2.809 |
|        0.75 |            0.035 |  0.055 |          0.034 |  0.047 |  1.930 |
|        1.00 |            0.034 |  0.051 |          0.031 |  0.047 |  0.855 |


### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0