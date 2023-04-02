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
- The keys of `0xFFFFFFFFFFFFFFFF` and `0xFFFFFFFFFFFFFFFE` are reserved for internal using.

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
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode: 

* dim = 4
* capacity = 64 Million-KV
* HBM = 32 GB
* HMEM = 0 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign | insert_and_evict |
|------------:|:----------------:|-------:|:--------------:|-------:|:----------------:|
|        0.50 |            1.443 |  2.708 |          1.279 |  2.348 |            0.866 |
|        0.75 |            0.810 |  1.769 |          0.820 |  1.601 |            0.603 |
|        1.00 |            0.187 |  0.582 |          0.277 |  0.191 |            0.149 |

* dim = 64
* capacity = 64 Million-KV
* HBM = 16 GB
* HMEM = 0 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign | insert_and_evict |
|------------:|:----------------:|-------:|:--------------:|-------:|:----------------:|
|        0.50 |            0.671 |  1.296 |          0.792 |  1.118 |            0.610 |
|        0.75 |            0.546 |  0.912 |          0.528 |  0.833 |            0.421 |
|        1.00 |            0.171 |  0.380 |          0.224 |  0.163 |            0.103 |

### On HBM+HMEM hybrid mode: 

* dim = 64
* capacity = 128 Million-KV
* HBM = 16 GB
* HMEM = 16 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |
|------------:|:----------------:|-------:|:--------------:|-------:|
|        0.50 |            0.086 |  0.128 |          0.112 |  0.135 |
|        0.75 |            0.083 |  0.126 |          0.109 |  0.133 |
|        1.00 |            0.064 |  0.111 |          0.078 |  0.080 |

* dim = 64
* capacity = 1024 Million-KV
* HBM = 56 GB
* HMEM = 200 GB

| load_factor | insert_or_assign |   find | find_or_insert | assign |
|------------:|:----------------:|-------:|:--------------:|-------:|
|        0.50 |            0.039 |  0.055 |          0.035 |  0.052 |
|        0.75 |            0.038 |  0.055 |          0.034 |  0.051 |
|        1.00 |            0.033 |  0.052 |          0.031 |  0.041 |




### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0