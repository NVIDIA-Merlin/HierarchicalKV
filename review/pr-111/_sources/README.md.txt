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

| dim | Capacity<br>(M-KVs) | HBM/HMEM<br>(GB) | load_factor | insert_or_assign |   find | find_or_insert | assign |
|----:|:-------------------:|:----------------:|------------:|:----------------:|-------:|:--------------:|-------:|
|   4 |                  64 |          32 /  0 |        0.50 |            1.481 |  2.799 |          1.440 |  1.816 |
|   4 |                  64 |          32 /  0 |        0.75 |            0.872 |  1.749 |          0.784 |  1.217 |
|   4 |                  64 |          32 /  0 |        1.00 |            0.190 |  0.671 |          0.267 |  0.248 |
|  16 |                  64 |          16 /  0 |        0.50 |            1.221 |  2.020 |          1.100 |  1.453 |
|  16 |                  64 |          16 /  0 |        0.75 |            0.795 |  1.299 |          0.712 |  0.968 |
|  16 |                  64 |          16 /  0 |        1.00 |            0.190 |  0.647 |          0.266 |  0.247 |
|  64 |                  64 |          16 /  0 |        0.50 |            0.685 |  0.922 |          0.694 |  0.736 |
|  64 |                  64 |          16 /  0 |        0.75 |            0.573 |  0.683 |          0.517 |  0.554 |
|  64 |                  64 |          16 /  0 |        1.00 |            0.174 |  0.548 |          0.237 |  0.234 |
| 128 |                 128 |          64 /  0 |        0.50 |            0.411 |  0.527 |          0.428 |  0.423 |
| 128 |                 128 |          64 /  0 |        0.75 |            0.388 |  0.413 |          0.363 |  0.345 |
| 128 |                 128 |          64 /  0 |        1.00 |            0.157 |  0.447 |          0.211 |  0.213 |

### On HBM+HMEM hybrid mode:

| dim | Capacity<br>(M-KVs) | HBM/HMEM<br>(GB) | load_factor | insert_or_assign |   find | find_or_insert | assign |
|----:|:-------------------:|:----------------:|------------:|:----------------:|-------:|:--------------:|-------:|
|  64 |                 128 |          16 / 16 |        0.50 |            0.086 |  0.082 |          0.119 |  0.086 |
|  64 |                 128 |          16 / 16 |        0.75 |            0.083 |  0.081 |          0.113 |  0.085 |
|  64 |                 128 |          16 / 16 |        1.00 |            0.064 |  0.076 |          0.072 |  0.052 |
|  64 |                1024 |          56 /200 |        0.50 |            0.038 |  0.042 |          0.034 |  0.039 |
|  64 |                1024 |          56 /200 |        0.75 |            0.038 |  0.042 |          0.034 |  0.038 |
|  64 |                1024 |          56 /200 |        1.00 |            0.033 |  0.041 |          0.029 |  0.030 |
| 128 |                  64 |          16 / 16 |        0.50 |            0.045 |  0.042 |          0.064 |  0.045 |
| 128 |                  64 |          16 / 16 |        0.75 |            0.044 |  0.042 |          0.062 |  0.042 |
| 128 |                  64 |          16 / 16 |        1.00 |            0.038 |  0.041 |          0.048 |  0.034 |
| 128 |                 512 |          56 /200 |        0.50 |            0.029 |  0.028 |          0.042 |  0.029 |
| 128 |                 512 |          56 /200 |        0.75 |            0.029 |  0.028 |          0.041 |  0.029 |
| 128 |                 512 |          56 /200 |        1.00 |            0.026 |  0.027 |          0.034 |  0.024 |



### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0