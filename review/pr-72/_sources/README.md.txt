# [NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV)

[![Version](https://img.shields.io/github/v/release/NVIDIA-Merlin/HierarchicalKV?color=orange)](https://github.com/NVIDIA-Merlin/HierarchicalKV/releases)
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

* GPU: 1 x NVIDIA A100-SXM4-80GB: 8.0
* Key Type = uint64_t
* Value Type = float32 * {dim}
* Key-Values per OP = 1,048,576
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |   find |  erase |
|----:|------------:|------------:|--------:|---------:|-------:|-------:|-------:|
|   4 |    67108864 |        0.50 |      32 |        0 |  1.118 |  1.478 |  0.482 |
|   4 |    67108864 |        0.75 |      32 |        0 |  0.902 |  1.242 |  0.472 |
|   4 |    67108864 |        1.00 |      32 |        0 |  0.178 |  0.683 |  0.487 |
|  16 |    67108864 |        0.50 |      16 |        0 |  1.104 |  1.334 |  0.485 |
|  16 |    67108864 |        0.75 |      16 |        0 |  0.894 |  1.145 |  0.476 |
|  16 |    67108864 |        1.00 |      16 |        0 |  0.178 |  0.654 |  0.486 |
|  64 |    67108864 |        0.50 |      16 |        0 |  0.875 |  0.849 |  0.481 |
|  64 |    67108864 |        0.75 |      16 |        0 |  0.723 |  0.777 |  0.482 |
|  64 |    67108864 |        1.00 |      16 |        0 |  0.168 |  0.536 |  0.481 |
| 128 |   134217728 |        0.50 |      64 |        0 |  0.634 |  0.577 |  0.393 |
| 128 |   134217728 |        0.75 |      64 |        0 |  0.555 |  0.545 |  0.365 |
| 128 |   134217728 |        1.00 |      64 |        0 |  0.147 |  0.417 |  0.352 |

### On HBM+HMEM hybrid mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |   find |  erase |
|----:|------------:|------------:|--------:|---------:|-------:|-------:|-------:|
|  64 |   134217728 |        0.50 |      16 |       16 |  0.114 |  0.106 |  0.456 |
|  64 |   134217728 |        0.75 |      16 |       16 |  0.113 |  0.107 |  0.430 |
|  64 |   134217728 |        1.00 |      16 |       16 |  0.075 |  0.115 |  0.421 |
|  64 |  1073741824 |        0.50 |      56 |      200 |  0.038 |  0.040 |  0.293 |
|  64 |  1073741824 |        0.75 |      56 |      200 |  0.037 |  0.040 |  0.291 |
|  64 |  1073741824 |        1.00 |      56 |      200 |  0.033 |  0.041 |  0.278 |
| 128 |    67108864 |        0.50 |      16 |       16 |  0.079 |  0.073 |  0.562 |
| 128 |    67108864 |        0.75 |      16 |       16 |  0.078 |  0.078 |  0.550 |
| 128 |    67108864 |        1.00 |      16 |       16 |  0.061 |  0.111 |  0.576 |
| 128 |   536870912 |        0.50 |      56 |      200 |  0.041 |  0.041 |  0.313 |
| 128 |   536870912 |        0.75 |      56 |      200 |  0.041 |  0.041 |  0.305 |
| 128 |   536870912 |        1.00 |      56 |      200 |  0.035 |  0.045 |  0.296 |


### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0