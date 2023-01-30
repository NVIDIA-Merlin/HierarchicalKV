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

* GPU: 1 x NVIDIA A100-PCIE-80GB: 8.0
* Key Type = uint64_t
* Value Type = float32 * {dim}
* Key-Values per OP = 1,048,576
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |   find |  erase |
|----:|------------:|------------:|--------:|---------:|-------:|-------:|-------:|
|   4 |    67108864 |        0.50 |      32 |        0 |  1.756 |  3.043 |  4.157 |
|   4 |    67108864 |        0.75 |      32 |        0 |  1.036 |  1.998 |  2.867 |
|   4 |    67108864 |        1.00 |      32 |        0 |  0.919 |  1.132 |  0.558 |
|  16 |    67108864 |        0.50 |      16 |        0 |  1.397 |  2.220 |  4.335 |
|  16 |    67108864 |        0.75 |      16 |        0 |  1.012 |  1.393 |  2.940 |
|  16 |    67108864 |        1.00 |      16 |        0 |  0.924 |  1.135 |  0.559 |
|  64 |    67108864 |        0.50 |      16 |        0 |  0.736 |  0.939 |  4.276 |
|  64 |    67108864 |        0.75 |      16 |        0 |  0.716 |  0.688 |  2.911 |
|  64 |    67108864 |        1.00 |      16 |        0 |  0.922 |  1.135 |  0.561 |
| 128 |   134217728 |        0.50 |      64 |        0 |  0.406 |  0.559 |  3.821 |
| 128 |   134217728 |        0.75 |      64 |        0 |  0.565 |  0.435 |  2.708 |
| 128 |   134217728 |        1.00 |      64 |        0 |  0.824 |  1.114 |  0.549 |

### On HBM+HMEM hybrid mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |   find |  erase |
|----:|------------:|------------:|--------:|---------:|-------:|-------:|-------:|
|  64 |   134217728 |        0.50 |      16 |       16 |  0.111 |  0.112 |  3.659 |
|  64 |   134217728 |        0.75 |      16 |       16 |  0.110 |  0.110 |  2.574 |
|  64 |   134217728 |        1.00 |      16 |       16 |  0.338 |  0.293 |  0.543 |
|  64 |  1073741824 |        0.50 |      56 |      200 |  0.038 |  0.041 |  2.372 |
|  64 |  1073741824 |        0.75 |      56 |      200 |  0.036 |  0.041 |  2.056 |
|  64 |  1073741824 |        1.00 |      56 |      200 |  0.295 |  0.274 |  0.533 |
| 128 |    67108864 |        0.50 |      16 |       16 |  0.063 |  0.068 |  4.062 |
| 128 |    67108864 |        0.75 |      16 |       16 |  0.065 |  0.067 |  2.765 |
| 128 |    67108864 |        1.00 |      16 |       16 |  0.316 |  0.260 |  0.547 |
| 128 |   536870912 |        0.50 |      56 |      200 |  0.037 |  0.041 |  2.789 |
| 128 |   536870912 |        0.75 |      56 |      200 |  0.037 |  0.041 |  2.175 |
| 128 |   536870912 |        1.00 |      56 |      200 |  0.290 |  0.255 |  0.534 |


### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0