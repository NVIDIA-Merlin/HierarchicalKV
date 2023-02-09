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
|   4 |    67108864 |        0.50 |      32 |        0 |  1.620 |  3.156 |  4.026 |
|   4 |    67108864 |        0.75 |      32 |        0 |  1.052 |  2.159 |  2.901 |
|   4 |    67108864 |        1.00 |      32 |        0 |  0.225 |  0.891 |  0.805 |
|  16 |    67108864 |        0.50 |      16 |        0 |  1.402 |  2.237 |  4.219 |
|  16 |    67108864 |        0.75 |      16 |        0 |  0.999 |  1.636 |  2.863 |
|  16 |    67108864 |        1.00 |      16 |        0 |  0.224 |  0.856 |  0.819 |
|  64 |    67108864 |        0.50 |      16 |        0 |  0.686 |  0.859 |  4.200 |
|  64 |    67108864 |        0.75 |      16 |        0 |  0.637 |  0.705 |  2.911 |
|  64 |    67108864 |        1.00 |      16 |        0 |  0.205 |  0.718 |  0.804 |
| 128 |   134217728 |        0.50 |      64 |        0 |  0.403 |  0.472 |  3.783 |
| 128 |   134217728 |        0.75 |      64 |        0 |  0.445 |  0.425 |  2.684 |
| 128 |   134217728 |        1.00 |      64 |        0 |  0.183 |  0.557 |  0.766 |

### On HBM+HMEM hybrid mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |   find |  erase |
|----:|------------:|------------:|--------:|---------:|-------:|-------:|-------:|
|  64 |   134217728 |        0.50 |      16 |       16 |  0.105 |  0.125 |  3.620 |
|  64 |   134217728 |        0.75 |      16 |       16 |  0.107 |  0.123 |  2.610 |
|  64 |   134217728 |        1.00 |      16 |       16 |  0.073 |  0.112 |  0.770 |
|  64 |  1073741824 |        0.50 |      56 |      200 |  0.037 |  0.042 |  2.362 |
|  64 |  1073741824 |        0.75 |      56 |      200 |  0.037 |  0.042 |  2.106 |
|  64 |  1073741824 |        1.00 |      56 |      200 |  0.031 |  0.040 |  0.727 |
| 128 |    67108864 |        0.50 |      16 |       16 |  0.064 |  0.072 |  4.039 |
| 128 |    67108864 |        0.75 |      16 |       16 |  0.069 |  0.071 |  2.834 |
| 128 |    67108864 |        1.00 |      16 |       16 |  0.052 |  0.067 |  0.805 |
| 128 |   536870912 |        0.50 |      56 |      200 |  0.037 |  0.042 |  2.866 |
| 128 |   536870912 |        0.75 |      56 |      200 |  0.038 |  0.043 |  2.225 |
| 128 |   536870912 |        1.00 |      56 |      200 |  0.033 |  0.041 |  0.734 |



### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0