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

| dim | Capacity<br>(M-KVs) | HBM/HMEM<br>(GB) | load_factor | insert or<br>assign |   find | find or<br>insert | assign | insert and<br>evict |
|----:|:-------------------:|:----------------:|------------:|:-------------------:|-------:|:-----------------:|-------:|:-------------------:|
|   4 |                  64 |          32 /  0 |        0.50 |               1.484 |  2.441 |             1.269 |  2.114 |               0.091 |
|   4 |                  64 |          32 /  0 |        0.75 |               0.853 |  1.737 |             0.856 |  1.593 |               0.629 |
|   4 |                  64 |          32 /  0 |        1.00 |               0.189 |  0.579 |             0.284 |  0.191 |               0.153 |
|  16 |                  64 |          16 /  0 |        0.50 |               1.304 |  2.390 |             1.238 |  2.135 |               0.844 |
|  16 |                  64 |          16 /  0 |        0.75 |               0.806 |  1.490 |             0.790 |  1.396 |               0.625 |
|  16 |                  64 |          16 /  0 |        1.00 |               0.189 |  0.525 |             0.276 |  0.185 |               0.147 |
|  64 |                  64 |          16 /  0 |        0.50 |               0.686 |  1.293 |             0.821 |  1.116 |               0.645 |
|  64 |                  64 |          16 /  0 |        0.75 |               0.560 |  0.872 |             0.531 |  0.799 |               0.436 |
|  64 |                  64 |          16 /  0 |        1.00 |               0.173 |  0.379 |             0.229 |  0.163 |               0.105 |
| 128 |                 128 |          64 /  0 |        0.50 |               0.547 |  0.793 |             0.548 |  0.664 |               0.429 |
| 128 |                 128 |          64 /  0 |        0.75 |               0.384 |  0.603 |             0.378 |  0.538 |               0.316 |
| 128 |                 128 |          64 /  0 |        1.00 |               0.156 |  0.282 |             0.189 |  0.136 |               0.075 |
   
### On HBM+HMEM hybrid mode:   

| dim | Capacity<br>(M-KVs) | HBM/HMEM<br>(GB) | load_factor | insert or<br>assign |   find | find or<br>insert | assign |
|----:|:-------------------:|:----------------:|------------:|:-------------------:|-------:|:-----------------:|-------:|
|  64 |                 128 |          16 / 16 |        0.50 |               0.086 |  0.128 |             0.111 |  0.135 |
|  64 |                 128 |          16 / 16 |        0.75 |               0.083 |  0.125 |             0.106 |  0.132 |
|  64 |                 128 |          16 / 16 |        1.00 |               0.064 |  0.110 |             0.076 |  0.080 |
|  64 |                1024 |          56 /200 |        0.50 |               0.038 |  0.056 |             0.035 |  0.052 |
|  64 |                1024 |          56 /200 |        0.75 |               0.038 |  0.055 |             0.034 |  0.051 |
|  64 |                1024 |          56 /200 |        1.00 |               0.033 |  0.052 |             0.030 |  0.040 |
| 128 |                  64 |          16 / 16 |        0.50 |               0.045 |  0.067 |             0.066 |  0.072 |
| 128 |                  64 |          16 / 16 |        0.75 |               0.044 |  0.067 |             0.064 |  0.071 |
| 128 |                  64 |          16 / 16 |        1.00 |               0.038 |  0.062 |             0.053 |  0.053 |
| 128 |                 512 |          56 /200 |        0.50 |               0.029 |  0.045 |             0.042 |  0.048 |
| 128 |                 512 |          56 /200 |        0.75 |               0.029 |  0.045 |             0.042 |  0.047 |
| 128 |                 512 |          56 /200 |        1.00 |               0.026 |  0.043 |             0.036 |  0.038 |




### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0