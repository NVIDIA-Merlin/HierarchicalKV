# [NVIDIA Merlin-KV](https://github.com/NVIDIA-Merlin/merlin-kv)

![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/NVIDIA-Merlin/merlin-kv?sort=semver)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/merlin-kv)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/merlin-kv/master/README.html)

## About Merlin-KV

Merlin Key-Value is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements.

The key capability of Merlin-KV is to store key-value feature-embeddings on high-bandwidth memory (HBM) of GPUs and in host memory.

You can also use the library for generic key-value storage.

## Benefits of Merlin-KV

When building large recommender systems, machine learning (ML) engineers face the following challenges:

- GPUs are needed, but HBM on a single GPU is too small for the large DLRMs that scale to several terabytes.
- Improving communication performance is getting more difficult in larger and larger CPU clusters.
- It is difficult to efficiently control consumption growth of limited HBM with customized strategies.
- Most generic key-value libraries provide low HBM and host memory utilization.

Merlin-KV alleviates these challenges and helps the machine learning engineers in RecSys with the following benefits:

- Supports training large RecSys models on HBM and host memory at the same time.
- Provides better performance by bypassing CPUs and reducing the communication workload.
- Implements model-size restraint strategies that are based on timestamp or occurrences.
  The strategies are implemented by CUDA kernels and are customizable.
- Operates at a high working-status load factor that is close to 1.0.

Merlin-KV makes NVIDIA GPUs more suitable for training large and super-large models of search, recommendations, and advertising.
The library simplifies the common challenges to building, evaluating, and serving sophisticated recommenders models.

## Tutorials and Demos

See the [tutorials](docs/api_docs/html/index.html) and [demo(TBD)](cpp/tests/merlin_hashtable_test.cc.cu) for end-to-end examples of each subpackage.

## API Documentation

Refer to [API Docs](docs/api_docs/)

## Requirements

Merlin-KV is a header-only library. Your environment must meet the following requirements:

- CUDA version >= 11.2
- NVIDIA GPU with compute capbility 8.0, 8.6, or 8.7

## Benchmarks

* Update time: June 10, 2022
* version: tag [r0.3.0](https://github.com/NVIDIA-Merlin/merlin-kv/tree/r0.3.0)

***Througput Unit: Million-KV/second***
- **CPU**: libcuckoo
- **cudf**: [cudf::concurrent_unordered_map](https://github.com/rapidsai/cudf/blob/branch-22.08/cpp/src/hash/concurrent_unordered_map.cuh) on pure HBM
- **Merlin-KV**: Merlin-KV on hybrid of HBM & HMEM

### **lookup**

|   dim |   keys num |    CPU |    cudf |    MKVS |
|------:|-----------:|-------:|--------:|--------:|
|     8 |       1024 | 29.188 |   4.923 |   4.294 |
|     8 |       8192 | 19.912 |   9.708 |  34.874 |
|     8 |      65536 | 21.126 | 417.335 |  89.332 |
|     8 |     131072 | 21.653 | 204.309 | 110.943 |
|     8 |    1048576 | 18.142 | 440.483 | 136.452 |
|    64 |       1024 |  2.291 |  17.232 |   1.568 |
|    64 |       8192 |  1.803 |  46.959 |   14.01 |
|    64 |      65536 |  3.074 |  38.849 |   37.85 |
|    64 |     131072 |  3.143 |  56.098 |  42.124 |
|    64 |    1048576 |  2.641 |  75.714 |  55.921 |
|   128 |       1024 |  0.782 |   1.065 |   1.464 |
|   128 |       8192 |  3.204 |  24.895 |  18.506 |
|   128 |      65536 |  3.013 |  26.216 |   25.69 |
|   128 |     131072 |  2.289 |  27.607 |  27.361 |
|   128 |    1048576 |  2.695 |  36.501 |  35.449 |

### **upsert**

|   dim |   keys num |    CPU |    cudf |    MKVS |
|------:|-----------:|-------:|--------:|--------:|
|     8 |       1024 | 32.661 |   0.704 |   6.437 |
|     8 |       8192 | 31.775 |   4.321 |  11.792 |
|     8 |      65536 | 38.731 |  41.261 |  43.005 |
|     8 |     131072 | 40.009 |  80.422 |  60.495 |
|     8 |    1048576 | 31.708 | 304.959 |  43.513 |
|    64 |       1024 |  7.865 |   0.563 |   1.985 |
|    64 |       8192 |  9.300 |   4.683 |   8.435 |
|    64 |      65536 | 12.796 |  27.293 |  27.508 |
|    64 |     131072 | 13.302 |  56.027 |  34.556 |
|    64 |    1048576 | 10.858 | 162.962 |  32.050 |
|   128 |       1024 |  0.784 |   0.373 |   1.354 |
|   128 |       8192 |  5.001 |   3.602 |   8.692 |
|   128 |      65536 |  5.057 |  22.744 |  26.745 |
|   128 |     131072 |  4.909 |  31.414 |  26.063 |
|   128 |    1048576 |  4.686 |  93.366 |  24.197 |

## Contributing to Merlin-KV

Merlin-KV is maintianed by [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin)
and also open for public contributions, bug fixes, and documentation. [[Contribute](CONTRIBUTING.md)]