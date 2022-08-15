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

## API Documentation

Refer to [API Docs](https://nvidia-merlin.github.io/merlin-kv/master/README.html)


## Contributors

Merlin-KV is co-maintianed by [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin) and NVIDIA product end-users,
and also open for public contributions, bug fixes, and documentation. [[Contribute](CONTRIBUTING.md)]

## How to build
  
Basically, Merlin-KV is a headers only library, the commands below only create binaries for benchmark and unit testing.

```bash
git clone https://github.com/NVIDIA-Merlin/merlin-kv.git
cd merlin-kv && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSM=80 .. && make -j

// for Benchmark
./merlin_hashtable_benchmark

// for Unit Test
./merlin_hashtable_test
```

Your environment must meet the following requirements:

- CUDA version >= 11.2
- NVIDIA GPU with compute capbility 8.0, 8.6, or 8.7


## Benchmark

* GPU: 1 x NVIDIA A100-SXM4-80GB: 8.0
* Key Type = uint64_t
* Value Type = float32 * dim
* Key-Values per OP = 1,048,576
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode:

| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |  find |
|----:|------------:|------------:|--------:|---------:|-------:|------:|
|   4 |    67108864 |        0.50 |      16 |        0 |  1.167 | 1.750 |
|   4 |    67108864 |        0.75 |      16 |        0 |  0.897 | 1.386 |
|   4 |    67108864 |        1.00 |      16 |        0 |  0.213 | 0.678 |
|  16 |    67108864 |        0.50 |      16 |        0 |  1.114 | 1.564 |
|  16 |    67108864 |        0.75 |      16 |        0 |  0.894 | 1.258 |
|  16 |    67108864 |        1.00 |      16 |        0 |  0.215 | 0.640 |
|  64 |    67108864 |        0.50 |      16 |        0 |  0.873 | 0.915 |
|  64 |    67108864 |        0.75 |      16 |        0 |  0.767 | 0.823 |
|  64 |    67108864 |        1.00 |      16 |        0 |  0.206 | 0.492 |
| 128 |   134217728 |        0.50 |      64 |        0 |  0.664 | 0.613 |
| 128 |   134217728 |        0.75 |      64 |        0 |  0.593 | 0.560 |
| 128 |   134217728 |        1.00 |      64 |        0 |  0.191 | 0.387 |

### On HBM+HMEM hybrid mode:
| dim |    capacity | load_factor | HBM(GB) | HMEM(GB) | insert |  find |
|----:|------------:|------------:|--------:|---------:|-------:|------:|
|  64 |   134217728 |        0.50 |      16 |       16 |  0.107 | 0.103 |
|  64 |   134217728 |        0.75 |      16 |       16 |  0.106 | 0.101 |
|  64 |   134217728 |        1.00 |      16 |       16 |  0.077 | 0.094 |
|  64 |  1073741824 |        0.50 |      56 |      200 |  0.037 | 0.040 |
|  64 |  1073741824 |        0.75 |      56 |      200 |  0.037 | 0.040 |
|  64 |  1073741824 |        1.00 |      56 |      200 |  0.030 | 0.036 |
| 128 |    67108864 |        0.50 |      16 |       16 |  0.076 | 0.072 |
| 128 |    67108864 |        0.75 |      16 |       16 |  0.071 | 0.071 |
| 128 |    67108864 |        1.00 |      16 |       16 |  0.059 | 0.068 |
| 128 |   536870912 |        0.50 |      56 |      200 |  0.039 | 0.040 |
| 128 |   536870912 |        0.75 |      56 |      200 |  0.041 | 0.040 |
| 128 |   536870912 |        1.00 |      56 |      200 |  0.035 | 0.038 |

