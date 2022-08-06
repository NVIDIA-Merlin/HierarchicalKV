# Merlin-KV
-----------------
![Merlin KV logo](assets/merlin-hkvs.png)

[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](docs/api_docs/)

# What's the Merlin-KV?
[Merlin-KV](https://github.com/NVIDIA-Merlin/merlin-kv) is an open source library of NVIDIA Merlin.

- Merlin-KV is a generic Key-Value library designed with full considerations of RecSys requirements.
- Storing the key-value (feature-embedding) on both of the HBM and Host memory (SSD/NVMe is coming soon)
- Can also be used as a generic Key-Value storage

# Benefits

When building large recommender systems, machine learning (ML) engineers have been faced with the following challenges:

- GPUs are needed, but HBM on a single GPU is too small for the large DLRMs with several Tera Bytes
- Improving communication performance is getting harder in larger and larger CPU clusters
- Difficult to efficiently control consumption growth of limited HBM with customized strategies
- Low HBM/HMEM utilization when using the generic Key-Value library

Merlin-KV alleviates these challenges and helps the machine learning (ML) engineers in RecSys:

- Training the large RecSys models on HBM and Host memory at the same time
- Better performance by CPUs fully bypass and less communication workload
- Model size restraint strategies based on timestamp or occurrences, implemented by CUDA kernels and customizable
- High working status load factor which is close to 1.0

Merlin-KV makes NVIDIA GPUs more suitable for training large and super-large models of **Search, Recommendations, and Advertising** and 
makes building, evaluating, and serving sophisticated recommenders models easy.

## Contributors

Merlin-KV is maintianed by [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin) 
and also open for public contributions, bug fixes, and documentation. [[Contribute](CONTRIBUTING.md)]

<a href="https://github.com/NVIDIA-Merlin">
  <kbd> <img src="./assets/merilin.png" height="70" /> </kbd>
</a>

## Benchmark

* Update time: Aug 4, 2022
* version: tag [r0.1.0](https://github.com/NVIDIA-Merlin/merlin-kv/tree/r0.1.0)
* Key Type = uint64_t
* Value Type = float32 * dim

***Througput Unit: Billion-KV/second***


### On pure HBM mode:

| dim |    capacity | keys_num_per_op | load_factor | HBM(GB) | HMEM(GB) | insert |  find |
|----:|------------:|----------------:|------------:|--------:|---------:|-------:|------:|
|   4 |    67108864 |         1048576 |        0.50 |      16 |        0 |  1.167 | 1.750 |
|   4 |    67108864 |         1048576 |        0.75 |      16 |        0 |  0.897 | 1.386 |
|   4 |    67108864 |         1048576 |        1.00 |      16 |        0 |  0.213 | 0.678 |
|  16 |    67108864 |         1048576 |        0.50 |      16 |        0 |  1.114 | 1.564 |
|  16 |    67108864 |         1048576 |        0.75 |      16 |        0 |  0.894 | 1.258 |
|  16 |    67108864 |         1048576 |        1.00 |      16 |        0 |  0.215 | 0.640 |
|  64 |    67108864 |         1048576 |        0.50 |      16 |        0 |  0.873 | 0.915 |
|  64 |    67108864 |         1048576 |        0.75 |      16 |        0 |  0.767 | 0.823 |
|  64 |    67108864 |         1048576 |        1.00 |      16 |        0 |  0.206 | 0.492 |
| 128 |   134217728 |         1048576 |        0.50 |      64 |        0 |  0.664 | 0.613 |
| 128 |   134217728 |         1048576 |        0.75 |      64 |        0 |  0.593 | 0.560 |
| 128 |   134217728 |         1048576 |        1.00 |      64 |        0 |  0.191 | 0.387 |

### On HBM+HMEM hybrid mode:

| dim |    capacity | keys_num_per_op | load_factor | HBM(GB) | HMEM(GB) | insert |  find |
|----:|------------:|----------------:|------------:|--------:|---------:|-------:|------:|
|  64 |   134217728 |         1048576 |        0.50 |      16 |       16 |  0.107 | 0.103 |
|  64 |   134217728 |         1048576 |        0.75 |      16 |       16 |  0.106 | 0.101 |
|  64 |   134217728 |         1048576 |        1.00 |      16 |       16 |  0.077 | 0.094 |
|  64 |  1073741824 |         1048576 |        0.50 |      56 |      200 |  0.037 | 0.040 |
|  64 |  1073741824 |         1048576 |        0.75 |      56 |      200 |  0.037 | 0.040 |
|  64 |  1073741824 |         1048576 |        1.00 |      56 |      200 |  0.030 | 0.036 |
| 128 |    67108864 |         1048576 |        0.50 |      16 |       16 |  0.076 | 0.072 |
| 128 |    67108864 |         1048576 |        0.75 |      16 |       16 |  0.071 | 0.071 |
| 128 |    67108864 |         1048576 |        1.00 |      16 |       16 |  0.059 | 0.068 |
| 128 |   536870912 |         1048576 |        0.50 |      56 |      200 |  0.039 | 0.040 |
| 128 |   536870912 |         1048576 |        0.75 |      56 |      200 |  0.041 | 0.040 |
| 128 |   536870912 |         1048576 |        1.00 |      56 |      200 |  0.035 | 0.038 |


## Tutorials & Demos

Merlin-KV is positioned as a header-only library. The environment requirement is :

- CUDA version >= 11.2
- NVIDIA GPU with Compute capbility 8.0 8.6 or 8.7

See [tutorials](docs/api_docs/html/index.html) and [demo(TBD)](cpp/tests/merlin_hashtable_test.cc.cu) for end-to-end examples of each subpackages.

## API docs

Refer to [API Docs](docs/api_docs/)