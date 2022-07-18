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


## Tutorials & Demos

Merlin-KV is positioned as a header-only library. The environment requirement is :

- CUDA version >= 11.2
- NVIDIA GPU with Compute capbility 8.0 8.6 or 8.7

See [tutorials](docs/api_docs/html/index.html) and [demo(TBD)](cpp/tests/merlin_hashtable_test.cc.cu) for end-to-end examples of each subpackages.

## API docs

Refer to [API Docs](docs/api_docs/)