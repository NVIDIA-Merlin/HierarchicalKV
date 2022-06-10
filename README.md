# Merlin Hierarchical Key-Value Storage
-----------------
![TensorFlow Recommenders logo](assets/merlin-hkvs.png)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](docs/api_docs/)

TensorFlow Recommenders Addons(TFRA) are a collection of projects related to large-scale recommendation systems 
built upon TensorFlow by introducing the **Dynamic Embedding Technology** to TensorFlow 
that makes TensorFlow more suitable for training models of **Search, Recommendations, and Advertising** and 
makes building, evaluating, and serving sophisticated recommenders models easy. 
See approved TensorFlow RFC #[313](https://github.com/tensorflow/community/pull/313).
Those contributions will be complementary to TensorFlow Core and TensorFlow Recommenders etc. 

For Apple silicon(M1), please refer to [Apple Silicon Support](#apple-silicon-support-beta-release).

## Main Features
                                                                                  
Merlin Hierarchical Key-Value Storage

## How to build with TFRA
```shell
git clone -b rhdong/merlin https://github.com/rhdong/recommenders-addons.git
cd recommenders-addons/tensorflow_recommenders_addons/dynamic_embedding/core/lib/
git clone -b master https://github.com/rhdong/merlin-hkvs.git
cd ../../../../
PY_VERSION="3.8" TF_VERSION="2.5.1" TF_NEED_CUDA=1 sh .github/workflows/make_wheel_Linux_x86.sh
```

wheel file will be created in the [TFRA root]/wheelhouse/
## Contributors

Merlin-HKVS depends on public contributions, bug fixes, and documentation.
This project exists thanks to all the people and organizations who contribute. [[Contribute](CONTRIBUTING.md)]

<a href="https://github.com/tensorflow/recommenders-addons/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tensorflow/recommenders-addons" />
</a>


\
<a href="https://github.com/tencent">
  <kbd> <img src="./assets/tencent.png" height="70" /> </kbd>
</a><a href="https://github.com/alibaba">
  <kbd> <img src="./assets/alibaba.jpg" height="70" /> </kbd>
</a><a href="https://vip.com/"> 
  <kbd> <img src="./assets/vips.jpg" height="70" /> </kbd>
</a><a href="https://www.zhipin.com//">
  <kbd> <img src="./assets/boss.svg" height="70" /> </kbd>
</a>

\
A special thanks to [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin) and NVIDIA China DevTech Team, 
who have provided GPU acceleration technology support and code contribution.

<a href="https://github.com/NVIDIA-Merlin">
  <kbd> <img src="./assets/merilin.png" height="70" /> </kbd>
</a>

## Tutorials & Demos
See [tutorials](docs/tutorials/) and [demo](demo/) for end-to-end examples of each subpackages.

## benchmark

### Merlin
* Update time: June 10, 2022
* version: tag [v0.2.0](https://github.com/rhdong/merlin-hkvs/releases/tag/v0.2.0)

Througput Unit: Million-KV/second
CPU: libcuckoo
cudf: pure HBM
MKVS: HBM & HMEM hybird

- Upsert

|   dim |   keys num |   test_times | CPU    |  cudf   | MKVS   |
|-------|------------|--------------|--------|---------|--------|
|     8 |       1024 |           20 | 32.661 |   0.704 | 6.437  |
|     8 |       8192 |           20 | 31.775 |   4.321 | 11.792 |
|     8 |      16384 |           20 | 31.812 |  12.159 | 13.466 |
|     8 |      32768 |           20 | 93.662 |  22.393 | 28.133 |
|     8 |      65536 |           20 | 38.731 |  41.261 | 43.005 |
|     8 |     131072 |           20 | 40.009 |  80.422 | 60.495 |
|     8 |    1048576 |           20 | 31.708 | 304.959 | 43.513 |
|    64 |       1024 |           20 | 7.865  |   0.563 | 1.985  |
|    64 |       8192 |           20 | 9.3    |   4.683 | 8.435  |
|    64 |      16384 |           20 | 9.346  |   9.099 | 9.274  |
|    64 |      32768 |           20 | 11.438 |  17.117 | 17.315 |
|    64 |      65536 |           20 | 12.796 |  27.293 | 27.508 |
|    64 |     131072 |           20 | 13.302 |  56.027 | 34.556 |
|    64 |    1048576 |           20 | 10.858 | 162.962 | 32.05  |
|   128 |       1024 |           20 | 0.784  |   0.373 | 1.354  |
|   128 |       8192 |           20 | 5.001  |   3.602 | 8.692  |
|   128 |      16384 |           20 | 4.376  |   5.313 | 13.919 |
|   128 |      32768 |           20 | 4.81   |  13.742 | 21.107 |
|   128 |      65536 |           20 | 5.057  |  22.744 | 26.745 |
|   128 |     131072 |           20 | 4.909  |  31.414 | 26.063 |
|   128 |    1048576 |           20 | 4.686  |  93.366 | 24.197 |

- lookup

|   dim |   keys num |   test_times | CPU     |  cudf    | MKVS    |
|-------|------------|--------------|---------|----------|---------|
|     8 |       1024 |           20 | -29.188 |    4.923 |   4.294 |
|     8 |       8192 |           20 |  19.912 |    9.708 |  34.874 |
|     8 |      16384 |           20 |  23.018 |  675.708 |  21.335 |
|     8 |      32768 |           20 |  33.189 |  416.292 |  73.87  |
|     8 |      65536 |           20 |  21.126 |  417.335 |  89.332 |
|     8 |     131072 |           20 |  21.653 |  204.309 | 110.943 |
|     8 |    1048576 |           20 |  18.142 |  440.483 | 136.452 |
|    64 |       1024 |           20 |   2.291 |   17.232 |   1.568 |
|    64 |       8192 |           20 |   1.803 |   46.959 |  14.01  |
|    64 |      16384 |           20 |   1.783 |   45.991 |  10.791 |
|    64 |      32768 |           20 |   2.548 |   44.505 |  20.714 |
|    64 |      65536 |           20 |   3.074 |   38.849 |  37.85  |
|    64 |     131072 |           20 |   3.143 |   56.098 |  42.124 |
|    64 |    1048576 |           20 |   2.641 |   75.714 |  55.921 |
|   128 |       1024 |           20 |   0.782 |    1.065 |   1.464 |
|   128 |       8192 |           20 |   3.204 |   24.895 |  18.506 |
|   128 |      16384 |           20 |   2.292 |   12.334 |  31.249 |
|   128 |      32768 |           20 |   2.471 |   27.061 |  30.724 |
|   128 |      65536 |           20 |   3.013 |   26.216 |  25.69  |
|   128 |     131072 |           20 |   2.289 |   27.607 |  27.361 |
|   128 |    1048576 |           20 |   2.695 |   36.501 |  35.449 |