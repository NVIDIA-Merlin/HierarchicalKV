load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "types_and_utils",
    srcs = [
    ],
    hdrs = [
        "types.cuh",
        "utils.cuh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@local_config_cuda//cuda",
    ],
)

cuda_library(
    name = "merlin",
    srcs = [
    ],
    hdrs = [
        "array_kernels.cuh",
        "core_kernels.cuh",
        "debug.hpp",
        "flexible_buffer.cuh",
        "initializers.cuh",
        "memory_pool.cuh",
        "optimizers.cuh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//include/merlin:types_and_utils",
        "//include/merlin/core_kernels",
        "@local_config_cuda//cuda",
    ],
)
