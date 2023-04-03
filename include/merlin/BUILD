load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

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
