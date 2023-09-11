load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "core_kernels",
    srcs = [],
    hdrs = glob([
        "**/*.cuh",
    ]),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//include/merlin:types_and_utils",
        "@local_config_cuda//cuda",
    ],
)
