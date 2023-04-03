load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_binary(
    name = "benchmark_util",
    deps = [
        ":benchmark_lib",
    ],
)

cuda_library(
    name = "benchmark_lib",
    srcs = [
        "merlin_hashtable_benchmark.cc.cu",
    ],
    hdrs = [
        "benchmark_util.cuh",
    ],
    copts = ["-Iinclude/"],
    linkopts = ["-pthread"],
    deps = [
        "//include:merlin_hashtable",
        "@local_config_cuda//cuda",
    ],
)
