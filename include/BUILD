load("@local_config_cuda//cuda:build_defs.bzl", "cuda_cc_library")

cuda_cc_library(
    name = "merlin_localfile",
    hdrs = [
        "merlin_localfile.hpp",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//include/merlin",
        "@local_config_cuda//cuda",
    ],
)

cuda_cc_library(
    name = "merlin_hashtable",
    hdrs = [
        "merlin_hashtable_base.hpp",
        "merlin_hashtable.cuh",
        "merlin_hashtable_device.cuh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//include/merlin",
        "@local_config_cuda//cuda",
    ],
)
