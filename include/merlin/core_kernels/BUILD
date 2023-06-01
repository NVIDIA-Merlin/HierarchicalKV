load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "core_kernels", srcs = [],
    hdrs = [
      "find_or_insert.cuh",
      "find_ptr_or_insert.cuh",
      "kernel_utils.cuh",
      "lookup_ptr.cuh",
      "lookup.cuh",
      "update.cuh",
      "upsert_and_evict.cuh",
      "upsert.cuh",
    ],
    visibility = [
      "//visibility:public",
    ],
    deps = [
      "//include/merlin:types_and_utils",
      "@local_config_cuda//cuda",
    ], 
)