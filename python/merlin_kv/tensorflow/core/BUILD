licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//merlin_kv:merlin_kv.bzl", "custom_op_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

custom_op_library(
    name = "_merlin_kv_ops.so",
    cuda_srcs = [
        "kernels/merlin_kv_op_gpu.h",
        "kernels/merlin_kv_op_gpu.cu.cc",
        "ops/merlin_kv_ops.cc",
        "utils/types.h",
    ] + glob(["kernels/lookup_impl/lookup_table_op_gpu*"]),
    cuda_deps = [
        "//merlin_kv/tensorflow/core/lib/merlin-kv:merlinkv"
    ],
)
