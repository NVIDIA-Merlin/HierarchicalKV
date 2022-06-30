# Placeholder for internal Python library.

# NVIDIA Merlin-KV, a hierarchical Key-Value storage for GPU on HBM/HostMEM/SSD.
licenses(["notice"])  # Apache 2.0

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "merlin_kv",
    srcs = glob(["**/*.py"]),
    deps = [
        "//merlin_kv/tensorflow/python/ops:merlin_kv",
        # "//merlin_kv/pytorch/python/ops:merlin_kv,
        "//merlin_kv/utils",
    ],
)

py_test(
    name = "merlin_kv_test",
    size = "small",
    srcs = glob(["**/*test.py"]),
    main = "tests/run_all_test.py",
    deps = [
        ":merlin_kv",
    ],
)
