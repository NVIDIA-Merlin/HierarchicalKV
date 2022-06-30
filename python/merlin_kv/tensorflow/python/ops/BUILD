# Placeholder for internal Python library.

package(
    default_visibility = [
        "//merlin_kv:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "merlin_kv",
    srcs = glob(["*.py"]),
    data = [
        "//merlin_kv/tensorflow/core:_merlin_kv_ops.so",
    ],
    srcs_version = "PY2AND3",
)
