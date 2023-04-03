workspace(name = "HierarchicalKV")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/gpus:configure.bzl", "cuda_configure")

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)

cuda_configure(name = "local_config_cuda")
