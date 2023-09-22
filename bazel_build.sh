#!/bin/bash

# Usage : `./bazel_build.sh` or `bash bazel_build.sh`
set -e
export $(cat .bazeliskrc | xargs)

bazel build --config=cuda //...
