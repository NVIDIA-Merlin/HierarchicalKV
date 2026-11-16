# GPU CI

HKV already uses [Blossom-CI](../.github/workflows/blossom-ci.yml).
An allowlisted maintainer requests a build with `/blossom-ci` on a pull request.
The `blossom` runner handles authorization and backend dispatch; its label does
not identify a GPU. The vulnerability scan runs separately on `ubuntu-latest`.
The `workflow_dispatch` path uploads backend results; it is not a standalone
GPU-test trigger.

A successful **Start ci job** means dispatch succeeded. Require the downstream
GPU job's terminal result and tested commit before treating GPU CI as passing.
[Lint](../.github/workflows/lint.yml) provides a separate CPU formatting check.

## Backend test command

Use an isolated checkout of the maintainer-approved, immutable commit with its
pinned submodules, a fresh `build/` directory, and a GPU worker with the CUDA/C++
toolchain described in the [build instructions](../README.md#how-to-build).
Run these commands from the repository root; set `HKV_CUDA_SM` to the worker's
verified CUDA architecture, such as `80` for the documented A100 build:

```bash
set -euo pipefail
: "${HKV_CUDA_SM:?Set the verified worker CUDA architecture}"
git rev-parse HEAD
git submodule status --recursive
nvidia-smi
nvcc --version
cmake --version
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dsm="$HKV_CUDA_SM"
cmake --build build --parallel
bash run_all_tests.sh
```

`run_all_tests.sh` runs executable `*_test` files under `build/`, aggregates test
failures, and fails when discovery fails or finds no test binaries. Backend jobs
must propagate the configure, build, and test exit codes without allowing failure.
Archive the command log, commit, GPU/driver/toolchain identity, and terminal
result. Record any skipped tests; report sanitizer results only if executed.

## Service prerequisites

The repository/service owners must verify the existing runner's health,
repository-scoped `BLOSSOM_KEY` and `CI_SERVER` configuration, backend dispatch
registration, and a GPU worker that invokes the command above. Secret names or
an online control runner alone do not establish a working GPU backend.

Preserve the workflow's trusted-maintainer gate. The backend must bind work to
the approved commit and execute contributed code on an isolated, unprivileged
GPU worker without repository, organization, or service credentials. Retrieve
code through a separate credentialed control step and remove credentials before
running it. Keep reporting credentials in the control plane. Do not introduce
`pull_request_target` execution of contributed code.
