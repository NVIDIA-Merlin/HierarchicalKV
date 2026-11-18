# GPU CI

## Automatic H100 NVL check

The [HKV H100 NVL workflow](../.github/workflows/hkv-gpu-ci.yml) reports the
`hkv/H100-NVL` commit status on the exact pull-request head. Its link opens the
public Actions run. Passing requires the isolated host pipeline's terminal
success, exactly one successful `hkv:gpu` job, and matching source artifacts;
dispatch alone cannot pass the check.

Pushes to repository branches with open PRs and PR opened/reopened/ready/synchronize
events use the existing trusted maintainers: `EmmaQiaoCh`, `rhdong`,
`Ranjeet-Nvidia`, and `jiashuy`. Both the event actor and PR author must be on that
list with current write/maintain/admin permission for automatic dispatch.
Other PRs remain pending until a trusted maintainer with current write permission
comments `/hkv-gpu-ci <complete-40-hex-head-sha>`. The event and current comment
must both approve the current head; approval never transfers to a changed commit.
The Actions **Run workflow** input `pull_request` allows a trusted retry; external
PR retries still require an existing approval for that exact head.
Non-authorizing events preserve any existing status on the same immutable head;
they neither erase a proved result nor authorize new GPU work.

Events for a PR are serialized. Duplicate automatic events reuse the exact
deployment/head pipeline, including its terminal failure; successful reuse still
validates the job and artifact evidence. Explicit comments, manual requests and
trusted Actions **Re-run all jobs** attempts retry completed work while reusing
an active matching run. Current permissions and exact-head approval remain
required. Host reuse lookup covers up to 50 pages of 20 pipelines; an incomplete
search fails without creating work. The controller checks for head changes while
polling and cancels only a pipeline it created on
handled interruption, timeout or stale head. Its wait is bounded to 205 minutes;
the host's 195-minute job limit remains the last resort for abrupt controller loss.
These limits allow full tests without promising a completion time.

The `blossom` runner executes only trusted controller code from the workflow's own
immutable SHA. Checkout is sparse, fixed to this repository, without submodules
or persisted credentials. PR files are never checked out or executed in this
credentialed control path. Contributed code runs only on the existing isolated
host; this extension changes no NRT jobs, resource settings or GPU test scope.
PR-target, comment and manual triggers become available after the workflow is on
the default branch; the branch-push path also works before that merge.

Repository owners configure Actions secrets `HKV_GITLAB_TOKEN` and
`HKV_GITLAB_URL` (HTTPS origin, kept secret to suppress private URLs in job logs),
plus variables `HKV_GITLAB_PROJECT_ID` and `HKV_GITLAB_DEPLOYMENT_SHA`.
The deployment must equal `f00e3879cacd5adcc2830f0c669a437670cea872` on the fixed
`hrong/hkv-ci` ref; moving or replacing it requires review and controller update.
Use a dedicated expiring project API token, with the existing Maintainer minimum
needed to pass pipeline variables. Its API scope covers the project, not only
that branch; the fixed ref/input restrictions are enforced by this trusted
controller. Workflow, secret and variable administration remain trusted controls.
No personal token or reporting credential enters the GPU workload.

After a genuine automatic run passes, normal Code Owner approval permits merging
the exact reviewed workflow onto `master`. Only once that workflow is available
on `master` should owners require `hkv/H100-NVL` using the observed
GitHub Actions application while preserving existing review, strictness and
administrator policies. Code Owner approval remains independently required.
Local controller checks are `python3 tests/hkv_gpu_ci_test.py`; they use mock HTTP
and establish no GPU result. The separate hosted **HKV GPU controller tests**
workflow runs them for relevant PR/push changes with read-only permissions and no
host credentials; contributed tests never run on the privileged blossom path.

## Existing Blossom adapter

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
