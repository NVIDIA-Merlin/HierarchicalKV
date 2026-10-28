# Native embedding-cache service comparison

This optional benchmark uses the **real** FBGEMM GPU table-batched embedding
operator through the frozen RecSys adapter. `benchmark/tbe_like_cache_baseline.cc.cu`
is a separate synthetic pipeline ablation; it is not FBGEMM and must never be
reported as its result.

The adapter originates from NVIDIA `recsys-examples` at commit
`250aaa8bbbda2b6dd7995ef7f363754e4571066e`. The byte-preserved
compiled/runtime changes are distributed as
`third_party/recsys-examples/legacy-250aaa.patch` rather than active `.cuh`
files under this repository's benchmark tree. In a separate, clean checkout of
that exact commit, run:

```sh
git -C "$RECSYS_ROOT" apply --unidiff-zero --check "$HKV_ROOT/third_party/recsys-examples/legacy-250aaa.patch"
git -C "$RECSYS_ROOT" apply --unidiff-zero "$HKV_ROOT/third_party/recsys-examples/legacy-250aaa.patch"
```

Set `HKV_ROOT` to this repository root and `RECSYS_ROOT` to the separate
RecSys checkout. The patch creates/updates 27 files; `provenance.json` pins
SHA-256 for every resulting path. A dry run against the exact base reproduced
all 27 bytes. The active driver is then
`$RECSYS_ROOT/corelib/dynamicemb/benchmark/criteo_experiment/native_tbe_comparison.py`.
The benchmark-only SDK extension is distinct from the historical f52 SDK.
The recorded source-bound build used this benchmark branch at
`cc797335de5cef92be6f604f07c92e2b38133e45` (core parent `d1cb1a2c`),
the overlay SHA-256 `14f23aa9…d0f1a`, and a native ELF SHA-256
`b928115e…0ffc1bc2` (build ID `c97ba0b3…209ab744`). Rebuild and revalidate
for any different source or environment. The target Python
environment was 3.10, Torch `2.7.0+cu126`, and FBGEMM GPU `1.2.0+cu126`.
Use a matching CUDA toolchain and build the DynamicEmb extension against the
HKV source revision under test. No wheel, dataset, native ELF, profile, or
production log is bundled here.
The public overlay's `setup.py` no longer runs `pip uninstall dynamicemb`.
Set `DYNAMICEMB_HKV_SOURCE` to the exact HKV checkout
before building so the extension receives that checkout's `include/` tree
without initializing the bundled HKV submodule. Build and import an isolated
source copy, then record its actual ELF hash and loaded path; the historical
f52 extension pin is not a binary for this new source.

The checked-in `pins-legacy-f52.json` and external
`validate_native_result.py` are historical f52-specific checks. They were
**not** applied unchanged to the new build. The recorded run also used a
source-rebound per-arm/paired validator; the public reproducer uses the overlay's
native driver SHA-256 `dcfa420d…a717`, its internal exact preflight/full
output/miss/dirty26/backing/flush oracles, and `summarize --raw` over all ten
processes. A new build must record and check its own HKV/RecSys/Python/FBGEMM
identities and rerun the full contract; do not substitute current RecSys main,
a different wheel, or the synthetic ablation.

The existing `native_tbe_comparison.py` commands are `prepare`, `run`, and
`summarize`. For the fixed service cell, prepare five independent inputs with
seeds 42–46, `--cache-rows 65536 --vocab-rows 262144 --batch-size 8192
--batches 1 --distribution uniform --target-miss-percent 25 --occupancy
full-overfill --eviction lru`. Run HKV and the **unchanged** FBGEMM backend
as separate fresh processes per seed, alternating order. Both arms must keep
the original 26 logical FP32/dim32 tables, full output, misses, dirty restore,
backing, and final flush; TBE's native source and prefetch remain unchanged.
`input-contract-legacy.json` contains only the prior five seed/trace/weight
digests and no input payload or machine-specific paths. Regenerate with the
exact pinned `prepare` code, verify every digest, and make a new contract for
any changed source or workload; do not borrow old input hashes by config name.
For the new-SDK record, use the declarative
[`shared-cache-five-pair.json`](shared-cache-five-pair.json) to bind the
existing `prepare`, `run`, and `summarize` CLI and the five frozen input
digests. The ten fresh processes alternate backend order by seed. The driver's
internal oracles and five-pair summary are the reproducer path; the legacy
external checker is not a new-build qualification. Missing or failed arms
stay in the record and cannot be replaced.

The recorded new-SDK five-pair, fixed-entry full-service comparison is
[`timings-new-sdk-five-pair.csv`](timings-new-sdk-five-pair.csv): TBE/HKV
host-wall geometric mean **1.9563×**, paired-seed 95% bootstrap interval
**[1.6109, 2.8016]** (five independent pairs, 10,000 fixed resamples,
RNG seed 20261020). All five HKV cells were faster; seed 44 has an unusually
high 3.969× ratio and remains in the result. This is a configured-native
comparison: HKV shared 128-slot buckets, TIMESTAMP/native hit refresh and
full-HBM backing versus unchanged TBE LRU/32-way/MANAGED_CACHING with
prefetch off. The original four-GiB total-memory gate is **not evaluated**;
equal capacity rows do not establish equal actual memory or policy. Neither
the result nor the historical f52 binary isolates a single SDK change's
effect. `test_real_tbe_interface.py` checks native forward, cold/hot lookup, SGD dirty
update, complete backing flush, and reset/restore on a small 26-table fixture;
it explicitly reports `NOT_RUN` until `HKV_RUN_NATIVE_TBE_TEST=1`, the pinned
RecSys checkout, FBGEMM/Torch and CUDA are available. Existing RecSys CPU tests
are in the patched checkout's `test_native_tbe_comparison.py` and
`test_native_tbe_fairness.py`; their fake fixtures are not GPU evidence.

FBGEMM GPU is a third-party BSD-licensed dependency, not vendored code.
The benchmark uses Meta's table-batched embedding implementation described by
Khudia *et al.*, “FBGEMM: Enabling High-Performance Low-Precision Deep Learning
Inference,” 2019; cite its [repository](https://github.com/pytorch/FBGEMM)
and the exact wheel version when reporting results. RecSys and HKV sources
retain their Apache-2.0 headers. See `provenance.json` for source identities.

The fixed `run` path rejects `TBE_EXPERIMENTAL_UVM_PREFETCH` unless it is unset or `0`; the explicit UVM-prefetch experiment was removed. This keeps the native baseline unchanged. The GPU interface test also applies disjoint LRU pressure after SGD, requires an observed dirty victim writeback before explicit flush, then checks full readback and flush/reset recovery. These are correctness checks, not performance samples or a four-GiB qualification.
