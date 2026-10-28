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
The benchmark-only SDK extension is a later, separate change and is not the
previously measured f52 SDK; rebuild and revalidate before reporting results.
The target Python
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

The patched `native_tbe_comparison.py` and checked-in `pins-legacy-f52.json` retain
identity checks for the previously validated native extension; the latter's
three Python source pins reflect the portable data-path defaults in this
patch. They do **not**
prove that a newly built upstream extension is equivalent. Before a new run,
record the exact HKV/RecSys/Python/FBGEMM build identities, update the runtime
identity pins and revalidate the untouched product contract. If any dependency
or identity is unavailable, report **NOT_RUN**; do not silently use current
RecSys main, a different wheel, or the synthetic ablation as fallback.

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
Use the exact result filenames in `pins-legacy-f52.json`, the preserved
`validate_native_result.py` for each arm and its paired counterpart, then the
driver's five-pair `summarize` command. Missing or failed arms stay in the
record and cannot be replaced to manufacture a five-pair result.

This directory provides code and checks, **not** a qualified performance
claim. The original four-GiB total memory commitment gate remains open; equal
cache-entry counts do not equalize memory or replacement policy. The frozen
legacy executable measured previously must not be described as a new upstream
build. `test_real_tbe_interface.py` checks native forward, cold/hot lookup, SGD dirty
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
