# Native embedding-cache comparison

This optional suite calls real FBGEMM GPU table-batched embedding. The separate
`benchmark/tbe_like_cache_baseline.cc.cu` is synthetic, not FBGEMM. TBE source
and native prefetch behavior are unchanged; nonzero
`TBE_EXPERIMENTAL_UVM_PREFETCH` is rejected.

## Pinned adapter and build

Use a separate clean NVIDIA `recsys-examples` checkout at
`250aaa8bbbda2b6dd7995ef7f363754e4571066e`. Current RecSys main uses a different
backend and is not interchangeable. Set `HKV_ROOT` to this repository and
`RECSYS_ROOT` to that checkout:

```sh
git -C "$RECSYS_ROOT" apply --unidiff-zero --check "$HKV_ROOT/third_party/recsys-examples/legacy-250aaa.patch"
git -C "$RECSYS_ROOT" apply --unidiff-zero "$HKV_ROOT/third_party/recsys-examples/legacy-250aaa.patch"
export PYTHONPATH="$RECSYS_ROOT/corelib/dynamicemb${PYTHONPATH:+:$PYTHONPATH}"
```

The existing 27-file overlay is pinned in `provenance.json`. The recorded stack
is Python 3.10, Torch 2.7.0+cu126 and FBGEMM GPU 1.2.0+cu126. Build/import an isolated
SDK copy with `DYNAMICEMB_HKV_SOURCE="$HKV_ROOT"`; setup does not uninstall an
existing DynamicEmb package. The tested native core is PR #255 commit `d1cb1a2c`,
built in HKV tree `cc797335`, ELF SHA256 `b928115e…0ffc1bc2` and build ID
`c97ba0b3…209ab744`. This update changes Python/tests only; native C++/CUDA
bytes are unchanged. Different native sources/builds require fresh bindings
and validation. No ELF, wheels, input payloads, profiles or credentials are
bundled. Remote source/build/cache/results belong in a leased node-local root.

## Generic DynamicEmb options

These keyword-only constructor options also pass through the HKV factory:

| Setting | Contract |
| --- | --- |
| `native_lru_hit_update="off"/"auto"/"on"` | Capability selection; on rejects unsupported caches/calls, auto preserves fallback. Required LRU hit refresh also occurs on off/dedup/training paths; this switch is not a speed ablation. |
| `backing_placement="host"/"hbm"` | Native backing-value placement; default host. Existing `enforce_hbm=True` aliases HBM and conflicts with explicit host. |
| `num_of_buckets_per_alloc=1024` | Positive power-of-two metadata allocation group, clamped to each physical table's initial bucket count. Existing default grouping is unchanged; this does not change capacity or remove metadata. |

Explicit arguments override legacy environment flags. Unset options read
`DYNAMICEMB_EXPERIMENTAL_NATIVE_LRU_HIT_UPDATE` (0=off,1=auto),
`DYNAMICEMB_EXPERIMENTAL_BACKING_HBM_FULL` (0=host,1=HBM), and
`DYNAMICEMB_BUCKETS_PER_ALLOCATION`.

The module accepts arbitrary logical-feature counts and multiple compatible
physical caches through ordered contiguous `feature_table_map` groups.
Shared versus partitioned layout is explicit. Incompatible dimensions,
optimizer/policy groups and shuffled mappings are not silently merged.
The selector retains FP32/int64, TIMESTAMP/KLru, bucket128 and eval/no-grad
input restrictions; fallback still has correct LRU semantics. Generic factory
prefix offsets are for direct callers: the Criteo data/replay/frozen-weight
helpers retain their original 26-feature uniform shared-vocabulary contract.

## Tests

Run the portable 25-case source/CPU suite after applying the overlay, then the
existing adapter CPU tests with Torch/pytest available:

```sh
HKV_RECSYS_ROOT="$RECSYS_ROOT" python -m unittest discover -s "$HKV_ROOT/benchmark/native_tbe" -p 'test_dynamicemb_*cpu.py'
python -m pytest -q "$RECSYS_ROOT/corelib/dynamicemb/benchmark/criteo_experiment/test_contract.py" "$RECSYS_ROOT/corelib/dynamicemb/benchmark/criteo_experiment/test_native_tbe_comparison.py" "$RECSYS_ROOT/corelib/dynamicemb/benchmark/criteo_experiment/test_native_tbe_fairness.py"
```

CPU fixtures are not GPU/performance evidence. The historical None-to-const LRU
fixture is an exact two-method base250aaa excerpt, not an active SDK module.
The adapter tests retain full-service timing order and check that the required
all-hit counter probe precedes both timed paths.

On a separately reserved GPU, `RUN_ROOT` is the leased result directory; record
actual SDK/ELF paths and hashes plus GPU identity before running:

```sh
HKV_RUN_NATIVE_TBE_TEST=1 python "$HKV_ROOT/benchmark/native_tbe/test_dynamicemb_generic_interface.py" --output "$RUN_ROOT/dynamicemb-generic-tests.json"
HKV_RUN_NATIVE_TBE_TEST=1 python -m unittest discover -s "$HKV_ROOT/benchmark/native_tbe" -p 'test_real_tbe_interface.py'
```

The first command covers ten cases: 1/3/26 features, dim16/32/64, independent
caches, host/HBM, off/auto/on/dedup, full outputs, empty inputs, real miss/restore,
mandatory LRU/no-touch reads, natural full-bucket eviction, Adagrad payload/flush,
and metadata clamp. Repeated `--case NAME` explicitly selects subsets; output
lists planned/executed cases separately. The second checks unchanged FBGEMM
cold/hot lookup, SGD dirty-victim writeback, complete flush and reset/restore.
Without opt-in they report NOT_RUN/skip, never GPU PASS. Neither contributes
performance samples.

## Separate recorded scopes

| Record | Meaning |
| --- | --- |
| `shared-cache-five-pair.json`, `timings-new-sdk-five-pair.csv` | Historical full-service lookup/miss/backing/D2H/final-flush timing; TBE/HKV GM 1.9563, nominal 95% CI [1.6109,2.8016], five independent pairs. Seed44's high ratio remains retained. |
| `cache-forward-host-five-pair.json`, `timings-cache-forward-host.csv` | Five independent paired seeds/processes, 20 warm-ups and 200 forwards/process in groups of eight; one resident batch replayed, 2,000 complete outputs checked outside timing. TBE/HKV GM 2.0451, nominal 95% CI [2.0279,2.0624]. |

Both use frozen synthetic inputs, not Criteo: 8,192 logical requests across 26
FP32/dim32 tables. The current cache-forward recipe keeps 1,703,936 slots and 208 MiB
value capacity per cache, HKV pinned-host backing/metadata group 1024, and native TBE
LRU/32-way/MANAGED_CACHING. Repeated forwards are not independent samples. This
serial resident-batch scope does not measure changing traces, overlapped
pipelines or end-to-end miss service; it does not isolate an eviction/fusion
kernel or establish equal total memory/policies.

The historical full-service recipe pins the old overlay; retrieve that patch
from HKV commit `cb04241` and verify its SHA before applying to a separate clean
base250aaa checkout. Do not relabel the current Python update as that cohort.
The f52 `pins-legacy-f52.json`/`validate_native_result.py` remain historical;
current source-bound checks use `pins-cache-forward-host.json` and
`validate_cache_forward_result.py`. Different builds must rebind actual
source/library identities; old budget labels do not certify equal memory.
The existing prepare/run/summarize commands and five-pair alternating order
are recorded in the JSON recipes. Verify frozen input digests and all full
output/miss/dirty/backing/flush checks, summarize all ten fresh processes,
and retain failed arms without replacement.

For example, the current HKV seed42 arm is:

```sh
DYNAMICEMB_ALLOCATION_IDENTITIES=0 DYNAMICEMB_EXPERIMENTAL_NATIVE_LRU_HIT_UPDATE=1 DYNAMICEMB_EXPERIMENTAL_BACKING_HBM_FULL=0 DYNAMICEMB_BUCKETS_PER_ALLOCATION=1024 TBE_EXPERIMENTAL_UVM_PREFETCH=0 python -m benchmark.criteo_experiment.native_tbe_comparison run --input "$INPUT_ROOT/seed42.pt" --backend dynamicemb --state warm --cuda-event-service --cache-rows 65536 --hkv-layout shared --diagnostic-flush-repeats 1 --cache-only --cache-only-warmup-iterations 20 --cache-only-iterations 200 --cache-only-output-group-size 8 --output "$RUN_ROOT/42-dynamicemb-warm.jsonl"
```

Use the recipe's native TBE arguments and alternating order for the other arms.
For the recorded source/build, the current checker accepts a raw result, backend,
input, input checksum, CUDA UUID, seed, input contract and identity pins:

```sh
python "$HKV_ROOT/benchmark/native_tbe/validate_cache_forward_result.py" "$RUN_ROOT/42-dynamicemb-warm.jsonl" dynamicemb "$INPUT_ROOT/seed42.pt" "$INPUT_ROOT/seed42.pt.sha256" "$GPU_UUID" 42 "$HKV_ROOT/benchmark/native_tbe/input-contract-legacy.json" "$HKV_ROOT/benchmark/native_tbe/pins-cache-forward-host.json"
```

An optional final positional argument supplies the previous backend's raw result
for that seed and additionally checks pair identity/output parity.

FBGEMM GPU is a third-party BSD dependency, not vendored here. Cite Khudia et al.,
“FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference,” 2019,
its [repository](https://github.com/pytorch/FBGEMM), and the exact wheel version.
RecSys/HKV headers retain Apache-2.0 licensing.
