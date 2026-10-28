# Reproducible HKV benchmarks

The optional targets in this directory exercise load factor, value size,
lookup mode, cache quality, hash choice, admission, concurrency, and
single/dual-bucket behavior. They were migrated from an older HKV research
branch, but this repository does **not** attach old performance numbers to a
new source or dependency revision. Every reported result needs its own source
commit, dependency identities, GPU/driver/toolchain, parameters, unit,
successful-operation denominator, output checks, and independent-run count.

Existing core targets such as `merlin_hashtable_benchmark` and
`dual_bucket_benchmark` remain registered in the root CMake file. Additional
targets are opt-in:

```sh
cmake -S . -B build-repro -Dsm=90 -DCMAKE_BUILD_TYPE=Release \
  -DHKV_BUILD_REPRO_BENCHMARKS=ON
cmake --build build-repro --target hkv_lookup_modes_benchmark
```

`benchmark/CMakeLists.txt` keeps digest and R/W-lock ablation definitions
target-local (`digest_ablation_no_digest`, `concurrency_rw_lock`) rather than
altering every library/test build. `HKV_REPRO_*` cache options apply only to
the corresponding sensitivity target. GPU runs are **not** performed by the
source migration; missing CUDA/GoogleTest/dependencies are `NOT_RUN`, not a
benchmark pass.

`baselines/` contains standalone value-returning WarpCore, BGHT/P2BHT, and
cuCollections comparisons. Its CMake dependency revisions are immutable
commits resolved on 2026-10-27 for **new** runs, not proof of the historical
baselines' revisions. The external sources keep their own licenses and must
be cited. Do not compare 8-byte scalar lookup with a 128-byte full-value
return as one workload. The comparison code records full-payload oracles and
failed/overflow behavior; do not drop unsuccessful operations from the
denominator without reporting them.

`scripts/run_repro_*.sh` are migrated single-purpose commands. They no
longer clone or delete dependencies, hardcode a particular host's CUDA path,
or assume a specific conference workflow. Set `RESULTS`/`RESULTS_DIR`,
`BUILD_DIR`, `SM`, and any workload options explicitly. The old one-shot
figure/expected-number runner was intentionally omitted because it mixed
venue-specific claims with destructive rebuilds. A failed run remains a
failed run; the scripts do not fill missing samples.

The `tbe_like_cache_baseline` target and `run_repro_tbe_like.sh` implement a
**synthetic** staged cache-pipeline ablation. They are not the real FBGEMM
TBE. The real native TBE service path, dependency pin, unchanged baseline,
full output/miss/dirty/backing/flush checks, and five-seed summary are in
[`native_tbe/README.md`](native_tbe/README.md). Its four-GiB total memory
qualification remains open; this migration does not claim a ranking.

The opt-in CUDA benchmark targets explicitly use `-O3` for host optimization, because the existing root CMake file sets the build type to Debug while declaring tests. This does not change native test flags. Record generated compiler flags and the new executable identity for each run; these builds are not the historical measured binaries.
