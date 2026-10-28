#!/usr/bin/env bash
set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RESULTS="${RESULTS:-$HKV_ROOT/results/repro-lookup/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
BUILD_DIR="${BUILD_DIR:-$RESULTS/build}"
SM="${SM:-90}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
  echo "This runner requires one visible GPU" >&2
  exit 1
fi
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if [[ -e "$RESULTS/run-info.txt" || -e "$RESULTS/raw/hkv_lookup_modes.csv" ]]; then
  echo "Refusing to overwrite prior artifacts in $RESULTS" >&2
  exit 1
fi
mkdir -p "$RESULTS"/{raw,logs}
BUILD_ARGS=(
  -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON -S "$HKV_ROOT" -B "$BUILD_DIR"
  -DHKV_REPRO_LOOKUP_CAPACITY="${CAPACITY:-65536}" -DHKV_REPRO_LOOKUP_BATCH_SIZE="${BATCH_SIZE:-1024}"
  -DHKV_REPRO_LOOKUP_DIM="${DIM:-32}" -DHKV_REPRO_LOOKUP_WARMUP="${WARMUP:-1}"
  -DHKV_REPRO_LOOKUP_RUNS="${RUNS:-2}" -DHKV_REPRO_LOOKUP_HBM_GB="${HBM_GB:-1}"
)
if [[ "$#" -eq 0 ]]; then set -- --mode all --seed 12345 0.5; fi
{
  echo "Lookup-only immutable-table diagnostics; no cache-service speedup."
  echo "Started: $(date -u +%FT%TZ)"
  echo "Source: $(git -C "$HKV_ROOT" rev-parse HEAD)"
  printf 'Build: '; printf '%q ' cmake "${BUILD_ARGS[@]}"; echo
  printf 'Run: '; printf '%q ' timeout "$TIMEOUT_SECONDS" "$BUILD_DIR/hkv_lookup_modes_benchmark" "$@"; echo
} > "$RESULTS/run-info.txt"
if [[ "$DRY_RUN" -eq 1 ]]; then cat "$RESULTS/run-info.txt"; exit 0; fi
if [[ ! -f "$HKV_ROOT/tests/googletest/CMakeLists.txt" ]]; then
  echo "Missing GoogleTest; preserve pinned/dirty dependency and ask coordinator to restore it" >&2
  exit 1
fi
cmake "${BUILD_ARGS[@]}" > "$RESULTS/logs/configure.log" 2>&1
cmake --build "$BUILD_DIR" --target hkv_lookup_modes_benchmark -j"${JOBS:-8}" \
  > "$RESULTS/logs/build.log" 2>&1
timeout "$TIMEOUT_SECONDS" "$BUILD_DIR/hkv_lookup_modes_benchmark" "$@" \
  > "$RESULTS/raw/hkv_lookup_modes.csv.partial" 2> "$RESULTS/logs/run.log"
python3 - "$RESULTS/raw/hkv_lookup_modes.csv.partial" <<'PY'
import csv
import math
import sys
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "No benchmark rows"
for row in rows:
    assert row["oracle_status"] == "PASS"
    assert row["timing_scope"] == "lookup_only_host_wall"
    completed = int(row["completed_requests"])
    seconds = float(row["elapsed_ms"]) / 1000
    assert completed > 0 and math.isfinite(seconds) and seconds > 0
    assert math.isclose(float(row["throughput_grequests_s"]), completed / seconds / 1e9,
                        rel_tol=1e-5, abs_tol=1e-6)
PY
mv "$RESULTS/raw/hkv_lookup_modes.csv.partial" "$RESULTS/raw/hkv_lookup_modes.csv"
echo "Validated lookup CSV: $RESULTS/raw/hkv_lookup_modes.csv"
