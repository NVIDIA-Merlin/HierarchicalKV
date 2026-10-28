#!/usr/bin/env bash
# Reproduction TBE-like baseline runner.
#
# Produces validated CSV artifacts for a device-only TBE-like simulation:
#   - 32-way set-associative cache by default
#   - 128B value rows
#   - native batch wall timing and asynchronous GPU-stage diagnostics
#   - LF sweep: 0.25, 0.50, 0.75, 1.00
#
# Usage:
#   RESULTS=$PWD/results/repro-tbe-like SM=90 \
#     bash scripts/run_repro_tbe_like.sh --mode quick --policy lru
#
#   RESULTS=$PWD/results/repro-tbe-like SM=90 \
#     bash scripts/run_repro_tbe_like.sh --mode full --policy all

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/repro-tbe-like/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
  echo "This runner requires one visible GPU" >&2
  exit 1
fi
MODE="quick"
POLICY="lru"
LFS="0.25,0.50,0.75,1.00"
MISS_RATE="0.25"
ASSOC="32"
VALUE_BYTES="128"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_repro_tbe_like.sh [options]

Options:
  --mode quick|full       quick uses small local sanity settings; full uses
                          Config-B-sized capacity/batch for GPU runs.
  --quick                 Alias for --mode quick.
  --full                  Alias for --mode full.
  --policy lru|lfu|all    Default: lru.
  --lf LIST               Comma-separated LF sweep. Default: 0.25,0.50,0.75,1.00.
  --miss-rate P           Default: 0.25.
  --assoc N               Default: 32.
  --value-bytes N         Default: 128.
  --dry-run               Print planned build/run commands without executing.
  --                      Pass remaining args to tbe_like_cache_baseline.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --policy=*)
      POLICY="${1#--policy=}"
      shift
      ;;
    --lf)
      LFS="$2"
      shift 2
      ;;
    --lf=*)
      LFS="${1#--lf=}"
      shift
      ;;
    --miss-rate)
      MISS_RATE="$2"
      shift 2
      ;;
    --miss-rate=*)
      MISS_RATE="${1#--miss-rate=}"
      shift
      ;;
    --assoc)
      ASSOC="$2"
      shift 2
      ;;
    --assoc=*)
      ASSOC="${1#--assoc=}"
      shift
      ;;
    --value-bytes)
      VALUE_BYTES="$2"
      shift 2
      ;;
    --value-bytes=*)
      VALUE_BYTES="${1#--value-bytes=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  quick|full) ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

case "$POLICY" in
  lru|lfu|all) ;;
  *)
    echo "Invalid --policy: $POLICY" >&2
    exit 1
    ;;
esac


if command -v nproc >/dev/null 2>&1; then
  JOBS="${JOBS:-$(nproc)}"
elif command -v sysctl >/dev/null 2>&1; then
  JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
else
  JOBS="${JOBS:-8}"
fi

BUILD_DIR="${BUILD_DIR:-$RESULTS/build}"
RUN_CMD=(
  "$BUILD_DIR/tbe_like_cache_baseline"
  --mode "$MODE"
  --policy "$POLICY"
  --lf "$LFS"
  --miss-rate "$MISS_RATE"
  --assoc "$ASSOC"
  --value-bytes "$VALUE_BYTES"
  "${EXTRA_ARGS[@]}"
)

if [[ -e "$RESULTS/run-info.txt" || -e "$RESULTS/raw/tbe_like_cache_baseline.csv" ]]; then
  echo "Refusing to overwrite prior artifacts in $RESULTS" >&2
  exit 1
fi
mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV reproduction TBE-like baseline"
  echo "  HKV_ROOT    = $HKV_ROOT"
  echo "  SM          = $SM"
  echo "  RESULTS     = $RESULTS"
  echo "  MODE        = $MODE"
  echo "  POLICY      = $POLICY"
  echo "  LFS         = $LFS"
  echo "  MISS_RATE   = $MISS_RATE"
  echo "  ASSOC       = $ASSOC"
  echo "  VALUE_BYTES = $VALUE_BYTES"
  echo "  EXTRA_ARGS  = ${EXTRA_ARGS[*]:-}"
  echo "  Started     = $(date)"
} | tee "$RESULTS/run-info.txt"

if [ "$DRY_RUN" -eq 1 ]; then
  {
    echo "Dry run: commands not executed."
    echo "cd $HKV_ROOT"
    echo "cmake -Dsm=\"$SM\" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON -S . -B \"$BUILD_DIR\""
    echo "cmake --build \"$BUILD_DIR\" --target tbe_like_cache_baseline -j\"$JOBS\""
    echo "timeout $TIMEOUT_SECONDS $BUILD_DIR/tbe_like_cache_baseline --self-test"
    printf '%q ' "${RUN_CMD[@]}"
    echo "> \"$RESULTS/raw/tbe_like_cache_baseline.csv\" 2> \"$RESULTS/logs/tbe_like_cache_baseline.log\""
  } | tee -a "$RESULTS/run-info.txt"
  exit 0
fi

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  echo "Missing GoogleTest; preserve pinned/dirty dependency and ask coordinator to restore it" >&2
  exit 1
fi

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON -S . -B "$BUILD_DIR" \
  > "$RESULTS/logs/cmake-tbe-like.log" 2>&1
cmake --build "$BUILD_DIR" --target tbe_like_cache_baseline -j"$JOBS" \
  > "$RESULTS/logs/build-tbe-like.log" 2>&1

timeout "$TIMEOUT_SECONDS" "$BUILD_DIR/tbe_like_cache_baseline" --self-test \
  > "$RESULTS/logs/self-test.stdout" 2> "$RESULTS/logs/self-test.log"
grep -Eq '^SELF_TEST PASS cases=[1-9][0-9]* ' "$RESULTS/logs/self-test.log"

timeout "$TIMEOUT_SECONDS" "${RUN_CMD[@]}" \
  > "$RESULTS/raw/tbe_like_cache_baseline.csv.partial" \
  2> "$RESULTS/logs/tbe_like_cache_baseline.log"
python3 - "$RESULTS/raw/tbe_like_cache_baseline.csv.partial" <<'PY'
import csv
import math
import sys
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "No benchmark rows"
totals = [row for row in rows if row["stage"] == "total_wall"]
assert totals, "No completed service rows"
for row in rows:
    assert row["oracle_status"] == "PASS", "Failed oracle"
    assert int(row["completed_requests"]) == int(row["batch_size"]) > 0
    assert int(row["hits"]) + int(row["misses"]) == int(row["completed_requests"])
    if row["stage"] != "total_wall":
        assert row["throughput_grequests_s"] == "", "Stage is not completed service"
for row in totals:
    seconds = float(row["elapsed_ms"]) / 1000
    assert math.isfinite(seconds) and seconds > 0
    expected = int(row["completed_requests"]) / seconds / 1e9
    assert math.isclose(float(row["throughput_grequests_s"]), expected, rel_tol=1e-5, abs_tol=1e-9)
PY
mv "$RESULTS/raw/tbe_like_cache_baseline.csv.partial" "$RESULTS/raw/tbe_like_cache_baseline.csv"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/tbe_like_cache_baseline.csv"
  echo "  $RESULTS/logs/tbe_like_cache_baseline.log"
} | tee -a "$RESULTS/run-info.txt"
