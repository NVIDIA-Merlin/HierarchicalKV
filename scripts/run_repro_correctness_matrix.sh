#!/usr/bin/env bash
# Reproduction correctness/adversarial CSV matrix.
#
# Default quick run:
#   RESULTS=$PWD/results/repro-correctness-matrix SM=90 \
#     bash scripts/run_repro_correctness_matrix.sh
#
# Full scalable run:
#   MODE=full TABLE_MODE=both RESULTS=$PWD/results/full-correctness-matrix \
#     bash scripts/run_repro_correctness_matrix.sh --streams=8

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SM="${SM:-90}"
MODE="${MODE:-quick}"
TABLE_MODE="${TABLE_MODE:-both}"
SCENARIO="${SCENARIO:-all}"
BUILD_DIR="${BUILD_DIR:-$HKV_ROOT/build-repro-correctness-matrix}"
RESULTS="${RESULTS:-$HKV_ROOT/results/repro-correctness-matrix}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
BENCH_ARGS=("--mode=$MODE" "--table_mode=$TABLE_MODE" "--scenario=$SCENARIO")

if [ "$#" -gt 0 ]; then
  BENCH_ARGS+=("$@")
fi

if [[ -e "$RESULTS/run-info.txt" ]]; then
  echo "Refusing to overwrite evidence: $RESULTS" >&2
  exit 2
fi

mkdir -p "$RESULTS"/{logs,raw}
printf 'step\tstatus\texit_code\n' > "$RESULTS/status.tsv"
trap 'result=$?; printf "runner\tfinished\t%s\n" "$result" >> "$RESULTS/status.tsv"' EXIT
command -v timeout >/dev/null

run_step() {
  local name="$1" limit="$2" stdout="$3" stderr="$4" result=0 status=pass
  shift 4
  timeout --signal=TERM --kill-after=10s "$limit" "$@" > "$stdout" 2> "$stderr" || result=$?
  if (( result != 0 )); then
    status=fail
    [[ "$result" == 124 || "$result" == 137 ]] && status=timeout
  fi
  printf '%s\t%s\t%s\n' "$name" "$status" "$result" >> "$RESULTS/status.tsv"
  return "$result"
}

{
  echo "HKV reproduction correctness/adversarial matrix"
  echo "  HKV_ROOT   = $HKV_ROOT"
  echo "  SM         = $SM"
  echo "  MODE       = $MODE"
  echo "  TABLE_MODE = $TABLE_MODE"
  echo "  SCENARIO   = $SCENARIO"
  echo "  BUILD_DIR  = $BUILD_DIR"
  echo "  RESULTS    = $RESULTS"
  echo "  BENCH_ARGS = ${BENCH_ARGS[*]}"
  echo "  Started    = $(date)"
  git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/  Commit     = /' || true
  nvidia-smi -L 2>/dev/null | sed 's/^/  GPU        = /' || true
} > "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
RESULTS_DIR="$RESULTS/tests" BUILD_DIR="$BUILD_DIR" SM="$SM" JOBS="$JOBS" \
  bash "$HKV_ROOT/scripts/run_repro_correctness_stress.sh" \
  > "$RESULTS/logs/regression.log" 2>&1

run_step build "$BUILD_TIMEOUT" "$RESULTS/logs/build.log" "$RESULTS/logs/build-stderr.log" \
  cmake --build "$BUILD_DIR" --target hkv_correctness_stress_benchmark -j"$JOBS"
run_step matrix "$BENCH_TIMEOUT" "$RESULTS/raw/hkv_correctness_matrix.csv" \
  "$RESULTS/logs/hkv_correctness_matrix.log" \
  "$BUILD_DIR/hkv_correctness_stress_benchmark" "${BENCH_ARGS[@]}"
rows=$(awk 'NR > 1 && NF { count++ } END { print count+0 }' "$RESULTS/raw/hkv_correctness_matrix.csv")
if (( rows == 0 )); then
  printf 'matrix\tempty_selection\t2\n' >> "$RESULTS/status.tsv"
  echo "Matrix produced zero cases" >&2
  exit 2
fi

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/tests/raw/correctness_stress_summary.csv"
  echo "  $RESULTS/raw/hkv_correctness_matrix.csv"
  echo "  $RESULTS/logs/hkv_correctness_matrix.log"
} >> "$RESULTS/run-info.txt"
echo "Wrote $RESULTS"
