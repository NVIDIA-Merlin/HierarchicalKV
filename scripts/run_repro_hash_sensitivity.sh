#!/usr/bin/env bash
# Reproduction hash sensitivity microbenchmark.
#
# This script builds the standalone hash_sensitivity_benchmark target and runs
# a quick matrix by default.  It does not change HKV's core hash path.
#
# Usage:
#   RESULTS_DIR=$PWD/results/hash-sensitivity SM=90 \
#     scripts/run_repro_hash_sensitivity.sh --mode quick
#   scripts/run_repro_hash_sensitivity.sh --mode full

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-repro-hash-sensitivity}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/repro-hash-sensitivity}"
SM="${SM:-90}"
MODE="quick"
RUN_ARGS=("$@")
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [ "${ARGS[$i]}" = "--mode" ] && [ $((i + 1)) -lt "${#ARGS[@]}" ]; then
    MODE="${ARGS[$((i + 1))]}"
  fi
done
if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  RUN_ARGS=(--mode "$MODE")
fi

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/raw"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "build_dir=$BUILD_DIR"
  echo "sm=$SM"
  echo "mode=$MODE"
  echo "args=${ARGS[*]}"
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  nvidia-smi -L 2>/dev/null | sed 's/^/gpu=/' || true
} > "$RESULTS_DIR/run-info.txt"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON \
  > "$RESULTS_DIR/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR" --target hash_sensitivity_benchmark -j"$JOBS" \
  > "$RESULTS_DIR/logs/build.log" 2>&1

"$BUILD_DIR/hash_sensitivity_benchmark" "${RUN_ARGS[@]}" \
  > "$RESULTS_DIR/raw/hash_sensitivity_${MODE}.csv" \
  2> "$RESULTS_DIR/logs/hash_sensitivity_${MODE}.log"

{
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "csv=$RESULTS_DIR/raw/hash_sensitivity_${MODE}.csv"
  echo "log=$RESULTS_DIR/logs/hash_sensitivity_${MODE}.log"
} >> "$RESULTS_DIR/run-info.txt"

echo "Wrote $RESULTS_DIR/raw/hash_sensitivity_${MODE}.csv"
