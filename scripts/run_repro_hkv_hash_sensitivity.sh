#!/usr/bin/env bash
# HKV-level alternate-hash reproduction experiment.
#
# Rebuilds the real HKV kernels with several standard 64-bit hash finalizers
# and runs insert/find/admission/eviction checks through HKV itself.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/repro-hkv-hash-sensitivity}"
HASH_MODE="${HASH_MODE:-quick}"
HASH_SEED="${HASH_SEED:-0}"
HASH_EXACT_EVENTS="${HASH_EXACT_EVENTS:-0}"
HASH_ONLY="${HASH_ONLY:-0}"
HKV_INCLUDE_OVERRIDE="${HKV_INCLUDE_OVERRIDE:-}"
for argument in "$@"; do
  if [[ "$argument" == --exact-events ]]; then HASH_EXACT_EVENTS=1; fi
  if [[ "$argument" == --hash-only ]]; then HASH_ONLY=1; fi
done
if [[ ! "$HASH_EXACT_EVENTS" =~ ^[01]$ || ! "$HASH_ONLY" =~ ^[01]$ ]] ||
   [[ "$HASH_EXACT_EVENTS" == 1 && "$HASH_ONLY" == 1 ]]; then
  echo 'HASH_EXACT_EVENTS/HASH_ONLY must be 0 or 1 and mutually exclusive' >&2
  exit 1
fi
HASH_ARGS=(--mode "$HASH_MODE" --seed "$HASH_SEED")
suffix=""
if [[ "$HASH_EXACT_EVENTS" == 1 ]]; then
  HASH_ARGS+=(--exact-events)
  suffix="_exact_diagnostic"
elif [[ "$HASH_ONLY" == 1 ]]; then
  HASH_ARGS+=(--hash-only --hash-warmup "${HASH_WARMUP:-3}" --hash-runs "${HASH_RUNS:-5}")
  suffix="_hash_only"
fi
HASH_ARGS+=("$@")

if [ "${1:-}" = "--cpu-regression" ]; then
  if [ "$#" -ne 1 ]; then
    echo "--cpu-regression takes no additional arguments" >&2
    exit 1
  fi
  mkdir -p "$RESULTS/logs"
  "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
    -DHKV_HASH_ACCOUNTING_TEST -x c++ \
    "$HKV_ROOT/benchmark/hkv_hash_sensitivity_benchmark.cc.cu" \
    -o "$RESULTS/hash-accounting-cpu-test" \
    > "$RESULTS/logs/cpu-regression-build.log" 2>&1
  "$RESULTS/hash-accounting-cpu-test" \
    > "$RESULTS/logs/cpu-regression.log" 2>&1
  echo "CPU regression PASS; GPU NOT_RUN; logs: $RESULTS/logs"
  exit 0
fi


if [[ -e "$RESULTS" ]]; then
  echo "Refusing existing RESULTS: $RESULTS; use a fresh directory" >&2
  exit 1
fi
include_root="$HKV_ROOT/include"
if [[ -n "$HKV_INCLUDE_OVERRIDE" ]]; then
  include_root="$(realpath "$HKV_INCLUDE_OVERRIDE")"
  for header in merlin_hashtable.cuh merlin/core_kernels/upsert.cuh merlin/core_kernels/dual_bucket_upsert.cuh; do
    test -f "$include_root/$header" || { echo "Missing override header: $header" >&2; exit 1; }
  done
fi
mkdir -p "$RESULTS"/{logs,raw}
find "$include_root" -type f -print0 | sort -z | xargs -0 sha256sum > "$RESULTS/include-source.sha256"

{
  echo "HKV-level hash sensitivity"
  echo "  HKV_ROOT   = $HKV_ROOT"
  echo "  git_commit = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM         = $SM"
  echo "  RESULTS    = $RESULTS"
  echo "  HASH_ARGS  = ${HASH_ARGS[*]}"
  echo "  timing     = batch insert/find plus terminal synchronization; decimal 1e9 attempts/queries"
  echo "  accounting = same-run exact diagnostics only when enabled; null-hook run counts stay unknown"
  echo "  exact_events = $HASH_EXACT_EVENTS (instrumented timing is diagnostic only; no replay attribution)"
  echo "  hash_only  = $HASH_ONLY (actual finalizer plus input/output traffic, not arithmetic latency)"
  echo "  include_root = $include_root (read-only; include-source.sha256)"
  echo "  formal_accepted = false; FullV0/V5 gates remain required"
  sha256sum "$HKV_ROOT/benchmark/hkv_hash_sensitivity_benchmark.cc.cu" \
    "$HKV_ROOT/scripts/run_repro_hkv_hash_sensitivity.sh"
  echo "  Started    = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  echo "Missing preserved GoogleTest source; ask the coordinator to restore it" >&2
  exit 1
fi

variants=(
  "0:murmur3"
  "1:splitmix64"
  "2:xxhash_avalanche"
  "3:wyhash_final"
)

if [ "${INCLUDE_IDENTITY:-0}" = "1" ]; then
  variants+=("4:identity")
fi

raw_paths=()
for entry in "${variants[@]}"; do
  variant_id="${entry%%:*}"
  variant_name="${entry#*:}"
  build_dir="$RESULTS/build-hash-${variant_name}"
  raw_path="$RESULTS/raw/hkv_hash_sensitivity_${variant_name}${suffix}.csv"
  if [ -e "$raw_path" ] || [ -e "$raw_path.partial" ]; then
    echo "Refusing to overwrite prior run: $raw_path; use a fresh RESULTS directory" >&2
    exit 1
  fi

  echo "Building hash variant ${variant_name} (${variant_id})" | tee -a "$RESULTS/run-info.txt"
  if [[ -n "$HKV_INCLUDE_OVERRIDE" ]]; then
    mkdir -p "$build_dir"
    nvcc_path="$(command -v "${NVCC:-nvcc}")"
    cuda_include="$(dirname "$(dirname "$(realpath "$nvcc_path")")")/include"
    command=("$nvcc_path" --std=c++17 -O3 -DNDEBUG "-arch=sm_$SM"
      "-DHKV_REPRO_HASH_VARIANT=$variant_id" "-DHKV_REPRO_HASH_OUTCOMES_AVAILABLE=$HASH_EXACT_EVENTS"
      -I "$include_root" -I "$cuda_include/cccl"
      "$HKV_ROOT/benchmark/hkv_hash_sensitivity_benchmark.cc.cu"
      -o "$build_dir/hkv_hash_sensitivity_benchmark")
    printf '%q ' "${command[@]}" > "$RESULTS/logs/build-${variant_name}.command"
    printf '\n' >> "$RESULTS/logs/build-${variant_name}.command"
    "$nvcc_path" --version > "$RESULTS/logs/nvcc-${variant_name}.log" 2>&1
    "${command[@]}" > "$RESULTS/logs/build-${variant_name}.log" 2>&1
  else
    cmake -S "$HKV_ROOT" -B "$build_dir" \
      -Dsm="$SM" \
      -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON \
      -DHKV_REPRO_HASH_VARIANT="$variant_id" \
      -DCMAKE_CUDA_FLAGS="-DHKV_REPRO_HASH_OUTCOMES_AVAILABLE=$HASH_EXACT_EVENTS" \
      > "$RESULTS/logs/cmake-${variant_name}.log" 2>&1
    cmake --build "$build_dir" --target hkv_hash_sensitivity_benchmark \
      -j"$(nproc)" \
      > "$RESULTS/logs/build-${variant_name}.log" 2>&1
  fi
  sha256sum -c "$RESULTS/include-source.sha256" > "$RESULTS/logs/include-check-${variant_name}.log"

  echo "Running hash variant ${variant_name}" | tee -a "$RESULTS/run-info.txt"
  sha256sum "$build_dir/hkv_hash_sensitivity_benchmark" \
    >> "$RESULTS/run-info.txt"
  "$build_dir/hkv_hash_sensitivity_benchmark" "${HASH_ARGS[@]}" \
    > "$raw_path.partial" \
    2> "$RESULTS/logs/hkv_hash_sensitivity_${variant_name}.log"
  awk 'END { exit NR < 2 }' "$raw_path.partial"
  mv "$raw_path.partial" "$raw_path"
  raw_paths+=("$raw_path")
done

awk 'FNR == 1 && NR != 1 { next } { print }' \
  "${raw_paths[@]}" \
  > "$RESULTS/raw/hkv_hash_sensitivity_combined${suffix}.csv"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/hkv_hash_sensitivity_combined${suffix}.csv"
} | tee -a "$RESULTS/run-info.txt"
