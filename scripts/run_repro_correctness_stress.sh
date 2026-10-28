#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-repro-correctness}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/repro-correctness-stress}"
SM="${SM:-90}"
TEST_TIMEOUT="${TEST_TIMEOUT:-900}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"
GTEST_FILTER="${GTEST_FILTER:-*}"
JOBS="${JOBS:-$(nproc)}"
TARGETS=(correctness_stress_test dual_bucket_test merlin_hashtable_test
         insert_and_evict_test group_lock_test assign_score_test)

if [[ -e "$RESULTS_DIR/run-info.txt" ]]; then
  echo "Refusing to overwrite evidence: $RESULTS_DIR" >&2
  exit 2
fi
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/raw"
printf 'step\tstatus\texit_code\n' > "$RESULTS_DIR/status.tsv"
trap 'result=$?; printf "runner\tfinished\t%s\n" "$result" >> "$RESULTS_DIR/status.tsv"' EXIT
command -v timeout >/dev/null
command -v python3 >/dev/null
[[ -f "$ROOT_DIR/tests/googletest/CMakeLists.txt" ]] || {
  echo "Missing preserved GoogleTest checkout; refusing clone/reset" >&2
  exit 2
}

run_step() {
  local name="$1" limit="$2" log="$3" result=0 status=pass
  shift 3
  timeout --signal=TERM --kill-after=10s "$limit" "$@" > "$log" 2>&1 || result=$?
  if (( result != 0 )); then
    status=fail
    [[ "$result" == 124 || "$result" == 137 ]] && status=timeout
  fi
  printf '%s\t%s\t%s\n' "$name" "$status" "$result" >> "$RESULTS_DIR/status.tsv"
  return "$result"
}

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "build_dir=$BUILD_DIR"
  echo "sm=$SM"
  echo "gtest_filter=$GTEST_FILTER"
  echo "test_timeout=$TEST_TIMEOUT"
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  nvidia-smi -L 2>/dev/null | sed 's/^/gpu=/' || true
} > "$RESULTS_DIR/run-info.txt"

run_step configure "$BUILD_TIMEOUT" "$RESULTS_DIR/logs/cmake.log" \
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON
run_step build "$BUILD_TIMEOUT" "$RESULTS_DIR/logs/build.log" \
  cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS"

printf 'suite,status,executed_tests,log\n' > "$RESULTS_DIR/raw/correctness_stress_summary.csv"
for target in "${TARGETS[@]}"; do
  run_step "$target-list" "$TEST_TIMEOUT" "$RESULTS_DIR/logs/$target-list.log" \
    "$BUILD_DIR/$target" --gtest_filter="$GTEST_FILTER" --gtest_list_tests
  listed=$(awk '/^  [A-Za-z0-9_]/ { count++ } END { print count+0 }' "$RESULTS_DIR/logs/$target-list.log")
  if (( listed == 0 )); then
    printf '%s\tempty_selection\t2\n' "$target" >> "$RESULTS_DIR/status.tsv"
    echo "No tests selected: $target" >&2
    exit 2
  fi
  run_step "$target" "$TEST_TIMEOUT" "$RESULTS_DIR/logs/$target.log" \
    "$BUILD_DIR/$target" --gtest_filter="$GTEST_FILTER" \
    --gtest_output="xml:$RESULTS_DIR/raw/$target.xml"
  run_step "$target-count" "$TEST_TIMEOUT" "$RESULTS_DIR/logs/$target-count.log" \
    python3 - "$RESULTS_DIR/raw/$target.xml" "$listed" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
executed = sum(case.get('status') == 'run' and case.get('result', 'completed') == 'completed'
               for case in root.iter('testcase'))
if int(root.get('tests', 0)) == 0 or executed == 0:
    raise SystemExit('Zero executed tests')
if int(root.get('failures', 0)) or int(root.get('errors', 0)):
    raise SystemExit('Test report contains failures')
listed = int(sys.argv[2])
if int(root.get('tests', 0)) != listed or executed != listed:
    raise SystemExit('Selected test count does not match completed tests')
print(executed, flush=True)
print('listed_tests=' + sys.argv[2], file=sys.stderr)
PY
  executed=$(head -n 1 "$RESULTS_DIR/logs/$target-count.log")
  printf '%s,pass,%s,logs/%s.log\n' "$target" "$executed" "$target" \
    >> "$RESULTS_DIR/raw/correctness_stress_summary.csv"
done

echo "Wrote $RESULTS_DIR"
