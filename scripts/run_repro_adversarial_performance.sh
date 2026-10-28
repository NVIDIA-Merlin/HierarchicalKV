#!/usr/bin/env bash
set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/repro-adversarial-performance}"
BUILD_DIR="${BUILD_DIR:-$HKV_ROOT/build-adversarial-performance}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
JOBS="${JOBS:-$(nproc)}"
RUN_ARGS=(--mode="${RUN_MODE:-quick}" "$@")

if [[ -d "$RESULTS" ]] && [[ -n "$(find "$RESULTS" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite evidence: $RESULTS" >&2
  exit 2
fi
mkdir -p "$RESULTS/logs" "$RESULTS/raw"
printf 'step\tstatus\texit_code\n' > "$RESULTS/status.tsv"
trap 'result=$?; printf "runner\tfinished\t%s\n" "$result" >> "$RESULTS/status.tsv"' EXIT
command -v timeout >/dev/null
command -v python3 >/dev/null
[[ -f "$HKV_ROOT/tests/googletest/CMakeLists.txt" ]] || {
  echo "Missing preserved GoogleTest checkout; refusing clone/reset" >&2
  exit 2
}

run_step() {
  local name="$1" limit="$2" output="$3" errors="$4" result=0 status=pass
  shift 4
  timeout --signal=TERM --kill-after=10s "$limit" "$@" > "$output" 2> "$errors" || result=$?
  if (( result != 0 )); then
    status=fail
    [[ "$result" == 124 || "$result" == 137 ]] && status=timeout
  fi
  printf '%s\t%s\t%s\n' "$name" "$status" "$result" >> "$RESULTS/status.tsv"
  return "$result"
}

{
  printf 'date=%s\nroot=%s\nbuild_dir=%s\nsm=%s\n' "$(date -u +%FT%TZ)" "$HKV_ROOT" "$BUILD_DIR" "$SM"
  printf 'evidence_status=DEVELOPMENT_ONLY\nnetwork_included=false\n'
  printf 'argv='; printf '%q ' "${RUN_ARGS[@]}"; printf '\n'
  git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  sha256sum "$HKV_ROOT/benchmark/hkv_adversarial_performance_benchmark.cc.cu" "$HKV_ROOT/scripts/run_repro_adversarial_performance.sh"
} > "$RESULTS/run-info.txt"

run_step configure "$BUILD_TIMEOUT" "$RESULTS/logs/cmake.log" "$RESULTS/logs/cmake.stderr" \
  cmake -S "$HKV_ROOT" -B "$BUILD_DIR" -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON
run_step build "$BUILD_TIMEOUT" "$RESULTS/logs/build.log" "$RESULTS/logs/build.stderr" \
  cmake --build "$BUILD_DIR" --target hkv_adversarial_performance_benchmark -j"$JOBS"
run_step benchmark "$BENCH_TIMEOUT" "$RESULTS/raw/hkv_adversarial_performance.csv" "$RESULTS/logs/benchmark.stderr" \
  "$BUILD_DIR/hkv_adversarial_performance_benchmark" "${RUN_ARGS[@]}"
run_step validate 60 "$RESULTS/logs/validation.log" "$RESULTS/logs/validation.stderr" \
  python3 -I - "$RESULTS/raw/hkv_adversarial_performance.csv" "${RUN_ARGS[@]}" <<'PY'
import csv
import math
import sys

options = dict(argument.removeprefix('--').split('=', 1) for argument in sys.argv[2:])
mode = options.get('mode', 'quick')
runs = int(options.get('runs', 5 if mode == 'full' else 3))
batch = int(options.get('batch', 1048576 if mode == 'full' else 65536))
capacity = int(options.get('capacity', 16777216 if mode == 'full' else 1048576))
adversarial_batch = int(options.get('adversarial_batch', 262144 if mode == 'full' else 32768))
seed = int(options.get('seed', 20260611))
load_factor = float(options.get('load_factor', 0.5))
total = int(options.get('total_requests', batch))
request_batch = int(options.get('request_batch', 128))
scenario = options.get('scenario', 'all')
roles = options.get('roles', 'RUI')
streams = int(options.get('streams', 6))
overlap = int(options.get('overlap_pct', 50))
kind = options.get('overlap_kind', 'hot_key')
ingress = int(options.get('ingress_pct', 50))
expected = set()
role_counts = dict.fromkeys('RUI', 0)
full_batches, tail = divmod(total, request_batch)
for index, role in enumerate(roles):
    role_counts[role] += ((full_batches + len(roles) - 1 - index) // len(roles)) * request_batch
role_counts[roles[full_batches % len(roles)]] += tail

def expect(name, operation, workers=1, overlap_kind='none', overlap_pct=0, ingress_pct=0):
    for run in range(1, runs + 1):
        expected.add((name, operation, workers, overlap_kind, overlap_pct, ingress_pct, run))

if scenario in ('all', 'uniform'):
    expect('uniform', 'insert_or_assign')
    expect('uniform', 'find')
if scenario in ('all', 'bucket_skew'):
    for divisor in (4, 16, 64, 256):
        expect(f'bucket_skew_1over{divisor}', 'insert_or_assign')
if scenario in ('all', 'single_bucket'):
    expect('single_bucket_pressure', 'insert_or_assign')
if scenario in ('all', 'zipf'):
    expect('zipf_hotset', 'insert_or_assign')
if scenario in ('all', 'rui'):
    expect('rui_serial', 'fixed_work_find_assign_insert', 1, kind, overlap, ingress)
    expect('rui_overlap', 'fixed_work_find_assign_insert', streams, kind, overlap, ingress)
for group in ('fixed_work', 'streaming'):
    if scenario not in ('all', group):
        continue
    for worker_count in ([streams] if 'streams' in options else (1, 2, 4, 8)):
        for overlap_count in ([overlap] if 'overlap_pct' in options else (0, 50, 100)):
            for overlap_type in ([kind] if 'overlap_kind' in options else ('hot_key', 'same_bucket')):
                expect(group, 'fixed_work_find_assign_insert', worker_count, overlap_type,
                       overlap_count, ingress if group == 'streaming' else 0)
assert expected, 'Empty or invalid expected selection'
seen = set()
fingerprints = {}
with open(sys.argv[1], newline='') as source:
    rows = list(csv.DictReader(source))
assert rows, 'Empty benchmark result'
for row in rows:
    assert None not in row and None not in row.values(), 'Malformed CSV'
    identity = (row['scenario'], row['operation'], int(row['streams']), row['overlap_kind'],
                int(row['overlap_pct']), int(row['ingress_pct']), int(row['run']))
    assert identity in expected and identity not in seen, f'Unexpected/duplicate row: {identity}'
    seen.add(identity)
    assert row['status'] == 'PASS' and int(row['oracle_errors']) == 0 and int(row['worker_errors']) == 0
    assert row['mode'] == mode and row['table_mode'] == 'throughput'
    assert int(row['seed']) == seed and int(row['capacity']) == capacity, 'Wrong workload configuration'
    assert math.isclose(float(row['target_load_factor']), load_factor, rel_tol=1e-8, abs_tol=1e-9)
    for field in ('read_requests', 'update_requests', 'insert_requests', 'read_hits', 'read_misses',
                  'initial_size', 'final_size', 'cross_worker_key_requests', 'cross_worker_bucket_requests'):
        assert int(row[field]) >= 0, 'Negative counter'
    completed = int(row['completed_requests'])
    assert completed == int(row['attempted']) and completed > 0, 'Incomplete fixed work'
    assert sum(int(row[field]) for field in ('read_requests', 'update_requests', 'insert_requests')) == completed
    assert int(row['read_hits']) + int(row['read_misses']) == int(row['read_requests'])
    assert int(row['oracle_checked_rows']) > 0, 'Empty oracle'
    seconds = float(row['seconds'])
    rate = float(row['throughput_grequest_s'])
    assert math.isfinite(seconds) and seconds > 0 and math.isfinite(rate) and rate >= 0
    assert math.isclose(rate, completed / seconds / 1e9, rel_tol=1e-5, abs_tol=1e-8), 'Wrong decimal throughput'
    for prefix in ('', 'service_'):
        quantiles = [float(row[prefix + field]) for field in ('p50_us', 'p95_us', 'p99_us')]
        assert all(math.isfinite(value) and value >= 0 for value in quantiles)
        assert quantiles == sorted(quantiles), 'Invalid quantiles'
    assert int(row['latency_samples']) > 0
    if row['operation'] == 'fixed_work_find_assign_insert':
        assert completed == total and row['roles'] == roles
        assert int(row['request_batch']) == request_batch
        assert int(row['latency_samples']) == (total + request_batch - 1) // request_batch
        assert int(row['initial_size']) == int(load_factor * 128) * (capacity // 128)
        for role, field in (('R', 'read_requests'), ('U', 'update_requests'), ('I', 'insert_requests')):
            assert int(row[field]) == role_counts[role], 'Wrong fixed role budget'
        assert int(row['final_size']) <= capacity
        assert int(row['cross_worker_key_requests']) <= total
        assert int(row['cross_worker_bucket_requests']) <= total
        group = (row['scenario'], row['overlap_kind'], row['overlap_pct'], row['ingress_pct'], row['run'])
        prior = fingerprints.setdefault(group, row['trace_hash'])
        assert row['trace_hash'] == prior, 'Worker sweep changed workload'
    else:
        planned = min(adversarial_batch, 512) if row['scenario'] == 'single_bucket_pressure' else batch
        assert completed == planned and int(row['request_batch']) == planned, 'Wrong original batch budget'
        assert int(row['latency_samples']) == 1
assert seen == expected, f'Missing rows: {expected - seen}'
print(f'{len(rows)} source-scoped rows passed completion/oracle/count/rate gates; DEVELOPMENT_ONLY')
PY
printf 'Wrote %s\n' "$RESULTS"
