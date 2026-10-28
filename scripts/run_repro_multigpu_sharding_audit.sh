#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RESULTS="${RESULTS:-$REPO_ROOT/outputs/repro-multigpu-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export BUILD_DIR="${BUILD_DIR:-$RESULTS/build}"
export GPU_COUNTS="${GPU_COUNTS:-1 2 4}"
export SCALING="${SCALING:-strong weak}"
export TRAFFIC="${TRAFFIC:-uniform hot}"
export LF="${LF:-0.75}" DIM="${DIM:-32}" CAPACITY="${CAPACITY:-65536}"
export BATCH_SIZE="${BATCH_SIZE:-1024}" WORKING_SET="${WORKING_SET:-131072}"
export CACHE_BYTES="${CACHE_BYTES:-67108864}" WARMUP="${WARMUP:-1}" RUNS="${RUNS:-3}"
export SEED="${SEED:-12345}" TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
export MODE="${MODE:-value_returning}" QUERY="${QUERY:-random}" HOT_FRACTION="${HOT_FRACTION:-0.9}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}" API_LOCK="${API_LOCK:-1}"
export SHARD_BINARY="$BUILD_DIR/hkv_lookup_modes_benchmark" CONTROL_TEST=0
if [[ "${1:-}" == "--control-test" && "$#" -eq 2 ]]; then
  export CONTROL_TEST=1 SHARD_BINARY="$2"
elif [[ "$#" -ne 0 ]]; then
  echo 'Usage: run_repro_multigpu_sharding_audit.sh [--control-test MOCK_WORKER]' >&2
  exit 1
fi
if [[ -e "$RESULTS" ]]; then echo "Refusing existing RESULTS: $RESULTS" >&2; exit 1; fi
mkdir -p "$RESULTS"/{logs,raw}
if [[ "$CONTROL_TEST" == 0 ]]; then
  if [[ ! -f "$REPO_ROOT/tests/googletest/CMakeLists.txt" ]]; then
    echo 'Missing preserved GoogleTest; ask coordinator, do not fetch/reset it' >&2
    exit 1
  fi
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -Dsm="${SM:-90}" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON \
    -DHKV_REPRO_LOOKUP_DIM="$DIM" -DHKV_REPRO_LOOKUP_CAPACITY="$CAPACITY" \
    -DHKV_REPRO_LOOKUP_BATCH_SIZE="$BATCH_SIZE" -DHKV_REPRO_LOOKUP_WARMUP="$WARMUP" \
    -DHKV_REPRO_LOOKUP_RUNS="$RUNS" -DHKV_REPRO_LOOKUP_BLOCK_SIZE="${BLOCK_SIZE:-128}" \
    -DHKV_REPRO_LOOKUP_API_LOCK="${API_LOCK:-1}" > "$RESULTS/logs/configure.log" 2>&1
  cmake --build "$BUILD_DIR" --target hkv_lookup_modes_benchmark -j"${JOBS:-8}" \
    > "$RESULTS/logs/build.log" 2>&1
fi
python3 - <<'PY'
import array
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time

root = pathlib.Path(os.environ['RESULTS'])
repo = pathlib.Path(os.environ['REPO_ROOT'])
mock = os.environ['CONTROL_TEST'] == '1'
kind = 'cpu_mock_control_only' if mock else 'development_gpu_runtime'
if sys.byteorder != 'little' or array.array('Q').itemsize != 8 or array.array('f').itemsize != 4:
    raise SystemExit('Worker protocol requires little-endian uint64/FP32')

def positive(name):
    value = int(os.environ[name])
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value

counts = sorted(set(int(value) for value in os.environ['GPU_COUNTS'].split()))
scaling_modes = os.environ['SCALING'].split()
patterns = os.environ['TRAFFIC'].split()
capacity, batch, working = (positive(name) for name in ('CAPACITY', 'BATCH_SIZE', 'WORKING_SET'))
budget, dim, runs = (positive(name) for name in ('CACHE_BYTES', 'DIM', 'RUNS'))
block_size, api_lock = int(os.environ['BLOCK_SIZE']), int(os.environ['API_LOCK'])
warmup, seed = int(os.environ['WARMUP']), int(os.environ['SEED'])
timeout = float(os.environ['TIMEOUT_SECONDS'])
load_factor, hot_fraction = float(os.environ['LF']), float(os.environ['HOT_FRACTION'])
if not counts or counts[0] != 1 or any(count not in (1, 2, 4, 8) for count in counts):
    raise ValueError('GPU_COUNTS must include 1 and use 1/2/4/8; unavailable counts fail, never skip')
if not scaling_modes or not set(scaling_modes) <= {'strong', 'weak'}:
    raise ValueError('SCALING must contain strong and/or weak')
if not patterns or not set(patterns) <= {'uniform', 'hot'}:
    raise ValueError('TRAFFIC must contain uniform and/or hot')
if os.environ['MODE'] != 'value_returning' or os.environ['QUERY'] not in ('random', 'sequential'):
    raise ValueError('Routed service requires value_returning and random/sequential queries')
if not math.isfinite(timeout) or timeout <= 0 or not 0 <= load_factor <= 1 or not 0 <= hot_fraction <= 1:
    raise ValueError('Invalid deadline/load factor/skew')
if warmup < 0 or seed < 0 or working < capacity or dim > 1024 or api_lock not in (0, 1) or block_size <= 0:
    raise ValueError('Invalid warmup/seed/working set/dimension')
if capacity & (capacity - 1) or capacity // max(counts) < 128 or budget % max(counts):
    raise ValueError('Capacity must be power-of-two with >=128 slots/shard; byte budget must divide all counts')
binary = pathlib.Path(os.environ['SHARD_BINARY']).resolve()
if not os.access(binary, os.X_OK):
    raise ValueError(f'Missing executable: {binary}')
gpu_metadata = []
if mock:
    devices = [str(index) for index in range(max(counts))]
else:
    selected = os.environ.get('GPU_IDS', os.environ.get('CUDA_VISIBLE_DEVICES'))
    if selected is None:
        selected = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True, timeout=timeout).strip().replace('\n', ',')
    devices = selected.split(',')
    if len(devices) < max(counts) or len(set(devices)) != len(devices) or any(not device or device == '-1' for device in devices):
        raise ValueError('Insufficient/duplicate visible GPUs; no count is silently skipped')
    for device in devices[:max(counts)]:
        metadata = subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=uuid,name,pci.bus_id,memory.total', '--format=csv,noheader'], text=True, timeout=timeout).strip()
        gpu_metadata.append(next(csv.reader([metadata], skipinitialspace=True)))
    if len({row[0] for row in gpu_metadata}) != max(counts) or len({row[1] for row in gpu_metadata}) != 1:
        raise ValueError('Requires distinct GPUs of the same model on this host')
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True, timeout=timeout)
    (root / 'topology.txt').write_text(topology)

def digest(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

manifest = dict(evidence_kind=kind, gpu_runs=0 if mock else None, hostname=os.uname().nodename,
                gpu_devices=devices[:max(counts)], gpu_metadata=gpu_metadata, seed=seed,
                scaling=scaling_modes, traffic=patterns, capacity=capacity, batch_size=batch,
                working_set=working, cache_allocation_budget_bytes=budget, dim=dim,
                block_size=block_size, api_lock=api_lock,
                load_factor=load_factor, warmup=warmup, runs=runs, hot_fraction=hot_fraction,
                query=os.environ['QUERY'], binary_sha256=digest(binary),
                source_sha256={name: digest(repo / name) for name in ('benchmark/hkv_lookup_modes_benchmark.cc.cu', 'scripts/run_repro_multigpu_sharding_audit.sh')},
                service='immutable_lookup_host_backing_readthrough_no_admission_no_writeback',
                timing='route_pack_pipe_H2D_find_D2H_pipe_unpack_host_miss_gather',
                memory_scope='allocator_requested_cache_plus_explicit_nonusable_padding_excludes_cuda_pool_runtime',
                transfer_scope='logical_copy_payload_bytes_not_measured_link_transactions',
                tail_scope='batch_arrival_to_shard_rows_merged_upper_bounds_no_network_queue',
                independence='in_process_batches_not_independent_experiments', formal_accepted=False)
(root / 'run-info.json').write_text(json.dumps(manifest, indent=2) + '\n')
mask64 = (1 << 64) - 1

def value_for(key, component):
    if component < 4:
        return float((key >> (16 * component)) & 65535)
    mixed = ((key ^ component) + 0x9e3779b97f4a7c15) & mask64
    mixed = ((mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9) & mask64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94d049bb133111eb) & mask64
    return float((mixed ^ (mixed >> 31)) & 0xffffff)

def exact_read(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            raise RuntimeError('Truncated worker response')
        chunks.extend(part)
    return chunks

def fields(process, expected, width):
    line = process.stdout.readline(4096)
    if not line.endswith(b'\n'):
        raise RuntimeError(f'Missing/broken {expected}: {line!r}')
    parts = line.decode('ascii').split()
    if len(parts) != width or parts[0] != expected:
        raise RuntimeError(f'Unexpected {expected}: {parts}')
    return parts[1:]

def prepare(process, sequence):
    process.stdin.write(f'PREPARE {sequence}\n'.encode())
    process.stdin.flush()
    if fields(process, 'PREPARED', 2) != [str(sequence)]:
        raise RuntimeError('Stale preparation acknowledgement')

def exchange(process, sequence, keys):
    process.stdin.write(f'GO {sequence} {len(keys)}\n'.encode())
    process.stdin.write(array.array('Q', keys).tobytes())
    process.stdin.flush()
    record = fields(process, 'DONE', 7)
    observed_sequence, completed, hits, start, stop = map(int, record[:5])
    gpu_ms = float(record[5])
    if observed_sequence != sequence or completed != len(keys) or not 0 <= hits <= completed:
        raise RuntimeError('Stale/wrong completion count')
    if start > stop or not math.isfinite(gpu_ms) or gpu_ms < 0:
        raise RuntimeError('Invalid worker interval')
    found = exact_read(process.stdout, completed)
    values = array.array('f')
    values.frombytes(exact_read(process.stdout, completed * dim * 4))
    return dict(completed=completed, hits=hits, start_ns=start, stop_ns=stop, gpu_ms=gpu_ms,
                receive_ns=time.monotonic_ns(), found=found, values=values)

def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]

raw_path = root / 'raw' / 'batches.jsonl.partial'
summary = []
with raw_path.open('x') as raw:
    for scaling in scaling_modes:
        for traffic in patterns:
            baseline_rate = None
            baseline_trace = {}
            for count in counts:
                factor = count if scaling == 'weak' else 1
                global_capacity, request_count, working_count, global_budget = capacity * factor, batch * factor, working * factor, budget * factor
                local_capacity, local_budget = global_capacity // count, global_budget // count
                initial_count = int(global_capacity * load_factor)
                backing = array.array('f', (value_for(key, component) for key in range(working_count) for component in range(dim)))
                processes, logs = [], []
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=count)
                measured = []
                try:
                    for rank in range(count):
                        resident = len(range(rank, initial_count, count))
                        log = (root / 'logs' / f'{scaling}-{traffic}-{count}-{rank}.log').open('xb')
                        logs.append(log)
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], CUDA_DEVICE_ORDER='PCI_BUS_ID')
                        command = [str(binary), '--shard-worker', str(rank), str(count), str(local_capacity), str(resident), str(request_count), str(local_budget)]
                        processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=environment))
                    futures = [pool.submit(fields, process, 'READY', 15) for process in processes]
                    ready = [list(map(int, future.result(timeout=timeout))) for future in futures]
                    for rank, record in enumerate(ready):
                        if record[:4] != [rank, count, local_capacity, len(range(rank, initial_count, count))]:
                            raise RuntimeError('Wrong initial shard state')
                        if any(value < 0 for value in record) or sum(record[4:8]) != local_budget or record[8] != local_budget:
                            raise RuntimeError('Cache-byte budget violated')
                        if record[9:11] != [request_count * (8 + dim * 4 + 1)] * 2 or record[11:] != [dim, block_size, api_lock]:
                            raise RuntimeError('Worker buffer/build configuration mismatch')
                    for sequence in range(warmup + runs):
                        rng = random.Random(seed + sequence)
                        requests = []
                        for index in range(request_count):
                            if traffic == 'hot' and rng.random() < hot_fraction:
                                key = rng.randrange(len(range(0, working_count, max(counts)))) * max(counts)
                            else:
                                key = index % working_count if os.environ['QUERY'] == 'sequential' else rng.randrange(working_count)
                            requests.append(key)
                        trace = hashlib.sha256(array.array('Q', requests).tobytes()).hexdigest()
                        if count == 1:
                            baseline_trace[sequence] = trace
                        elif scaling == 'strong' and baseline_trace[sequence] != trace:
                            raise RuntimeError('Strong-scaling request trace changed')
                        merged = array.array('f', [0.0]) * (request_count * dim)
                        for future in [pool.submit(prepare, process, sequence) for process in processes]:
                            future.result(timeout=timeout)
                        started = time.monotonic_ns()
                        indices, shards = [[] for unused in range(count)], [[] for unused in range(count)]
                        for index, key in enumerate(requests):
                            rank = key % count
                            indices[rank].append(index)
                            shards[rank].append(key)
                        routed = time.monotonic_ns()
                        pending = {pool.submit(exchange, process, sequence, shards[rank]): rank for rank, process in enumerate(processes)}
                        completions, latencies, shard_records = 0, [], []
                        host_miss_bytes = 0
                        gather_ns = 0
                        for future in concurrent.futures.as_completed(pending, timeout=timeout):
                            gather_started = time.monotonic_ns()
                            rank = pending[future]
                            record = future.result()
                            found, returned = record.pop('found'), record.pop('values')
                            if any(flag not in (0, 1) for flag in found) or sum(found) != record['hits']:
                                raise RuntimeError('Invalid membership/count response')
                            for offset, index in enumerate(indices[rank]):
                                key = requests[index]
                                if found[offset]:
                                    merged[index * dim:(index + 1) * dim] = returned[offset * dim:(offset + 1) * dim]
                                else:
                                    merged[index * dim:(index + 1) * dim] = backing[key * dim:(key + 1) * dim]
                                    host_miss_bytes += dim * 4
                            finished = time.monotonic_ns()
                            gather_ns += finished - gather_started
                            record.update(rank=rank, merged_ns=finished, request_count=len(indices[rank]),
                                          membership=list(found))
                            shard_records.append(record)
                            latencies.extend([(finished - started) / 1e6] * len(indices[rank]))
                            completions += record['completed']
                        stopped = time.monotonic_ns()
                        if completions != request_count:
                            raise RuntimeError('Incomplete global gather')
                        for record in shard_records:
                            rank = record['rank']
                            if not started <= record['start_ns'] <= record['stop_ns'] <= record['receive_ns'] <= stopped:
                                raise RuntimeError('Worker interval outside common host window')
                            for offset, index in enumerate(indices[rank]):
                                key = requests[index]
                                if record['membership'][offset] != int(key < initial_count):
                                    raise RuntimeError('Membership differs from frozen initial cache')
                                if merged[index * dim:(index + 1) * dim] != backing[key * dim:(key + 1) * dim]:
                                    raise RuntimeError('Gathered payload/order oracle failed')
                            del record['membership']
                        elapsed = (stopped - started) / 1e9
                        hits = sum(record['hits'] for record in shard_records)
                        gpu_start = min(record['start_ns'] for record in shard_records)
                        gpu_stop = max(record['stop_ns'] for record in shard_records)
                        row = dict(evidence_kind=kind, scaling=scaling, traffic=traffic, gpus=count,
                                   run=sequence - warmup, warmup=sequence < warmup, seed=seed + sequence,
                                   request_sha256=trace, global_capacity=global_capacity, global_working_set=working_count,
                                   initial_residents=initial_count, requested_cache_bytes=global_budget,
                                   nonusable_padding_bytes=sum(record[7] for record in ready),
                                   cache_device_bytes=sum(record[4] + record[7] for record in ready),
                                   cache_pinned_bytes=sum(record[5] for record in ready), cache_host_bytes=sum(record[6] for record in ready),
                                   request_device_bytes=sum(record[9] for record in ready), request_pinned_bytes=sum(record[10] for record in ready),
                                   backing_payload_bytes=len(backing) * 4, completed_requests=completions, hits=hits,
                                   misses=completions-hits, hit_rate=hits/completions,
                                   common_start_ns=started, common_stop_ns=stopped, common_wall_seconds=elapsed,
                                   throughput_grequests_s=completions/elapsed/1e9, route_ms=(routed-started)/1e6,
                                   gather_ms=gather_ns/1e6,
                                   gpu_host_service_envelope_ms=(gpu_stop-gpu_start)/1e6,
                                   batch_arrival_request_p50_ms=percentile(latencies, 0.5),
                                   batch_arrival_request_p95_ms=percentile(latencies, 0.95),
                                   batch_arrival_request_p99_ms=percentile(latencies, 0.99),
                                   max_to_mean_requests=max(len(shard) for shard in shards)/(request_count/count),
                                   h2d_logical_bytes=completions*8, d2h_logical_bytes=completions*(1+dim*4),
                                   ipc_request_bytes=completions*8, ipc_response_bytes=completions*(1+dim*4),
                                   host_backing_read_bytes=host_miss_bytes, p2p_bytes=0,
                                   cache_mutation='none', oracle_status='PASS', per_shard=shard_records,
                                   shard_initial_allocations=ready)
                        raw.write(json.dumps(row) + '\n')
                        raw.flush()
                        if sequence >= warmup:
                            measured.append(row)
                    for process in processes:
                        process.stdin.write(b'STOP\n')
                        process.stdin.flush()
                        process.stdin.close()
                    for process in processes:
                        if process.wait(timeout=timeout) != 0:
                            raise RuntimeError('Worker nonzero terminal status')
                    total = sum(row['completed_requests'] for row in measured)
                    wall = sum(row['common_wall_seconds'] for row in measured)
                    rate = total / wall / 1e9
                    if count == 1:
                        baseline_rate = rate
                    summary.append(dict(evidence_kind=kind, scaling=scaling, traffic=traffic, gpus=count,
                                        completed_requests=total, common_wall_seconds=wall, throughput_grequests_s=rate,
                                        speedup_vs_1gpu=rate/baseline_rate, efficiency=rate/baseline_rate/count,
                                        global_capacity=global_capacity, global_working_set=working_count,
                                        requested_cache_bytes=global_budget, seed=seed, in_process_batches=runs,
                                        independent_experiments=1, formal_accepted=False))
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    for process in processes:
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    pool.shutdown(wait=True, cancel_futures=True)
                    for process in processes:
                        for stream in (process.stdout, process.stdin):
                            try:
                                stream.close()
                            except OSError:
                                pass
                    for log in logs:
                        log.close()
summary_path = root / ('cpu_mock_control_summary.csv' if mock else 'multigpu_sharding_summary.csv')
with summary_path.with_suffix('.csv.partial').open('x', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
os.replace(summary_path.with_suffix('.csv.partial'), summary_path)
os.replace(raw_path, raw_path.with_suffix(''))
print(f'{kind}: validated controller output {summary_path}; formal acceptance remains blocked')
PY
