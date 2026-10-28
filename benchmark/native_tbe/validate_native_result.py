import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shared_dirty_evidence(dirty, required_tables):
    """Check concrete vectors, residency and the finite pressure trace."""
    require(required_tables == 26, "shared dirty coverage pin must require 26 tables")
    spec = dirty["spec"]
    base_initial = [(component + 8) / 8 for component in range(32)]
    gradient = [
        (component + 1) / 16 * (-1 if component % 2 else 1)
        for component in range(32)
    ]
    base_resident = [-(component + 8) / 16 for component in range(32)]
    resident_gradient = [-(32 - component) / 32 for component in range(32)]
    rate = 0.125
    base_expected = [value - rate * grad for value, grad in zip(base_initial, gradient)]
    base_resident_expected = [
        value - rate * grad
        for value, grad in zip(base_resident, resident_gradient)
    ]
    require(
        spec == {
            "table": 0,
            "eviction_key": 0,
            "resident_key": 1,
            "learning_rate": rate,
            "initial": base_initial,
            "gradient": gradient,
            "expected": base_expected,
            "resident_initial": base_resident,
            "resident_gradient": resident_gradient,
            "resident_expected": base_resident_expected,
            "pressure_first_key": 2,
            "pressure_end_key": 262144,
            "pressure_requests": 262142,
        },
        "shared dirty oracle changed the original table-zero spec",
    )
    require(dirty["table_coverage"] == list(range(26)), "dirty table coverage incomplete")
    require(dirty["backing_batch_reads"] == 4 and dirty["backing_keys_per_read"] == 26, "dirty shared backing read coverage incomplete")
    rows = dirty["table_results"]
    require(isinstance(rows, list) and len(rows) == 26, "dirty table results incomplete")
    observed_fields = {
        "backing_old_after_dirty": "initial",
        "backing_after_pressure": "expected",
        "restored_after_real_miss": "expected",
        "backing_old_resident": "resident_initial",
        "backing_after_flush": "resident_expected",
        "restored_after_flush_reset": "resident_expected",
    }
    for table, row in enumerate(rows):
        require(row["table"] == table, "dirty table results reordered or duplicated")
        expected = {
            "initial": [value + table for value in base_initial],
            "gradient": gradient,
            "expected": [value + table for value in base_expected],
            "resident_initial": [value + table for value in base_resident],
            "resident_gradient": resident_gradient,
            "resident_expected": [
                value + table for value in base_resident_expected
            ],
        }
        require(row["vectors"] == expected, f"dirty table {table} vector spec mismatch")
        for observed, field in observed_fields.items():
            require(
                row[observed] == expected[field],
                f"dirty table {table} {observed} full vector mismatch",
            )
    expected_snapshots = {
        "dirty_eviction_resident": (0, True),
        "evicted_after_pressure": (0, False),
        "restored_after_real_miss": (0, True),
        "dirty_resident_before_flush": (1, True),
        "absent_after_reset": (1, False),
        "restored_after_flush_reset": (1, True),
    }
    require(set(dirty["snapshots"]) == set(expected_snapshots), "dirty snapshot stages incomplete")
    for name, (key, present) in expected_snapshots.items():
        snapshot = dirty["snapshots"][name]
        require(snapshot["key"] == key and snapshot["full_cache_exports"] == 1, "dirty snapshot repeated or wrong key")
        require(
            snapshot["targets"]
            == [
                {"table": table, "key": key, "present": present}
                for table in range(26)
            ],
            f"dirty snapshot {name} lacks 26 exact target states",
        )
    require(
        dirty["pressure_trace_scope"]
        == "one complete ascending sweep of keys[2,vocab_rows) in every table; identical finite trace, no adaptive victim search",
        "dirty pressure scope changed",
    )
    digest = hashlib.sha256()
    for key in range(2, 262144):
        digest.update(struct.pack("<q", key) * 26)
    require(dirty["pressure_trace_sha256"] == digest.hexdigest(), "dirty pressure trace differs from finite ascending sweep")


def validate_flush_repeat_cell(path, result, pins):
    allowed_flush_counts = pins["flush_repeat_cells"]
    require(type(allowed_flush_counts) is dict and len(allowed_flush_counts) == 10, "flush-repeat whitelist must name exactly ten paired cells")
    expected_flush_count = allowed_flush_counts.get(Path(path).name)
    require(type(expected_flush_count) is int and expected_flush_count == 1, "unlisted or invalid flush-repeat cell")
    require(type(result["requested_flush_repeats"]) is int and result["requested_flush_repeats"] == expected_flush_count, "wrong requested flush count")
    require(type(result["flush_count"]) is int and result["flush_count"] == expected_flush_count, "wrong completed native flush count")
    require(result["diagnostic_work_scaling"] is (expected_flush_count == 2), "wrong diagnostic work-scaling label")


def validate_posttimer_budget_diagnostic(result, backend):
    observation = result["memory_final"].get("post_timer_budget_bound_candidate")
    require(type(observation) is dict, "missing post-timer budget field")
    require(observation.get("status") == "capture_only_coverage_incomplete",
            "unexpected budget capture status")
    require(observation.get("budget_qualification") == "HOLD" and
            observation.get("coverage_complete") is False,
            "budget must remain unqualified")
    require(observation.get("phase") ==
            "after_final_flush_device_sync_and_elapsed_timestamp" and
            observation.get("post_timer_not_in_service_denominator") is True and
            observation.get("allocation_identities_during_timed_service") is False,
            "post-timer mode0 boundary changed")
    for key in ("map_node_provider_binding_f52", "torch_pinned_allocator_route",
                "native_live_rosters", "static_timestamp", "native_live_cuda_extents"):
        part = observation.get(key)
        require(type(part) is dict and type(part.get("status")) is str,
                f"missing post-timer component {key}")
    provider = observation["map_node_provider_binding_f52"]
    if backend == "torchrec":
        require(provider["status"] == "not_applicable", "TBE provider changed")
    elif provider["status"] == "exact_f52_got_chain_observed":
        require(provider.get("map_node_provider_bound_to_glibc") is True and
                provider.get("f52_loaded_sha256") ==
                "dc3b4000dac266d0b559b65b9a3a500dfabcb3e9681cf25f06915f4670578a63" and
                provider.get("f52_build_id") ==
                "2f36cfb73774aad1e9ef466f502e69efc1c212dd",
                "wrong observed f52 provider identity")
    route = observation["torch_pinned_allocator_route"]
    if route["status"] == "observed_unqualified":
        require(type(route.get("pinned_use_cuda_host_register")) is bool and
                route.get("route_stable_whole_interval_proved") is False,
                "wrong observed Torch pinned route")
    timestamp = observation["static_timestamp"]
    rosters = observation["native_live_rosters"]
    extents = observation["native_live_cuda_extents"]
    if backend == "torchrec":
        require(timestamp["status"] == "not_applicable" and
                rosters["status"] == "not_applicable" and
                extents["status"] == "not_applicable",
                "TBE native-owner diagnostics changed")
    else:
        if timestamp["status"] in ("not_initialized_absent", "observed_live_unqualified"):
            require(timestamp.get("f52_loaded_sha256") ==
                    "dc3b4000dac266d0b559b65b9a3a500dfabcb3e9681cf25f06915f4670578a63" and
                    timestamp.get("f52_build_id") ==
                    "2f36cfb73774aad1e9ef466f502e69efc1c212dd" and
                    timestamp.get("guard_symbol_offset") == "0x292e0988" and
                    timestamp.get("object_symbol_offset") == "0x292e0990",
                    "wrong observed f52 timestamp identity")
            rows = timestamp.get("external_direct_rows")
            require(type(rows) is list and len(rows) ==
                    (1 if timestamp["status"] == "observed_live_unqualified" else 0),
                    "wrong timestamp owner count")
            if rows:
                row = rows[0]
                require(type(row.get("pointer")) is int and row["pointer"] > 0 and
                        row.get("requested_bytes") == 8 and
                        row.get("origin") == "f52_static_DeviceTimestamp_cudaMalloc",
                        "wrong timestamp owner")
        if rosters["status"] == "observed_unqualified":
            owners = rosters.get("rosters")
            require(type(owners) is list and len(owners) == 2,
                    "wrong shared native owner count")
            group = [row for owner in owners
                     for row in owner["native"]["direct_group_lock_rows"]]
            require(len(group) == 6 and len({row["pointer"] for row in group}) == 6 and
                    all(type(row.get("pointer")) is int and row["pointer"] > 0 and
                        row.get("allocation_api") == "cudaMalloc" and
                        row.get("counted") is False and
                        row.get("in_allocator_map") is False for row in group) and
                    all(owner["native"].get("group_lock_source_sha256") ==
                        "c12d77929dc2124a2c8dd724f2fde4f16d6ec59fc16389421e7d6dea6e89a1fa"
                        for owner in owners), "wrong typed group-lock owners")
        if extents["status"] == "observed_unqualified":
            extra = 6 + (1 if timestamp["status"] == "observed_live_unqualified" else 0)
            union = extents.get("legacy_api_allocation_region_union")
            require(extents.get("extra_direct_row_count") == extra and
                    type(union) is dict and type(union.get("bytes")) is int and
                    union["bytes"] > 0 and type(union.get("intervals")) is list,
                    "wrong direct legacy-region diagnostic")
    require(type(observation.get("unproved_coverage")) is list and
            bool(observation["unproved_coverage"]), "missing coverage gaps")


def read_result(path, backend, input_hash, gpu_uuid, expected_input, pins):
    payload = Path(path).read_bytes()
    require(payload.endswith(b"\n"), "truncated JSONL: no final newline")
    events = [json.loads(line) for line in payload.splitlines()]
    require(events and events[-1]["kind"] == "result", "no final result")
    require(sum(event["kind"] == "result" for event in events) == 1, "duplicate result")
    result = events[-1]["value"]
    require(result["format"] == "hkv-native-tbe-v2-1", "unexpected format")
    require(result["backend"] == backend and result["state"] == "warm", "wrong arm")
    require(result["status"] == "complete" and result["parity"] == "exact", "failed parity")
    require(result["formal_performance_admitted"] is False, "formal admission forbidden")
    validate_flush_repeat_cell(path, result, pins)
    require(result["input_sha256"] == input_hash, "wrong input")
    require(result["hardware"]["uuid"].removeprefix("GPU-").lower() == gpu_uuid.removeprefix("GPU-").lower(), "wrong GPU")
    cell = result["cell"]
    require(cell == {"seed": expected_input["cell"]["seed"], "distribution": "uniform", "target_miss_percent": 25, "occupancy": "full-overfill", "eviction": "lru", "cache_rows": 65536, "batch_size": 8192, "batches": 1}, "unexpected command defaults")
    require(result["workload"] == {"vocab_rows_per_table": 262144, "reference_cache_rows_per_table": 65536, "working_rows_per_table": 262144}, "wrong frozen workload")
    capacity = result["native_capacity"]
    requested_rows = 65536
    hkv_layout = pins.get("hkv_layout", "partitioned")
    shared_hkv = backend == "dynamicemb" and hkv_layout == "shared"
    require(
        result["hkv_layout"] == (hkv_layout if backend == "dynamicemb" else "partitioned"),
        "wrong HKV layout opt-in",
    )
    require(result["config"]["hkv_layout"] == result["hkv_layout"], "config layout mismatch")
    require(result["native"]["hkv_layout"] == result["hkv_layout"], "loaded backend layout mismatch")
    partitions = [65536] * 26 if backend == "dynamicemb" and not shared_hkv else [1703936]
    partition_kind = "per_table" if backend == "dynamicemb" and not shared_hkv else "shared_across_tables"
    native_layout = result["native_layout"]
    require(native_layout["logical_tables"] == 26, "wrong logical table count")
    require(native_layout["physical_tables"] == (1 if shared_hkv else 26), "wrong physical table count")
    require(native_layout["cache_partitions"] == len(partitions), "wrong cache partition count")
    require(native_layout["partition"] == partition_kind, "wrong native cache layout")
    policy = result["native_policy"]
    require(policy["logical_tables"] == 26, "wrong native policy logical table count")
    require(policy["physical_tables"] == native_layout["physical_tables"], "wrong native policy physical table count")
    require(policy["partition"] == partition_kind, "wrong native policy partition")
    if shared_hkv:
        require(native_layout["key_encoding"] == "table_id * 262144 + local_id", "wrong shared key encoding")
    require(capacity["explicit_controls"] is True and capacity["request_satisfied"] is True, "capacity request not satisfied")
    require(capacity["matched_total_budget"] is False, "calibration cannot establish matched budget")
    require(capacity["partition"] == partition_kind, "wrong native partition")
    require(capacity["requested_rows_per_table_equivalent"] == requested_rows, "wrong requested capacity")
    require(capacity["effective_rows_by_native_partition"] == partitions, "wrong effective partition capacities")
    total_rows = 26 * requested_rows
    require(capacity["requested_total_rows"] == capacity["effective_total_rows"] == total_rows, "wrong total native capacity")
    require(capacity["requested_payload_capacity_bytes"] == capacity["effective_payload_capacity_bytes"] == total_rows * 128, "wrong payload capacity")
    expected = cell["batch_size"] * cell["batches"]
    require(type(result["expected_requests"]) is int and result["expected_requests"] == expected, "wrong expected requests")
    measurement = result["measurement"]
    elapsed = measurement["host_wall_seconds"]
    require(math.isfinite(elapsed) and elapsed > 0, "invalid interval")
    records = result["batches"]
    require(len(records) == cell["batches"], "incomplete batches")
    completed = 0
    previous = 0
    for index, record in enumerate(records):
        require(record["batch"] == index and record["first_request_id"] == completed, "reordered completion")
        require(type(record["rows"]) is int and record["rows"] == cell["batch_size"], "incomplete batch")
        boundary = record["completed_seconds"]
        require(math.isfinite(boundary) and previous < boundary <= elapsed, "invalid completion boundary")
        completed += record["rows"]
        previous = boundary
    require(completed == expected == measurement["completed_requests"], "wrong completed denominator")
    require(measurement["embedding_rows"] == completed * 26, "wrong embedding denominator")
    require(math.isclose(measurement["requests_per_second"], completed / elapsed, rel_tol=1e-12), "wrong rate denominator")
    weight_digests = result["weights_sha256"]
    require(isinstance(weight_digests, list) and len(weight_digests) == 26, "wrong weight digest count")
    digests = [result["trace_sha256"], *weight_digests, *result["output_sha256"]]
    require(len(result["output_sha256"]) == len(records), "missing output digests")
    require(all(isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest) for digest in digests), "invalid digest")
    require(result["trace_sha256"] == expected_input["trace_sha256"], "wrong frozen trace")
    require(weight_digests == expected_input["weights_sha256"], "wrong frozen weights")
    require(result["native"]["scaffold_sha256"] == pins["scaffold_sha256"], "wrong scaffold source")
    loaded = {Path(name).name: digest for name, digest in result["native"]["loaded_objects_sha256"].items()}
    require(loaded == pins["loaded_objects_sha256"][backend], "wrong native loaded libraries")
    require(result["native"]["backend_module_sha256"] == pins["backend_module_sha256"][backend], "wrong backend source")
    require("allocation_identity_observation" not in result["memory_initial"] and
            "allocation_identity_observation" not in result["memory_final"], "observer must be off")
    event = result["cuda_event_service"]
    require(math.isfinite(event["seconds"]) and event["seconds"] > 0, "invalid secondary service event")
    require(all(event.get(k) == v for k, v in pins["event_contract"].items()), "secondary event contract changed")
    require(result["dirty_update_restore"]["status"] == "PASS" and result["dirty_update_restore"]["exact_full_vectors"] is True, "dirty full-vector oracle failed")
    require(result["dirty_update_restore"]["native_policy"]["partition"] == partition_kind, "dirty oracle used the wrong cache layout")
    if shared_hkv:
        validate_shared_dirty_evidence(
            result["dirty_update_restore"],
            pins["shared_dirty_oracle_logical_tables"],
        )
    require(any(e["kind"] == "full_backing_oracle" and e["value"]["exact"] is True for e in events), "missing full backing oracle")
    require(len([e for e in events if e["kind"] == "preflight" and e["value"]["exact"] is True]) == 3, "missing three preflight oracles")
    if backend == "dynamicemb":
        require("dc3b4000dac266d0b559b65b9a3a500dfabcb3e9681cf25f06915f4670578a63" in result["native"]["loaded_objects_sha256"].values(), "wrong loaded f52 extension")
    validate_posttimer_budget_diagnostic(result, backend)
    return result


def main():
    report, backend, inputs, checksum, gpu_uuid, seed, metadata, pinfile, *pair = sys.argv[1:]
    expected_input = next(r["input"] for r in json.loads(Path(metadata).read_text())["records"] if r["seed"] == int(seed))
    pins = json.loads(Path(pinfile).read_text())
    require(len(pair) <= 1, "one previous arm only")
    input_hash = Path(checksum).read_text().split()[0]
    require(file_hash(inputs) == input_hash == expected_input["sha256"], "frozen input changed")
    current = read_result(report, backend, input_hash, gpu_uuid, expected_input, pins)
    if pair:
        other = "torchrec" if backend == "dynamicemb" else "dynamicemb"
        previous = read_result(pair[0], other, input_hash, gpu_uuid, expected_input, pins)
        for field in ("cell", "workload", "input_sha256", "trace_sha256", "weights_sha256", "output_sha256", "expected_requests", "service_contract"):
            require(current[field] == previous[field], "pair mismatch: " + field)
    print("PAIR_EXACT_CONTROL_ONLY_NOT_EQUAL_MEMORY" if pair else "ARM_EXACT_CONTROL_ONLY_NOT_EQUAL_MEMORY")


if __name__ == "__main__":
    main()
