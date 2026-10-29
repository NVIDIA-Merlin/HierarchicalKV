# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CUDA unit coverage using the existing V2 constructor/insert/export API.

This is correctness-only; root stages it with the existing SDK and b928 ELF.
No tests in this file are run during CPU preparation.
"""

import argparse
import os
import unittest
import gc
import hashlib
import json
from pathlib import Path

def load_native_dependencies():
    global torch, DynamicEmbPoolingMode, DynamicEmbScoreStrategy, DynamicEmbTableOptions, EmbOptimType, tables_module, BatchedDynamicEmbeddingTablesV2, batched_export_keys_values, extension
    import torch
    from dynamicemb import (
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
        EmbOptimType,
    )
    import dynamicemb.batched_dynamicemb_tables as tables_module
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
    from dynamicemb.key_value_table import batched_export_keys_values
    import dynamicemb_extensions as extension


def snapshot(cache):
    result = {}
    for keys, values, _, scores in batched_export_keys_values(cache.table, torch.device("cpu"), batch_size=1024):
        for key, value, score in zip(keys.tolist(), values.tolist(), scores.tolist()):
            result[key] = (value, score)
    return result


def expected_values(keys, dim, physical_table):
    return keys.to(torch.float32)[:, None] * 0.5 + torch.arange(dim, device=keys.device, dtype=torch.float32)[None, :] / 256 + physical_table * 100


def assert_lru_refreshed(before, after):
    for old, new in zip(before, after):
        assert old.keys() == new.keys()
        assert all(old[key][0] == new[key][0] for key in old)
        assert all(new[key][1] > old[key][1] for key in old)


def run_case(name, mapping, dim, placement="host", mode="auto", policy="timestamp", dedup=False):
    nphysical = max(mapping) + 1
    device = torch.device("cuda:0")
    score_strategy = DynamicEmbScoreStrategy.TIMESTAMP if policy == "timestamp" else DynamicEmbScoreStrategy.LFU
    options = [DynamicEmbTableOptions(
        dim=dim, init_capacity=4096, max_capacity=4096,
        index_type=torch.int64, embedding_dtype=torch.float32,
        device_id=0, score_strategy=score_strategy, caching=True,
        local_hbm_for_values=1024 * dim * 4, bucket_capacity=128,
    ) for _ in range(nphysical)]
    metadata_kwargs = {"num_of_buckets_per_alloc": 1024} if name == "three_independent_dim32_host" else {}
    tables = BatchedDynamicEmbeddingTablesV2(
        table_names=[f"physical_{i}" for i in range(nphysical)],
        table_options=options, feature_table_map=mapping,
        pooling_mode=DynamicEmbPoolingMode.NONE, optimizer=EmbOptimType.EXACT_SGD,
        use_index_dedup=dedup, output_dtype=torch.float32, device=device,
        native_lru_hit_update=mode, backing_placement=placement,
        **metadata_kwargs,
    )
    if metadata_kwargs:
        assert all(cache.options.num_of_buckets_per_alloc == 8 for cache in tables.caches)
        assert all(storage.options.num_of_buckets_per_alloc == 32 for storage in tables.tables)
    tables.eval()
    per_feature_keys = [torch.tensor([1, 2, 1, 3], device=device, dtype=torch.int64) + feature * 64 for feature in range(len(mapping))]
    indices = torch.cat(per_feature_keys)
    offsets = torch.arange(indices.numel() + 1, device=device, dtype=torch.int64)
    expected = torch.cat([expected_values(keys, dim, physical) for keys, physical in zip(per_feature_keys, mapping)])
    for physical, storage in enumerate(tables.tables):
        keys = torch.unique(torch.cat([keys for keys, group in zip(per_feature_keys, mapping) if group == physical]))
        if policy == "lfu":
            # Direct insert requires an initial scalar score for scored tables;
            # regular model forwards set this through their table-score loop.
            storage.set_score(1)
        storage.insert(keys, expected_values(keys, dim, physical))

    tables.set_record_cache_metrics(True)
    with torch.no_grad():
        first = tables(indices, offsets)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        initial_metrics = [cache.cache_metrics.tolist() for cache in tables.caches]
        assert sum(row[0] - row[1] for row in initial_metrics) == len(mapping) * 3, initial_metrics
        before = [snapshot(cache) for cache in tables.caches]
        selected = tables._native_lru_hit_update_call_mask()
        second = tables(indices, offsets)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, expected, rtol=0, atol=0)
        metrics = [cache.cache_metrics.tolist() for cache in tables.caches]
        assert all(row[0] == row[1] for row in metrics), metrics
        after = [snapshot(cache) for cache in tables.caches]
        score_refresh = []
        for physical, (old, new) in enumerate(zip(before, after)):
            assert old.keys() == new.keys()
            assert all(old[key][0] == new[key][0] for key in old)
            refreshed = sum(new[key][1] > old[key][1] for key in old)
            if policy == "timestamp":
                assert refreshed == len(old), (name, physical, refreshed, len(old))
            score_refresh.append({"keys": len(old), "increased": refreshed})
        readonly_before = [snapshot(cache) for cache in tables.caches]
        for physical, cache in enumerate(tables.caches):
            keys = torch.tensor(sorted(readonly_before[physical]), device=device, dtype=torch.int64)
            values = torch.empty((keys.numel(), dim), device=device, dtype=torch.float32)
            cache.score_update = False
            missing, *_ = cache.find_embeddings(keys, values)
            assert missing == 0
            torch.testing.assert_close(values, expected_values(keys, dim, physical), rtol=0, atol=0)
        torch.cuda.synchronize()
        assert readonly_before == [snapshot(cache) for cache in tables.caches]
        empty = tables(indices[:0], torch.zeros_like(offsets))
        torch.cuda.synchronize()
        assert tuple(empty.shape) == (0, dim)
        assert all(cache.cache_metrics.tolist() == [0] * 10 for cache in tables.caches)
        empty_verified = True
        tables.reset_cache_states()
        restored = tables(indices, offsets)
        torch.cuda.synchronize()
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        assert sum(int(cache.cache_metrics[0] - cache.cache_metrics[1]) for cache in tables.caches) == len(mapping) * 3
        tables.flush()
        for physical, storage in enumerate(tables.tables):
            keys = torch.unique(torch.cat([keys for keys, group in zip(per_feature_keys, mapping) if group == physical]))
            values = torch.empty((keys.numel(), dim), device=device, dtype=torch.float32)
            missing, *_ = storage.find_embeddings(keys, values)
            assert missing == 0
            torch.testing.assert_close(values, expected_values(keys, dim, physical), rtol=0, atol=0)

    # auto preserves training/grad calls; on rejects these before native dispatch.
    if mode in ("auto", "off"):
        tables.train()
        with torch.enable_grad():
            assert not any(tables._native_lru_hit_update_call_mask())
            before_train = [snapshot(cache) for cache in tables.caches]
            training_output = tables(indices, offsets)
            torch.testing.assert_close(training_output.detach(), expected, rtol=0, atol=0)
            if policy == "timestamp":
                assert_lru_refreshed(before_train, [snapshot(cache) for cache in tables.caches])
        tables.eval()
        with torch.enable_grad():
            assert not any(tables._native_lru_hit_update_call_mask())
            before_grad = [snapshot(cache) for cache in tables.caches]
            grad_output = tables(indices, offsets)
            torch.testing.assert_close(grad_output.detach(), expected, rtol=0, atol=0)
            if policy == "timestamp":
                assert_lru_refreshed(before_grad, [snapshot(cache) for cache in tables.caches])
    elif mode == "on":
        tables.train()
        with torch.no_grad():
            try:
                tables(indices, offsets)
            except RuntimeError as error:
                assert "eval/no-grad" in str(error)
            else:
                raise AssertionError("on accepted a training forward")
        tables.eval()

    torch.cuda.synchronize()
    record = dict(
        case=name, status="PASS", logical_features=len(mapping), feature_table_map=mapping,
        physical_caches=nphysical, dim=dim, mode=tables.native_lru_hit_update,
        selected=list(selected), strategy=policy, dedup=dedup, backing_placement=placement,
        requested_backing_capacity=[opt.max_capacity for opt in options],
        actual_backing_capacity=[extension.dyn_emb_capacity(storage.table) for storage in tables.tables],
        actual_cache_capacity=[extension.dyn_emb_capacity(cache.table) for cache in tables.caches],
        backing_value_budget_bytes=[storage.options.local_hbm_for_values for storage in tables.tables],
        requested_metadata_buckets_per_alloc=metadata_kwargs.get("num_of_buckets_per_alloc"),
        actual_cache_metadata_buckets_per_alloc=[cache.options.num_of_buckets_per_alloc for cache in tables.caches],
        actual_backing_metadata_buckets_per_alloc=[storage.options.num_of_buckets_per_alloc for storage in tables.tables],
        full_outputs=True, initial_true_miss_restore=True, reset_true_miss_restore=True,
        final_flush_backing=True, duplicate_keys=True, empty_keys=empty_verified,
        pure_read_no_score_touch=True,
        empty_skip_reason=None,
        score_refresh=score_refresh,
    )
    del tables, options, first, second, restored
    gc.collect()
    torch.cuda.empty_cache()
    return record


def select_cases(cases, requested):
    if requested is None:
        return cases
    names = {case[0] for case in cases}
    unknown = set(requested) - names
    if unknown:
        raise ValueError(f"unknown CUDA correctness cases: {sorted(unknown)}")
    return [case for case in cases if case[0] in requested]


def run_full_bucket_eviction(name):
    device = torch.device("cuda:0")
    dim = 16
    tables = BatchedDynamicEmbeddingTablesV2(
        table_names=["single_bucket"],
        table_options=[DynamicEmbTableOptions(
            dim=dim, init_capacity=256, max_capacity=256,
            index_type=torch.int64, embedding_dtype=torch.float32,
            device_id=0, score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=True, local_hbm_for_values=128 * dim * 4, bucket_capacity=128,
        )],
        pooling_mode=DynamicEmbPoolingMode.NONE, optimizer=EmbOptimType.EXACT_SGD,
        device=device, native_lru_hit_update="off", backing_placement="host",
    )
    tables.eval()
    cache = tables.caches[0]
    capacity = extension.dyn_emb_capacity(cache.table)
    assert capacity == 128, capacity
    arms = []
    with torch.no_grad():
        for touch in (False, True):
            cache.reset()
            # Individual natural insertions establish an unambiguous oldest key.
            for key in range(capacity):
                keys = torch.tensor([key], device=device, dtype=torch.int64)
                cache.insert(keys, expected_values(keys, dim, 0))
                torch.cuda.synchronize()
            before = snapshot(cache)
            assert len(before) == capacity
            hot = min(before, key=lambda key: before[key][1])
            assert sum(row[1] == before[hot][1] for row in before.values()) == 1
            keys = torch.tensor([hot], device=device, dtype=torch.int64)
            values = torch.empty((1, dim), device=device)
            cache.score_update = touch
            missing, *_ = cache.find_embeddings(keys, values)
            cache.score_update = False
            torch.cuda.synchronize()
            assert missing == 0
            torch.testing.assert_close(values, expected_values(keys, dim, 0), rtol=0, atol=0)
            after_read = snapshot(cache)
            if touch:
                assert after_read[hot][1] > max(row[1] for row in before.values())
            else:
                assert after_read == before
            newcomer = torch.tensor([capacity], device=device, dtype=torch.int64)
            count, evicted_keys, evicted_values, _ = cache.insert_and_evict(
                newcomer, expected_values(newcomer, dim, 0)
            )
            torch.cuda.synchronize()
            final = snapshot(cache)
            assert count == 1 and len(final) == capacity
            victim = int(evicted_keys[0])
            assert capacity in final
            torch.testing.assert_close(evicted_values, expected_values(evicted_keys, dim, 0), rtol=0, atol=0)
            assert (hot in final) == touch, (touch, hot, victim)
            assert (victim != hot) == touch
            assert all(final[key][0] == before[key][0] for key in final if key in before)
            arms.append(dict(mutable_hit_refresh=touch, oldest_hot_key=hot, victim=victim, hot_retained=hot in final))
    del tables, cache
    gc.collect()
    torch.cuda.empty_cache()
    return dict(case=name, status="PASS", physical_caches=1, actual_cache_capacity=capacity,
                natural_eviction=True, readonly_negative=True, optimization_mode="off", arms=arms)


def run_adagrad_payload(name):
    device = torch.device("cuda:0")
    dim = 16
    tables = BatchedDynamicEmbeddingTablesV2(
        table_names=["adagrad"], table_options=[DynamicEmbTableOptions(
            dim=dim, init_capacity=1024, max_capacity=1024,
            index_type=torch.int64, embedding_dtype=torch.float32,
            device_id=0, score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=True, local_hbm_for_values=256 * dim * 2 * 4, bucket_capacity=128,
        )], pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.EXACT_ADAGRAD, device=device,
        native_lru_hit_update="off", backing_placement="host",
    )
    tables.eval()
    cache, storage = tables.caches[0], tables.tables[0]
    keys = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
    embeddings = expected_values(keys, dim, 0)
    state_dim = storage.value_dim() - dim
    assert state_dim == dim
    state = torch.arange(keys.numel() * state_dim, device=device, dtype=torch.float32).reshape(keys.numel(), state_dim) / 128 + 0.5
    payload = torch.cat((embeddings, state), dim=1)
    storage.insert(keys, payload)
    offsets = torch.arange(keys.numel() + 1, device=device, dtype=torch.int64)
    def full_state(cache):
        result = {}
        for exported_keys, values, states, scores in batched_export_keys_values(cache.table, torch.device("cpu"), batch_size=1024):
            for key, value, row, score in zip(exported_keys.tolist(), values.tolist(), states.tolist(), scores.tolist()):
                result[key] = (value, row, score)
        return result
    with torch.no_grad():
        first = tables(keys, offsets)
        torch.testing.assert_close(first, embeddings, rtol=0, atol=0)
        before = full_state(cache)
        second = tables(keys, offsets)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, embeddings, rtol=0, atol=0)
        after = full_state(cache)
        assert before.keys() == after.keys()
        assert all(before[key][:2] == after[key][:2] and after[key][2] > before[key][2] for key in before)
        for index, key in enumerate(keys.tolist()):
            assert after[key][1] == state[index].tolist()
        tables.flush()
        restored = torch.empty_like(payload)
        missing, *_ = storage.find(keys, restored)
        assert missing == 0
        torch.testing.assert_close(restored, payload, rtol=0, atol=0)
    del tables, cache, storage
    gc.collect()
    torch.cuda.empty_cache()
    return dict(case=name, status="PASS", optimizer="EXACT_ADAGRAD", state_dim=state_dim,
                score_refresh=True, optimizer_payload_unchanged=True, full_flush_payload=True,
                optimization_mode="off", backing_placement="host")


def test_generic_cache_cuda(output=None, requested_cases=None):
    cases = [
        ("one_feature_dim16_host", [0], 16, "host", "auto", "timestamp", False),
        ("three_independent_dim32_host", [0, 1, 2], 32, "host", "auto", "timestamp", False),
        ("twentysix_shared_dim64_hbm", [0] * 26, 64, "hbm", "auto", "timestamp", False),
        ("twentysix_shared_dim32_hbm", [0] * 26, 32, "hbm", "auto", "timestamp", False),
        ("three_features_two_physical_dim16", [0, 0, 1], 16, "host", "on", "timestamp", False),
        ("explicit_off", [0, 0, 0], 32, "host", "off", "timestamp", False),
        ("auto_lfu_fallback", [0, 1, 2], 32, "host", "auto", "lfu", False),
        ("auto_dedup_fallback", [0, 0, 0], 32, "host", "auto", "timestamp", True),
        ("lru_full_bucket_eviction",),
        ("eval_adagrad_state_payload",),
    ]
    records = []
    for case in select_cases(cases, requested_cases):
        if case[0] == "lru_full_bucket_eviction":
            record = run_full_bucket_eviction(case[0])
        elif case[0] == "eval_adagrad_state_payload":
            record = run_adagrad_payload(case[0])
        else:
            record = run_case(*case)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    summary = dict(
        status="PASS", cases=records, case_count=len(records), module=tables_module.__file__,
        module_sha256=hashlib.sha256(Path(tables_module.__file__).read_bytes()).hexdigest(),
        extension=extension.__file__, independent_performance_samples=0,
        executed_cases=[record["case"] for record in records],
        planned_cases=[case[0] for case in cases],
        all_cases_executed=len(records) == len(cases),
    )
    if output is not None:
        Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "case_count": len(records)}, sort_keys=True), flush=True)


class DynamicEmbGenericInterfaceTest(unittest.TestCase):
    def test_generic_shapes_lru_and_payloads(self):
        if os.environ.get("HKV_RUN_NATIVE_TBE_TEST") != "1":
            self.skipTest("NOT_RUN: set HKV_RUN_NATIVE_TBE_TEST=1 for CUDA correctness")
        load_native_dependencies()
        self.assertTrue(torch.cuda.is_available(), "CUDA device unavailable")
        test_generic_cache_cuda(output=os.environ.get("HKV_DYNAMICEMB_TEST_OUTPUT"))


if __name__ == "__main__":
    if os.environ.get("HKV_RUN_NATIVE_TBE_TEST") != "1":
        print("NOT_RUN: set HKV_RUN_NATIVE_TBE_TEST=1 for CUDA correctness", flush=True)
        raise SystemExit(0)
    load_native_dependencies()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--case", action="append", dest="requested_cases")
    args = parser.parse_args()
    test_generic_cache_cuda(args.output, args.requested_cases)
