# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute real candidate methods with CPU fakes; never import torch/CUDA."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch
from typing import Tuple

if not os.environ.get("HKV_RECSYS_ROOT"):
    raise unittest.SkipTest("NOT_RUN: set HKV_RECSYS_ROOT to the applied RecSys250aaa checkout")
BASE = Path(os.environ["HKV_RECSYS_ROOT"]).resolve() / "corelib/dynamicemb"


def method(path, cls, name, namespace):
    tree = ast.parse((BASE / path).read_text())
    owner = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    node.returns = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(BASE / path), "exec"), namespace)
    return namespace[name]


class FakeKV:
    def __init__(self, option=None, optimizer=None):
        self.options = option or NS(bucket_capacity=128)
        self.strategy = "lru"

    def evict_strategy(self):
        return self.strategy


GRAD = False
TORCH = NS(int64="i64", float32="fp32", is_grad_enabled=lambda: GRAD)
ENV = dict(
    os=os, torch=TORCH, KeyValueTable=FakeKV,
    DynamicEmbPoolingMode=NS(NONE="none"),
    DynamicEmbScoreStrategy=NS(TIMESTAMP="timestamp"),
    EvictStrategy=NS(KLru="lru"), Tuple=Tuple,
)
PATH = "dynamicemb/batched_dynamicemb_tables.py"
CLASS = "BatchedDynamicEmbeddingTablesV2"
RESOLVE = method(PATH, CLASS, "_resolve_cache_setting", ENV)
CLAMP = method(PATH, None, "_clamp_bucket_allocation_group", ENV)
GROUPS = method(PATH, CLASS, "_feature_groups", ENV)
SUPPORTS = method(PATH, CLASS, "_supports_native_lru_hit_update", ENV)
MASK = method(PATH, CLASS, "_native_lru_hit_update_call_mask", ENV)


def instance(n=1, dim=32, mode="auto"):
    obj = NS(
        _caches=[FakeKV() for _ in range(n)], _storages=[FakeKV() for _ in range(n)],
        _dynamicemb_options=[NS(index_type="i64", embedding_dtype="fp32", dim=dim, score_strategy="timestamp") for _ in range(n)],
        pooling_mode="none", _caching=True, _enable_prefetch=False, use_index_dedup=False,
        output_dtype="fp32", native_lru_hit_update=mode, training=False,
    )
    obj._supports_native_lru_hit_update = lambda i: SUPPORTS(obj, i)
    return obj


class Settings(unittest.TestCase):
    def test_metadata_group_preserves_capacity_and_clamps(self):
        for capacity, expected in ((1024,8),(4096,32),(1703936,1024),(13631488,1024)):
            self.assertEqual(CLAMP(1024, capacity, 128), expected)
            self.assertEqual(CLAMP(1, capacity, 128), 1)
        for invalid in (0,-1,3,True,1.5,"1024"):
            with self.assertRaises(ValueError): CLAMP(invalid, 4096, 128)

    def test_legacy_training_compatibility(self):
        with patch.dict(os.environ, {"TEST_SWITCH": "1"}):
            mode = RESOLVE(None, "TEST_SWITCH", {"0": "off", "1": "auto"}, ("off", "auto", "on"))
        obj = instance(mode=mode)
        obj.training = True
        self.assertEqual(MASK(obj), (False,))

    def test_constructor_precedes_invalid_legacy(self):
        with patch.dict(os.environ, {"TEST_SWITCH": "invalid"}):
            self.assertEqual(RESOLVE("off", "TEST_SWITCH", {"0": "off", "1": "auto"}, ("off", "auto", "on")), "off")
            with self.assertRaises(ValueError):
                RESOLVE(None, "TEST_SWITCH", {"0": "off", "1": "auto"}, ("off", "auto", "on"))
            with self.assertRaises(ValueError):
                RESOLVE(True, "TEST_SWITCH", {}, ("off", "auto", "on"))

    def test_backing_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(RESOLVE(None, "TEST_SWITCH", {"0": "host", "1": "hbm"}, ("host", "hbm")), "host")
        with patch.dict(os.environ, {"TEST_SWITCH": "1"}):
            self.assertEqual(RESOLVE("host", "TEST_SWITCH", {"0": "host", "1": "hbm"}, ("host", "hbm")), "host")

    def test_existing_enforce_hbm_alias(self):
        tree = ast.parse((BASE / PATH).read_text())
        init = next(n for n in next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS).body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        conflict = next(n for n in init.body if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp) and "enforce_hbm" in ast.unparse(n.test))
        assignment = next(n for n in init.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "backing_placement" for t in n.targets))
        code = compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(conflict), deepcopy(assignment)], type_ignores=[])), "constructor-backing-setting", "exec")
        for enforce, explicit, expected in ((True, None, "hbm"), (True, "hbm", "hbm"), (False, "host", "host")):
            obj = NS(_resolve_cache_setting=RESOLVE)
            exec(code, dict(self=obj, enforce_hbm=enforce, backing_placement=explicit))
            self.assertEqual(obj.backing_placement, expected)
        with self.assertRaises(ValueError):
            exec(code, dict(self=NS(), enforce_hbm=True, backing_placement="host"))

    def test_arbitrary_logical_features(self):
        for n in (1, 3, 26, 31):
            self.assertEqual(GROUPS([0] * n, 1), [0, n])
        self.assertEqual(GROUPS([0, 0, 1, 2, 2], 3), [0, 2, 3, 5])
        for invalid in ([0, 1, 0], [1, 0], [-1], [0, 2], [True], [], [0.0]):
            with self.assertRaises(ValueError, msg=repr(invalid)):
                GROUPS(invalid, 2)

    def test_dimensions_not_benchmark_identity(self):
        for dim in (8, 16, 32, 64, 128):
            self.assertEqual(MASK(instance(n=3, dim=dim)), (True, True, True))

    def test_per_physical_cache_fallback(self):
        obj = instance(n=3)
        obj._caches[1].strategy = "lfu"
        obj._dynamicemb_options[1].score_strategy = "lfu"
        self.assertEqual(MASK(obj), (True, False, True))
        obj.native_lru_hit_update = "on"
        with self.assertRaises(RuntimeError):
            MASK(obj)

    def test_safety_constraints_preserved(self):
        for name, value in (("_enable_prefetch", True), ("use_index_dedup", True), ("output_dtype", "fp16"), ("pooling_mode", "sum"), ("_caching", False)):
            obj = instance()
            setattr(obj, name, value)
            self.assertEqual(MASK(obj), (False,), name)
        for field, value in (("index_type", "i32"), ("embedding_dtype", "fp16"), ("dim", 0)):
            obj = instance()
            setattr(obj._dynamicemb_options[0], field, value)
            self.assertEqual(MASK(obj), (False,), field)
        obj = instance()
        obj._caches[0].options.bucket_capacity = 1024
        self.assertEqual(MASK(obj), (False,))
        obj = instance()
        obj._storages[0] = object()
        self.assertEqual(MASK(obj), (False,))

    def test_calltime_reject_and_fallback(self):
        global GRAD
        for field, value in (("training", True), ("_enable_prefetch", True), ("use_index_dedup", True)):
            for mode in ("auto", "on"):
                obj = instance(mode=mode)
                setattr(obj, field, value)
                if mode == "on":
                    with self.assertRaises(RuntimeError):
                        MASK(obj)
                else:
                    self.assertEqual(MASK(obj), (False,))
        GRAD = True
        try:
            self.assertEqual(MASK(instance()), (False,))
            with self.assertRaises(RuntimeError):
                MASK(instance(mode="on"))
        finally:
            GRAD = False
        self.assertEqual(MASK(instance(), per_sample_weights=object()), (False,))
        with self.assertRaises(RuntimeError):
            MASK(instance(mode="on"), feature_requires_grad=object())
        self.assertEqual(MASK(instance(n=3, mode="off")), (False, False, False))

    def test_explicit_layout_and_prefixes(self):
        layout = method("benchmark/criteo_experiment/embedding_config.py", None, "_hkv_layout", {})
        self.assertEqual(layout([7, 13, 5], "shared"), ([25], [0, 0, 0], [0, 7, 20]))
        self.assertEqual(layout([7, 13, 5], "partitioned"), ([7, 13, 5], [0, 1, 2], [0, 0, 0]))
        for n in (1, 3, 26):
            self.assertEqual(layout([262144] * n, "shared")[0], [n * 262144])
        for invalid in ([0], [-1], [True], [], [1.1], [(1 << 63) - 3]):
            with self.assertRaises(ValueError):
                layout(invalid, "shared")

    def test_backing_value_budget_native_formula(self):
        captured = []
        def kv(option, optimizer):
            captured.append(option)
            return FakeKV(option, optimizer)
        namespace = dict(ENV, deepcopy=deepcopy, KeyValueTable=kv,
                         OptimizerType=NS(Null="null"), convert_optimizer_type=lambda _: "sgd",
                         get_constraint_capacity=lambda *args: 1024,
                         dtype_to_bytes=lambda dtype: {"fp32": 4, "fp16": 2}[dtype],
                         _print_memory_consume=lambda *args: None)
        create = method(PATH, CLASS, "_create_cache_storage", namespace)
        for n, dim, state_dim, max_cap, placement in ((1, 32, 0, 13631488, "hbm"), (3, 16, 16, 2048, "hbm"), (3, 64, 0, 4096, "host")):
            captured.clear()
            options = [NS(caching=True, training=True, optimizer_type="sgd", bucket_capacity=128, local_hbm_for_values=2**20, dim=dim, max_capacity=max_cap, init_capacity=max_cap, num_of_buckets_per_alloc=1, embedding_dtype="fp32", external_storage=None, io_by_cpu=False) for _ in range(n)]
            obj = NS(_dynamicemb_options=options, _optimizer=NS(get_state_dim=lambda _: state_dim), _optimizer_type="sgd", _table_names=[str(i) for i in range(n)], device_id=0, backing_placement=placement, requested_num_of_buckets_per_alloc=1024)
            create(obj)
            for storage in captured[1::2]:
                self.assertEqual(storage.local_hbm_for_values, max_cap * (dim + state_dim) * 4 if placement == "hbm" else 0)
            if max_cap == 13631488:
                self.assertEqual(captured[1].local_hbm_for_values, 1744830464)
            self.assertEqual(len(obj._caches), n)
            self.assertEqual(len(obj._storages), n)

    def test_empty_real_find_returns_without_native_launch(self):
        namespace = dict(ENV, torch=NS(Tensor=object), Optional=lambda: None)
        node = ast.parse((BASE / "dynamicemb/key_value_table.py").read_text())
        cls = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == "KeyValueTable")
        f = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "find_impl"))
        # Strip annotations only; execute the actual early return unchanged.
        for a in f.args.args + f.args.kwonlyargs:
            a.annotation = None
        f.returns = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), "key_value_table.py", "exec"), namespace)
        class Empty:
            dtype = "i64"
            def size(self, *args): return 0
            def __getitem__(self, key): return self
        class Values:
            dtype = "fp32"
            def dim(self): return 2
            def size(self, i): return 0 if i == 0 else 32
        metrics = NS(zero_=lambda: None)
        obj = NS(key_type=lambda: "i64", value_type=lambda: "fp32", _record_cache_metrics=True, _cache_metrics=metrics)
        result = namespace["find_impl"](obj, Empty(), Values(), native_lru_hit_update_from_unique=True)
        self.assertEqual(result[0], 0)
        self.assertIsNone(result[3])

    def test_full_function_routes_mask_per_physical_cache(self):
        # Execute the unmodified V2 forward with minimal CPU tensor/lookup fakes.
        import numpy as np
        class Tensor:
            def __init__(self, data): self.a = np.asarray(data); self.device = "cpu"
            @property
            def shape(self): return self.a.shape
            def __getitem__(self, key): return Tensor(self.a[key])
            def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value
        stores = [FakeKV(), FakeKV(), FakeKV()]
        for store in stores:
            store.embedding_dtype = lambda: "fp32"
            store.embedding_dim = lambda: 16
        flags = []
        def lookup(cache, storage, keys, values, *args, **kwargs):
            flags.append(kwargs["native_lru_hit_update_from_unique"])
            values.a[:] = keys.a[:, None]
        namespace = dict(torch=NS(Tensor=Tensor, dtype=object, empty=lambda *shape, **kw: Tensor(np.empty(shape, np.float32))),
                         List=list, Optional=object, Cache=object, Storage=object, BaseDynamicEmbInitializer=object, BaseDynamicEmbeddingOptimizer=object,
                         KeyValueTable=FakeKV, EvictStrategy=NS(KLfu="lfu"),
                         get_table_range=lambda *args: None,
                         segmented_unique=lambda *args: (args[0], None, None, [0, 2, 3, 5], None),
                         KeyValueTableCachingFunction=NS(lookup=lookup),
                         gather_embedding=lambda source, target, inverse: target.a.__setitem__(slice(None), source.a))
        tree = ast.parse((BASE / "dynamicemb/batched_dynamicemb_function.py").read_text())
        f = deepcopy(next(n for n in next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamicEmbeddingFunctionV2").body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
        f.decorator_list = []
        for a in f.args.args + f.args.kwonlyargs: a.annotation = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), "batched_dynamicemb_function.py", "exec"), namespace)
        result = namespace["forward"](NS(), Tensor([1, 2, 3, 4, 5]), None, stores, stores, None, "fp32", [None] * 3, None, None, False, False, False, None, None, (True, False, True))
        self.assertEqual(flags, [True, False, True])
        np.testing.assert_array_equal(result.a, np.repeat(np.arange(1, 6)[:, None], 16, axis=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
