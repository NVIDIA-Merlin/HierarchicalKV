# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real find_impl/create_scores AST routing regression, without torch/CUDA."""

import ast
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

if not os.environ.get("HKV_RECSYS_ROOT"):
    raise unittest.SkipTest("NOT_RUN: set HKV_RECSYS_ROOT to the applied RecSys250aaa checkout")
ROOT = Path(os.environ["HKV_RECSYS_ROOT"]).resolve() / "corelib/dynamicemb"
NEW = ROOT / "dynamicemb/key_value_table.py"
OLD = Path(__file__).resolve().parent / "fixtures/legacy_250aaa_key_value_methods.txt"


class Tensor:
    def __init__(self, data, dtype="i64"):
        self.data = np.asarray(data)
        self.dtype = dtype
        self.device = "cpu"
    def size(self, i): return self.data.shape[i]
    def dim(self): return self.data.ndim
    def numel(self): return self.data.size
    def to(self, dtype): return Tensor(self.data, dtype)
    def __getitem__(self, key):
        if isinstance(key, Tensor): key = key.data
        return Tensor(self.data[key], self.dtype)
    def __setitem__(self, key, value): self.data[key] = value.data if isinstance(value, Tensor) else value
    def sum(self): return Tensor(self.data.sum())
    def cpu(self): return self
    def item(self): return self.data.item()
    def fill_(self, value): self.data.fill(value); return self
    def zero_(self): self.data.fill(0); return self


def empty(*shape, dtype="fp32", device=None):
    return Tensor(np.zeros(shape, dtype=bool if dtype == "bool" else np.int64 if dtype in ("i64", "u64") else np.float32), dtype)


def select(mask, source, result, count):
    selected = source.data[mask.data]
    result.data[:len(selected)] = selected
    count.data[0] = len(selected)


def select_index(mask, result, count):
    selected = np.flatnonzero(mask.data)
    result.data[:len(selected)] = selected
    count.data[0] = len(selected)


def compile_methods(path, calls):
    def scalar(table, batch, keys, pointers, founds, *token):
        calls.append(("scalar", token))
        founds.data[:] = True
    def vector(table, batch, keys, pointers, founds, scores):
        calls.append(("vector", scores))
        founds.data[:] = True
    torch = NS(Tensor=Tensor, bool="bool", long="i64", uint64="u64", empty=empty,
               empty_like=lambda value: empty(*value.data.shape, dtype=value.dtype),
               logical_not=lambda value: Tensor(np.logical_not(value.data), "bool"),
               ones=lambda n, **kw: Tensor(np.ones(n), kw.get("dtype", "i64")))
    namespace = dict(torch=torch, find_pointers=scalar, find_pointers_with_scores=vector,
                     load_from_pointers=lambda pointers, values: values.data.fill(7),
                     select=select, select_index=select_index,
                     EvictStrategy=NS(KLru="lru", KLfu="lfu", KCustomized="custom"))
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KeyValueTable")
    for name in ("find_impl", "create_scores"):
        node = deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        for arg in node.args.args + node.args.kwonlyargs: arg.annotation = None
        node.returns = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class Routing(unittest.TestCase):
    def obj(self, namespace, strategy="lru", update=True, score=None):
        obj = NS(
            table=object(), key_type=lambda: "i64", value_type=lambda: "fp32", value_dim=lambda: 16,
            options=NS(bucket_capacity=128), _score_update=update, score=score,
            evict_strategy=lambda: strategy, _record_cache_metrics=True,
            _cache_metrics=Tensor(np.full(10, 99)),
        )
        obj.create_scores = lambda *args: namespace["create_scores"](obj, *args)
        return obj

    def run_find(self, path=NEW, strategy="lru", update=True, hint=False, n=3, score=None, key_dtype="i64", value_dtype="fp32", input_scores=None):
        calls = []
        namespace = compile_methods(path, calls)
        obj = self.obj(namespace, strategy, update, score)
        keys = Tensor(np.arange(n), key_dtype)
        values = empty(n, 16, dtype=value_dtype)
        kwargs = {"input_scores": input_scores}
        if "native_lru_hit_update_from_unique" in namespace["find_impl"].__code__.co_varnames:
            kwargs["native_lru_hit_update_from_unique"] = hint
        result = namespace["find_impl"](obj, keys, values, **kwargs)
        return calls, result, obj, values

    def test_reproduce_old_none_to_const(self):
        calls, _, _, _ = self.run_find(path=OLD, hint=False)
        self.assertEqual(calls, [("vector", None)])
        # Proven C++ contract: vector(None) chooses the const overload.

    def test_off_and_dedup_hint_false_mutable_lru(self):
        calls, result, _, values = self.run_find(hint=False)
        self.assertEqual(calls, [("scalar", (0,))])
        self.assertEqual(result[0], 0)
        self.assertIsNone(result[3])
        np.testing.assert_array_equal(values.data, np.full((3, 16), 7))

    def test_old_hint_same_primitive_no_ablation(self):
        a, *_ = self.run_find(hint=False)
        b, *_ = self.run_find(hint=True)
        self.assertEqual(a, b)

    def test_pure_read_no_token(self):
        for hint in (False, True):
            calls, *_ = self.run_find(update=False, hint=hint)
            self.assertEqual(calls, [("scalar", ())])

    def test_lru_scalar_score_value_is_not_a_gate(self):
        for score in (None, -1, 2**100, object()):
            calls, *_ = self.run_find(score=score)
            self.assertEqual(calls, [("scalar", (0,))])

    def test_lfu_vector_unchanged(self):
        counts = Tensor([2, 3, 7])
        calls, *_ = self.run_find(strategy="lfu", input_scores=counts)
        self.assertEqual(calls[0][0], "vector")
        self.assertIs(calls[0][1], counts)
        calls, *_ = self.run_find(strategy="lfu")
        np.testing.assert_array_equal(calls[0][1].data, np.ones(3))

    def test_custom_vector_unchanged(self):
        calls, *_ = self.run_find(strategy="custom", score=42)
        self.assertEqual(calls[0][0], "vector")
        np.testing.assert_array_equal(calls[0][1].data, np.full(3, 42))

    def test_empty_all_paths_clear_metrics_without_lookup(self):
        for strategy in ("lru", "lfu", "custom"):
            for hint in (False, True):
                calls, result, obj, _ = self.run_find(strategy=strategy, n=0, hint=hint)
                self.assertEqual(calls, [])
                self.assertEqual(result[0], 0)
                np.testing.assert_array_equal(obj._cache_metrics.data, np.zeros(10))

    def test_invalid_embedding_dtype_before_native(self):
        with self.assertRaisesRegex(RuntimeError, "Embedding dtype not match"):
            self.run_find(value_dtype="fp16")

    def test_key_conversion_kept(self):
        calls, *_ = self.run_find(key_dtype="i32")
        self.assertEqual(calls, [("scalar", (0,))])

    def test_original_native_failure_propagates(self):
        calls = []
        namespace = compile_methods(NEW, calls)
        obj = self.obj(namespace)
        obj.options.bucket_capacity = 1
        def unsupported(*args): raise RuntimeError("Not support update score when keys are not unique or bucket capacity is small.")
        namespace["find_pointers"] = unsupported
        with self.assertRaisesRegex(RuntimeError, "Not support update score"):
            namespace["find_impl"](obj, Tensor([1]), empty(1, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
