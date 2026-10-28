"""Small native FBGEMM interface check; the formal service checker is separate."""

import os
import unittest


class NativeTBEInterfaceTest(unittest.TestCase):
    def test_full_vectors_miss_hit_update_and_flush(self):
        if os.environ.get("HKV_RUN_NATIVE_TBE_TEST") != "1":
            self.skipTest("NOT_RUN: set HKV_RUN_NATIVE_TBE_TEST=1 for GPU test")
        try:
            import torch
            from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
                EmbeddingLocation,
            )
            from benchmark.criteo_experiment.config import ExperimentConfig
            from benchmark.criteo_experiment.embedding_config import (
                create_criteo_torchrec_tables,
            )
            from benchmark.criteo_experiment.serving_replay import _lookup
        except ImportError as error:
            self.skipTest(f"NOT_RUN: pinned RecSys/Torch/FBGEMM dependency: {error}")
        if not torch.cuda.is_available():
            self.skipTest("NOT_RUN: CUDA device unavailable")

        device = torch.device("cuda:0")
        config = ExperimentConfig(
            backend="torchrec",
            vocab_sizes=[128] * 26,
            embedding_dim=32,
            optimizer="exact_sgd",
            eviction="lru",
            gpu_ratio=0.25,
            batch_size=2,
            learning_rate=0.125,
        )
        native = create_criteo_torchrec_tables(config, device)
        self.assertEqual(
            type(native).__module__,
            "fbgemm_gpu.split_table_batched_embeddings_ops_training",
        )
        self.assertEqual(len(native.embedding_specs), 26)
        self.assertTrue(
            all(
                spec[2] == EmbeddingLocation.MANAGED_CACHING
                for spec in native.embedding_specs
            )
        )
        self.assertEqual(native.lxu_cache_state.shape[1], 32)
        self.assertIs(native.uvm_host_mapped, False)
        self.assertIs(native.prefetch_pipeline, False)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(native.weights_uvm))

        row = torch.arange(128, dtype=torch.float32).reshape(128, 1) / 256
        col = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 1024
        weights = [row + col + float(table) for table in range(26)]
        with torch.no_grad():
            for target, weight in zip(native.split_embedding_weights(), weights):
                self.assertEqual(tuple(target.shape), (128, 32))
                target.copy_(weight)
            native.reset_cache_states()

        sparse = torch.tensor([[1] * 26, [2] * 26], dtype=torch.int64)
        expected = torch.cat(
            [torch.stack([weight[1], weight[2]]) for weight in weights], dim=1
        )
        with torch.no_grad():
            cold = _lookup(native, config, sparse, device).cpu()
            hot = _lookup(native, config, sparse, device).cpu()
        torch.testing.assert_close(cold, expected, rtol=0, atol=0)
        torch.testing.assert_close(hot, expected, rtol=0, atol=0)
        self.assertTrue(bool((native.lxu_cache_state >= 0).any()))

        native.train()
        updated = _lookup(native, config, sparse, device)
        updated.sum().backward()

        # Native LRU pressure must write back dirty victims without an explicit
        # flush; two hot rows alone fit in the actual 832-slot cache.
        self.assertEqual(native.lxu_cache_state.numel(), 832)
        offsets = native.cache_hash_size_cumsum.cpu().tolist()
        self.assertEqual(len(offsets), 27)
        self.assertTrue(all(b - a == 128 for a, b in zip(offsets, offsets[1:])))
        resident = set(native.lxu_cache_state.cpu().flatten().tolist())
        dirty = {
            offsets[table] + key: (table, key)
            for table in range(26)
            for key in (1, 2)
            if offsets[table] + key in resident
        }
        self.assertTrue(dirty)
        for table, key in dirty.values():
            torch.testing.assert_close(
                native.split_embedding_weights()[table][key].cpu(),
                weights[table][key],
                rtol=0,
                atol=0,
            )
        pressure_keys = [key for key in range(128) if key not in (1, 2)]
        self.assertGreater(len(pressure_keys) * 26, native.lxu_cache_state.numel())
        with torch.no_grad():
            for key in pressure_keys:
                request = torch.full((1, 26), key, dtype=torch.int64)
                output = _lookup(native, config, request, device).cpu()
                oracle = torch.cat([weight[key : key + 1] for weight in weights], dim=1)
                torch.testing.assert_close(output, oracle, rtol=0, atol=0)
        after_pressure = set(native.lxu_cache_state.cpu().flatten().tolist())
        evicted = [
            pair for cache_id, pair in dirty.items() if cache_id not in after_pressure
        ]
        self.assertTrue(evicted, "pressure did not naturally evict a dirty row")
        for table, key in evicted:
            torch.testing.assert_close(
                native.split_embedding_weights()[table][key].cpu(),
                weights[table][key] - 0.125,
                rtol=0,
                atol=0,
            )
        with torch.no_grad():
            restored_before_flush = _lookup(native, config, sparse, device).cpu()
        torch.testing.assert_close(
            restored_before_flush, expected - 0.125, rtol=0, atol=0
        )

        native.flush()
        for target, before in zip(native.split_embedding_weights(), weights):
            expected_backing = before.clone()
            expected_backing[1:3] -= 0.125
            torch.testing.assert_close(target.cpu(), expected_backing, rtol=0, atol=0)
        native.reset_cache_states()
        with torch.no_grad():
            restored = _lookup(native, config, sparse, device).cpu()
        torch.testing.assert_close(restored, expected - 0.125, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
