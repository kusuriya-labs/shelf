import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/hy4_flash"))
from convert_engram import convert
from octave import Cache, selected_tokens
import placement
from placement import EXPERT_BYTES, EXPERT_BYTES_3BIT, assign


class Hy4FlashTests(unittest.TestCase):
    def test_engram(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            x = torch.linspace(-1, 1, 63).bfloat16().reshape(7, 9)
            x[2].zero_()
            save_file({"table": x}, directory / "source")
            out = load_file(convert(directory / "source", "table", [7, 9], directory, chunk=2))
            scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
            expected = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16() * scale
            self.assertTrue(torch.equal(expected, out["table"].bfloat16() * out["table.scale"]))
            with self.assertRaises(FileExistsError):
                convert(directory / "source", "table", [7, 9], directory)

    def test_rejection_at_ring_wrap(self):
        cache = Cache(1024, True, "cpu")
        x = torch.arange(512).bfloat16()
        for i in range(515):
            cache.append(x + i, x[:64] - i)
        positions = torch.arange(3, 515)
        expected = cache.get(positions)
        for i in range(6):
            cache.append(x - i, x[:64])
        cache.rewind(515)
        restored = cache.get(positions)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, restored)))

    def test_sparse_boundary(self):
        for pos in (2047, 2048, 2049, 2051, 2052):
            ids = selected_tokens(torch.arange(pos // 4).float(), pos)
            self.assertEqual(len(ids.unique()), len(ids))
            self.assertLessEqual(len(ids), 2048)
            self.assertEqual(ids.max().item(), pos)
            if pos >= 2048:
                self.assertEqual(ids.min().item(), (pos // 4 - 511) * 4)

    def test_ep_partition_sum(self):
        generator = torch.Generator().manual_seed(19)
        histogram = torch.randint(0, 1000, (512,), generator=generator).tolist()
        experts = torch.randn(512, 24, generator=generator, dtype=torch.float64)
        selected = torch.randperm(512, generator=generator)[:16]
        weights = torch.rand(16, generator=generator, dtype=torch.float64)
        reference = (experts[selected] * weights[:, None]).sum(dim=0)
        for ranks in (1, 2, 3, 4):
            owners = assign([512 * EXPERT_BYTES] * ranks, [1] * ranks, [histogram])[0]
            parts = []
            for rank in range(ranks):
                indices = [j for j, expert in enumerate(selected.tolist()) if owners[expert] == rank]
                parts.append((experts[selected[indices]] * weights[indices, None]).sum(dim=0))
            torch.testing.assert_close(torch.stack(parts).sum(dim=0), reference, rtol=1e-12, atol=1e-12)
            self.assertEqual(len(owners), 512)
            self.assertTrue(all(0 <= rank < ranks for rank in owners))

    def test_ep_balances_total_memory(self):
        base = [8 * EXPERT_BYTES, 0, 0]
        rows = assign([512 * EXPERT_BYTES] * 3, [1] * 3, [[1] * 512], base)
        counts = [rows[0].count(r) for r in range(3)]
        peaks = [base[r] + counts[r] * EXPERT_BYTES for r in range(3)]
        self.assertLessEqual(max(peaks) - min(peaks), EXPERT_BYTES)
        self.assertLess(counts[0], counts[1])

    def test_graph_headroom_at_two_million_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = {"quantization_config": {"linear_storage": {}}}
            (directory / "config.json").write_text(json.dumps(config))
            experts = dict.fromkeys(range(41 * 512), EXPERT_BYTES)
            for ranks in (3, 4):
                owners = directory / f"owners-{ranks}.json"
                argv = ["placement.py", tmp, "--tokens", "2097152", "--capacity", *(["32"] * ranks),
                        "--output", str(owners)]
                output = io.StringIO()
                with patch.object(sys, "argv", argv), patch.object(placement, "checkpoint_sizes",
                        return_value=(4 << 30, experts)), contextlib.redirect_stdout(output):
                    placement.main()
                result = json.loads(output.getvalue())
                layout = json.loads(owners.read_text())
                self.assertEqual(len(layout), 41)
                self.assertEqual(result["graph_bytes_per_rank"], 256 << 20)
                for rank in range(ranks):
                    fixed = (2 << 30) if not rank else (512 << 20)
                    if not rank:
                        fixed += (4 << 30) + result["routing_bytes"] + sum(placement.cache_bytes(2097152))
                    count = sum(row.count(rank) for row in layout)
                    peak = fixed + result["matrix_bytes"] + (256 << 20) + count * EXPERT_BYTES
                    self.assertEqual(result["peak_estimate"][rank], peak)
                    self.assertEqual(result["free_after"][rank], (32 << 30) - (512 << 20) - peak)
                    self.assertGreater(result["free_after"][rank], 0)

    def test_batch_expert_balance(self):
        experts = dict.fromkeys(range(41 * 512), EXPERT_BYTES)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            storage = {f"model.layers.{i}.self_attn.kv_b_proj.weight":
                       {"original_shape": [512, 14336 if i == 0 or i % 4 == 1 else 28672]} for i in range(42)}
            (directory / "config.json").write_text(json.dumps({"quantization_config": {"linear_storage": storage}}))
            for ranks in (3, 4):
                for tokens, slots in ((2097152, 1), (4096, 16), (8192, 16)):
                    results = []
                    for flags in (["--no-balance-experts"], []):
                        argv = ["placement.py", tmp, "--draft", tmp, "--tokens", str(tokens),
                                "--max-concurrent-requests", str(slots), "--capacity", *(["32"] * ranks), *flags]
                        output = io.StringIO()
                        with patch.object(sys, "argv", argv), patch.object(placement, "checkpoint_sizes",
                                side_effect=[(4329651680, experts), (424428112, {})]), contextlib.redirect_stdout(output):
                            placement.main()
                        result = json.loads(output.getvalue())
                        self.assertGreaterEqual(min(result["free_after"]), 0)
                        self.assertLessEqual(max(result["peak_estimate"]), (32 << 30) - (512 << 20))
                        results.append(result["rank_experts"])
                    if slots == 1:
                        self.assertEqual(*results)
                    else:
                        before, after = results
                        self.assertEqual(sum(after), 41 * 512)
                        self.assertLess(max(after), max(before))
                        self.assertAlmostEqual(after[0] / max(after), 0.9, places=3)

    def test_balanced_routing_frequency(self):
        histogram = [1 + ((e * 193) % 512) for e in range(512)]
        owners = assign([512 * EXPERT_BYTES] * 4, [1] * 4, [histogram], balance_experts=True)[0]
        loads = [sum(cost for cost, rank in zip(histogram, owners) if rank == i) for i in range(4)]
        self.assertAlmostEqual(loads[0] / max(loads), 1.0, delta=0.025)
        self.assertLess(max(loads[1:]) - min(loads[1:]), 512)

    def test_ep_capacity(self):
        budgets = [n * EXPERT_BYTES for n in (2, 1022, 0)]
        for balance in (False, True):
            rows = assign(budgets, [2, 1, 1], [list(range(512))] * 2, balance_experts=balance)
            for row in rows:
                self.assertEqual(row.count(0), 1)
                self.assertEqual(row[511], 0)
            with self.assertRaises(MemoryError):
                assign([0, 0, 0], [2, 1, 1], [[1] * 512], balance_experts=balance)

    def test_mixed_35bpw_capacity_and_c32(self):
        experts = {
            (layer, expert): EXPERT_BYTES_3BIT if (layer + expert) % 2 == 0 else EXPERT_BYTES
            for layer in range(1, 42) for expert in range(512)
        }
        expected_total = sum(experts.values())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            storage = {f"model.layers.{i}.self_attn.kv_b_proj.weight":
                       {"original_shape": [512, 14336 if i == 0 or i % 4 == 1 else 28672]} for i in range(42)}
            (directory / "config.json").write_text(json.dumps({"quantization_config": {"linear_storage": storage}}))
            for ranks, tokens, slots in ((3, 8192, 32), (4, 4096, 32), (3, 2097152, 1)):
                argv = ["placement.py", tmp, "--draft", tmp, "--tokens", str(tokens),
                        "--max-concurrent-requests", str(slots), "--capacity", *(["32"] * ranks)]
                output = io.StringIO()
                with patch.object(sys, "argv", argv), patch.object(placement, "checkpoint_sizes",
                        side_effect=[(4329651680, experts), (424428112, {})]), contextlib.redirect_stdout(output):
                    placement.main()
                result = json.loads(output.getvalue())
                self.assertEqual(sum(result["rank_experts"]), 41 * 512)
                self.assertEqual(sum(result["rank_expert_bytes"]), expected_total)
                self.assertGreaterEqual(min(result["free_after"]), 0)
                self.assertLessEqual(max(result["peak_estimate"]), (32 << 30) - (512 << 20))
                if slots > 1:
                    self.assertAlmostEqual(result["rank_experts"][0] / max(result["rank_experts"]), 0.9, places=3)

    def test_lm_head_tp_placement(self):
        experts = {(layer, expert): EXPERT_BYTES for layer in range(1, 42) for expert in range(512)}
        lm_total = 278609920
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            storage = {f"model.layers.{i}.self_attn.kv_b_proj.weight":
                       {"original_shape": [512, 14336 if i == 0 or i % 4 == 1 else 28672]} for i in range(42)}
            (directory / "config.json").write_text(json.dumps({"quantization_config": {"linear_storage": storage}}))
            results = {}
            for enabled in ("0", "1"):
                argv = ["placement.py", tmp, "--draft", tmp, "--tokens", "8192",
                        "--max-concurrent-requests", "32", "--capacity", "32", "32", "32", "32"]
                output = io.StringIO()
                with patch.dict("os.environ", {"SHELF_HY4_LM_HEAD_TP": enabled}), \
                     patch.object(sys, "argv", argv), \
                     patch.object(placement, "checkpoint_sizes", side_effect=[(4329651680, experts), (424428112, {})]), \
                     patch.object(placement, "lm_head_size", return_value=lm_total), \
                     contextlib.redirect_stdout(output):
                    placement.main()
                results[enabled] = json.loads(output.getvalue())
            self.assertEqual(results["0"]["lm_head_bytes_per_rank"], lm_total)
            self.assertEqual(results["1"]["lm_head_bytes_per_rank"], lm_total // 4)
            self.assertLess(results["1"]["peak_estimate"][0], results["0"]["peak_estimate"][0])

    def test_tp4_placement(self):
        experts = dict.fromkeys(range(41 * 512), EXPERT_BYTES)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            storage = {f"model.layers.{i}.self_attn.kv_b_proj.weight":
                       {"original_shape": [512, 14336 if i == 0 or i % 4 == 1 else 28672]} for i in range(42)}
            (directory / "config.json").write_text(json.dumps({"quantization_config": {"linear_storage": storage}}))
            argv = ["placement.py", tmp, "--draft", tmp, "--tensor-parallel", "4",
                    "--tokens", "32768", "--max-concurrent-requests", "16", "--capacity", "32", "32", "32", "32"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(placement, "checkpoint_sizes",
                    side_effect=[(4329651680, experts), (424428112, {})]), patch.object(
                    placement, "split_checkpoint_sizes",
                    side_effect=[(2900 << 20, 1400 << 20), (350 << 20, 74 << 20)]), contextlib.redirect_stdout(output):
                placement.main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["expert_bytes"], EXPERT_BYTES)
            self.assertEqual(result["graph_bytes_per_rank"], 256 << 20)
            self.assertEqual(sum(result["rank_experts"]), 41 * 512)
            self.assertEqual(max(result["rank_experts"]) - min(result["rank_experts"]), 0)
            self.assertGreater(min(result["free_after"]), 0)
            self.assertLessEqual(max(result["peak_estimate"]), (32 << 30) - (512 << 20))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
