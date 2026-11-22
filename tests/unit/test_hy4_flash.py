import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/hy4_flash"))
from convert_engram import convert
from octave import Cache, selected_tokens
from placement import EXPERT_BYTES, assign


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

    def test_ep_capacity(self):
        budgets = [n * EXPERT_BYTES for n in (2, 1022, 0)]
        rows = assign(budgets, [2, 1, 1], [list(range(512))] * 2)
        for row in rows:
            self.assertEqual(row.count(0), 1)
            self.assertEqual(row[511], 0)
        with self.assertRaises(MemoryError):
            assign([0, 0, 0], [2, 1, 1], [[1] * 512])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
