import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/quantize'))
import ngram_quantize as tool


class QuantizeToolTest(unittest.TestCase):
    def test_model_family_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = [1, 11, 21, 31]
            config = {'engram_layer_ids': [x + 1 for x in layers],
                      'engram_layer_head_vocab_sizes': {str(x + 1): [[2] * 8, [3] * 8] for x in layers}}
            (root / 'config.json').write_text(json.dumps({'text_config': config}))
            src = {}
            for layer in layers:
                name = f'model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight'
                src[name] = (root / 'source', 0, {'dtype': 'BF16', 'shape': [40, 96]})
            specs = tool.automatic_specs(root, src, 'hy4')
            self.assertEqual(len(specs), 64)
            self.assertEqual((specs[8]['layer'], specs[8]['ngram_order'], specs[8]['head'], specs[8]['segments'][0][1:]), (1, 3, 8, (16, 3)))
            self.assertEqual(tool.automatic_specs(root, src, 'auto'), specs)
            config = {'ple_layer_ids': [2], 'split_ngram_parts': 128, 'heads_per_ngram': 8,
                      'ngram_size': 3, 'ngram_vocab_size_base': 2,
                      'make_ngram_vocab_size_divisible_by': 128}
            (root / 'config.json').write_text(json.dumps({'text_config': config}))
            src = {}
            for i in range(128):
                name = f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight'
                src[name] = (root / 'source', 0, {'dtype': 'BF16', 'shape': [3, 160]})
            specs = tool.automatic_specs(root, src, 'qwen4exp')
            self.assertEqual(len(specs), 16)
            self.assertEqual((specs[8]['layer'], specs[8]['ngram_order'], specs[8]['head']), (1, 3, 8))
            self.assertEqual(sum(segment[2] for segment in specs[8]['segments']), 23)
            self.assertGreater(len(specs[8]['segments']), 1)
            self.assertEqual(tool.automatic_specs(root, src, 'auto'), specs)
            src.pop(next(iter(src)))
            with self.assertRaises(ValueError):
                tool.automatic_specs(root, src, 'qwen4exp')

    def test_rows_cross_shard_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = {}
            for i in range(2):
                data = np.arange(i * 12, (i + 1) * 12, dtype='<u2').reshape(4, 3)
                path = root / str(i)
                path.write_bytes(data.tobytes())
                src[str(i)] = (path, 0, {'dtype': 'BF16', 'shape': [4, 3], 'data_offsets': [0, data.nbytes]})
            rows = tool.source_rows(src, [('0', 2, 2), ('1', 0, 3)], 1, 3)
            np.testing.assert_array_equal(rows, np.arange(9, 18, dtype='<u2').reshape(3, 3))

    def test_automatic_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(17)
            for family in ('hy4', 'qwen4exp'):
                source = root / family
                source.mkdir()
                arrays = {}
                if family == 'hy4':
                    layers = [1, 11, 21, 31]
                    config = {'engram_layer_ids': [x + 1 for x in layers],
                              'engram_layer_head_vocab_sizes': {str(x + 1): [[2] * 8, [2] * 8] for x in layers}}
                    for layer in layers:
                        arrays[f'model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight'] = (32, 96)
                else:
                    config = {'ple_layer_ids': [2], 'split_ngram_parts': 128, 'heads_per_ngram': 8,
                              'ngram_size': 3, 'ngram_vocab_size_base': 2,
                              'make_ngram_vocab_size_divisible_by': 128}
                    for i in range(128):
                        arrays[f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight'] = (3, 160)
                (source / 'config.json').write_text(json.dumps({'text_config': config}))
                header, payload, offset = {}, [], 0
                for name, shape in arrays.items():
                    data = (rng.normal(size=shape).astype(np.float32).view(np.uint32) >> 16).astype('<u2').tobytes()
                    header[name] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
                    payload.append(data)
                    offset += len(data)
                raw = json.dumps(header).encode()
                raw += b' ' * (-len(raw) % 8)
                with (source / 'model.safetensors').open('wb') as f:
                    f.write(struct.pack('<Q', len(raw)) + raw)
                    for data in payload:
                        f.write(data)
                output = root / (family + '-out')
                subprocess.run([sys.executable, str(ROOT / 'tools/quantize/ngram_quantize.py'), 'convert',
                                str(source), str(output), '--model-family', family, '--max-rows', '2',
                                '--threads', '1', '--sample-rows', '2'], check=True, capture_output=True)
                tables = json.loads((output / 'ngram_manifest.json').read_text())['tables']
                self.assertEqual(len(tables), 64 if family == 'hy4' else 16)
                for i, table in enumerate(tables):
                    self.assertEqual((table['head'], table['ngram_order']), (i % 16, 2 + (i % 16) // 8))
                    self.assertEqual(table['layer'], ([1, 11, 21, 31][i // 16] if family == 'hy4' else 1))

    def test_profiles_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            rng = np.random.default_rng(7)
            arrays = [(rng.normal(size=(67, 19)).astype(np.float32).view(np.uint32) >> 16).astype('<u2') for _ in range(2)]
            header = {}
            offset = 0
            for i, a in enumerate(arrays):
                header[f'ngram.embedding.{i}'] = {'dtype': 'BF16', 'shape': list(a.shape), 'data_offsets': [offset, offset + a.nbytes]}
                offset += a.nbytes
            raw = json.dumps(header).encode()
            raw += b' ' * (-len(raw) % 8)
            with (source / 'model.safetensors').open('wb') as f:
                f.write(struct.pack('<Q', len(raw)) + raw)
                for a in arrays:
                    f.write(a.tobytes())
            spec = root / 'spec.json'
            spec.write_text(json.dumps([{'name': name, 'layer': i, 'ngram_order': 2, 'head': i} for i, name in enumerate(header)]))
            bits = root / 'bits.json'
            bits.write_text(json.dumps({list(header)[0]: 4, list(header)[1]: 5}))
            lib = tool.library()
            for profile in ('ngram4', 'ngram5', 'mixed'):
                out = root / profile
                command = [sys.executable, str(ROOT / 'tools/quantize/ngram_quantize.py'), 'convert', str(source), str(out), '--tensor-spec', str(spec), '--profile', profile, '--bits-map', str(bits) if profile == 'mixed' else str(root / 'empty.json'), '--threads', '2', '--chunk-rows', '13']
                (root / 'empty.json').write_text('{}')
                subprocess.run(command, check=True, capture_output=True)
                subprocess.run([sys.executable, command[1], 'verify', str(out)], check=True, capture_output=True)
                m = json.loads((out / 'ngram_manifest.json').read_text())
                for i, t in enumerate(m['tables']):
                    p = out / t['shard']
                    with p.open('rb') as f:
                        n = struct.unpack('<Q', f.read(8))[0]
                        h = json.loads(f.read(n))
                    base = n + 8
                    def read(name, dtype):
                        a, b = h[name]['data_offsets']
                        return np.memmap(p, dtype=dtype, mode='r', offset=base+a, shape=((b-a)//np.dtype(dtype).itemsize,))
                    codes = read(t['codes_tensor'], 'u1'); scales = read(t['scale_tensor'], '<u2'); book = read(t['codebook_tensor'], '<u2')
                    decoded = np.empty(arrays[i].shape, dtype=np.float32)
                    tool.call(lib.shelf_ngram_decode_rows_f32, tool.ptr(codes), tool.ptr(scales), tool.ptr(book), 67, 19, t['bits'], tool.ptr(decoded))
                    original = (arrays[i].astype(np.uint32) << 16).view(np.float32)
                    self.assertLess(np.linalg.norm(original-decoded) / np.linalg.norm(original), 0.16)
                t = m['tables'][0]
                p = out / t['shard']
                with p.open('r+b') as f:
                    n = struct.unpack('<Q', f.read(8))[0]
                    h = json.loads(f.read(n))
                    f.seek(8+n+h[t['scale_tensor']]['data_offsets'][0])
                    f.write(b'\x00\xfc')
                result = subprocess.run([sys.executable, command[1], 'verify', str(out)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
