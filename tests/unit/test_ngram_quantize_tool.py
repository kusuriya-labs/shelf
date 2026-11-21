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
