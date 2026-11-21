import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('validate_exl3', ROOT / 'tools/quantize/validate_exl3.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
        self.items = {'x.trellis': ('I16', [8, 8, 16], bytes(2048)),
                      'x.suh': ('F16', [128], bytes(256)),
                      'x.svh': ('F16', [128], bytes(256)),
                      'x.mul1': ('I32', [], struct.pack('<I', validator.MUL24))}

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        header, data = {}, bytearray()
        for name, (dtype, shape, value) in self.items.items():
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(data), len(data) + len(value)]}
            data.extend(value)
        raw = json.dumps(header).encode()
        (self.root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + data)
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: 'model.safetensors' for name in self.items}}))

    def rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            validator.validate(self.root)

    def test_valid(self):
        self.write()
        self.assertEqual(validator.validate(self.root), 1)

    def test_legacy_marker(self):
        self.items['x.mul1'] = ('I32', [], struct.pack('<I', 0x83DCD12D))
        self.rejected()

    def test_s0(self):
        self.items['x.s0'] = ('F16', [1], bytes(2))
        self.rejected()

    def test_unindexed_s0(self):
        self.write()
        header = json.dumps({'x.s0': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
        (self.root / 'unindexed.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(2))
        with self.assertRaisesRegex(ValueError, r'\.s0'):
            validator.validate(self.root)

    def test_missing(self):
        for name in list(self.items):
            with self.subTest(name=name):
                item = self.items.pop(name)
                self.rejected()
                self.items[name] = item

    def test_shape_dtype_alignment_bitrate(self):
        original = self.items['x.trellis']
        for item in [('F16', [8, 8, 16], bytes(2048)), ('I16', [8, 8, 16, 1], bytes(2048)),
                     ('I16', [7, 8, 16], bytes(1792)), ('I16', [8, 7, 16], bytes(1792)),
                     ('I16', [8, 8, 9], bytes(1152))]:
            with self.subTest(item=item[:2]):
                self.items['x.trellis'] = item
                self.rejected()
        self.items['x.trellis'] = original
        for name in ('x.suh', 'x.svh'):
            old = self.items[name]
            self.items[name] = ('I16', [128], bytes(256))
            self.rejected()
            self.items[name] = ('F16', [127], bytes(254))
            self.rejected()
            self.items[name] = old
        self.items['x.mul1'] = ('I32', [1], struct.pack('<I', validator.MUL24))
        self.rejected()

    def test_path_escape(self):
        self.write()
        path = self.root / 'model.safetensors.index.json'
        path.write_text(json.dumps({'weight_map': {'x.trellis': '../outside.safetensors'}}))
        with self.assertRaises(ValueError):
            validator.validate(self.root)


if __name__ == '__main__':
    unittest.main()
