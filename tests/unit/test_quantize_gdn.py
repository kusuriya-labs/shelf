import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('shelf_quantize', ROOT / 'tools/quantize/shelf_quantize.py')
quantize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize)


class NormalizeGdnTest(unittest.TestCase):
    def test_gdn_f32_conversion_preserves_other_tensors_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = 'model.language_model.layers.0.linear_attn.A_log'
            values = torch.tensor([1.2345, -2.3456], dtype=torch.float32)
            save_file({name: values, 'other': values.clone()}, str(root / 'model.safetensors'), metadata={'key': 'value'})
            config = {'tensor_storage': {'m': {'stored_tensors': {name: {'dtype': 'torch.float32', 'n_bytes': 8}}}}}
            (root / 'quantization_config.json').write_text(json.dumps(config))
            quantize.normalize_gdn_weights(root)
            result = load_file(str(root / 'model.safetensors'))
            self.assertTrue(torch.equal(result[name], values.to(torch.bfloat16)))
            self.assertTrue(torch.equal(result['other'], values))
            saved = json.loads((root / 'quantization_config.json').read_text())
            self.assertEqual(saved['tensor_storage']['m']['stored_tensors'][name],
                             {'dtype': 'torch.bfloat16', 'n_bytes': 4})
            from safetensors import safe_open
            with safe_open(root / 'model.safetensors', framework='pt') as source:
                self.assertEqual(source.metadata(), {'key': 'value'})


if __name__ == '__main__':
    unittest.main()
