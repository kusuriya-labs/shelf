import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('shelf_quantize', ROOT / 'tools/quantize/shelf_quantize.py')
quantize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize)


class DraftLinearTest(unittest.TestCase):
    def test_linear_selection_matches_dflash2_contract(self):
        quantized = [
            'fc.weight',
            'layers.0.self_attn.q_proj.weight',
            'layers.0.self_attn.k_proj.weight',
            'layers.0.self_attn.v_proj.weight',
            'layers.0.self_attn.o_proj.weight',
            'layers.0.mlp.gate_proj.weight',
            'layers.0.mlp.up_proj.weight',
            'layers.0.mlp.down_proj.weight',
        ]
        kept = [
            'hidden_norm.weight',
            'norm.weight',
            'layers.0.input_layernorm.weight',
            'layers.0.self_attn.q_norm.weight',
            'layers.0.attention_conv.base_kernel',
            'layers.0.attention_conv.kernel_projection.weight',
            'layers.0.mlp_conv.kernel_projection.weight',
            'candidate_selector.hidden_projection.weight',
            'candidate_selector.predecessor_codebook',
        ]
        self.assertTrue(all(quantize.is_draft_linear(name) for name in quantized))
        self.assertFalse(any(quantize.is_draft_linear(name) for name in kept))

    def test_non_dflash2_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({'architectures': ['Qwen3ForCausalLM']}))
            args = types.SimpleNamespace(model=model, output=Path(temp) / 'out', bpw=4.0, seed=0, device=0)
            with self.assertRaises(ValueError):
                quantize.convert_draft(args)


if __name__ == '__main__':
    unittest.main()
