#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import save_file

build = Path(os.environ['SHELF_BUILD_DIR'])
model = Path(os.environ['SHELF_HUNYUAN_MODEL_DIR'])
with tempfile.TemporaryDirectory(prefix='hunyuan-exl3-', dir=build) as tmp:
    root = Path(tmp)
    config = json.loads((model / 'config.json').read_text())
    config.update(
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=512,
        vocab_size=110,
        quantization_config={'quant_method': 'exl3'},
    )
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'input.txt').write_text('1 2 3 4')
    h, q, kv, f, padded_v = 256, 256, 128, 512, 128
    prefix = 'model.layers.0.'
    tensors = {
        'model.embed_tokens.weight': torch.zeros((110, h), dtype=torch.bfloat16),
        'model.norm.weight': torch.ones(h, dtype=torch.bfloat16),
    }
    for name in ('input_layernorm', 'post_attention_layernorm'):
        tensors[prefix + name + '.weight'] = torch.ones(h, dtype=torch.bfloat16)
    for name in ('self_attn.query_layernorm', 'self_attn.key_layernorm'):
        tensors[prefix + name + '.weight'] = torch.ones(128, dtype=torch.bfloat16)

    shapes = [
        ('self_attn.q_proj', q, h),
        ('self_attn.k_proj', kv, h),
        ('self_attn.v_proj', kv, h),
        ('self_attn.o_proj', h, q),
        ('mlp.gate_proj', f, h),
        ('mlp.up_proj', f, h),
        ('mlp.down_proj', h, f),
    ]
    for name, rows, cols in shapes:
        key = prefix + name
        tensors[key + '.trellis'] = torch.zeros((cols // 16, rows // 16, 64), dtype=torch.int16)
        tensors[key + '.suh'] = torch.ones(cols, dtype=torch.float16)
        tensors[key + '.svh'] = torch.ones(rows, dtype=torch.float16)
        tensors[key + '.mul1'] = torch.tensor(0xdcd12d, dtype=torch.int32)

    tensors['lm_head.trellis'] = torch.zeros((h // 16, padded_v // 16, 96), dtype=torch.int16)
    tensors['lm_head.suh'] = torch.ones(h, dtype=torch.float16)
    tensors['lm_head.svh'] = torch.ones(padded_v, dtype=torch.float16)
    tensors['lm_head.mul1'] = torch.tensor(0xdcd12d, dtype=torch.int32)

    checkpoint = root / 'model.safetensors'
    save_file(tensors, str(checkpoint))
    command = [
        str(build / 'shelf-compute'),
        '--model-dir', str(root),
        '--input-ids-file', str(root / 'input.txt'),
        '--max-new-tokens', '4',
        '--max-seq-len', '128',
        '--arena-gib', '2',
        '--device', '0',
        '--temperature', '0',
    ]
    good = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if good.returncode or 'GENERATED_IDS=0,0,0,0' not in good.stdout:
        raise RuntimeError(f'valid fixture failed: {good.stdout}\n{good.stderr}')
    tensors[prefix + 'self_attn.q_proj.mul1'] = torch.tensor(0x83dcd12d - (1 << 32), dtype=torch.int32)
    save_file(tensors, str(checkpoint))
    bad = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if bad.returncode == 0 or 'hf exl3 codebook is not mul24' not in bad.stderr:
        raise RuntimeError(f'invalid marker accepted: {bad.stdout}\n{bad.stderr}')
    print('HunYuan EXL3 synthetic padded lm_head and marker rejection passed')
