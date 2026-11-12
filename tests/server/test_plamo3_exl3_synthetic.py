#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import save_file

build = Path(os.environ['SHELF_BUILD_DIR'])
model = Path(os.environ['SHELF_PLAMO_MODEL_DIR'])
with tempfile.TemporaryDirectory(prefix='plamo-exl3-', dir=build) as tmp:
    root = Path(tmp)
    config = json.loads((model / 'config.json').read_text())
    config.update(num_hidden_layers=1, vocab_size=128, quantization_config={'quant_method': 'exl3'})
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'input.txt').write_text('1 2 3 4')
    h, q, kv, f = 2560, 3072, 512, 10240
    prefix = 'model.layers.layers.0.'
    tensors = {
        'model.embed_tokens.weight': torch.zeros((128, h), dtype=torch.bfloat16),
        'lm_head.weight': torch.zeros((128, h), dtype=torch.bfloat16),
        'model.norm.weight': torch.zeros(h, dtype=torch.bfloat16),
    }
    for name in ('pre_mixer_norm', 'post_mixer_norm', 'pre_mlp_norm', 'post_mlp_norm'):
        tensors[prefix + name + '.weight'] = torch.zeros(h, dtype=torch.bfloat16)
    for name in ('mixer.q_norm', 'mixer.k_norm'):
        tensors[prefix + name + '.weight'] = torch.zeros(128, dtype=torch.float16)
    shapes = [('mixer.q_proj', q, h), ('mixer.k_proj', kv, h), ('mixer.v_proj', kv, h),
              ('mixer.o_proj', h, q), ('mlp.gate_proj', f, h), ('mlp.up_proj', f, h),
              ('mlp.down_proj', h, f)]
    for name, rows, cols in shapes:
        key = prefix + name
        tensors[key + '.trellis'] = torch.zeros((cols // 16, rows // 16, 64), dtype=torch.int16)
        tensors[key + '.suh'] = torch.ones(cols, dtype=torch.float16)
        tensors[key + '.svh'] = torch.ones(rows, dtype=torch.float16)
        tensors[key + '.mul1'] = torch.tensor(0xdcd12d, dtype=torch.int32)
    checkpoint = root / 'model.safetensors'
    save_file(tensors, str(checkpoint))
    command = [str(build / 'shelf-compute'), '--model-dir', str(root), '--input-ids-file',
               str(root / 'input.txt'), '--max-new-tokens', '4', '--max-seq-len', '128',
               '--arena-gib', '2', '--device', '0', '--temperature', '0']
    good = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if good.returncode or 'GENERATED_IDS=0,0,0,0' not in good.stdout:
        raise RuntimeError(f'valid fixture failed: {good.stdout}\n{good.stderr}')
    tensors[prefix + 'mixer.q_proj.mul1'] = torch.tensor(0x83dcd12d - (1 << 32), dtype=torch.int32)
    save_file(tensors, str(checkpoint))
    bad = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if bad.returncode == 0 or 'EXL3 requires integer-rate mul24' not in bad.stderr:
        raise RuntimeError(f'invalid marker accepted: {bad.stdout}\n{bad.stderr}')
    print('PLaMo EXL3 synthetic prefill/decode and marker rejection passed')
