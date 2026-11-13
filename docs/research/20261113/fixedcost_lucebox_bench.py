import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
tmp = Path('/tmp/fc_lucebox')
tmp.mkdir(parents=True, exist_ok=True)
base = Path('/home/homelab1/coding-local/shelf/.worktrees/integ-final5/build-gfx1201/shelf-compute')
candidate = root / 'build-gfx1201/shelf-compute'
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
drafters = {
    'bf16': '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2',
    'mul24': '/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24',
}
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib')
result_path = tmp / 'bench.json'
rows = json.loads(result_path.read_text()) if result_path.exists() else []
for kind in ('none', 'bf16', 'mul24'):
    for repeat in range(3):
        for label in (('base', 'candidate') if repeat % 2 == 0 else ('candidate', 'base')):
            if any(r['kind'] == kind and r['repeat'] == repeat and r['variant'] == label and r['exit'] == 0 for r in rows):
                continue
            exe = base if label == 'base' else candidate
            cmd = ['flock', '-w', '900', '/tmp/r9700.lock', 'timeout', '240', str(exe), '--model-dir', model, '--input-ids-file', str(root / 'docs/research/20261113/exl3_general/prompt_code.ids'), '--max-new-tokens', '128', '--max-seq-len', '1024', '--temperature', '0']
            if kind != 'none':
                cmd += ['--dflash2-model-dir', drafters[kind], '--dflash2-drafts', '7']
            p = subprocess.run(cmd, env=env, text=True, capture_output=True)
            fields = dict(re.findall(r'^([A-Z_]+)=(.*)$', p.stdout + '\n' + p.stderr, re.M))
            row = dict(kind=kind, repeat=repeat, variant=label, exit=p.returncode, fields=fields, stderr_tail=p.stderr[-2000:])
            rows.append(row)
            result_path.write_text(json.dumps(rows, indent=2) + '\n')
            print(kind, repeat, label, p.returncode, fields.get('DECODE_TOKENS_PER_SEC'), flush=True)
            if p.returncode:
                raise SystemExit(p.returncode)
