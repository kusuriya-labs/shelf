import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASELINE = Path('/home/homelab1/coding-local/shelf/.worktrees/integ-fc/build-gfx1201/shelf-compute')
MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFTERS = {
    'bf16': '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2',
    'mul24': '/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24',
}
LIBS = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', LD_LIBRARY_PATH=LIBS)
env.pop('SHELF_GRAPH_DEBUG', None)
env.pop('SHELF_GRAPH_VERIFY', None)
records = []
for name, drafter in DRAFTERS.items():
    for repeat in range(3):
        for variant, binary in [('baseline', BASELINE), ('candidate', ROOT / 'build-gfx1201/shelf-compute')]:
            cmd = ['flock', '/tmp/r9700.lock', 'timeout', '-k', '5', '270', str(binary),
                   '--model-dir', MODEL, '--input-ids-file', str(ROOT / 'docs/research/20261113/exl3_general/prompt_code.ids'),
                   '--max-new-tokens', '128', '--max-seq-len', '1024', '--temperature', '0',
                   '--dflash2-model-dir', drafter, '--dflash2-drafts', '7']
            run = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            fields = dict(re.findall(r'^([A-Z0-9_]+)=(.*)$', run.stdout, re.M))
            record = {'drafter': name, 'repeat': repeat, 'variant': variant,
                      'returncode': run.returncode, 'fields': fields, 'stderr': run.stderr}
            records.append(record)
            args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
            print(name, repeat, variant, run.returncode, fields.get('DECODE_TOKENS_PER_SEC'), flush=True)
            if run.returncode or 'DECODE_TOKENS_PER_SEC' not in fields:
                sys.exit(1)
