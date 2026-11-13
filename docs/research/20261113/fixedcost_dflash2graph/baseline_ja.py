import os
from pathlib import Path
import re
import subprocess
import sys

root = Path('/home/homelab1/coding-local/shelf/.worktrees/fc-dflash2graph')
out = Path('/tmp/fc_dflash2graph')
out.mkdir(parents=True, exist_ok=True)
binary = '/home/homelab1/coding-local/shelf/.worktrees/integ-fc/build-gfx1201/shelf-compute'
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
drafts = {'bf16': '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2',
          'mul24': '/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24'}
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
           SHELF_GRAPH_VERIFY='0',
           LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib')


def run(name, extra):
    cmd = ['flock', '/tmp/r9700.lock', 'timeout', '-k', '5', '270', binary,
           '--model-dir', model, '--input-ids-file', str(root / 'docs/research/20261113/exl3_general/prompt_ja.ids'),
           '--max-new-tokens', '256', '--max-seq-len', '1024', '--temperature', '0', *extra]
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    (out / f'{name}.stdout').write_text(result.stdout)
    (out / f'{name}.stderr').write_text(result.stderr)
    match = re.search(r'^GENERATED_IDS=(.*)$', result.stdout, re.M)
    if result.returncode or not match:
        print('FAIL', name, result.returncode, result.stderr[-1000:], flush=True)
        sys.exit(1)
    return match.group(1).split(',')


greedy = run('baseline_greedy_ja', [])
for draft, path in drafts.items():
    for mode in ('tree', 'chain'):
        extra = ['--dflash2-model-dir', path, '--dflash2-drafts', '7']
        if mode == 'chain':
            extra += ['--dflash2-tree-budget', '0']
        ids = run(f'baseline_{draft}_ja_{mode}', extra)
        first = next((i for i, (a, b) in enumerate(zip(greedy, ids)) if a != b), None)
        print(draft, mode, 'n', len(ids), 'first_diff', first, flush=True)
