import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path('/tmp/fc_hipfire/runs')
out.mkdir(parents=True, exist_ok=True)
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
baseline = '/home/homelab1/coding-local/shelf/.worktrees/integ-final5/build-gfx1201/shelf-compute'
candidate = str(root / 'build-gfx1201/shelf-compute')
server = str(root / 'build-gfx1201/shelf-server')
lib = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
for rep in range(3):
    for name, binary in [('baseline', baseline), ('candidate', candidate)]:
        tag = f'{name}-{rep}'
        env = os.environ.copy()
        env.update(SHELF_COMPUTE=binary, ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', LD_LIBRARY_PATH=lib,
                   PATH='/tmp/venv-plamo/bin:' + env['PATH'])
        cmd = ['flock', '/tmp/r9700.lock', 'timeout', '260', '/tmp/venv-plamo/bin/python',
               str(root / 'docs/research/20261112/plamo_closed_loop.py'), '--server', server,
               '--model', model, '--concurrency', '1', '--warmup', '20', '--duration', '90',
               '--port', '18472', '--log', str(out / f'{tag}.log')]
        print('start', tag, flush=True)
        result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
        data = json.loads(result.stdout)
        (out / f'{tag}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        print(tag, data['completion_tokens_per_s'], flush=True)
