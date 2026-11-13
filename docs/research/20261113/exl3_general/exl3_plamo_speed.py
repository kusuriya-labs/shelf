import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
           LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',
           PATH='/tmp/venv-plamo/bin:' + env['PATH'])
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
results = []
for concurrency in [1, 64]:
    for repeat in range(2):
        order = ['base', 'candidate'] if repeat % 2 == 0 else ['candidate', 'base']
        for variant in order:
            env['SHELF_COMPUTE'] = str(folder / ('shelf-compute.base' if variant == 'base' else 'shelf-compute.optimized'))
            log = folder / f'plamo-speed-{concurrency}-{repeat}-{variant}.log'
            cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '-k', '10', '--signal=INT', '150',
                   '/tmp/venv-plamo/bin/python', 'docs/research/20261112/plamo_closed_loop.py',
                   '--server', str(root / 'build-gfx1201/shelf-server'), '--model', model,
                   '--concurrency', str(concurrency), '--warmup', '5', '--duration', '30',
                   '--port', '18472', '--log', str(log)]
            run = subprocess.run(cmd, env=env, text=True, capture_output=True)
            record = {'concurrency': concurrency, 'repeat': repeat, 'variant': variant,
                      'exit': run.returncode, 'stderr': run.stderr[-2000:]}
            if run.returncode == 0:
                record['result'] = json.loads(run.stdout)
            results.append(record)
            (folder / 'plamo-speed.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
            print(concurrency, repeat, variant, run.returncode,
                  record.get('result', {}).get('completion_tokens_per_s'), flush=True)
            if run.returncode:
                raise RuntimeError(str(record))
