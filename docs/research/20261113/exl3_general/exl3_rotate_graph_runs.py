import json
import os
import subprocess
from pathlib import Path

folder = Path.cwd() / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
results = []
for rows, splits in [(128, 5), (255, 1), (255, 5)]:
    cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '90',
           str(folder / 'rotate-micro-graph'), str(rows), '5120', str(splits), '1']
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    results.append(json.loads(result.stdout))
    (folder / 'rotate-graph-retry.json').write_text(json.dumps(results, indent=2) + '\n')
    print(rows, splits, flush=True)
