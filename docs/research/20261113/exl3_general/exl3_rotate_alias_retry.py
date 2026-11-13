import json
import os
import subprocess
from pathlib import Path

folder = Path.cwd() / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
cases = [('rotate-micro-graph', 128, 10240, 1, 0), ('rotate-micro-graph', 128, 10240, 3, 1),
         ('rotate-micro-graph', 64, 20480, 2, 1), ('rotate-micro-graph', 64, 20480, 4, 1),
         ('rotate-micro-graph', 32, 40960, 4, 0), ('rotate-micro-graph', 32, 40960, 5, 1),
         ('rotate-micro-graph', 16, 81920, 4, 1)]
cases += [('rotate-micro-perop', 256, 5120, splits, bf16) for splits in [2, 3, 4] for bf16 in [0, 1]]
records = []
for exe, rows, n, splits, bf16 in cases:
    cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '90', str(folder / exe),
           str(rows), str(n), str(splits), str(bf16)]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    records.append({'method': exe, 'result': json.loads(result.stdout)})
    (folder / 'rotate-alias-retry.json').write_text(json.dumps(records, indent=2) + '\n')
    print(exe, rows, n, splits, bf16, flush=True)
