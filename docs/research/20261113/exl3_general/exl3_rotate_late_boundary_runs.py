import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
cases = [(m, n, s, b) for n in [5120] for m in [320, 384, 448]
         for s in [1, 5] for b in [0, 1]]
records = []
for m, n, splits, bf16 in cases:
    cmd = ['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '90',
           str(folder / 'rotate-micro-perop'), str(m), str(n), str(splits), str(bf16)]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    record = {'rows': m, 'n': n, 'splits': splits, 'bf16': bool(bf16),
              'exit': result.returncode, 'stderr': result.stderr}
    if result.returncode == 0:
        record['result'] = json.loads(result.stdout)
    records.append(record)
    (folder / 'rotate-late-boundary.json').write_text(json.dumps(records, indent=2) + '\n')
    print(m, n, splits, bf16, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(str(record))
