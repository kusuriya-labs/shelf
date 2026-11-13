import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
pairs = [(128,8192),(64,16384),(32,32768),(16,65536),(8,131072), (256,6144),(128,12288),(64,24576),(32,49152),(16,98304),(8,196608)]
cases = [(m + offset, n, split, bf16) for m,n in pairs for offset in [-1,1] for split in [1,5] for bf16 in [0,1]]
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
    (folder / 'rotate-four-six-near.json').write_text(json.dumps(records, indent=2) + '\n')
    print(m, n, splits, bf16, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(str(record))
