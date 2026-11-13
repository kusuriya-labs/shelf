import fcntl
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
records = []
for n, k, bits in [(1024,5120,4),(10240,5120,4),(17408,5120,4),(5120,17408,4),(248320,5120,6)]:
    for rows in [1,2,4,8]:
        with open('/tmp/r9700.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for repeat in range(5):
                for variant in (['base', 'candidate'] if repeat % 2 == 0 else ['candidate', 'base']):
                    exe = folder / ('shelf-bench.rotate-group' if variant == 'base' else 'shelf-bench.optimized')
                    cmd = ['timeout', '60', str(exe), 'exl3-int8', '--n', str(n), '--k', str(k),
                           '--bits', str(bits), '--rows', str(rows), '--scratch-rows', '1024',
                           '--working-mib', '256', '--iterations', '16']
                    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                    item = {'n': n, 'k': k, 'bits': bits, 'rows': rows, 'repeat': repeat,
                            'variant': variant, 'exit': result.returncode,
                            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
                    records.append(item)
                    (folder / 'gemm-unchanged.json').write_text(json.dumps(records, indent=2) + '\n')
                    if result.returncode:
                        raise RuntimeError(str(item))
        print(n, k, bits, rows, flush=True)
