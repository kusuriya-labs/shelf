import fcntl
import hashlib
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
for bits, n, k, row_set in [(4, 512, 17408, [49, 64]), (4, 1024, 8192, [49, 56, 64]),
                             (5, 512, 17408, [17, 64]), (5, 1024, 8192, [17, 32]),
                             (5, 1024, 17408, [17, 32, 33, 48, 64])]:
    for rows in row_set:
        for splits in [1, 5]:
            hashes = {}
            with open('/tmp/r9700.lock', 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for variant in ['base', 'candidate']:
                    exe = folder / ('shelf-bench.rotate-group' if variant == 'base' else 'shelf-bench.full-v1')
                    dump = folder / f'extra-exact-{bits}-{n}-{k}-{rows}-{splits}-{variant}.bin'
                    cmd = ['timeout', '60', str(exe), 'exl3-int8', '--n', str(n), '--k', str(k),
                           '--bits', str(bits), '--rows', str(rows), '--scratch-rows', '1024',
                           '--split', str(splits), '--working-mib', '8', '--iterations', '2',
                           '--random', '1', '--dump', str(dump)]
                    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(result.stderr + result.stdout)
                    hashes[variant] = hashlib.sha256(dump.read_bytes()).hexdigest()
            item = {'bits': bits, 'n': n, 'k': k, 'rows': rows, 'splits': splits,
                    'sha256': hashes, 'equal': hashes['base'] == hashes['candidate']}
            records.append(item)
            (folder / 'tile-exact-extra.json').write_text(json.dumps(records, indent=2) + '\n')
            print(bits, n, rows, splits, item['equal'], flush=True)
            if not item['equal']:
                raise RuntimeError(str(item))
