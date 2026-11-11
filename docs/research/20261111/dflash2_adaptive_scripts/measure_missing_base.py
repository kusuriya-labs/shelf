import collections
import os
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'STEPWALL rows=(\d+) ')
for batch in range(1, 65):
    counts = collections.Counter()
    for path in root.glob('base_*.err'):
        result = path.with_suffix('.out')
        if not result.exists() or (not result.read_text().startswith('{') or 'Traceback (most recent call last)' in result.read_text()):
            continue
        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if match:
                counts[int(match[1])] += 1
    if counts[batch] >= 5:
        continue
    tag = f'base_{batch}_missing'
    env = dict(os.environ, ERRF=str(root / f'{tag}.err'), GENF=str(root / f'{tag}.json'),
               SHELF_BATCH_STEPWALL='1')
    command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
               '/tmp/agentx/conc3.py', str(batch), '160', '--kv-cache-dtype', 'fp8',
               '--max-num-batched-tokens', '2048']
    print('start', batch, 'previous_samples', counts[batch], flush=True)
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    (root / f'{tag}.out').write_text(result.stdout + result.stderr)
    print('end', batch, 'status', result.returncode, result.stdout.strip(), flush=True)
