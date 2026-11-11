import json
import os
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
failures = 0
for sessions in range(1, 9):
    found = False
    for path in root.glob(f'base_{sessions}*.json'):
        output = path.with_suffix('.out')
        if not output.exists() or not output.read_text().startswith('{') or 'Traceback (most recent call last)' in output.read_text():
            continue
        try:
            data = json.loads(path.read_text())
            found |= len(data) == sessions and all(isinstance(ids, list) and ids for ids in data.values())
        except (OSError, ValueError):
            pass
    if found:
        continue
    tag = f'base_{sessions}_exact'
    env = dict(os.environ, ERRF=str(root / f'{tag}.err'), GENF=str(root / f'{tag}.json'),
               SHELF_BATCH_STEPWALL='1')
    command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
               '/tmp/agentx/conc3.py', str(sessions), '160', '--kv-cache-dtype', 'fp8',
               '--max-num-batched-tokens', '2048']
    print('start', sessions, flush=True)
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    (root / f'{tag}.out').write_text(result.stdout + result.stderr)
    print('end', sessions, 'status', result.returncode, result.stdout.strip(), flush=True)
    failures += result.returncode != 0
raise SystemExit(1 if failures else 0)
