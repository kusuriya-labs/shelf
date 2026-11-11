import json
import os
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
baseline_root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path('/tmp/dflash2-adaptive-measure')
root.mkdir(parents=True, exist_ok=True)
draft = os.environ['SHELF_DFLASH2_MODEL_DIR']
mismatches = 0
for sessions in range(1, 9):
    baseline = None
    for path in sorted(baseline_root.glob(f'base_{sessions}*.json')):
        output = path.with_suffix('.out')
        if not output.exists() or not output.read_text().startswith('{') or 'Traceback (most recent call last)' in output.read_text():
            continue
        try:
            candidate = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if len(candidate) == sessions and all(isinstance(ids, list) and ids for ids in candidate.values()):
            baseline = candidate
            break
    if baseline is None:
        raise SystemExit(f'missing target-only reference for S={sessions}')
    for drafts in range(1, 8):
        tag = f's{sessions}k{drafts}'
        env = dict(os.environ, ERRF=str(root / f'{tag}.err'), GENF=str(root / f'{tag}.json'),
                   SHELF_DFLASH2_SERVE_STATS='1', SHELF_DFLASH2_COST_TRACE='1',
                   SHELF_DFLASH2_MEASURE_VERIFY_DRAFTS=str(drafts))
        command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
                   '/tmp/agentx/conc3.py', str(sessions), '160', '--kv-cache-dtype', 'fp8',
                   '--max-num-batched-tokens', '2048', '--dflash2-model-dir', draft,
                   '--dflash2-drafts', '7', '--dflash2-max-sessions', str(sessions)]
        output_path = root / f'{tag}.out'
        error_path = root / f'{tag}.err'
        generation_path = root / f'{tag}.json'
        complete = (output_path.exists() and error_path.exists() and generation_path.exists() and
                    output_path.read_text().startswith('{') and
                    'Traceback (most recent call last)' not in output_path.read_text())
        if not complete:
            print('start', sessions, drafts, flush=True)
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            output_path.write_text(result.stdout + result.stderr)
            print('end', sessions, drafts, result.returncode, result.stdout.strip(), flush=True)
            if result.returncode != 0:
                raise SystemExit(f'prefix measurement failed S={sessions} K={drafts}')
        else:
            print('reuse', sessions, drafts, flush=True)
        try:
            generated = json.loads((root / f'{tag}.json').read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(f'missing output S={sessions} K={drafts}: {error}')
        if generated != baseline:
            print(f'greedy output mismatch S={sessions} K={drafts}', flush=True)
            mismatches += 1
        target_rows = sessions * (drafts + 1)
        inner = outer = 0
        for line in (root / f'{tag}.err').read_text().splitlines():
            match = re.search(r'(DFCOST|DFHOST) sessions=(\d+) rows=(\d+) ', line)
            if match and int(match[2]) == sessions and int(match[3]) == target_rows:
                inner += match[1] == 'DFCOST'
                outer += match[1] == 'DFHOST'
        if min(inner, outer) < 5:
            raise SystemExit(f'insufficient samples S={sessions} K={drafts}: {inner}/{outer}')
env = dict(os.environ, ERRF=str(root / 'base_64_host.err'), GENF=str(root / 'base_64_host.json'),
           SHELF_BATCH_STEPWALL='1')
command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
           '/tmp/agentx/conc3.py', '64', '160', '--kv-cache-dtype', 'fp8',
           '--max-num-batched-tokens', '2048']
print('start normal CPU B=64', flush=True)
result = subprocess.run(command, env=env, text=True, capture_output=True)
(root / 'base_64_host.out').write_text(result.stdout + result.stderr)
if result.returncode != 0:
    raise SystemExit('normal CPU measurement B=64 failed')
print('end normal CPU B=64', flush=True)
missing = pathlib.Path(__file__).with_name('measure_missing_base.py')
subprocess.run([sys.executable, '-I', str(missing), str(root)], env=os.environ, check=True)
coverage = pathlib.Path(__file__).with_name('check_baseline_coverage.py')
subprocess.run([sys.executable, '-I', str(coverage), str(root)], check=True)
print('target-only greedy mismatches', mismatches, flush=True)
raise SystemExit(1 if mismatches else 0)
