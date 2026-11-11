import os
import pathlib
import subprocess
import sys

prompts = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
script = pathlib.Path(__file__).with_name('compare.py')
draft = os.environ['SHELF_DFLASH2_MODEL_DIR']
for repeat in (1, 2):
    for count in (8, 32):
        for mode in ('off', 'fixed', 'adaptive'):
            tag = f'r{repeat}_{count}_{mode}'
            env = dict(os.environ, ERRF=str(out / f'{tag}.err'), GENF=str(out / f'{tag}.json'),
                       SHELF_DFLASH2_SERVE_STATS='1')
            flags = ['--kv-cache-dtype', 'fp8', '--max-num-batched-tokens', '2048']
            if mode != 'off':
                flags += ['--dflash2-model-dir', draft, '--dflash2-drafts', '7']
            if mode == 'adaptive':
                flags += ['--dflash2-adaptive']
            command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
                       str(script), str(count), '160', str(prompts / 'benchmark.json')] + flags
            print('start', tag, flush=True)
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            (out / f'{tag}.out').write_text(result.stdout + result.stderr)
            print('end', tag, 'status', result.returncode, result.stdout.strip(), flush=True)
            if result.returncode != 0:
                raise SystemExit(f'repeat failed: {tag}')
