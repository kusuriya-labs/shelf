import json
import os
import pathlib
import subprocess
import sys

prompts = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
script = pathlib.Path(__file__).with_name('compare.py')
draft = os.environ['SHELF_DFLASH2_MODEL_DIR']
for name in ('benchmark', 'japanese', 'shell', 'code', 'reasoning', 'mixed'):
    for count in (1, 2, 4, 8, 16, 32):
        for mode in ('off', 'fixed', 'adaptive'):
            tag = f'{name}_{count}_{mode}'
            env = dict(os.environ, ERRF=str(out / f'{tag}.err'), GENF=str(out / f'{tag}.json'),
                       SHELF_DFLASH2_SERVE_STATS='1')
            flags = ['--kv-cache-dtype', 'fp8', '--max-num-batched-tokens', '2048']
            if mode != 'off':
                flags += ['--dflash2-model-dir', draft, '--dflash2-drafts', '7']
            if mode == 'adaptive':
                flags += ['--dflash2-adaptive']
            command = ['flock', '/tmp/r9700.lock', 'timeout', '1500', sys.executable, '-I',
                       str(script), str(count), '160', str(prompts / f'{name}.json')] + flags
            output = out / f'{tag}.out'
            generation = out / f'{tag}.json'
            if output.exists() and generation.exists():
                try:
                    previous = output.read_text()
                    decoded = json.loads(previous)
                    tokens = json.loads(generation.read_text())
                    if ('all_active_tps' in decoded and len(tokens) == count and
                            all(isinstance(ids, list) and ids for ids in tokens.values())):
                        print('reuse', tag, flush=True)
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            print('start', tag, flush=True)
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            output.write_text(result.stdout + result.stderr)
            print('end', tag, 'status', result.returncode, result.stdout.strip(), flush=True)
            if result.returncode != 0:
                raise SystemExit(f'comparison failed: {tag}')
