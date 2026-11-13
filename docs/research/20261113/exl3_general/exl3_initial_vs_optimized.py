import json
import os
import re
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / 'build-gfx1201/exl3-general'
env = os.environ.copy()
env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
           LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',
           PATH='/tmp/venv-so/bin:' + env['PATH'])
model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
drafters = {'bf16': '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2',
            'mul24': '/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24'}
records = []


def run(mode, concurrency, repeat, variant, prompt):
    exe = folder / ('shelf-compute.base' if variant == 'group' else 'shelf-compute.optimized')
    if concurrency == 1:
        cmd = [str(exe), '--model-dir', model, '--input-ids-file',
               str(root / f'docs/research/20261113/exl3_general/prompt_{prompt}.ids'), '--max-new-tokens', '128',
               '--max-seq-len', '1024', '--temperature', '0']
        if mode != 'none':
            cmd += ['--dflash2-model-dir', drafters[mode], '--dflash2-drafts', '7']
    else:
        cmd = ['/tmp/venv-so/bin/python', '-I', 'docs/research/20261113/exl3_general/exl3_batch.py',
               '--compute', str(exe), '--model', model, '--concurrency', str(concurrency),
               '--runs', '1', '--tokens', '128', '--prompts',
               'docs/research/20261113/exl3_general/prompt_code.ids', 'docs/research/20261113/exl3_general/prompt_prose.ids',
               'docs/research/20261113/exl3_general/prompt_ja.ids']
        if mode != 'none':
            cmd += ['--spec', '--drafter', drafters[mode]]
    result = subprocess.run(['flock', '-w', '600', '/tmp/r9700.lock', 'timeout', '-k', '10',
                             '--signal=INT', '150', *cmd], env=env, text=True, capture_output=True)
    record = {'mode': mode, 'concurrency': concurrency, 'repeat': repeat, 'variant': variant,
              'prompt': prompt, 'exit': result.returncode, 'stderr_tail': result.stderr[-2000:]}
    if concurrency == 1:
        record['fields'] = dict(re.findall(r'^([A-Z_]+)=(.*)$', result.stdout + '\n' + result.stderr, re.M))
    else:
        try:
            record['result'] = json.loads(result.stdout)
        except ValueError:
            record['stdout'] = result.stdout
    records.append(record)
    (folder / 'initial-vs-optimized.json').write_text(json.dumps(records, indent=2) + '\n')
    speed = record.get('fields', {}).get('DECODE_TOKENS_PER_SEC')
    if concurrency > 1:
        speed = record.get('result', {}).get('runs', [{}])[0].get('tokens_per_s')
    print(mode, concurrency, repeat, variant, prompt, result.returncode, speed, flush=True)
    if result.returncode:
        raise RuntimeError(str(record))


for mode in ['none', 'bf16']:
    for concurrency in [1, 2, 4]:
        for repeat in range(3):
            order = ['group', 'candidate'] if repeat % 2 == 0 else ['candidate', 'group']
            for variant in order:
                run(mode, concurrency, repeat, variant, 'code' if concurrency == 1 else 'mixed')
