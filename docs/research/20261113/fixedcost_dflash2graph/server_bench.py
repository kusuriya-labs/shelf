import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/server'))
from support import ServerHarness, http_json

MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFTERS = {'bf16': '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2',
            'mul24': '/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24'}
LIBS = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
BASE = '/home/homelab1/coding-local/shelf/.worktrees/integ-fc/build-gfx1201'
OUT = Path('/tmp/fc_dflash2graph')
parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True)
parser.add_argument('--one', nargs=2, metavar=('DRAFTER', 'VARIANT'))
args = parser.parse_args()
if args.one:
    draft, variant = args.one
    os.environ['SHELF_BUILD_DIR'] = BASE if variant == 'baseline' else str(ROOT / 'build-gfx1201')
    os.environ['SHELF_MODEL_DIR'] = MODEL
    os.environ['SHELF_TEST_DEVICE'] = '0'
    tag = f'{args.phase}_server_{draft}_{variant}'
    token_path = OUT / f'{tag}.tokens'
    server = ServerHarness(log_path=OUT / f'{tag}.log', arena_gib=31, max_seq_len=512,
                           dflash2_model_dir=DRAFTERS[draft], dflash2_tree_budget=10,
                           env={'SHELF_SERVER_TOKEN_LOG': str(token_path)})
    payload = {'model': 'shelf', 'messages': [{'role': 'user', 'content': 'Write a Python module with functions for reading, writing and validating JSON files, including detailed examples and tests.'}],
               'temperature': 0, 'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}}
    try:
        server.start()
        runs = []
        for _ in range(2):
            start = time.monotonic()
            result = http_json(f'{server.base_url}/chat/completions', payload)
            elapsed = time.monotonic() - start
            count = result['usage']['completion_tokens']
            tokens = token_path.read_text().splitlines()[-1]
            runs.append({'elapsed_s': elapsed, 'completion_tokens': count,
                         'tokens_per_s': count / elapsed, 'finish_reason': result['choices'][0]['finish_reason'],
                         'token_ids_sha256': hashlib.sha256(tokens.encode()).hexdigest()})
        print(json.dumps({'drafter': draft, 'variant': variant, 'runs': runs}), flush=True)
    finally:
        server.close()
else:
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', LD_LIBRARY_PATH=LIBS,
               PATH='/tmp/venv-so/bin:' + env['PATH'], TMPDIR=str(OUT))
    records = []
    for draft in DRAFTERS:
        for variant in ('baseline', 'candidate'):
            cmd = ['flock', '/tmp/r9700.lock', 'timeout', '-k', '5', '270', '/tmp/venv-so/bin/python', '-I',
                   str(Path(__file__).resolve()), '--phase', args.phase, '--one', draft, variant]
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            if p.returncode:
                print('FAIL', draft, variant, p.returncode, p.stderr[-2000:], flush=True)
                sys.exit(1)
            record = json.loads(p.stdout.strip().splitlines()[-1])
            records.append(record)
            (OUT / f'{args.phase}_server.json').write_text(json.dumps(records, indent=2) + '\n')
            print(draft, variant, record['runs'][-1]['tokens_per_s'], flush=True)
