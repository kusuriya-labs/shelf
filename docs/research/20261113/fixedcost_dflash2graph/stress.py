import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFTER = '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--log', required=True, type=Path)
parser.add_argument('--requests', type=int, default=64)
args = parser.parse_args()
prompts = [[int(x) for x in (ROOT / f'docs/research/20261113/exl3_general/prompt_{name}.ids').read_text().split()] for name in ('code', 'prose', 'ja')]
cmd = [str(ROOT / 'build-gfx1201/shelf-compute'), '--serve-stdio', '--model-dir', MODEL,
       '--max-seq-len', '512', '--device', '0', '--kv-cache-dtype', 'bf16',
       '--max-concurrent-requests', '1', '--prefix-cache', '0', '--dflash2-model-dir', DRAFTER,
       '--dflash2-drafts', '7', '--dflash2-tree-budget', '10']
results = []
start = time.monotonic()
with args.log.open('w') as log:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    try:
        for i in range(args.requests):
            request = {'op': 'generate', 'request_id': i + 1, 'input_ids': prompts[i % len(prompts)],
                       'max_new_tokens': 256, 'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'seed': 0}
            proc.stdin.write(json.dumps(request) + '\n')
            proc.stdin.flush()
            streamed = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'compute closed stdout, rc={proc.poll()}')
                event = json.loads(line)
                if event.get('event') == 'error':
                    raise RuntimeError(str(event))
                if event.get('request_id') != i + 1:
                    continue
                if event.get('event') == 'token':
                    streamed.append(event['token_id'])
                elif event.get('event') == 'done':
                    if streamed != event['generated_ids']:
                        raise RuntimeError('streamed token mismatch')
                    results.append(event['generated_ids'])
                    args.output.write_text(json.dumps(results) + '\n')
                    print(i + 1, len(streamed), f'{time.monotonic() - start:.1f}', flush=True)
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
