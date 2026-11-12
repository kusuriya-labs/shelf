#!/usr/bin/env python3
"""Run inside the designated worktree; each server holds the GPU flock until exit.

Results are checkpointed after every trial. Repeat order reverses within each
prompt/concurrency group; `--limit` allows a smoke test. Logs stay under build/.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import aiohttp

ROOT = Path(__file__).resolve().parents[4]
MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFT = '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
OUT = Path(os.environ.get('FOREST_RESULTS', ROOT / 'docs/research/20261111/forest_concurrency_bottleneck_results.json'))
RUNS = ROOT / 'build-gfx1201/forest-c2-runs'
LIB = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
spec = importlib.util.spec_from_file_location('measure_prompts', ROOT/'docs/research/20261111/gdn_recurrence_scripts/measure.py')
# Avoid executing the original measurement module and its environment dependencies.
import ast
module = ast.parse(spec.origin and Path(spec.origin).read_text())
prompts = next(ast.literal_eval(x.value) for x in module.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPTS' for t in x.targets))


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def request(session, addr, prompt):
    payload = {'model': 'shelf', 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': 512, 'temperature': 0, 'seed': 0, 'stream': True,
               'chat_template_kwargs': {'enable_thinking': False},
               'stream_options': {'include_usage': True}}
    usage = None
    finish = None
    async with session.post(addr + '/v1/chat/completions', json=payload) as res:
        if res.status != 200:
            raise RuntimeError(f'HTTP {res.status}: {await res.text()}')
        async for raw in res.content:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            ev = json.loads(line[5:])
            if ev.get('usage'): usage = ev['usage']
            for choice in ev.get('choices') or []:
                if choice.get('finish_reason'): finish = choice['finish_reason']
    if not usage or usage['completion_tokens'] != 512 or finish != 'length':
        raise RuntimeError(f'incomplete response: {usage}, {finish}')
    return {'usage': usage, 'finish': finish}


async def trial(c, kind, budget, repeat, diagnostic):
    ids = ([1, 13, 12, 21] if kind == 'A' else [1, 0, 3, 2])[:c]
    # c=1 uses the same code prompt for A and B; there is no duplicated B trial.
    tag = f'{"sync" if diagnostic == "sync" else "diag" if diagnostic else "speed"}-c{c}-{kind}-n{budget}-r{repeat}'
    trace = RUNS / (tag + '.tokens.jsonl')
    log = RUNS / (tag + '.log')
    trace.unlink(missing_ok=True)
    number = port()
    args = ['flock', '/tmp/r9700.lock', sys.executable,
            str(ROOT/'src/apps/server/shelf_server.py'), '--compute', str(ROOT/'build-gfx1201/shelf-compute'),
            '--model-dir', MODEL, '--dflash2-model-dir', DRAFT, '--dflash2-drafts', '7',
            '--dflash2-tree-budget', str(budget), '--max-seq-len', '1024',
            '--max-concurrent-requests', str(c), '--kv-cache-dtype', 'bf16', '--device', '0', '--port', str(number)]
    env = dict(os.environ, LD_LIBRARY_PATH=LIB, ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
               SHELF_DFLASH2_TOKEN_TIMESTAMPS=str(trace), PYTHONUNBUFFERED='1')
    for key in ('SHELF_DFLASH2_SERVE_STATS', 'SHELF_DFLASH2_COST_TRACE', 'SHELF_DFLASH2_BATCH_PHASE_TRACE', 'SHELF_DFLASH2_BATCH_PHASE_SYNC'):
        env.pop(key, None)
    if diagnostic:
        env.update(SHELF_DFLASH2_SERVE_STATS='1', SHELF_DFLASH2_COST_TRACE='1', SHELF_DFLASH2_BATCH_PHASE_TRACE='1')
        if diagnostic == 'sync': env['SHELF_DFLASH2_BATCH_PHASE_SYNC']='1'
    addr = f'http://127.0.0.1:{number}'
    started = time.monotonic()
    with log.open('w') as err:
        proc = subprocess.Popen(args, stdout=err, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
                for _ in range(10800):
                    if proc.poll() is not None: raise RuntimeError(f'server exit {proc.returncode}; {log}')
                    try:
                        async with session.get(addr + '/v1/models') as res:
                            if res.status == 200: break
                    except (aiohttp.ClientError, asyncio.TimeoutError): pass
                    await asyncio.sleep(1)
                else: raise TimeoutError('server startup / GPU lock')
                launch = time.monotonic_ns()
                responses = await asyncio.gather(*(request(session, addr, prompts[i]) for i in ids))
                end = time.monotonic_ns()
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            assert len(records) == c and all(len(r['timestamps_ns']) == 512 for r in records), (tag, [(r['request_id'],len(r['timestamps_ns'])) for r in records])
            low = max(r['timestamps_ns'][0] for r in records)
            high = min(r['timestamps_ns'][-1] for r in records)
            count = sum(sum(low <= t <= high for t in r['timestamps_ns']) for r in records)
            return {'c': c, 'kind': kind, 'budget': budget, 'repeat': repeat, 'diagnostic': diagnostic,
                    'prompt_indices': ids, 'responses': responses, 'records': records,
                    'all_active_tokens': count, 'all_active_seconds': (high-low)/1e9,
                    'all_active_tps': count/((high-low)/1e9),
                    'request_wall_seconds': (end-launch)/1e9,
                    'server_start_seconds': (time.monotonic()-started), 'log': str(log.relative_to(ROOT))}
        finally:
            os.killpg(proc.pid, 15)
            try: proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9); proc.wait()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--diagnostic-only', action='store_true')
    ap.add_argument('--speed-only', action='store_true')
    ap.add_argument('--sync-only', action='store_true', help='extra synchronized diagnostics for all cases')
    args=ap.parse_args(); RUNS.mkdir(parents=True, exist_ok=True)
    data=json.loads(OUT.read_text()) if OUT.exists() else {'description':'forest concurrency, token timestamps at server compute reader', 'trials':[]}
    completed={(x['c'],x['kind'],x['budget'],x['repeat'],x['diagnostic']) for x in data['trials']}
    done=0
    for diagnostic in (['sync'] if args.sync_only else [True] if args.diagnostic_only else [False] if args.speed_only else [False, True]):
        for c in (1,2,3,4):
            for kind in (['A'] if c==1 else ['A','B']):
                budgets=[0,10,12] if c==1 else [0,10]
                for repeat in range(2):
                    for budget in budgets if repeat==0 else budgets[::-1]:
                        key=(c,kind,budget,repeat,diagnostic)
                        if key in completed: continue
                        print('START', key, flush=True)
                        result=asyncio.run(trial(*key))
                        data['trials'].append(result)
                        tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(OUT)
                        print('DONE',key,'tps',round(result['all_active_tps'],2),flush=True)
                        done+=1
                        if args.limit and done>=args.limit: return

if __name__=='__main__': main()
