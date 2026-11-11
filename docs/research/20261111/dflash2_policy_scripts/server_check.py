#!/usr/bin/env python3
import argparse, asyncio, json, os, subprocess, sys, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument('--server', required=True)
ap.add_argument('--compute')
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--new', type=int, default=160)
ap.add_argument('--spec', type=int, default=0)
ap.add_argument('--adaptive', action='store_true')
ap.add_argument('--temp', type=float, default=0.0)
ap.add_argument('--kv', default='fp8')
ap.add_argument('--port', type=int, default=8211)
ap.add_argument('--err', default='/dev/null')
a = ap.parse_args()
MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFT = '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
argv = [a.server, '--model-dir', MODEL, '--port', str(a.port), '--max-seq-len', '4096',
        '--max-concurrent-requests', str(a.n), '--kv-cache-dtype', a.kv]
if a.compute: argv += ['--compute', a.compute]
if a.spec: argv += ['--dflash2-model-dir', DRAFT, '--dflash2-drafts', str(a.spec)]
if a.adaptive: argv += ['--dflash2-adaptive']
proc = subprocess.Popen(argv, stdout=open(a.err, 'w'), stderr=subprocess.STDOUT)
PROMPTS = [
    '日本の四季について、それぞれの特徴を説明してください。',
    'Explain the trade-offs of a database transaction isolation level.',
    'Pythonでクイックソートを実装して解説して。',
    'Write a shell script to find and sort large files safely.',
    'ブラックホールとは何か説明して。',
    'Write a Python function to calculate the longest common subsequence.',
    '機械学習の過学習とは何か。',
    'Explain why the square root of two is irrational.',
]
async def one(session, i, rec):
    body = {'model': 'shelf', 'messages': [{'role': 'user', 'content': PROMPTS[i % len(PROMPTS)] + f' ({i})'}], 'stream': True,
            'max_tokens': a.new, 'temperature': a.temp, 'seed': i, 'chat_template_kwargs': {'enable_thinking': False},
            'stream_options': {'include_usage': True}}
    stamps = []; usage = None
    async with session.post(f'http://127.0.0.1:{a.port}/v1/chat/completions', json=body) as r:
        if r.status != 200:
            rec[i] = {'error': f'{r.status} {(await r.text())[:120]}'}; return
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]': continue
            ev = json.loads(line[5:])
            if ev.get('usage'): usage = ev['usage']
            ch = (ev.get('choices') or [{}])[0]
            if (ch.get('delta') or {}).get('content'): stamps.append(time.monotonic())
    rec[i] = {'stamps': stamps, 'usage': usage}
async def main():
    for _ in range(600):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f'http://127.0.0.1:{a.port}/v1/models') as r:
                    if r.status == 200: break
        except Exception: pass
        if proc.poll() is not None: raise RuntimeError('server exited')
        await asyncio.sleep(1)
    rec = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
        async with s.post(f'http://127.0.0.1:{a.port}/v1/chat/completions', json={'model': 'shelf', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8}) as r:
            await r.read()
        t0 = time.monotonic()
        await asyncio.gather(*[one(s, i, rec) for i in range(a.n)])
    errs = [v['error'] for v in rec.values() if 'error' in v]
    ok = [v for v in rec.values() if 'stamps' in v and v['stamps']]
    out = {'n': a.n, 'spec': a.spec, 'kv': a.kv, 'temp': a.temp, 'errors': errs[:3], 'completed': len(ok)}
    if ok:
        lo = max(v['stamps'][0] for v in ok); hi = min(v['stamps'][-1] for v in ok)
        cnt = sum(sum(1 for x in v['stamps'] if lo <= x <= hi) for v in ok)
        out.update(all_active_chunks_per_s=round(cnt / (hi - lo), 1) if hi - lo > 0.5 else None, win_s=round(hi - lo, 2),
                   wall_s=round(max(v['stamps'][-1] for v in ok) - t0, 2),
                   completion_tokens=sum((v['usage'] or {}).get('completion_tokens', 0) for v in ok),
                   chunks=sum(len(v['stamps']) for v in ok))
    if ok and all(v['usage'] is not None for v in ok):
        out['whole_run_token_tps'] = round(out['completion_tokens'] / out['wall_s'], 2)
    print(json.dumps(out))
    if errs or len(ok) != a.n or any(v['usage'] is None for v in ok):
        raise SystemExit(1)
try:
    asyncio.run(main())
finally:
    proc.terminate()
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
