#!/usr/bin/env python3
# Validation harness for shelf-compute --serve-stdio (steady / mix modes).
import argparse, json, os, queue, statistics, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument('--bin', required=True)
ap.add_argument('--mode', default='steady')          # steady | mix
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--new', type=int, default=160)
ap.add_argument('--temp', type=float, default=0.0)
ap.add_argument('--kv', default='fp8')
ap.add_argument('--prefix', type=int, default=0)     # shared prefix tokens (steady)
ap.add_argument('--pc', type=int, default=0)         # prefix cache
ap.add_argument('--mnbt', type=int, default=2048)
ap.add_argument('--spec', type=int, default=0)       # DFlash2 drafts (0 = off)
ap.add_argument('--sessions', type=int, default=0)
ap.add_argument('--inject', type=int, default=8192)  # mix: tokens of the arriving long prompt
ap.add_argument('--err', default='/dev/null')
ap.add_argument('--tag', default='run')
a = ap.parse_args()

MODEL = '/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFT = '/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
HERE = os.path.dirname(os.path.abspath(__file__))
cases = [c['input_ids'] for c in json.load(open('/tmp/bench27/rows_cases.json')) if len(c['input_ids']) >= 16]
base = json.load(open('/tmp/bench27/prompt32k.json'))
def prefix_ids(n):
    out = []
    while len(out) < n: out += base
    return out[:n]

pref = prefix_ids(a.prefix) if a.prefix else []
longest = max(len(pref) + max(len(c) for c in cases), a.inject if a.mode == 'mix' else 0) + a.new + 256
args = [a.bin, '--model-dir', MODEL, '--serve-stdio', '--device', '0', '--max-seq-len', str(max(2048, longest)),
        '--max-concurrent-requests', str(a.n + (1 if a.mode == 'mix' else 0)), '--prefix-cache', str(a.pc),
        '--temperature', str(a.temp), '--kv-cache-dtype', a.kv, '--max-num-batched-tokens', str(a.mnbt)]
if a.spec:
    args += ['--dflash2-model-dir', DRAFT, '--dflash2-drafts', str(a.spec)]
    if a.sessions: args += ['--dflash2-max-sessions', str(a.sessions)]
p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(a.err, 'w'), text=True, bufsize=1)
q = queue.Queue()
def reader():
    for line in p.stdout:
        try: e = json.loads(line)
        except Exception: continue
        q.put((time.monotonic(), e))
    q.put((time.monotonic(), {'event': 'eof'}))
threading.Thread(target=reader, daemon=True).start()
def send(x): p.stdin.write(json.dumps(x) + '\n'); p.stdin.flush()
def gen(rid, ids, new):
    send({'op': 'generate', 'request_id': rid, 'input_ids': ids, 'max_new_tokens': new, 'temperature': a.temp, 'seed': rid})
def wait_for(pred, timeout=1800):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try: t, e = q.get(timeout=1)
        except queue.Empty: continue
        if e.get('event') == 'eof': raise RuntimeError('compute ended early')
        if e.get('event') == 'error': raise RuntimeError(str(e)[:300])
        yield t, e
        if pred(t, e): return
    raise RuntimeError('timeout')
result = {'tag': a.tag, 'mode': a.mode, 'n': a.n, 'kv': a.kv, 'temp': a.temp, 'spec': a.spec, 'prefix': a.prefix, 'mnbt': a.mnbt}
try:
    send({'op': 'ping'})
    for _ in wait_for(lambda t, e: e.get('event') == 'pong'): pass
    warm = (pref + cases[0]) if pref else cases[0]
    gen(900, warm, 8)
    for _ in wait_for(lambda t, e: e.get('request_id') == 900 and e.get('event') == 'done'): pass
    ts, first, done, toks = {}, {}, {}, {}
    def ingest(t, e):
        r = e.get('request_id'); ev = e.get('event')
        if ev == 'token' and r != 900:
            ts.setdefault(r, []).append(t); first.setdefault(r, t)
        elif ev == 'done' and r != 900:
            done[r] = t; toks[r] = e.get('generated_ids') or []
    if a.mode == 'steady':
        t0 = time.monotonic()
        for i in range(a.n):
            gen(i + 1, (pref + cases[i % len(cases)]) if pref else cases[i % len(cases)], a.new)
        for t, e in wait_for(lambda t, e: len(done) >= a.n and False):
            ingest(t, e)
            if len(done) >= a.n: break
        lo, hi = max(first.values()), min(done.values())
        cnt = sum(sum(1 for x in v if lo <= x <= hi) for v in ts.values())
        result.update(all_active_tps=round(cnt / (hi - lo), 1) if hi - lo > 0.5 else None, win_s=round(hi - lo, 2),
                      ttft_max_s=round(max(first.values()) - t0, 2), total_tokens=sum(len(ts[r]) for r in ts))
    else:  # mix
        for i in range(a.n):
            gen(i + 1, cases[i % len(cases)], a.new)
        inj_t = inj_first = None
        bg = set(range(1, a.n + 1))
        for t, e in wait_for(lambda t, e: False):
            ingest(t, e)
            if inj_t is None and len(first) >= a.n and all(len(ts.get(r, [])) >= 8 for r in bg):
                time.sleep(0.0)
                if t - max(first.values()) > 1.5 or True:
                    inj_t = time.monotonic(); gen(5000, prefix_ids(a.inject), 4)
            if e.get('event') == 'token' and e.get('request_id') == 5000 and inj_first is None:
                inj_first = t
            if all(r in done for r in bg): break
        gaps = []
        for r in bg:
            v = ts[r]; gaps += [(v[i], v[i] - v[i - 1]) for i in range(1, len(v))]
        pre = [g for (t, g) in gaps if inj_t - 3 <= t < inj_t]
        dur = [g for (t, g) in gaps if inj_t <= t <= inj_first + 1.0]
        cnt_pre = sum(1 for r in bg for x in ts[r] if inj_t - 3 <= x < inj_t)
        cnt_dur = sum(1 for r in bg for x in ts[r] if inj_t <= x <= inj_first)
        result.update(bg_pre_tps=round(cnt_pre / 3.0, 1), inject_tokens=a.inject,
                      inject_ttft_s=round(inj_first - inj_t, 2),
                      bg_tokens_during_prefill=cnt_dur, bg_tps_during=round(cnt_dur / max(inj_first - inj_t, 1e-3), 1),
                      itl_max_s=round(max(g for _, g in gaps), 2), itl_p99_s=round(sorted(g for _, g in gaps)[int(0.99 * len(gaps))], 3),
                      itl_pre_med_ms=round(1000 * statistics.median(pre), 1) if pre else None,
                      total_tokens=sum(len(ts[r]) for r in bg))
    # sanity: degenerate output check
    flat = [t for r in toks for t in toks[r]]
    if flat:
        result['distinct_ratio'] = round(len(set(flat)) / len(flat), 3)
        result['mean_len'] = round(statistics.mean(len(v) for v in toks.values()), 1)
except Exception as ex:
    result['error'] = str(ex)[:300]
finally:
    try:
        send({'op': 'shutdown'}); p.stdin.close(); p.wait(timeout=60)
    except Exception:
        p.kill()
print(json.dumps(result))
