#!/usr/bin/env python3
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = pathlib.Path(__file__).resolve().parent / 'real_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def corpus():
    manifest = json.loads((CORPUS / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest((CORPUS / name).read_bytes()) != expected:
            raise ValueError('corpus checksum mismatch: ' + name)
    return manifest, json.loads((CORPUS / 'tasks.json').read_text())


def loop_metrics(text):
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    grams = list(zip(*(tokens[i:] for i in range(5))))
    ratio = 1 - len(set(grams)) / len(grams) if grams else 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    longest = current = 0
    previous = None
    for line in lines:
        current = current + 1 if line == previous else 1
        longest = max(longest, current)
        previous = line
    period = None
    for n in range(1, min(64, len(tokens) // 4) + 1):
        if tokens[-n:] * 4 == tokens[-4*n:]:
            period = n
            break
    return {'loop_flagged': ratio >= 0.45 or longest >= 4 or period is not None,
            'repeated_5gram_ratio': ratio, 'max_identical_lines': longest,
            'tail_period_tokens': period}


def stream(endpoint, model, task, seed):
    context = (CORPUS / task['context_file']).read_text()
    body = {'model': model, 'messages': [{'role': 'user', 'content': task['prompt'] + '\n\n' + context}],
            'max_tokens': task['max_tokens'], 'temperature': task['temperature'],
            'top_p': task['top_p'], 'top_k': task['top_k'], 'seed': seed,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': task['thinking']}}
    request = urllib.request.Request(endpoint.rstrip('/') + '/chat/completions',
                                     json.dumps(body, ensure_ascii=False).encode(),
                                     {'Content-Type': 'application/json'}, method='POST')
    start = time.monotonic()
    first = last = None
    chunks, times = [], []
    usage = None
    finish = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('usage') is not None:
                    usage = event['usage']
                for choice in event.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta') or {}
                    chunk = (delta.get('reasoning_content') or '') + (delta.get('content') or '')
                    if chunk:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                        times.append(now)
                        chunks.append(chunk)
        end = time.monotonic()
        if not done:
            raise ValueError('stream ended without [DONE]')
        count = usage.get('completion_tokens') if usage else None
        approximate = count is None
        count = len(chunks) if count is None else count
        content = ''.join(chunks)
        result = {'ok': bool(chunks), 'error': None if chunks else 'empty output',
                  'ttft_ms': (first-start)*1000 if first is not None else None,
                  'e2e_ms': (end-start)*1000, 'completion_tokens': count,
                  'completion_tokens_approximate': approximate,
                  'decode_tok_s': (count-1)/(last-first) if first is not None and last > first and count > 1 else None,
                  'update_gaps_ms': [(b-a)*1000 for a,b in zip(times,times[1:])],
                  'tokens_per_update': count / len(chunks) if chunks and not approximate else None,
                  'finish_reason': finish, 'output_sha256': digest(content.encode()),
                  'output_text': content, 'usage': usage, 'context_sha256': digest(context.encode())}
        result.update(loop_metrics(content))
        return result
    except Exception as exc:
        return {'ok': False, 'error': str(exc), 'e2e_ms': (time.monotonic()-start)*1000,
                'loop_flagged': False}


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(manifest, tasks, results, non_loop=False):
    cells = []
    for cell in manifest['cells']:
        ids = [t['id'] for t in tasks if all(t[k] == v for k,v in cell.items())]
        per_task = []
        for task_id in ids:
            runs = [r['decode_tok_s'] for r in results if r['task_id'] == task_id and r.get('ok') and r.get('decode_tok_s') is not None and (not non_loop or not r['loop_flagged'])]
            per_task.append(median(runs))
        value = statistics.mean(per_task) if per_task and all(x is not None for x in per_task) else None
        cells.append({**cell, 'decode_tok_s': value})
    return {'cells': cells, 'macro_decode_tok_s': statistics.mean(c['decode_tok_s'] for c in cells) if cells and all(c['decode_tok_s'] is not None for c in cells) else None}


def metadata(args):
    def command(cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = command(['git', 'status', '--porcelain'])
    return {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'git_revision': command(['git', 'rev-parse', 'HEAD']),
            'dirty': bool(status) if status is not None else None,
            'gpu': command(['rocm-smi', '--showproductname', '--showuniqueid']),
            'rocm_path': os.path.realpath('/opt/rocm/core'), 'arguments': vars(args)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock-file')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--list', action='store_true')
    sub = p.add_subparsers(dest='track', required=True)
    c = sub.add_parser('track-c')
    c.add_argument('--endpoint', default='http://127.0.0.1:8080/v1')
    c.add_argument('--model', default='default')
    c.add_argument('--output')
    c.add_argument('--task', action='append')
    c.add_argument('--repeats', type=int, default=1)
    c.add_argument('--warmup', type=int, default=0)
    c.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p.error('repeats must be positive and warmup non-negative')
    manifest, tasks = corpus()
    if args.task:
        tasks = [t for t in tasks if t['id'] in args.task]
        if len(tasks) != len(set(args.task)):
            p.error('unknown task')
    if args.list or args.dry_run:
        print(json.dumps(tasks, ensure_ascii=False, indent=2))
        return
    lock = open(args.lock_file, 'a') if args.lock_file else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        records = []
        for task in tasks:
            for i in range(args.warmup + args.repeats):
                record = stream(args.endpoint, args.model, task, args.seed + i)
                if i >= args.warmup:
                    records.append({'task_id': task['id'], 'repeat': i-args.warmup, **record})
        result = {'schema_version': 1, 'track': 'C', 'metadata': metadata(args), 'corpus_version': manifest['version'],
                  'results': records, 'all': summarize(manifest, tasks, records),
                  'non_loop': summarize(manifest, tasks, records, True),
                  'failures': sum(not r['ok'] for r in records),
                  'truncated': sum(r.get('finish_reason') == 'length' for r in records)}
        data = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            pathlib.Path(args.output).write_text(data)
        else:
            print(data, end='')
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
