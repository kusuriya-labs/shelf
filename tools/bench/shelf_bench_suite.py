#!/usr/bin/env python3
import argparse
import concurrent.futures
import contextlib
import datetime
import fcntl
import hashlib
import http.client
import json
import math
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = pathlib.Path(__file__).resolve().parent / 'real_v1'
VENDOR = ROOT / 'vendor/BetterBench'
CATEGORIES = ('code', 'file_edit', 'json', 'math', 'prose', 'reasoning', 'summarization', 'chat')
ROUND_FIELDS = ('ms_per_round', 'accepted_drafts_per_round', 'committed_tokens_per_round')
WORKLOAD_WEIGHTS = (('tool', 45), ('reasoning', 25), ('coding', 20), ('doc', 10))
NEXT_SECTION = {
    'type': 'function',
    'function': {
        'name': 'read_next_section',
        'description': '調査資料の次のコード断片を取得する。',
        'parameters': {
            'type': 'object',
            'properties': {
                'topic': {
                    'type': 'string',
                    'enum': ['state', 'capacity', 'ownership', 'cleanup'],
                    'description': '次の断片で確認する観点。調査記録に添付する。',
                }
            },
            'required': ['topic'],
        },
    },
}
CONDITIONS = (
    'gpu',
    'weights',
    'quantization',
    'kv',
    'tokenizer',
    'chat_template',
    'thinking',
    'cache',
    'tp',
    'drafter',
    'eos',
    'context_limit',
    'corpus_sha256',
    'server_flags',
)
ENDPOINT = 'http://127.0.0.1:8080/v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def corpus():
    manifest = read_json(CORPUS / 'manifest.json')
    for name, expected in manifest['files'].items():
        if digest((CORPUS / name).read_bytes()) != expected:
            raise ValueError('資料の SHA-256 が一致しません: ' + name)
    return manifest, read_json(CORPUS / 'tasks.json')


def percentile(values, q):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = int(index)
    return values[lower] + (index - lower) * (values[math.ceil(index)] - values[lower])


def median(values):
    return percentile(values, 0.5)


def latency(values):
    values = list(values)
    return {'p50': median(values), 'p95': percentile(values, 0.95)}


def metadata(args):
    def command(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = command(['git', 'status', '--porcelain'])
    return {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_revision': command(['git', 'rev-parse', 'HEAD']),
        'dirty': bool(status) if status is not None else None,
        'gpu': command(['rocm-smi', '--showproductname', '--showuniqueid']),
        'rocr_visible_devices': os.getenv('ROCR_VISIBLE_DEVICES'),
        'rocm_path': os.path.realpath('/opt/rocm/core'),
        'arguments': vars(args),
    }


def loop_metrics(text):
    tokens = re.findall(r'[A-Za-z0-9_]+|[^\s]', text)
    grams = list(zip(*(tokens[i:] for i in range(5))))
    ratio = 1 - len(set(grams)) / len(grams) if grams else 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    longest = current = 0
    repeated_line = False
    previous = None
    for line in lines:
        current = current + 1 if line == previous else 1
        longest = max(longest, current)
        if current >= 4 and any(ch.isalnum() for ch in line):
            repeated_line = True
        previous = line
    period = None
    for n in range(1, min(64, len(tokens) // 4) + 1):
        if tokens[-n:] * 4 == tokens[-4 * n :] and any(t.isalnum() for t in tokens[-n:]):
            period = n
            break
    return {
        'loop_flagged': (len(grams) >= 40 and ratio >= 0.65) or repeated_line or period is not None,
        'repeated_5gram_ratio': ratio,
        'max_identical_lines': longest,
        'tail_period_tokens': period,
    }


def request_body(model, task, seed):
    context = (CORPUS / task['context_file']).read_text()
    if 'messages' in task:
        messages = [m.copy() for m in task['messages']]
    else:
        messages = [{'role': 'user', 'content': task['prompt']}]
    messages[0]['content'] += '\n\n' + context
    return {
        'model': model,
        'messages': messages,
        'max_tokens': task['max_tokens'],
        'temperature': task['temperature'],
        'top_p': task['top_p'],
        'top_k': task['top_k'],
        'seed': seed,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': task['thinking']},
    }


def stream(endpoint, body, on_first=None):
    url = endpoint.rstrip('/')
    if not url.endswith('/chat/completions'):
        url += '/chat/completions'
    request = urllib.request.Request(
        url, json.dumps(body, ensure_ascii=False).encode(), {'Content-Type': 'application/json'}
    )
    chunks, answers, reasonings, times = [], [], [], []
    calls = {}
    stats = dict.fromkeys(ROUND_FIELDS)
    spec_stats = None
    usage = finish = error = None
    done = False
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise ValueError(str(event['error']))
                usage = event.get('usage') or usage
                counters = event.get('speculative_stats')
                if isinstance(counters, dict):
                    spec_stats = dict(counters)
                    for key in ROUND_FIELDS:
                        if key in counters:
                            stats[key] = counters[key]
                for choice in event.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta') or {}
                    answer = delta.get('content') or ''
                    reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    tool_deltas = delta.get('tool_calls') or []
                    for fragment in tool_deltas:
                        call = calls.setdefault(
                            fragment['index'],
                            {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}},
                        )
                        call['id'] += fragment.get('id') or ''
                        fn = fragment.get('function') or {}
                        call['function']['name'] += fn.get('name') or ''
                        call['function']['arguments'] += fn.get('arguments') or ''
                    chunk = reasoning + answer
                    if chunk or tool_deltas:
                        times.append(time.monotonic())
                        if chunk:
                            chunks.append(chunk)
                        if answer:
                            answers.append(answer)
                        if reasoning:
                            reasonings.append(reasoning)
                        if len(times) == 1 and on_first:
                            on_first()
        if not done or not times or finish not in ('stop', 'length', 'tool_calls'):
            error = '生成が完了していません: ' + str(finish)
    except (OSError, ValueError, TypeError, EOFError, http.client.HTTPException) as exc:
        error = str(exc)
    end = time.monotonic()
    count = (usage or {}).get('completion_tokens')
    prompt_tokens = (usage or {}).get('prompt_tokens')
    cached_tokens = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    approximate = count is None
    if approximate:
        count = len(times)
    if spec_stats is not None:
        proposed = spec_stats.get('proposed_drafts')
        accepted = spec_stats.get('accepted_drafts')
        verify_rows = spec_stats.get('verify_rows')
        spec_stats['acceptance_rate'] = accepted / proposed if proposed and accepted is not None else None
        spec_stats['verify_rows_per_tok'] = (
            verify_rows / count if verify_rows is not None and not approximate and count else None
        )
    content = ''.join(chunks)
    duration = times[-1] - times[0] if times else 0
    assistant = {'role': 'assistant', 'content': ''.join(answers)}
    if calls:
        assistant['tool_calls'] = [calls[i] for i in sorted(calls)]
    if reasonings:
        assistant['reasoning_content'] = ''.join(reasonings)
    return {
        'ok': error is None,
        'error': error,
        'finish_reason': finish,
        'usage': usage,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'completion_tokens': count,
        'completion_tokens_approximate': approximate,
        'ttft_ms': (times[0] - start) * 1000 if times else None,
        'e2e_ms': (end - start) * 1000,
        'decode_tok_s': (count - 1) / duration if duration and count > 1 and not approximate else None,
        'update_gaps_ms': [(b - a) * 1000 for a, b in zip(times, times[1:])],
        'tokens_per_update': count / len(times) if times and not approximate else None,
        'output_sha256': digest(content.encode()),
        'output_text': content,
        'answer_text': ''.join(answers),
        'assistant_message': assistant,
        'speculative_stats': spec_stats,
        **stats,
    }


def summarize(manifest, tasks, results):
    cells = []
    for cell in manifest['cells']:
        ids = [t['id'] for t in tasks if all(t[k] == v for k, v in cell.items())]
        per_task = [
            median(r['decode_tok_s'] for r in results if r['task_id'] == task_id and r['ok'])
            for task_id in ids
        ]
        value = statistics.mean(per_task) if per_task and None not in per_task else None
        cells.append({**cell, 'decode_tok_s': value})
    values = [c['decode_tok_s'] for c in cells]
    return {'cells': cells, 'macro_decode_tok_s': statistics.mean(values) if None not in values else None}


def track_c(args, preview):
    manifest, tasks = corpus()
    selected = [t for t in tasks if not args.task or t['id'] in args.task]
    if args.task and len(selected) != len(set(args.task)):
        raise ValueError('未知の課題 ID が含まれています')
    if preview:
        return selected
    results, warmup_errors = [], []
    for task in selected:
        for i in range(-args.warmup, args.repeats):
            body = request_body(args.model, task, args.seed + i)
            if args.reasoning_effort and task['thinking']:
                body['reasoning_effort'] = args.reasoning_effort
            if args.no_answer_reserve:
                body['reserve_answer_tokens'] = False
            r = stream(args.endpoint, body)
            if i < 0:
                if not r['ok']:
                    warmup_errors.append({'task_id': task['id'], 'error': r['error']})
                continue
            results.append(
                {
                    'task_id': task['id'],
                    'repeat': i,
                    'seed': body['seed'],
                    'context_sha256': manifest['files'][task['context_file']],
                    'meets_min_tokens': not r['completion_tokens_approximate']
                    and r['completion_tokens'] >= task['min_expected_tokens'],
                    **r,
                    **loop_metrics(r['output_text']),
                }
            )
    qualified = [r for r in results if r['meets_min_tokens']]
    return {
        'corpus_version': manifest['version'],
        'manifest_sha256': digest((CORPUS / 'manifest.json').read_bytes()),
        'sampling': {
            t['id']: {
                k: t[k]
                for k in ('temperature', 'top_p', 'top_k', 'thinking', 'max_tokens', 'min_expected_tokens')
            }
            for t in selected
        },
        'results': results,
        'all': summarize(manifest, tasks, results),
        'non_loop': summarize(manifest, tasks, [r for r in results if not r['loop_flagged']]),
        'long_output': summarize(manifest, tasks, qualified),
        'long_non_loop': summarize(manifest, tasks, [r for r in qualified if not r['loop_flagged']]),
        'warmup_errors': warmup_errors,
        'failures': sum(not r['ok'] for r in results),
        'truncated': sum(r['finish_reason'] == 'length' for r in results),
        'loop_count': sum(r['loop_flagged'] for r in results),
        'long_completed': sum(r['ok'] for r in qualified),
    }


def parse_rounds(log):
    counters = {}
    for line in log.splitlines():
        match = re.search(
            r'\bDFLASH2_(ROUNDS|ACCEPTED_DRAFTS|GENERATED_TOKENS|ROUND_MS|MEAN_ACCEPTED|EMITTED_PER_ROUND)=([\d.]+)',
            line,
        )
        if match:
            key, value = match.groups()
            if key in counters:
                raise ValueError('複数の診断結果を単一の測定に割り当てられません')
            counters[key] = float(value)
        stage = re.search(r'\bDFLASH2_STAGE round=([\d.]+)', line)
        if stage:
            counters['stage'] = float(stage[1])
    rounds = counters.get('ROUNDS')
    stats = {}
    for field, total, average in zip(
        ROUND_FIELDS,
        ('ROUND_MS', 'ACCEPTED_DRAFTS', 'GENERATED_TOKENS'),
        ('stage', 'MEAN_ACCEPTED', 'EMITTED_PER_ROUND'),
    ):
        stats[field] = counters[total] / rounds if rounds and total in counters else counters.get(average)
        if rounds == 0:
            stats[field] = None
    return stats


def native_cases(args):
    cases = []
    prefix = [
        '--model-dir',
        args.model_dir or '<model-dir>',
        '--device',
        args.device,
        '--tensor-parallel',
        str(args.tensor_parallel),
    ]
    tokens_file = ['--tokens-file', args.tokens_file] if args.tokens_file else []
    for n in args.pp_lengths:
        cases.append(
            (
                f'pp/{n}',
                [
                    'pp',
                    *prefix,
                    '--prompt-tokens',
                    str(n),
                    '--mode',
                    'forward',
                    '--runs',
                    '1',
                    '--warmup',
                    str(args.warmup),
                    *tokens_file,
                ],
            )
        )
    for ctx in args.contexts:
        for size in args.batch_sizes:
            if size == 1:
                cases.append(
                    (
                        f'tg/{ctx}/1',
                        [
                            'tg',
                            *prefix,
                            '--context',
                            str(ctx),
                            '--tokens',
                            str(args.tg_tokens),
                            '--mode',
                            'greedy',
                            '--compute-logits',
                            '1',
                            '--warmup',
                            str(args.warmup),
                            *tokens_file,
                        ],
                    )
                )
            else:
                cmd = None
                if ctx >= 2:
                    cmd = [
                        'batch',
                        *prefix,
                        '--prefix-tokens',
                        str(ctx - 1),
                        '--suffix-tokens',
                        '1',
                        '--requests',
                        str(size),
                        '--max-new-tokens',
                        str(args.tg_tokens),
                        '--warmup',
                        str(args.warmup),
                    ]
                cases.append((f'batch/{ctx}/{size}', cmd))
    return cases


def measure_native(binary, cmd):
    if cmd is None:
        return {'ok': False, 'unsupported': True, 'error': 'batch は正の prefix と suffix を必要とします'}
    proc = subprocess.run([binary, *cmd], text=True, capture_output=True)
    output = proc.stdout
    result = {
        'argv': [binary, *cmd],
        'stdout': output,
        'stderr': proc.stderr,
        'ok': proc.returncode == 0,
        'error': None if not proc.returncode else proc.returncode,
        'gpu_tok_s': None,
        'wall_tok_s': None,
        'output_sha256': None,
    }
    if not result['ok']:
        return result
    if cmd[0] == 'batch':
        wall = re.search(r'^step_tokens=\d+ steps=\d+ wall_ms=([\d.]+)', output, re.M)
        generations = re.findall(r'^GEN\[\d+\]=([\d,-]*) \(n=(\d+)\)', output, re.M)
        expected = int(cmd[cmd.index('--requests') + 1])
        if wall and len(generations) == expected and float(wall[1]) > 0:
            count = sum(int(n) for _, n in generations)
            result.update(
                wall_tok_s=count * 1000 / float(wall[1]),
                generated_tokens=count,
                output_sha256=digest('\n'.join(ids for ids, _ in generations).encode()),
            )
    else:
        for key in ('prompt_tokens', 'context', 'final_position', 'tokens'):
            match = re.search(r'^' + key + r'=(\d+)$', output, re.M)
            if match:
                result[key] = int(match[1])
        for k in ('gpu', 'wall'):
            match = re.search(r'^' + k + r'_tokens_per_sec=([\d.]+)$', output, re.M)
            if match:
                result[k + '_tok_s'] = float(match[1])
    if result['wall_tok_s'] is None:
        result.update(ok=False, error='shelf-bench の測定値がありません')
    return result


def measure_spec(args, variant, body):
    cmd = args.baseline_spec_cmd if variant == 'baseline' else args.spec_cmd
    endpoint = args.baseline_spec_endpoint if variant == 'baseline' else args.spec_endpoint
    if cmd:
        proc = subprocess.run(shlex.split(cmd), input=json.dumps(body), text=True, capture_output=True)
        if proc.returncode:
            return {'ok': False, 'error': proc.stderr, 'meets_min_tokens': False}
        data = json.loads(proc.stdout)
        content = data['output_text']
        r = {
            **parse_rounds(proc.stderr),
            **{k: data[k] for k in ROUND_FIELDS if k in data},
            'ok': True,
            'error': None,
            'completion_tokens_approximate': False,
            'output_sha256': digest(content.encode()),
            'output_text': content,
            'completion_tokens': data['completion_tokens'],
            'decode_tok_s': data['decode_tok_s'],
        }
    else:
        r = stream(endpoint, body)
    r['meets_min_tokens'] = not r['completion_tokens_approximate'] and r['completion_tokens'] >= 1024
    if r['ok'] and not r['meets_min_tokens']:
        r.update(ok=False, error='実測出力が 1024 token に達していません')
    return r


def paired_comparison(records, speed_key):
    baseline, candidate = (next(r for r in records if r['variant'] == v) for v in ('baseline', 'candidate'))
    both = baseline['ok'] and candidate['ok']
    hashes = [r.get('output_sha256') for r in (baseline, candidate)]
    match = hashes[0] == hashes[1] if both and all(hashes) else None
    speeds = [r.get(speed_key) for r in (baseline, candidate)]
    ratio = speeds[1] / speeds[0] if both and all(speeds) else None
    return {
        'sha256_match': match,
        'raw_candidate_over_baseline': ratio,
        'candidate_over_baseline': ratio if match is not False else None,
    }


def track_a(args, preview):
    spec = args.spec_endpoint or args.spec_cmd
    ab_spec = args.baseline_spec_endpoint or args.baseline_spec_cmd
    if args.ab_compare != bool(ab_spec or args.baseline_bench):
        raise ValueError('--ab-compare と比較元の指定を組み合わせてください')
    if ab_spec and not spec:
        raise ValueError('比較先の投機 endpoint またはコマンドが必要です')
    if not args.model_dir and (args.baseline_bench or not spec) and not preview:
        raise ValueError('shelf-bench には --model-dir が必要です')
    cases = native_cases(args) if args.model_dir or (preview and not spec) else []
    if spec:
        cases.append(('spec', None))
    _, tasks = corpus()
    body = request_body(args.model, tasks[0], args.seed)
    body.update(temperature=0, max_tokens=args.spec_max_tokens)
    results, comparisons, plan = [], [], []
    for case, cmd in cases:
        is_spec = case == 'spec'
        compare = ab_spec if is_spec else args.baseline_bench
        for i in range(args.repeats):
            order = ['baseline', 'candidate'] if compare else ['candidate']
            if i % 2:
                order.reverse()
            plan.append({'case': case, 'repeat': i, 'order': order, 'arguments': cmd})
            pair = []
            for variant in order if not preview else []:
                try:
                    if is_spec:
                        r = measure_spec(args, variant, body)
                    else:
                        binary = args.baseline_bench if variant == 'baseline' else args.shelf_bench
                        r = measure_native(binary, cmd)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    r = {'ok': False, 'error': str(exc)}
                pair.append({'case': case, 'repeat': i, 'variant': variant, **r})
            results.extend(pair)
            if compare and pair:
                speed = 'decode_tok_s' if is_spec else 'wall_tok_s'
                comparisons.append(
                    {'case': case, 'repeat': i, 'order': order, **paired_comparison(pair, speed)}
                )
    if preview:
        return plan
    return {
        'sampling': {
            'native_pp': 'forward',
            'native_tg': 'greedy',
            'spec': {k: v for k, v in body.items() if k != 'messages'},
            'spec_task_id': tasks[0]['id'],
        },
        'batch_contract': {
            'max_scheduled_requests': 1,
            'timing': 'prefix_seed_and_serial_requests',
            'parallel_tg_supported': False,
        },
        'results': results,
        'comparisons': comparisons,
        'failures': sum(not r['ok'] for r in results),
        'unsupported': sum(bool(r.get('unsupported')) for r in results),
        'hash_mismatches': sum(c['sha256_match'] is False for c in comparisons),
    }


def distribution(values):
    values = sorted(x for x in values if x is not None)
    q1, q3 = (percentile(values, q) for q in (0.25, 0.75))
    return {
        'median': median(values),
        'iqr': q3 - q1 if values else None,
        'min': values[0] if values else None,
        'max': values[-1] if values else None,
    }


def category_logs(paths):
    scoped, global_log = {}, None
    for item in paths:
        label, sep, path = item.partition('=')
        if sep and label in CATEGORIES:
            if label in scoped:
                raise ValueError('カテゴリ別ログが重複しています: ' + label)
            scoped[label] = parse_rounds(pathlib.Path(path).read_text())
        else:
            if global_log is not None:
                raise ValueError('カテゴリ未指定ログは1ファイルだけ指定できます')
            global_log = parse_rounds(pathlib.Path(item).read_text())
    return scoped, global_log


def normalize_b(raw, args):
    config = raw.get('config', {})
    groups = raw['single_stream']
    passes, warmup = config.get('runs_per_category'), config.get('warmup')
    if args.import_json and args.quick and (passes, warmup) not in ((None, None), (5, 1)):
        raise ValueError('--quick と取り込み元の pass / warmup が一致しません')
    if args.quick or (passes, warmup) == (5, 1):
        mode = 'quick_smoke'
    elif (passes, warmup) == (20, 3) and all(len(groups.get(c, [])) == 20 for c in CATEGORIES):
        mode = 'full'
    else:
        mode = 'custom'
    scoped, general = category_logs(args.server_log)
    if general and len(groups) == 1 and next(iter(groups)) not in scoped:
        scoped[next(iter(groups))] = general
    categories = {}
    for category in CATEGORIES:
        runs = groups.get(category, [])
        valid = [r for r in runs if r['ok']]
        measured = [r for r in valid if r.get('completion_tokens') is not None]
        n = len(runs)
        failures = n - len(valid)
        truncated = sum(r.get('finish_reason') == 'length' for r in runs)
        categories[category] = {
            'runs': n,
            'failures': failures,
            'truncated': truncated,
            'failure_rate': failures / n if n else None,
            'truncation_rate': truncated / n if n else None,
            'missing_usage': len(valid) - len(measured),
            'long_completed': sum(r['completion_tokens'] >= 1024 for r in measured),
            'decode_tok_s': distribution(r.get('decode_tps') for r in measured),
            'ttft_ms': distribution(r.get('ttft_ms') for r in valid),
            **scoped.get(category, dict.fromkeys(ROUND_FIELDS)),
        }
    return {
        'mode': mode,
        'sampling': {k: config.get(k) for k in ('greedy', 'temperature', 'top_p', 'top_k', 'seed')},
        'categories': categories,
        'global_server_stats': general,
        'raw': raw,
        'conditions': read_json(args.conditions) if args.conditions else None,
    }


def track_b(args, preview):
    def argv(path):
        return [
            args.betterbench_python,
            '-m',
            'betterbench.cli',
            'run',
            '--decode',
            '--no-update-check',
            '--endpoint',
            args.endpoint,
            '--model',
            args.model,
            '--categories',
            *CATEGORIES,
            '--corpus',
            str(VENDOR / 'corpus/v1'),
            '--out',
            str(path),
            '--no-html',
            *(['--quick'] if args.quick else ['--passes', '20', '--warmup', '3']),
        ]

    if preview:
        return {
            'categories': CATEGORIES,
            'import': args.import_json,
            'command': None if args.import_json else argv('results.json'),
            'comparison_condition_keys': CONDITIONS,
        }
    if args.import_json:
        return normalize_b(read_json(args.import_json), args)
    with tempfile.TemporaryDirectory() as directory:
        out = pathlib.Path(directory) / 'results.json'
        proc = subprocess.run(argv(out), cwd=VENDOR, stdout=sys.stderr)
        if proc.returncode:
            raise ValueError(f'BetterBench が終了コード {proc.returncode} を返しました')
        return normalize_b(read_json(out), args)


def agentic_conversations(count, run_tag, seed, temperature):
    corpus()
    lines = (CORPUS / 'mid_8k.txt').read_text(encoding='utf-8').splitlines(keepends=True)
    counts = [0] * len(WORKLOAD_WEIGHTS)
    conversations = []
    for index in range(count):
        category = max(
            range(len(WORKLOAD_WEIGHTS)),
            key=lambda j: WORKLOAD_WEIGHTS[j][1] * (index + 1) - 100 * counts[j],
        )
        variant = counts[category]
        counts[category] += 1
        workload = WORKLOAD_WEIGHTS[category][0]
        body = {
            'temperature': temperature,
            'ignore_eos': False,
            'chat_template_kwargs': {'enable_thinking': workload == 'reasoning'},
        }
        ticket = f'調査票 {run_tag}/{index + 1}。'
        if workload == 'tool':
            body['tools'] = [NEXT_SECTION]
            body['tool_choice'] = {'type': 'function', 'function': {'name': 'read_next_section'}}
            turns = [
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': ticket + '実行時の状態管理を調査します。まず read_next_section を1回呼んでください。',
                        }
                    ]
                }
            ]
            cursor = (variant * 29 + seed) % len(lines)
            for step in range(5):
                excerpt = ''.join(lines[(cursor + offset) % len(lines)] for offset in range(24))
                cursor = (cursor + 24) % len(lines)
                followup = (
                    '取得した資料に基づき、不変条件を二つ簡潔にまとめてください。'
                    if step == 4
                    else 'これまでの断片から次に調べる観点を選び、read_next_section を1回呼んでください。'
                )
                turn = {
                    'messages': [
                        {'role': 'tool', 'name': 'read_next_section', 'content': excerpt},
                        {'role': 'user', 'content': followup},
                    ]
                }
                if step == 4:
                    turn['extra_body'] = {'tool_choice': 'auto'}
                turns.append(turn)
            limit = 192
        else:
            offset = (variant * 31 + seed) % len(lines)
            source = ''.join(lines[(offset + k) % len(lines)] for k in range(42))
            if workload == 'coding':
                prompts = [
                    '次のコードから境界条件を一つ選び、その条件を確認する C++ テストを実装してください。',
                    '先ほどのテストに失敗経路のケースを加え、unified diff で示してください。',
                ]
                turn_limits = [768, 512]
            elif workload == 'doc':
                prompts = [
                    '次のコードを使う開発者向けに、状態の変化と呼び出し条件を日本語の仕様書にしてください。',
                    '初めて利用する人向けに、具体例と失敗時の対処を含む日本語の手順書に書き換えてください。',
                ]
                turn_limits = [768, 512]
            else:
                prompts = [
                    '次の実装で途中失敗と再試行が重なったときの状態遷移を分析し、不変条件を導いてください。',
                    '先ほどの分析を踏まえ、検証アルゴリズムと反例を説明してください。',
                ]
                turn_limits = [1024, 512]
            prompts[0] = ticket + prompts[0] + '\n' + source
            turns = [
                {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': turn_limit}
                for prompt, turn_limit in zip(prompts, turn_limits)
            ]
            limit = turn_limits[0]
        conversations.append(
            {'workload_type': workload, 'max_tokens': limit, 'extra_body': body, 'turns': turns}
        )
    return conversations


def load_agentic_dataset(path):
    raw = read_json(path)
    if isinstance(raw, dict) and 'conversations' in raw:
        items = raw['conversations']
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError('agentic データセットの JSON 形式が不正です')
    conversations = []
    for conv in items:
        if isinstance(conv, dict) and 'turns' in conv:
            turns = conv['turns']
            settings = conv
        elif isinstance(conv, dict) and 'messages' in conv:
            turns = [{'messages': [m]} for m in conv['messages'] if m.get('role') == 'user']
            settings = conv
        elif isinstance(conv, list):
            turns = conv
            settings = {}
        else:
            raise ValueError('会話データの形式が不正です')
        if not turns:
            continue
        normalized_turns = []
        for turn in turns:
            msgs = turn['messages'] if isinstance(turn, dict) and 'messages' in turn else [turn]
            entry = {'messages': msgs}
            for key in ('extra_body', 'max_tokens'):
                if isinstance(turn, dict) and key in turn:
                    entry[key] = turn[key]
            normalized_turns.append(entry)
        item = {'workload_type': settings.get('workload_type', 'default'), 'turns': normalized_turns}
        for key in ('extra_body', 'max_tokens'):
            if key in settings:
                item[key] = settings[key]
        conversations.append(item)
    if not conversations:
        raise ValueError('agentic データセットに会話が含まれていません')
    return conversations


def merge_request_options(*levels):
    merged = {}
    for level in levels:
        for key, value in level.items():
            if key == 'chat_template_kwargs':
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
    return merged


def summarize_speculative(records):
    spec_records = [r for r in records if isinstance(r.get('speculative_stats'), dict)]
    stats = [r['speculative_stats'] for r in spec_records]

    def mean_key(key):
        values = [s[key] for s in stats if s.get(key) is not None]
        return statistics.mean(values) if values else None

    def sum_key(key):
        values = [s[key] for s in stats if s.get(key) is not None]
        return sum(values) if values else None

    proposed = sum_key('proposed_drafts')
    accepted = sum_key('accepted_drafts')
    verify_rows = sum_key('verify_rows')
    spec_tokens = sum(
        r['completion_tokens']
        for r in spec_records
        if not r.get('completion_tokens_approximate') and r.get('completion_tokens') is not None
    )
    return {
        'speculative_requests': len(stats),
        'mean_final_k': mean_key('final_k'),
        'mean_k_switches': mean_key('k_switches'),
        'mean_acc_ema': mean_key('acc_ema'),
        'draft_rounds': sum_key('draft_rounds'),
        'proposed_drafts': proposed,
        'accepted_drafts': accepted,
        'verify_rows': verify_rows,
        'acceptance_rate': accepted / proposed if proposed and accepted is not None else None,
        'verify_rows_per_tok': verify_rows / spec_tokens if verify_rows is not None and spec_tokens > 0 else None,
    }


def summarize_cache(records):
    measured = [r for r in records if r.get('prompt_tokens') is not None]
    total_prompt = sum(r['prompt_tokens'] for r in measured)
    total_cached = sum(r.get('cached_tokens') or 0 for r in measured)
    return {
        'prompt_tokens': total_prompt if measured else None,
        'cached_tokens': total_cached if measured else None,
        'kv_cache_hit_rate': total_cached / total_prompt if total_prompt > 0 else None,
    }


def summarize_workload(records):
    ok = [r for r in records if r['ok']]
    known = [r for r in ok if not r['completion_tokens_approximate'] and r.get('completion_tokens') is not None]
    return {
        'requests': len(records),
        'errors': len(records) - len(ok),
        'completion_tokens': sum(r['completion_tokens'] for r in known) if known else None,
        'decode_tok_s_median': median(r['decode_tok_s'] for r in ok),
        'ttft_ms': latency(r['ttft_ms'] for r in ok),
        **summarize_cache(ok),
        **summarize_speculative(ok),
    }


def summarize_level(mode, concurrency, records, seconds, *, conversations=None, by_workload=None):
    ok = [r for r in records if r['ok']]
    total_requests = len(records)
    known = all(not r['completion_tokens_approximate'] for r in records)
    long_completed = sum(
        not r['completion_tokens_approximate'] and r['completion_tokens'] >= 1024 for r in ok
    )
    total_completion = sum(r['completion_tokens'] for r in records) if known else None
    aggregate = total_completion / seconds if known and seconds > 0 else None
    result = {
        'mode': mode,
        'concurrency': concurrency,
        'requests': total_requests,
        'window_s': seconds,
        'results': records,
        'errors': total_requests - len(ok),
        'completion_rate': len(ok) / total_requests if total_requests else 0.0,
        'long_completed': long_completed,
        'completion_tokens': total_completion,
        'aggregate_tok_s': aggregate,
        'long_window_tok_s': aggregate if total_requests and long_completed == total_requests else None,
        'decode_tok_s_median': median(r['decode_tok_s'] for r in ok),
        'ttft_ms': latency(r['ttft_ms'] for r in ok),
        'e2e_ms': latency(r['e2e_ms'] for r in ok),
        'itl_ms': latency(g for r in ok for g in r.get('itl_ms', [])),
        'update_gap_ms': latency(g for r in ok for g in r.get('update_gap_ms', [])),
        'tokens_per_update': median(r['tokens_per_update'] for r in ok),
        **summarize_cache(ok),
        **summarize_speculative(ok),
    }
    if conversations is not None:
        result['conversations'] = conversations
    if by_workload is not None:
        result['by_workload'] = by_workload
    return result


def track_d(args, preview):
    modes = ['agentic', 'prefill-mix'] if args.mode == 'both' else [args.mode]
    if args.dataset and 'agentic' not in modes:
        raise ValueError('--dataset は agentic モードでのみ使用できます')
    if 'prefill-mix' in modes and args.max_tokens < 1024:
        raise ValueError('prefill-mix の出力上限は 1024 token 以上を指定してください')
    base = {}
    if 'prefill-mix' in modes:
        _, tasks = corpus()
        for name, tier in [('short', 'short_2k'), ('long', 'long_16k')]:
            task = next(
                t
                for t in tasks
                if t['domain'] == 'code_review' and t['language'] == 'ja' and t['length_tier'] == tier
            )
            base[name] = request_body(args.model, task, args.seed)
            base[name].update(max_tokens=args.max_tokens, temperature=args.temperature)
    dataset_conversations = load_agentic_dataset(args.dataset) if args.dataset else None
    run_id = uuid.uuid4().hex if not preview else None
    levels = []
    for mode in modes:
        for concurrency in args.concurrency:
            if mode == 'agentic':
                if dataset_conversations is not None:
                    n = args.requests or max(concurrency, len(dataset_conversations))
                    if n < concurrency:
                        raise ValueError('要求数は同時数以上を指定してください')
                    selected = [dataset_conversations[i % len(dataset_conversations)] for i in range(n)]
                else:
                    n = args.requests or concurrency * 4
                    if n < concurrency:
                        raise ValueError('要求数は同時数以上を指定してください')
                    run_tag = f'{run_id}/c{concurrency}' if run_id else f'preview/c{concurrency}'
                    selected = agentic_conversations(n, run_tag, args.seed, args.temperature)
                if preview:
                    workload_counts = {}
                    for conv in selected:
                        wt = conv.get('workload_type', 'default')
                        workload_counts[wt] = workload_counts.get(wt, 0) + 1
                    levels.append(
                        {
                            'mode': mode,
                            'concurrency': concurrency,
                            'conversations': len(selected),
                            'requests': sum(len(conv['turns']) for conv in selected),
                            'workload_counts': workload_counts,
                            'workload_types': [conv.get('workload_type', 'default') for conv in selected],
                        }
                    )
                    continue

                def run_conversation(item):
                    conv_idx, conv = item
                    messages = []
                    conv_records = []
                    workload = conv.get('workload_type', 'default')
                    for turn_idx, turn in enumerate(conv['turns']):
                        calls = messages[-1].get('tool_calls', []) if messages else []
                        offset = time.monotonic() - start
                        try:
                            for message in turn['messages']:
                                if message['role'] == 'tool' and 'tool_call_id' not in message:
                                    matches = [
                                        c
                                        for c in calls
                                        if not message.get('name')
                                        or c.get('function', {}).get('name') == message['name']
                                    ]
                                    if len(matches) != 1:
                                        raise ValueError('ツール結果に対応する関数呼び出しを一つ特定できません')
                                    message = {**message, 'tool_call_id': matches[0]['id']}
                                messages.append(message)
                            body = {
                                'model': args.model,
                                'messages': list(messages),
                                'temperature': args.temperature,
                                'seed': args.seed + conv_idx * 100 + turn_idx,
                                'stream': True,
                                'stream_options': {'include_usage': True},
                                'ignore_eos': False,
                            }
                            max_tokens = args.max_tokens
                            for level in (conv, turn):
                                extra = level.get('extra_body')
                                if extra:
                                    body = merge_request_options(body, extra)
                                if 'max_tokens' in level:
                                    max_tokens = level['max_tokens']
                            body['max_tokens'] = max_tokens
                            record = stream(args.endpoint, body)
                        except (OSError, ValueError, TypeError, KeyError) as exc:
                            conv_records.append(
                                {
                                    'conversation': conv_idx,
                                    'turn': turn_idx,
                                    'workload_type': workload,
                                    'start_offset_s': offset,
                                    'ok': False,
                                    'error': str(exc),
                                    'completion_tokens': 0,
                                    'completion_tokens_approximate': True,
                                    'prompt_tokens': None,
                                    'cached_tokens': None,
                                    'ttft_ms': None,
                                    'e2e_ms': None,
                                    'decode_tok_s': None,
                                    'tokens_per_update': None,
                                    'speculative_stats': None,
                                }
                            )
                            break
                        assistant = record.pop('assistant_message')
                        record.pop('output_text')
                        record.pop('answer_text')
                        gaps = record.pop('update_gaps_ms')
                        gap_key = 'itl_ms' if record['tokens_per_update'] == 1 else 'update_gap_ms'
                        conv_records.append(
                            {
                                'conversation': conv_idx,
                                'turn': turn_idx,
                                'workload_type': workload,
                                'start_offset_s': offset,
                                **record,
                                gap_key: gaps,
                            }
                        )
                        if not record['ok']:
                            break
                        messages.append(assistant)
                    return conv_records

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    start = time.monotonic()
                    sessions = list(pool.map(run_conversation, enumerate(selected)))
                    seconds = time.monotonic() - start
                records = [r for session in sessions for r in session]
                by_workload = {
                    wt: summarize_workload([r for r in records if r['workload_type'] == wt])
                    for wt in sorted({r['workload_type'] for r in records})
                }
                levels.append(
                    summarize_level(
                        mode,
                        concurrency,
                        records,
                        seconds,
                        conversations=len(selected),
                        by_workload=by_workload,
                    )
                )
            else:
                n = args.requests or concurrency * 4
                if n < concurrency or n < 2:
                    raise ValueError('要求数は同時数以上、prefill-mix では2件以上が必要です')
                interval = max(2, concurrency)
                kinds = [
                    'long' if i % interval == max(1, concurrency - 1) else 'short'
                    for i in range(n)
                ]
                if preview:
                    levels.append({'mode': mode, 'concurrency': concurrency, 'requests': n, 'kinds': kinds})
                    continue
                first_update = threading.Event()

                def run(i):
                    kind = kinds[i]
                    body = base[kind].copy()
                    msg = base[kind]['messages'][0].copy()
                    msg['content'] = f'{run_id}/{mode}/{concurrency}/{i}\n' + msg['content']
                    body['messages'] = [msg]
                    if kind == 'long':
                        first_update.wait()
                    offset = time.monotonic() - start
                    try:
                        record = stream(args.endpoint, body, first_update.set)
                    finally:
                        if kind == 'short':
                            first_update.set()
                    gaps = record.pop('update_gaps_ms')
                    gap_key = 'itl_ms' if record['tokens_per_update'] == 1 else 'update_gap_ms'
                    record.pop('output_text')
                    record.pop('answer_text')
                    record.pop('assistant_message')
                    return {'request': i, 'kind': kind, 'start_offset_s': offset, **record, gap_key: gaps}

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    start = time.monotonic()
                    records = list(pool.map(run, range(n)))
                    seconds = time.monotonic() - start
                levels.append(summarize_level(mode, concurrency, records, seconds))
    if preview:
        return levels
    sampling = (
        {k: v for k, v in base['short'].items() if k != 'messages'}
        if 'prefill-mix' in modes
        else {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'seed': args.seed,
            'dataset': str(args.dataset) if args.dataset else 'builtin',
        }
    )
    return {
        'run_id': run_id,
        'sampling': sampling,
        'levels': levels,
    }


def comparable_b(left, right):
    if left.get('mode') != 'full' or right.get('mode') != 'full':
        return False
    evidence = left.get('conditions') or {}
    if not all(k in evidence and evidence[k] not in (None, '', 'unknown', '未確認') for k in CONDITIONS):
        return False
    if evidence != right.get('conditions'):
        return False
    a, b = left['raw'], right['raw']
    for k in ('betterbench_version', 'corpus_version'):
        if a.get(k) is None or a[k] != b.get(k):
            return False
    for k in ('model', 'corpus_hash'):
        if a.get('env', {}).get(k) is None or a['env'][k] != b.get('env', {}).get(k):
            return False
    keys = ('runs_per_category', 'warmup', 'temperature', 'top_p', 'top_k', 'greedy', 'seed', 'unique_nonce')
    return all(k in a['config'] and k in b['config'] and a['config'][k] == b['config'][k] for k in keys)


def fmt(value):
    if value is None:
        return '未測定'
    return f'{value:.2f}' if isinstance(value, float) else str(value)


def quality_results(report, path):
    if not path:
        return None, None
    records = report.get('results', [])
    outputs = {(r['task_id'], r['output_sha256']) for r in records if r['ok']}
    verdicts = {}
    for review in read_json(path):
        key = (review['task_id'], review['output_sha256'])
        if key not in outputs or key in verdicts or type(review['passed']) is not bool:
            raise ValueError('評価対象の課題・出力 SHA-256 または合否が不正です')
        verdicts[key] = review['passed']
    judged = [
        verdicts[(r['task_id'], r['output_sha256'])]
        for r in records
        if r['ok'] and (r['task_id'], r['output_sha256']) in verdicts
    ]
    return len(judged), sum(judged)


def render(args, preview):
    if preview:
        return {'sections': ['C', 'B', 'A', 'D'], 'inputs': args.inputs}
    reports = {}
    for path in args.inputs:
        data = read_json(path)
        track = data['track']
        if track in reports or track not in 'ABCD' or len(track) != 1:
            raise ValueError('トラックの重複または不明な入力: ' + track)
        reports[track] = data
    c, b, a, d = (reports.get(track, {}) for track in 'CBAD')
    other = read_json(args.compare_b) if args.compare_b else {}
    compare = comparable_b(b, other)
    lines = []

    def row(*values):
        lines.append('| ' + ' | '.join(fmt(v) for v in values) + ' |')

    def section(title, *columns):
        lines.extend(['', '## ' + title, ''])
        row(*columns)
        row(*['---'] * len(columns))

    section(
        '① 実タスク代表値（Track C）',
        'real-v1',
        '全件 tok/s',
        'non-loop tok/s',
        '1024 token 以上の完了数',
        '失敗',
        'ループ',
        '上限到達',
    )
    row(
        '1024 token 以上',
        c.get('long_output', {}).get('macro_decode_tok_s'),
        c.get('long_non_loop', {}).get('macro_decode_tok_s'),
        c.get('long_completed'),
        c.get('failures'),
        c.get('loop_count'),
        c.get('truncated'),
    )
    row(
        '出力長を問わない集計',
        c.get('all', {}).get('macro_decode_tok_s'),
        c.get('non_loop', {}).get('macro_decode_tok_s'),
        '—',
        '—',
        '—',
        '—',
    )
    reviewed, passed = quality_results(c, args.reviews)
    lines.extend(['', f'固定 rubric による判定済み {fmt(reviewed)} 件、合格 {fmt(passed)} 件。'])
    section(
        '② 外部比較（Track B）',
        'カテゴリ（' + fmt(b.get('mode')) + '）',
        '中央値 tok/s',
        'IQR',
        'min–max',
        'TTFT 中央値 ms',
        '失敗率 / 打切り率',
        'ms / 受理 / 確定（round 当たり）',
        '同条件の比較先 tok/s',
        '速度比',
    )
    for category in CATEGORIES:
        cat = b.get('categories', {}).get(category, {})
        speed = cat.get('decode_tok_s', {})
        ref = other['categories'][category]['decode_tok_s']['median'] if compare else None
        value = speed.get('median')
        row(
            category,
            value,
            speed.get('iqr'),
            '–'.join(fmt(speed.get(k)) for k in ('min', 'max')),
            cat.get('ttft_ms', {}).get('median'),
            ' / '.join(fmt(cat.get(k)) for k in ('failure_rate', 'truncation_rate')),
            ' / '.join(fmt(cat.get(k)) for k in ROUND_FIELDS),
            ref,
            value / ref if ref and value is not None else None,
        )
    medians = [cat['decode_tok_s']['median'] for cat in b.get('categories', {}).values()]
    bounds = [min(medians), max(medians)] if len(medians) == 8 and None not in medians else [None, None]
    lines.extend(['', '8カテゴリの中央値の範囲（tok/s） ' + '–'.join(map(fmt, bounds))])
    section(
        '③ 基礎性能（Track A）',
        '条件',
        'GPU tok/s',
        'wall tok/s',
        'decode tok/s',
        'ms/round',
        '受理 draft/round',
        '確定 token/round',
    )
    groups = {}
    for record in a.get('results', []):
        key = (record['case'], record['variant'])
        groups.setdefault(key, [])
        if record['ok']:
            groups[key].append(record)
    for kind in ('pp', 'tg', 'batch', 'spec'):
        if not any(case.split('/')[0] == kind for case, _ in groups):
            groups[(kind, '')] = []
    series = {}
    for (case, variant), records in groups.items():
        kind, _, shape = case.partition('/')
        series.setdefault((kind, variant), []).append((shape, records))
    for (kind, variant), measurements in series.items():
        label = ' '.join((kind, variant)).strip()
        if kind == 'batch':
            label += '（逐次要求）'
        shapes = [shape for shape, _ in measurements if shape]
        if shapes:
            label += ' ' + ' · '.join(shapes)
        columns = []
        for key in ('gpu_tok_s', 'wall_tok_s', 'decode_tok_s', *ROUND_FIELDS):
            values = [median(r.get(key) for r in records) for _, records in measurements]
            columns.append(' · '.join(map(fmt, values)) if any(v is not None for v in values) else None)
        row(label, *columns)
    section(
        '④ サービング（Track D）',
        '負荷 / 同時数',
        'TTFT p50/p95 ms',
        'E2E p50/p95 ms',
        'ITL p50/p95 ms',
        '更新間隔 p50/p95 ms',
        'token/更新',
        '集約 tok/s（全件 / 全件1024以上）',
        '要求別 decode 中央値',
        'キャッシュ一致率',
        '平均 K / 検証行/tok',
        '完了率 / エラー',
        '1024 token 以上の完了数',
    )
    for level in d.get('levels', [{}]):
        row(
            fmt(level.get('mode')) + ' / ' + fmt(level.get('concurrency')),
            *(
                ' / '.join(fmt(level.get(k, {}).get(p)) for p in ('p50', 'p95'))
                for k in ('ttft_ms', 'e2e_ms', 'itl_ms', 'update_gap_ms')
            ),
            level.get('tokens_per_update'),
            ' / '.join(fmt(level.get(k)) for k in ('aggregate_tok_s', 'long_window_tok_s')),
            level.get('decode_tok_s_median'),
            level.get('kv_cache_hit_rate'),
            ' / '.join(fmt(level.get(k)) for k in ('mean_final_k', 'verify_rows_per_tok')),
            ' / '.join(fmt(level.get(k)) for k in ('completion_rate', 'errors')),
            level.get('long_completed'),
        )
    workload_rows = [
        (level.get('concurrency'), wt, metrics)
        for level in d.get('levels', [])
        for wt, metrics in sorted((level.get('by_workload') or {}).items())
    ]
    if workload_rows:
        lines.extend(['', '### Track D ワークロード別（agentic）', ''])
        row(
            '同時数 / 種別',
            '件数',
            '生成 tok',
            'TTFT p50/p95 ms',
            '要求別 decode 中央値',
            'キャッシュ一致率',
            '平均 K',
            '切替回数',
            'acc_ema',
            '受理率',
            '検証行/tok',
        )
        row(*['---'] * 11)
        for conc, wt, metrics in workload_rows:
            row(
                f'{fmt(conc)} / {wt}',
                metrics.get('requests'),
                metrics.get('completion_tokens'),
                ' / '.join(fmt(metrics.get('ttft_ms', {}).get(p)) for p in ('p50', 'p95')),
                metrics.get('decode_tok_s_median'),
                metrics.get('kv_cache_hit_rate'),
                metrics.get('mean_final_k'),
                metrics.get('mean_k_switches'),
                metrics.get('mean_acc_ema'),
                metrics.get('acceptance_rate'),
                metrics.get('verify_rows_per_tok'),
            )
    return '\n'.join(lines).strip() + '\n'


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('正の整数を指定してください')
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('0 以上の整数を指定してください')
    return n


def numbers(value, minimum=1):
    try:
        values = list(map(int, value.split(',')))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('整数をカンマで区切ってください') from exc
    if min(values) < minimum or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('範囲外の値または重複があります')
    return values


def long_budget(value):
    if int(value) < 1024:
        raise argparse.ArgumentTypeError('出力上限は 1024 token 以上を指定してください')
    return int(value)


def parser():
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument('--lock-file', help='測定全体で排他的に確保するファイル')
    common.add_argument(
        '--dry-run', '--list', dest='preview', action='store_true', help='実行計画を JSON で表示'
    )
    common.add_argument('--output', help='出力先（既定は標準出力）')
    p = argparse.ArgumentParser(parents=[common], description='統一ベンチマーク')
    sub = p.add_subparsers(dest='track', required=True)
    a, b, c, d = [sub.add_parser('track-' + name, parents=[common]) for name in 'abcd']
    for cmd in (a, b, c, d):
        cmd.add_argument('--model', default='default')
    for cmd in (b, c, d):
        cmd.add_argument('--endpoint', default=ENDPOINT)
    for cmd in (a, c, d):
        cmd.add_argument('--seed', type=int, default=42)
    for cmd, repeats, warmup in [(a, 5, 1), (c, 1, 0)]:
        cmd.add_argument('--repeats', type=positive, default=repeats)
        cmd.add_argument('--warmup', type=nonnegative, default=warmup)
    a.add_argument('--shelf-bench', default=str(ROOT / 'build/shelf-bench'))
    a.add_argument('--model-dir')
    a.add_argument('--tokens-file', help='pp と tg の PSKLDTOK 入力')
    a.add_argument('--device', default='0')
    a.add_argument('--tensor-parallel', type=int, choices=(1, 2, 4), default=1)
    a.add_argument('--pp-lengths', type=numbers, default=[128, 512, 2048, 8192])
    a.add_argument('--contexts', type=lambda v: numbers(v, 0), default=[0, 2048, 8192])
    a.add_argument('--batch-sizes', type=numbers, default=[1, 4, 8, 16])
    a.add_argument('--tg-tokens', type=positive, default=256)
    a.add_argument('--spec-max-tokens', type=long_budget, default=1024)
    for prefix in ('', 'baseline-'):
        g = a.add_mutually_exclusive_group()
        g.add_argument('--' + prefix + 'spec-endpoint')
        g.add_argument(
            '--' + prefix + 'spec-cmd',
            help='stdin に chat 要求 JSON、stdout に output_text, completion_tokens, decode_tok_s を持つ JSON',
        )
    a.add_argument('--ab-compare', action='store_true')
    a.add_argument('--baseline-bench')
    b.add_argument('--quick', action='store_true')
    b.add_argument('--betterbench-python', default=sys.executable)
    b.add_argument('--import-json')
    b.add_argument('--server-log', action='append', default=[], metavar='[CATEGORY=]PATH')
    b.add_argument('--conditions', help='同条件比較用の構成 JSON。必要なキーは --list で表示')
    c.add_argument('--task', action='append', metavar='ID')
    c.add_argument('--reasoning-effort', choices=('low', 'medium', 'xhigh'), help='思考を使う課題に渡す指示')
    c.add_argument('--no-answer-reserve', action='store_true', help='Shelf の回答用トークン予約を無効にする')
    d.add_argument('--concurrency', type=numbers, default=[1, 4, 8, 16])
    d.add_argument('--requests', type=positive, help='同時数ごとの測定件数（既定は agentic 既定では同時数の4倍、--dataset 指定時は会話数）')
    d.add_argument('--mode', choices=['agentic', 'prefill-mix', 'both'], default='agentic')
    d.add_argument('--dataset', help='agentic 用の凍結会話 JSON パス')
    d.add_argument('--max-tokens', type=positive, default=2048)
    d.add_argument('--temperature', type=float, default=0)
    md = sub.add_parser('render', parents=[common])
    md.add_argument('inputs', nargs='*', help='トラック別 JSON')
    md.add_argument('--reviews', help='real-v1 出力の評価 JSON 配列（task_id, output_sha256, passed）')
    md.add_argument('--compare-b', help='同条件で取得した比較先の Track B JSON')
    return p


def main():
    p = parser()
    args = p.parse_args(namespace=argparse.Namespace(lock_file=None, preview=False, output=None))
    run = {'track-a': track_a, 'track-b': track_b, 'track-c': track_c, 'track-d': track_d, 'render': render}[
        args.track
    ]
    try:
        with contextlib.ExitStack() as stack:
            if args.lock_file and not args.preview:
                handle = stack.enter_context(open(args.lock_file, 'a'))
                fcntl.flock(handle, fcntl.LOCK_EX)
            meta = metadata(args) if not args.preview and args.track != 'render' else None
            data = run(args, args.preview)
            if meta:
                data = {'schema_version': 1, 'track': args.track[-1].upper(), 'metadata': meta, **data}
            if not isinstance(data, str):
                data = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            if args.output:
                pathlib.Path(args.output).write_text(data)
            else:
                print(data, end='')
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
