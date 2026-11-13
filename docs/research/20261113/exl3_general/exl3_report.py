import hashlib
import json
import struct
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-gfx1201/exl3-general'
OUT = Path(__file__).with_name('exl3_general.json')
FILES = ['micro.json', 'micro-extra.json', 'tile.json', 'e2e.json',
         'group-e2e-code.json', 'group-e2e-extra.json', 'batch.json',
         'batch-extra.json', 'batch-initial-failure.json', 'plamo-dump.json',
         'final-e2e.json', 'final-batch.json', 'final-prompts.json', 'plamo-dump-final.json',
         'plamo-dump-full.json', 'tile-scoped.json', 'tile-boundary.json', 'tile-four-boundary.json',
         'tile-small-boundary.json', 'tile-exact.json', 'large-batch.json', 'cumulative.json',
         'plamo-byte.json', 'tile-exact-extra.json', 'rotate-high.json', 'rotate-perop.json',
         'mul24-initial-vs-full.json', 'mul24-prompts.json', 'tile-byte.json',
         'rotate-boundary.json', 'rotate-late-boundary.json', 'rotate-boundary-n17408.json',
         'tile-byte-extra.json', 'rotate-after256.json', 'rotate-near-retry.json', 'rotate-perop-retry.json', 'rotate-m64-recheck.json',
         'rotate-graph-retry.json', 'rotate-255-retry.json', 'plamo-dump-optimized.json',
         'optimized-vs-group.json', 'mul24-initial-vs-optimized.json', 'large-batch-optimized.json',
         'large-batch-repeat.json', 'optimized-c1-recheck.json', 'plamo-speed.json', 'gemm-unchanged.json',
         'initial-vs-optimized.json', 'rotate-middle-high.json', 'rotate-head-high.json',
         'rotate-plane-sweep.json', 'rotate-alias.json', 'rotate-alias-boundaries.json',
         'rotate-alias-retry.json', 'rotate-six-alias.json', 'rotate-four-six-near.json']


def exact_token_batches(before, after):
    if len(before) != len(after):
        return False
    return all(struct.pack('<' + 'i' * len(a), *a) == struct.pack('<' + 'i' * len(b), *b)
               for a, b in zip(before, after))


def render_records(data):
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    lines = ['{']
    items = list(data.items())
    for index, (key, value) in enumerate(items):
        suffix = ',' if index + 1 < len(items) else ''
        label = '  ' + encode(key) + ': '
        if isinstance(value, list):
            lines.append(label + '[')
            lines.extend('    ' + encode(item) + (',' if i + 1 < len(value) else '')
                         for i, item in enumerate(value))
            lines.append('  ]' + suffix)
        elif key == 'runtime_stage_timing':
            lines.append(label + '{')
            groups = list(value.items())
            for group_index, (name, records) in enumerate(groups):
                lines.append('    ' + encode(name) + ': [')
                lines.extend('      ' + encode(item) + (',' if i + 1 < len(records) else '')
                             for i, item in enumerate(records))
                lines.append('    ]' + (',' if group_index + 1 < len(groups) else ''))
            lines.append('  }' + suffix)
        else:
            lines.append(label + encode(value) + suffix)
    lines.append('}')
    return '\n'.join(lines) + '\n'


def main():
    data = {'revision': '2c1000a', 'gpu_uuid': 'GPU-a8e9ddefa2d60f55',
            'quantization': 'EXL3-INT8 mul24', 'optimized_revision': '8648b2c', 'missing': []}
    data['binaries'] = {}
    for name in ['shelf-compute.base', 'shelf-compute.final', 'shelf-compute.optimized', 'shelf-bench.optimized']:
        path = WORK / name
        if path.exists():
            data['binaries'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            data['missing'].append(name)
    data['timing_hook_binary'] = {}
    for name in ['shelf-compute.trace', 'shelf-compute.optimized']:
        path = WORK / name
        if path.exists():
            data['timing_hook_binary'][name] = b'EXL3_STAGE rows=' in path.read_bytes()
        else:
            data['missing'].append(name)
    for name in FILES:
        path = WORK / name
        if not path.exists():
            data['missing'].append(name)
        else:
            data[name.removesuffix('.json')] = json.loads(path.read_text())
    path = WORK / 'group.jsonl'
    if path.exists():
        data['group_micro'] = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        data['missing'].append(path.name)
    data['runtime_stage_timing'] = {}
    trace_names = ['trace-c1-0.log', 'trace-c1-1.log', 'trace-c4-0.log', 'trace-c4-1.log',
                   'trace-c1-fixed.log', 'trace-c2-0-extra.log', 'trace-c3-0-extra.log',
                   'trace-c5-0-extra.log', 'trace-c6-0-extra.log', 'trace-c7-0-extra.log', 'trace-c2-1-extra.log']
    for name in trace_names:
        path = WORK / name
        if not path.exists():
            data['missing'].append(name)
            continue
        lines = [line for line in path.read_text().splitlines() if line.startswith('EXL3_STAGE ')]
        data['runtime_stage_timing'][name] = [dict(re.findall(r'(\w+)=([\d.]+)', line)) for line in lines]
    for name in ['roofline.log', 'ctest.log', 'ctest-final.log', 'server-concurrency.log', 'server-final.log', 'rotate-kernel.log', 'tiled-kernel.log',
                 'full-v1-kernel.log', 'special-kernel.log', 'mul24-draft-smoke.log',
                 'ctest-v2.log', 'server-plamo-v2.log', 'ctest-optimized.log',
                 'server-optimized.log', 'server-plamo-optimized.log', 'drafter-quant-test.log']:
        path = WORK / name
        if path.exists():
            data[name.removesuffix('.log')] = path.read_text()
        else:
            data['missing'].append(name)
    expected = {'micro': 420, 'micro-extra': 175, 'tile': 280, 'e2e': 12,
                'group-e2e-code': 12, 'group-e2e-extra': 12, 'batch': 16,
                'batch-extra': 16, 'plamo-dump': 4, 'final-e2e': 12, 'final-batch': 24,
                'group_micro': 12, 'plamo-dump-final': 4, 'final-prompts': 4,
                'plamo-dump-full': 4, 'tile-scoped': 420, 'tile-boundary': 240,
                'tile-four-boundary': 60, 'tile-small-boundary': 140, 'tile-exact': 42,
                'large-batch': 2, 'cumulative': 66, 'plamo-byte': 6, 'tile-exact-extra': 28,
                'rotate-high': 56, 'rotate-perop': 24, 'mul24-initial-vs-full': 18, 'mul24-prompts': 4,
                'tile-byte': 42, 'rotate-boundary': 48, 'rotate-late-boundary': 12,
                'rotate-boundary-n17408': 12, 'tile-byte-extra': 28, 'rotate-after256': 16,
                'rotate-head-high': 16, 'rotate-plane-sweep': 140, 'rotate-alias': 24,
                'rotate-alias-boundaries': 70, 'rotate-alias-retry': 13, 'rotate-six-alias': 24,
                'rotate-four-six-near': 88, 'rotate-m64-recheck': 6, 'rotate-graph-retry': 3,
                'rotate-255-retry': 2, 'plamo-dump-optimized': 4, 'optimized-vs-group': 66,
                'mul24-initial-vs-optimized': 18, 'large-batch-optimized': 2,
                'large-batch-repeat': 6, 'optimized-c1-recheck': 10, 'plamo-speed': 8,
                'gemm-unchanged': 200, 'initial-vs-optimized': 36, 'rotate-middle-high': 24}
    data['incomplete'] = {name: {'expected': count, 'actual': len(data.get(name, []))}
                          for name, count in expected.items() if len(data.get(name, [])) != count}
    for name in trace_names:
        if not data['runtime_stage_timing'].get(name):
            data['incomplete'][name] = 'stage timing が未取得'
    rows_seen = {int(item['rows']) for items in data['runtime_stage_timing'].values() for item in items}
    if not set(range(1, 9)).issubset(rows_seen) or 16 not in rows_seen:
        data['incomplete']['trace_rows'] = {'required': list(range(1, 9)) + [16],
                                            'observed': sorted(rows_seen)}
    for name in expected:
        if any(record.get('exit', 0) != 0 for record in data.get(name, [])):
            data['incomplete'][name] = 'nonzero exit'
    for name in ['ctest', 'ctest-final', 'ctest-v2', 'ctest-optimized']:
        if '100% tests passed' not in data.get(name, ''):
            data['incomplete'][name] = '結果未取得または失敗'
    for name in ['server-concurrency', 'server-final', 'server-plamo-v2', 'server-optimized', 'server-plamo-optimized']:
        if 'failed=0' not in data.get(name, '') or re.search(r'failed=[1-9]', data.get(name, '')):
            data['incomplete'][name] = '結果未取得または失敗'
    drafter_test = data.get('drafter-quant-test', '')
    if 'SKIP' in drafter_test or any(f'rows={m} ' not in drafter_test for m in [1, 8, 16]):
        data['incomplete']['drafter-quant-test'] = '実 checkpoint の結果が不足'
    if data['timing_hook_binary'] != {'shelf-compute.trace': True, 'shelf-compute.optimized': False}:
        data['incomplete']['timing_hook_binary'] = '計測 hook の有無が不正'
    checks = []
    for a in data.get('final-e2e', []):
        if a['variant'] != 'base':
            continue
        candidates = [b for b in data['final-e2e'] if b['variant'] == 'candidate' and
                      (b['spec'], b['repeat'], b['prompt']) == (a['spec'], a['repeat'], a['prompt'])]
        if not candidates:
            continue
        before = a.get('fields', {}).get('GENERATED_IDS', '').encode()
        after = candidates[0].get('fields', {}).get('GENERATED_IDS', '').encode()
        checks.append({'kind': 'c1', 'spec': a['spec'], 'repeat': a['repeat'],
                       'prompt': a['prompt'], 'equal': bool(before) and before == after,
                       'before_sha256': hashlib.sha256(before).hexdigest(),
                       'after_sha256': hashlib.sha256(after).hexdigest()})
    for a in data.get('final-prompts', []):
        if a['variant'] != 'candidate' or a['prompt'] == 'code':
            continue
        bases = [b for b in data.get('e2e', []) if b['variant'] == 'base' and
                 (b['spec'], b['prompt']) == (a['spec'], a['prompt'])]
        if not bases:
            continue
        before = bases[0]['fields'].get('GENERATED_IDS', '').encode()
        after = a['fields'].get('GENERATED_IDS', '').encode()
        checks.append({'kind': 'c1_additional_prompt', 'spec': a['spec'], 'prompt': a['prompt'],
                       'equal': bool(before) and before == after,
                       'before_sha256': hashlib.sha256(before).hexdigest(),
                       'after_sha256': hashlib.sha256(after).hexdigest()})
    for a in data.get('final-batch', []):
        if a['variant'] != 'base':
            continue
        candidates = [b for b in data['final-batch'] if b['variant'] == 'candidate' and
                      (b['spec'], b['repeat'], b['concurrency']) ==
                      (a['spec'], a['repeat'], a['concurrency'])]
        if not candidates:
            continue
        before = a['result']['runs'][0]['generated_ids']
        after = candidates[0]['result']['runs'][0]['generated_ids']
        hashes = lambda ids: [hashlib.sha256(struct.pack('<' + 'i' * len(v), *v)).hexdigest() for v in ids]
        checks.append({'kind': 'batch', 'spec': a['spec'], 'repeat': a['repeat'],
                       'concurrency': a['concurrency'], 'equal': bool(before) and exact_token_batches(before, after),
                       'before_sha256': hashes(before), 'after_sha256': hashes(after)})
    for size in [1, 64]:
        records = [x for x in data.get('plamo-dump-final', []) if x['batch'] == size]
        if len(records) == 2:
            checks.append({'kind': 'plamo_dump', 'batch': size,
                           'equal': records[0]['sha256'] == records[1]['sha256'],
                           'before_sha256': records[0]['sha256'], 'after_sha256': records[1]['sha256']})
    for name in ['cumulative', 'optimized-vs-group', 'optimized-c1-recheck', 'initial-vs-optimized']:
        for a in data.get(name, []):
            if a['variant'] != 'group':
                continue
            matches = [b for b in data[name] if b['variant'] == 'candidate' and
                       (b['mode'], b['concurrency'], b['repeat'], b['prompt']) ==
                       (a['mode'], a['concurrency'], a['repeat'], a['prompt'])]
            if not matches:
                continue
            b = matches[0]
            if a['concurrency'] == 1:
                before = a.get('fields', {}).get('GENERATED_IDS', '').encode()
                after = b.get('fields', {}).get('GENERATED_IDS', '').encode()
                left = hashlib.sha256(before).hexdigest()
                right = hashlib.sha256(after).hexdigest()
            else:
                before = a['result']['runs'][0]['generated_ids']
                after = b['result']['runs'][0]['generated_ids']
                hashes = lambda ids: [hashlib.sha256(struct.pack('<' + 'i' * len(v), *v)).hexdigest() for v in ids]
                left, right = hashes(before), hashes(after)
            checks.append({'kind': name, 'mode': a['mode'], 'concurrency': a['concurrency'],
                           'repeat': a['repeat'], 'prompt': a['prompt'],
                           'equal': bool(before) and (before == after if a['concurrency'] == 1 else exact_token_batches(before, after)),
                           'before_sha256': left, 'after_sha256': right})
    for name in ['large-batch', 'large-batch-optimized']:
        large = data.get(name, [])
        if len(large) == 2 and all(x.get('exit') == 0 for x in large):
            before, after = (x['result']['runs'][0]['generated_ids'] for x in large)
            checks.append({'kind': name, 'concurrency': 64, 'equal': exact_token_batches(before, after)})
    for name in ['plamo-dump-full', 'plamo-dump-optimized']:
        for size in [1, 64]:
            records = [x for x in data.get(name, []) if x['batch'] == size]
            if len(records) == 2:
                checks.append({'kind': name, 'batch': size,
                               'equal': records[0]['sha256'] == records[1]['sha256'],
                               'before_sha256': records[0]['sha256'], 'after_sha256': records[1]['sha256']})
    for repeat in range(3):
        records = [x for x in data.get('large-batch-repeat', []) if x.get('repeat') == repeat]
        if len(records) == 2 and all(x.get('exit') == 0 for x in records):
            before, after = (x['result']['runs'][0]['generated_ids'] for x in records)
            checks.append({'kind': 'large-batch-repeat', 'repeat': repeat, 'equal': exact_token_batches(before, after)})
    for name in ['tile-exact', 'tile-exact-extra']:
        if any(not x.get('equal', False) for x in data.get(name, [])):
            data['incomplete'][name] = '一致しない出力がある'
    for name in ['tile-byte', 'tile-byte-extra']:
        if any(not x.get('byte_equal', False) for x in data.get(name, [])):
            data['incomplete'][name] = '一致しない出力がある'
    if any(not x.get('byte_equal', False) for x in data.get('plamo-byte', [])):
        data['incomplete']['plamo-byte'] = '一致しない dump がある'
    for name in ['mul24-initial-vs-full', 'mul24-prompts', 'mul24-initial-vs-optimized']:
        for a in data.get(name, []):
            if a['variant'] != 'group':
                continue
            matches = [b for b in data[name] if b['variant'] == 'candidate' and
                       (b['mode'], b['concurrency'], b['repeat'], b['prompt']) ==
                       (a['mode'], a['concurrency'], a['repeat'], a['prompt'])]
            if not matches:
                continue
            b = matches[0]
            if a['concurrency'] == 1:
                before = a.get('fields', {}).get('GENERATED_IDS', '').encode()
                after = b.get('fields', {}).get('GENERATED_IDS', '').encode()
            else:
                before = a['result']['runs'][0]['generated_ids']
                after = b['result']['runs'][0]['generated_ids']
            checks.append({'kind': name, 'concurrency': a['concurrency'], 'repeat': a['repeat'],
                           'prompt': a['prompt'],
                           'equal': bool(before) and (before == after if a['concurrency'] == 1 else exact_token_batches(before, after))})
    data['exact_checks'] = checks
    if any(not x['equal'] for x in checks):
        data['incomplete']['exact_checks'] = '一致しない出力がある'
    if len(checks) != 142:
        data['incomplete']['exact_checks_count'] = {'expected': 142, 'actual': len(checks)}
    OUT.write_text(render_records(data))
    print(json.dumps({'output': str(OUT), 'missing': data['missing'], 'incomplete': data['incomplete']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
