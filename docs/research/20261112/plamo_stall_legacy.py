#!/usr/bin/env python3
"""Summarize pre-existing STEPWALL logs, including prior occurrence of each shape."""
import argparse
import collections
import json
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument('logs', nargs='+')
a = p.parse_args()
results = {}
for path in a.logs:
    seen_rows = collections.Counter()
    seen_shapes = collections.Counter()
    steps, spikes = [], []
    with open(path) as f:
        for line in f:
            if 'STEPWALL ' not in line:
                continue
            values = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}
            if not {'rows', 'tokens', 'decode', 'total'} <= values.keys():
                continue
            rows, tokens, decode = (int(values[k]) for k in ('rows', 'tokens', 'decode'))
            shape = (rows, tokens, decode)
            if values['total'] >= 150:
                spikes.append({'step_index': len(steps), 'requests': rows, 'token_rows': tokens,
                               'decode_only': bool(decode), 'total_ms': values['total'],
                               'prior_token_rows': seen_rows[tokens], 'prior_shape': seen_shapes[shape]})
            seen_rows[tokens] += 1
            seen_shapes[shape] += 1
            steps.append(values)
    steady = steps[1:]  # exclude known synthetic 256-token startup, not the workload warmup
    decode = [s['total'] for s in steady if s['decode']]
    mixed = [s['total'] for s in steady if not s['decode']]
    median = statistics.median(decode)
    results[path] = {'steps': len(steady), 'step_wall_ms': sum(s['total'] for s in steady),
                     'decode_steps': len(decode), 'decode_total_ms': sum(decode),
                     'decode_median_ms': median, 'mixed_or_prefill_steps': len(mixed),
                     'mixed_or_prefill_total_ms': sum(mixed),
                     'mixed_or_prefill_gross_excess_vs_decode_median_ms': sum(x - median for x in mixed),
                     'spikes_including_startup': spikes}
print(json.dumps(results, ensure_ascii=False, indent=2))
