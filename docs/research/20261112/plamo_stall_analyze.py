#!/usr/bin/env python3
"""Summarize STEPDIAG logs and monotonic sysfs/host samples without discarding startup."""
import argparse
import collections
import json


def num(value):
    try:
        return float(value) if '.' in value else int(value)
    except ValueError:
        return value


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--telemetry', required=True)
    p.add_argument('--spike-ms', type=float, default=150)
    p.add_argument('--warmup-s', type=float, default=30)
    a = p.parse_args()
    steps = []
    window = None
    with open(a.log) as f:
        for line in f:
            marker_window = line.find('MEASURE start=')
            if marker_window >= 0:
                try:
                    window = {k: float(v) for k, v in (item.split('=', 1) for item in line[marker_window:].split()[1:])}
                except ValueError:
                    pass
            marker = line.find('STEPDIAG ')
            if marker < 0:
                continue
            try:
                steps.append({k: num(v) for k, v in
                              (item.split('=', 1) for item in line[marker:].split()[1:])})
            except ValueError:
                continue
    with open(a.telemetry) as f:
        samples = [json.loads(line) for line in f]
    if not steps:
        raise SystemExit('No STEPDIAG data; verify instrumented compute and environment')
    seen_rows = collections.Counter()
    seen_pairs = collections.Counter()
    first_t = steps[0]['t']
    for s in steps:
        shape = (s['rows'], s['tokens'], s['prefill'], s['decode'])
        s['prior_token_rows'] = seen_rows[s['tokens']]
        s['prior_shape'] = seen_pairs[shape]
        s['first_token_rows'] = s['prior_token_rows'] == 0
        s['first_shape'] = s['prior_shape'] == 0
        seen_rows[s['tokens']] += 1
        seen_pairs[shape] += 1
        s['kind'] = 'mixed' if s['prefill'] and s['decode'] else 'prefill' if s['prefill'] else 'decode'
        s['age_s'] = s['t'] - first_t
    steady = [s for s in steps if window['start'] <= s['t'] < window['end']] if window else [s for s in steps if s['age_s'] >= a.warmup_s]
    groups = collections.defaultdict(list)
    for s in steady:
        groups[s['kind']].append(s)
    summary = {}
    for name, group in groups.items():
        times = [s['total'] for s in group]
        summary[name] = {'n': len(group), 'median_ms': quantile(times, .5),
                         'p99_ms': quantile(times, .99), 'max_ms': max(times),
                         'spikes': sum(x >= a.spike_ms for x in times),
                         'total_ms': sum(times),
                         'prefill_rows': sum(s['tokens'] - s['decode'] for s in group),
                         'decode_rows': sum(s['decode'] for s in group)}
    correlations = {}
    classifiers = {
        'kind': lambda s: s['kind'],
        'first_token_rows': lambda s: str(s['first_token_rows']),
        'first_shape': lambda s: str(s['first_shape']),
        'new_admission': lambda s: str(s['admit'] > 0),
        'page_reserved_in_exec': lambda s: str(s['free_postexec'] < s['free_preexec']),
        'page_freed_after_exec': lambda s: str(s['free_after'] > s['free_postexec']),
        'token_rows_bucket_32': lambda s: str((s['tokens'] // 32) * 32),
    }
    for label, classify in classifiers.items():
        counts = collections.defaultdict(lambda: {'steps': 0, 'spikes': 0, 'max_ms': 0})
        for s in steady:
            entry = counts[classify(s)]
            entry['steps'] += 1
            entry['spikes'] += s['total'] >= a.spike_ms
            entry['max_ms'] = max(entry['max_ms'], s['total'])
        correlations[label] = dict(counts)
    spikes = []
    outside_spikes = []
    steady_ids = {id(s) for s in steady}
    for s in steps:
        if s['total'] < a.spike_ms:
            continue
        relevant = [x for x in samples if x['t'] <= s['t'] + s['total'] / 1000 and x.get('gpu_end', x['end']) >= s['t']]
        context = [x for x in samples if s['t'] - 1 <= x['t'] <= s['t'] + s['total'] / 1000 + 1 and 'host' in x]
        target = spikes if id(s) in steady_ids else outside_spikes
        target.append({'step': s, 'gpu_samples': relevant, 'host_samples': context})
    in_window = [x for x in samples if window is None or window['start'] <= x['t'] < window['end']]
    gpu_summary = {}
    for key in ('gpu_busy_percent', 'sclk_hz', 'mclk_hz', 'power_uw', 'edge_millic', 'junction_millic', 'memory_millic'):
        vals = [int(x['gpu'][key]) for x in in_window if x['gpu'].get(key, '').isdigit()]
        gpu_summary[key] = {'n': len(vals), 'min': min(vals) if vals else None,
                            'median': quantile(vals, .5), 'max': max(vals) if vals else None}
    host_samples = [x for x in in_window if 'host' in x]
    host_delta = {}
    if len(host_samples) >= 2:
        before, after = host_samples[0]['host']['vm'], host_samples[-1]['host']['vm']
        host_delta = {k: after[k] - v for k, v in before.items() if k in after}
    print(json.dumps({'measurement_window': window, 'fallback_warmup_s_from_first_step': a.warmup_s, 'all_steps': len(steps),
                      'steady_steps': len(steady), 'summary': summary, 'correlations': correlations,
                      'gpu_summary': gpu_summary, 'host_vm_delta': host_delta,
                      'first_steps': steps[:5], 'spikes': spikes, 'outside_window_spikes': outside_spikes,
                      'sampling_interval_ms': {'p50': quantile([(s['end'] - s['t']) * 1000 for s in samples], .5),
                                               'p99': quantile([(s['end'] - s['t']) * 1000 for s in samples], .99),
                                               'gap_p99': quantile([(b['t'] - a['t']) * 1000 for a, b in zip(samples, samples[1:])], .99)},
                      'measurement_limitations': 'first shape is rows/tokens/prefill/decode tuple only; age is from first recorded step'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
