#!/usr/bin/env python3
"""Correlate kcompactd CPU tick increments with traced steps; no GPU access."""
import argparse
import json
import os
import re


def ticks(stat):
    if not stat or ') ' not in stat:
        return None
    parts = stat.split(') ', 1)[1].split()
    try:
        return int(parts[11]) + int(parts[12])
    except (IndexError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--telemetry', required=True)
    parser.add_argument('--spike-ms', type=float, default=150)
    parser.add_argument('--gap-s', type=float, default=.15)
    args = parser.parse_args()
    steps = []
    with open(args.log) as f:
        for line in f:
            if 'STEPDIAG ' not in line:
                continue
            values = {key: float(value) for key, value in re.findall(r'(\w+)=([\d.]+)', line)}
            if 't' in values and 'total' in values:
                steps.append(values)
    with open(args.telemetry) as f:
        samples = [x for line in f if (x := json.loads(line)).get('kcompactd_stat')]
    if len(samples) < 2 or not steps:
        raise SystemExit('Need STEPDIAG and >=2 high-frequency kcompactd samples')
    hz = os.sysconf('SC_CLK_TCK')
    episodes = []
    for before, after in zip(samples, samples[1:]):
        a, b = ticks(before['kcompactd_stat']), ticks(after['kcompactd_stat'])
        if a is None or b is None or b <= a:
            continue
        start, end = before['t'], after['t']
        if episodes and start - episodes[-1]['end'] <= args.gap_s:
            episodes[-1]['end'] = end
            episodes[-1]['ticks'] += b - a
        else:
            episodes.append({'start': start, 'end': end, 'ticks': b - a})
    for episode in episodes:
        episode['cpu_ms'] = episode.pop('ticks') * 1000 / hz
        overlapping = [s for s in steps if s['t'] <= episode['end'] and s['t'] + s['total'] / 1000 >= episode['start']]
        episode['overlapping_steps'] = len(overlapping)
        episode['max_step_ms'] = max((s['total'] for s in overlapping), default=None)
    spikes = []
    for step in steps:
        if step['total'] < args.spike_ms:
            continue
        overlap = [e for e in episodes if e['start'] <= step['t'] + step['total'] / 1000 and e['end'] >= step['t']]
        spikes.append({'step': step, 'compaction_episodes': overlap})
    print(json.dumps({'clock_ticks_per_s': hz, 'sample_count': len(samples),
                      'sampling_limit': 'tick increases happened somewhere between successive samples',
                      'episodes': episodes, 'spikes_including_startup_and_warmup': spikes}, indent=2))


if __name__ == '__main__':
    main()
