#!/usr/bin/env python3
"""Collect compact, reproducible summaries from the guarded diagnostic runs."""
import argparse
import json
import os
import re
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    args = p.parse_args()
    runs = {}
    for mode, reps in [('trace', (1, 2)), ('events', (1, 2)), ('blocking', (1,)), ('nothp', (1, 2))]:
        for rep in reps:
            for c in (64, 128):
                tag = f'post-{mode}-{rep}-{c}'
                base = os.path.join(args.directory, tag)
                with open(base + '.analysis.json') as f:
                    analysis = json.load(f)
                with open(base + '.json') as f:
                    workload = json.load(f)
                def compact_spike(event):
                    return {'step': event['step'], 'gpu_samples': [
                        {'t': sample['t'], 'gpu': sample['gpu']}
                        for sample in event['gpu_samples']]}
                command = [sys.executable, os.path.join(os.path.dirname(__file__), 'plamo_compaction_analyze.py'),
                           '--log', base + '.server.log', '--telemetry', base + '.telemetry.jsonl']
                compaction = subprocess.run(command, capture_output=True, text=True, check=False)
                if compaction.returncode == 0:
                    obj = json.loads(compaction.stdout)
                    compaction_summary = {'episodes': [e for e in obj['episodes'] if e['cpu_ms'] >= 100],
                                          'spikes': obj['spikes_including_startup_and_warmup']}
                else:
                    compaction_summary = None  # early runs have only 1-second process samples
                event_lines = []
                with open(base + '.server.log') as f:
                    for line in f:
                        if 'GPUSTALL wait=' in line:
                            event_lines.append(line.strip())
                        elif 'GPUSTALL_SEG' in line:
                            match = re.search(r'ms=([\d.]+)', line)
                            if match and float(match.group(1)) >= 10:
                                event_lines.append(line.strip())
                runs[tag] = {'concurrency': c, 'mode': mode, 'rep': rep,
                             'completion_tokens_per_s': workload['completion_tokens_per_s'],
                             'measurement_steps': analysis['steady_steps'], 'summary': analysis['summary'],
                             'correlations': analysis['correlations'], 'gpu_summary': analysis['gpu_summary'],
                             'host_vm_delta': analysis['host_vm_delta'],
                             'sampling_interval_ms': analysis['sampling_interval_ms'],
                             'spikes': [compact_spike(x) for x in analysis['spikes']],
                             'outside_window_spikes': [compact_spike(x) for x in analysis['outside_window_spikes']],
                             'compaction': compaction_summary, 'long_gpu_event_segments': event_lines}
    print(json.dumps({'measurement_s': 90, 'warmup_s': 20,
                      'warning': 'diagnostic overhead; compare uninstrumented throughput separately',
                      'runs': runs}, indent=2))


if __name__ == '__main__':
    main()
