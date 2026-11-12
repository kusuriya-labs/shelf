#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


def read_rows(path, ids):
    with path.open('rb') as source:
        magic, version, vocab, count, _, score_start = struct.unpack('<8sIIQII', source.read(32))
        if magic != b'PSQUAL01' or version != 1 or count != len(ids) - 2 or score_start != 2:
            raise ValueError(f'invalid candidate dump: {path}')
        for index in range(1, len(ids) - 1):
            target, position = struct.unpack('<iI', source.read(8))
            raw = source.read(vocab * 4)
            if target != ids[index + 1] or position != index + 1 or len(raw) != vocab * 4:
                raise ValueError(f'invalid candidate row: {path}, index {index}')
            yield np.frombuffer(raw, dtype='<f4')


def compare(case, baseline, candidate):
    ref = case['logits'].numpy()
    changes = []
    for i, (before, after) in enumerate(zip(read_rows(baseline, case['ids']),
                                            read_rows(candidate, case['ids'])), start=1):
        reference = ref[i].astype(np.float64)
        before = before.astype(np.float64)
        after = after.astype(np.float64)
        if not (np.isfinite(reference).all() and np.isfinite(before).all() and np.isfinite(after).all()):
            raise ValueError(f'nonfinite logits: {case["name"]}, index {i}')
        p = np.exp(reference - reference.max())
        p /= p.sum()
        bmax, amax = before.max(), after.max()
        delta = np.dot(p, before - after) + (amax + np.log(np.exp(after - amax).sum()) -
                                              bmax - np.log(np.exp(before - bmax).sum()))
        changes.append(float(delta))
    return np.array(changes, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error('bootstrap-samples must be positive')
    torch.set_num_threads(1)
    groups = {'ja-en': [], 'en-ja': [], 'wiki': []}
    for case in torch.load(args.reference, map_location='cpu', weights_only=True):
        name = case['name']
        group = ('ja-en' if name.startswith('heldout-ja-en-') else
                 'en-ja' if name.startswith('heldout-en-ja-') else
                 'wiki' if name.startswith('general-corpus-') else None)
        if group:
            groups[group].append(compare(case, args.baseline / f'{name}.candidate',
                                         args.candidate / f'{name}.candidate'))
    rng = np.random.default_rng(args.seed)
    output = {}
    for group, cases in groups.items():
        if not cases:
            continue
        values = np.concatenate(cases)
        order = np.sort(values)
        cut = int(len(order) * 0.01)
        sums = np.array([v.sum() for v in cases])
        lengths = np.array([len(v) for v in cases])
        sample = rng.integers(0, len(cases), size=(args.bootstrap_samples, len(cases)))
        means = sums[sample].sum(axis=1) / lengths[sample].sum(axis=1)
        output[group] = {'cases': len(cases), 'positions': len(values),
                         'mean_delta_kl': float(values.mean()),
                         'median_delta_kl': float(np.median(values)),
                         'trimmed_1pct_mean_delta_kl': float(order[cut:-cut].mean() if cut else order.mean()),
                         'case_bootstrap_95pct_ci': np.quantile(means, [0.025, 0.975]).tolist(),
                         'bootstrap_samples': args.bootstrap_samples, 'seed': args.seed}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
