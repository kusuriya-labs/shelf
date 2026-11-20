#!/usr/bin/env python3
"""保存した位置別品質を対応付けて比較する。"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    b, a = [json.loads(Path(x).read_text()) for x in sys.argv[1:3]]
    for key in ('corpus_sha256', 'vocab_sha256', 'reference', 'model'):
        if a[key] != b[key]:
            raise ValueError(f'{key} が異なります')
    bc, ac = b['cases'], a['cases']
    if len(bc) != len(ac):
        raise ValueError('case 数が異なります')
    diffs, btops, atops, cases = [], [], [], []
    for x, y in zip(bc, ac):
        if (x['id'], x['positions'], x['prefill_positions'], x['reference_logits_sha256']) != (y['id'], y['positions'], y['prefill_positions'], y['reference_logits_sha256']):
            raise ValueError('case が対応していません')
        d = np.array(y['position_kld']) - np.array(x['position_kld'])
        bt = np.array(x['position_top1_agreement'], dtype=bool)
        at = np.array(y['position_top1_agreement'], dtype=bool)
        if len(d) != x['positions'] or len(bt) != len(d) or len(at) != len(d):
            raise ValueError('位置数が不正です')
        diffs.append(d); btops.append(bt); atops.append(at)
        cases.append({'id': x['id'], 'positions': len(d), 'delta_kld': float(d.mean()),
                      'kld_ratio': y['mean_kld'] / x['mean_kld'],
                      'delta_top1_pp': float(100 * (at.mean() - bt.mean())),
                      'b_only': int((bt & ~at).sum()), 'a_only': int((at & ~bt).sum())})
    d = np.concatenate(diffs); bt = np.concatenate(btops); at = np.concatenate(atops)
    rng = np.random.default_rng(20261120)
    sums = np.array([x.sum() for x in diffs]); counts = np.array([len(x) for x in diffs])
    idx = rng.integers(0, len(diffs), size=(100000, len(diffs)))
    bootstrap = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    out = {'b': b['configuration'], 'a': a['configuration'], 'positions': len(d),
           'delta_kld': float(d.mean()), 'kld_ratio': a['mean_kld'] / b['mean_kld'],
           'case_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
           'bootstrap_replicates': len(bootstrap), 'bootstrap_seed': 20261120,
           'delta_top1_pp': float(100 * (at.mean() - bt.mean())),
           'b_only': int((bt & ~at).sum()), 'a_only': int((at & ~bt).sum()), 'cases': cases}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
