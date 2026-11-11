import json
import math
import struct
import sys
from pathlib import Path
import numpy as np


def read(path):
    with open(path, 'rb') as f:
        magic, version, vocab, positions, dtype, start = struct.unpack('<8sIIQII', f.read(32))
    if magic != b'PSQUAL01' or version != 1 or not vocab or not positions:
        raise ValueError(f'invalid header: {path}')
    row = np.dtype([('target', '<i4'), ('position', '<u4'), ('logits', '<f4', vocab)])
    if Path(path).stat().st_size != 32 + positions * row.itemsize:
        raise ValueError(f'invalid file size: {path}')
    return (vocab, positions, dtype, start), np.memmap(path, dtype=row, mode='r', offset=32, shape=(positions,))


def compare(reference, candidate):
    rh, r = read(reference)
    ch, c = read(candidate)
    if rh != ch or not np.array_equal(r['target'], c['target']) or not np.array_equal(r['position'], c['position']):
        raise ValueError('dump metadata, target or position mismatch')
    kls, nll_r, nll_c, agrees, maxdiff = [], 0., 0., 0, 0.
    for rr, cc in zip(r, c):
        a, b = rr['logits'].astype(np.float64), cc['logits'].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'nonfinite logits at {rr["position"]}')
        az = np.max(a); bz = np.max(b)
        la = a - (az + np.log(np.exp(a - az).sum()))
        lb = b - (bz + np.log(np.exp(b - bz).sum()))
        kls.append(float(np.dot(np.exp(la), la - lb)))
        target = int(rr['target'])
        if target < 0 or target >= rh[0]:
            raise ValueError('target outside vocabulary')
        nll_r -= la[target]; nll_c -= lb[target]
        agrees += int(np.argmax(a) == np.argmax(b))
        maxdiff = max(maxdiff, float(np.max(np.abs(a - b))))
    n = rh[1]
    return {'positions': n, 'mean_kl': float(np.mean(kls)), 'p99_kl': float(np.quantile(kls, .99)),
            'top1': agrees / n, 'max_logit_diff': maxdiff,
            'ppl_reference': math.exp(nll_r / n), 'ppl_candidate': math.exp(nll_c / n),
            'delta_ppl': math.exp(nll_c / n) - math.exp(nll_r / n), 'delta_nll': (nll_c - nll_r) / n}


if __name__ == '__main__':
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
