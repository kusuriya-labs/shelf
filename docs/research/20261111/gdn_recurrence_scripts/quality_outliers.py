import json
import sys
from pathlib import Path
import numpy as np
from quality_score import read

root = Path(sys.argv[1])
result = {}
for corpus in ('prose', 'code', 'japanese', 'reasoning'):
    rh, ref = read(root / 'dumps' / f'{corpus}-base.psq')
    result[corpus] = {}
    for mode in (sys.argv[3].split(',') if len(sys.argv) > 3 else ('serial', 'chunk', 'exact', 'serial8', 'exact8', 'chunk8')):
        path = root / 'dumps' / f'{corpus}-{mode}.psq'
        if not path.exists():
            continue
        ch, cand = read(path)
        if rh != ch or not np.array_equal(ref['target'], cand['target']) or not np.array_equal(ref['position'], cand['position']):
            raise ValueError(f'metadata mismatch: {path}')
        kl, agrees, changed = [], [], []
        for arow, brow in zip(ref, cand):
            a, b = arow['logits'].astype(np.float64), brow['logits'].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite logits')
            la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
            lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
            kl.append(float(np.dot(np.exp(la), la - lb)))
            agrees.append(bool(a.argmax() == b.argmax()))
            changed.append(bool(np.any(a != b)))
        kl, agrees, changed = np.array(kl), np.array(agrees), np.array(changed)
        top = np.argsort(kl)[-5:]
        keep = np.ones(len(kl), dtype=bool)
        keep[top] = False
        result[corpus][mode] = {'mean_kl': float(kl.mean()), 'top1': float(agrees.mean()),
            'trimmed_mean_kl': float(kl[keep].mean()), 'trimmed_top1': float(agrees[keep].mean()),
            'top5_share': float(kl[top].sum() / kl.sum()) if kl.sum() else 0.,
            'first_different': int(ref['position'][np.flatnonzero(changed)[0]]) if changed.any() else None,
            'outliers': [{'position': int(ref['position'][i]), 'row_mod8': int(ref['position'][i]) % 8, 'verify_row': (int(ref['position'][i]) - rh[3] ) % 8,
                          'kl': float(kl[i]), 'top1_match': bool(agrees[i])} for i in np.flatnonzero(kl > .05)]}
        print(corpus, mode, len(result[corpus][mode]['outliers']), flush=True)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
