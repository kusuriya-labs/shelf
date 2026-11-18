import json
import sys
import numpy as np
from quality_score import read

rh, ref = read(sys.argv[1]); ch, cand = read(sys.argv[2])
if rh[0] != ch[0] or rh[2] != ch[2]:
    raise ValueError('vocabulary or KV dtype mismatch')
lookup = {int(r['position']): i for i, r in enumerate(ref)}
out = []
for c in cand:
    pos = int(c['position'])
    if pos not in lookup:
        continue
    r = ref[lookup[pos]]
    if r['target'] != c['target']:
        raise ValueError('target mismatch')
    a, b = r['logits'].astype(np.float64), c['logits'].astype(np.float64)
    la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
    lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
    out.append({'position': pos, 'kl': float(np.dot(np.exp(la), la - lb)),
                'top1_match': bool(a.argmax() == b.argmax()),
                'max_logit_diff': float(np.abs(a - b).max())})
if not out:
    raise ValueError('no overlapping positions')
print(json.dumps(out, indent=2))
