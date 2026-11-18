import json
import sys
import numpy as np
from quality_score import read

rh, ref = read(sys.argv[1])
ch, cand = read(sys.argv[2])
if rh != ch or not np.array_equal(ref['position'], cand['position']) or not np.array_equal(ref['target'], cand['target']):
    raise ValueError('metadata mismatch')
results = []
for arow, brow in zip(ref, cand):
    position = int(arow['position'])
    if not (2030 <= position <= 2060 or 4080 <= position <= 4108):
        continue
    a, b = arow['logits'].astype(np.float64), brow['logits'].astype(np.float64)
    la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
    lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
    results.append({'position': position, 'row_mod8': position % 8,
                    'kl': float(np.dot(np.exp(la), la - lb)),
                    'max_logit_diff': float(np.max(np.abs(a-b))),
                    'top1_match': bool(a.argmax() == b.argmax())})
print(json.dumps(results, indent=2))
