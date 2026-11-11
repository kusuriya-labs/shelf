import sys
import numpy as np
from quality_score import read

rh, r = read(sys.argv[1])
ch, c = read(sys.argv[2])
if rh != ch or not np.array_equal(r['target'], c['target']) or not np.array_equal(r['position'], c['position']):
    raise ValueError('dump metadata, target or position mismatch')
values = []
for rr, cc in zip(r, c):
    a, b = rr['logits'].astype(np.float64), cc['logits'].astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite logits')
    az, bz = a.max(), b.max()
    la = a - (az + np.log(np.exp(a - az).sum()))
    lb = b - (bz + np.log(np.exp(b - bz).sum()))
    values.append(float(np.dot(np.exp(la), la - lb)))
kls = np.array(values)
top = np.argsort(kls)[-5:][::-1]
print('top5:', [(int(r[i]['position']), float(kls[i])) for i in top])
print('top5 share of total:', float(kls[top].sum() / kls.sum()) if kls.sum() else 0.0)
