#!/usr/bin/env python3
"""shelf f32 logits vs f64 oracle (last prefill position)."""
import numpy as np, sys, json
shelf = np.fromfile(sys.argv[1], dtype=np.float32)
oracle = np.fromfile(sys.argv[2], dtype=np.float64)
V = 248320
T = oracle.size // V
orow = oracle.reshape(T, V)[-1]
assert shelf.size == V, f"shelf {shelf.size} != V {V}"
diff = np.abs(shelf - orow)
top_s = int(np.argmax(shelf)); top_o = int(np.argmax(orow))
print(f"top1 shelf={top_s} oracle={top_o} {'MATCH' if top_s==top_o else 'MISMATCH'}")
print(f"max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.6f}")
idx = np.argsort(orow)[-10:][::-1]
print("oracle top10:", [(int(i), round(float(orow[i]),3), round(float(shelf[i]),3)) for i in idx])
