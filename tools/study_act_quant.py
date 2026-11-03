#!/usr/bin/env python3
# Offline study of activation (A-side) quantization schemes for the MXFP4 W4A8 linear path.
# Reads the real BF16 activations captured by SHELF_LINEAR_INPUT_DUMP and the real
# MXFP4 weights from the checkpoint, then compares four A quantizations:
#   A  per-128-K group, FP32 scale            (production today)
#   B  per-token (whole row), FP32 scale
#   C  per-32-K group, E8M0 scale             (MXFP8, the OCP MX form)
#   D  per-128-K group, E8M0 scale            (current granularity, power-of-two scale)
import json, glob, sys, os
import numpy as np
import ml_dtypes

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/actdump"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "models/Qwen3.8-27B-Quark-AWQ-MXFP4"
MAX_ROWS = int(os.environ.get("STUDY_ROWS", 128))
E4M3_MAX = 448.0

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def e4m3_round(x):
    """Round-to-nearest-even to finite e4m3, saturating magnitude to 448 (the production
    quantizer divides by amax/448 so codes stay in range; overflow cannot happen)."""
    return np.asarray(np.asarray(x, dtype=np.float32).astype(ml_dtypes.float8_e4m3fn),
                      dtype=np.float64)


def scale_f32_group(a, gs):
    rows, k = a.shape
    ng = (k + gs - 1) // gs
    ap = a[:, :ng * gs].reshape(rows, ng, gs)
    amax = np.abs(ap).max(axis=2)
    s = np.where(amax > 0, amax / E4M3_MAX, 1.0)
    return s, ng


def scale_e8m0_group(a, gs):
    """scale = 2^ceil(log2(amax / 448)) so amax/scale <= 448 (OCP MX rounding)."""
    s, ng = scale_f32_group(a, gs)
    amax = s * E4M3_MAX
    with np.errstate(divide="ignore"):
        ex = np.ceil(np.log2(np.where(amax > 0, amax / E4M3_MAX, 1.0)))
    two = np.ldexp(np.ones_like(ex), ex.astype(np.int32))
    two = np.where(amax > 0, two, 1.0)
    return two, ng


def quant(a, kind):
    """Returns the dequantized A (f64) plus per-row diagnostics."""
    rows, k = a.shape
    if kind == "B":
        gs = k
    else:
        gs = {"A": 128, "C": 32, "D": 128}[kind]
    if kind in ("A", "B"):
        s, ng = scale_f32_group(a, gs)
    else:
        s, ng = scale_e8m0_group(a, gs)
    pad = ng * gs
    ap = np.zeros((rows, pad), dtype=np.float64)
    ap[:, :k] = a
    codes = e4m3_round(ap / np.repeat(s, gs, axis=1))
    deq = codes * np.repeat(s, gs, axis=1)
    deq = deq[:, :k]
    nz = a != 0
    sub = np.zeros(rows, dtype=np.int64)
    zeroed = np.zeros(rows, dtype=np.int64)
    if nz.any():
        mag = np.abs(a)
        # e4m3 subnormal magnitudes are < 2^-6 relative to the code scale
        subn = (np.abs(codes[:, :k]) < 2 ** -6) & nz
        sub = subn.sum(axis=1)
        zeroed = ((codes[:, :k] == 0) & (mag > 0)).sum(axis=1)
    span = np.zeros(rows, dtype=np.int64)
    for r in range(rows):
        row = a[r]
        pos = np.abs(row[row != 0])
        if pos.size:
            span[r] = int(np.floor(np.log2(pos.max())) - np.floor(np.log2(pos.min())))
    if kind in ("C", "D"):
        # distance the A-side E8M0 fold would need, per row
        ex = np.log2(s)
        fold_d = (ex.max(axis=1) - ex.min(axis=1)).astype(np.int64)
    else:
        fold_d = np.zeros(rows, dtype=np.int64)
    return deq, dict(subnormal=int(sub.sum()), zeroed=int(zeroed.sum()),
                     nz=int(nz.sum()), span_p50=int(np.median(span)),
                     span_p99=int(np.percentile(span, 99)), span_max=int(span.max()),
                     fold_d_max=int(fold_d.max()), fold_d_bad=int((fold_d > 8).sum()))


def weight_for(shape, occurrence, headers):
    """Picks the checkpoint tensor of this shape by occurrence (load order = layer order)."""
    rows, kp = shape
    names = headers["by_shape"].get((rows, kp // 32), [])
    if not names:
        return None
    return names[occurrence % len(names)]


def load_headers():
    by_shape = {}
    meta = {}
    for f in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            meta[name] = (f, m)
        for name, m in hdr.items():
            if name.endswith(".weight_scale") and len(m["shape"]) == 2:
                base = name[: -len(".weight_scale")]
                if base + ".weight" in hdr:
                    by_shape.setdefault(tuple(m["shape"]), []).append(name)
    for v in by_shape.values():
        v.sort()
    return {"by_shape": by_shape, "meta": meta}


def read_tensor(path, name):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
        base = 8 + n
        off, end = hdr[name]["data_offsets"]
        fh.seek(base + off)
        raw = fh.read(end - off)
    dt = {"U8": np.uint8, "BF16": ml_dtypes.bfloat16, "F32": np.float32}[hdr[name]["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(hdr[name]["shape"])


def dequant_w(scale_name, headers):
    code_name = scale_name[: -len(".weight_scale")] + ".weight"
    path = headers["meta"][scale_name][0]
    codes = read_tensor(path, code_name)
    scales = read_tensor(path, scale_name)
    rows, nb = scales.shape
    kp = nb * 32
    c = codes.reshape(rows, kp // 2).astype(np.uint8)
    lo = E2M1[(c & 0x0F)]
    hi = E2M1[(c >> 4)]
    out = np.empty((rows, kp), dtype=np.float64)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    out = out.reshape(rows, nb, 32) * np.ldexp(np.ones((rows, nb, 1)),
                                               scales.astype(np.int32)[:, :, None] - 127)
    return out.reshape(rows, kp)


def main():
    headers = load_headers()
    metas = [json.loads(l) for l in open(os.path.join(DUMP, "meta.jsonl")) if l.strip()]
    seen_shape = {}
    print(f"captures: {len(metas)}  rows_cap={MAX_ROWS}")
    agg = {}
    for m in metas:
        k = m["k"]
        path = os.path.join(DUMP, f"act_k{k}.bf16")
        if not os.path.exists(path):
            continue
        total_rows = os.path.getsize(path) // (2 * k)
        rows = min(total_rows, m["row_offset"] + m["dumped_rows"], MAX_ROWS)
        a = np.fromfile(path, dtype=ml_dtypes.bfloat16, count=rows * k).reshape(rows, k).astype(np.float64)
        shape = (m["weight_rows"], m["weight_cols"])
        key = shape
        occ = seen_shape.get(key, 0)
        seen_shape[key] = occ + 1
        wname = weight_for(shape, occ, headers)
        if wname is None:
            print("  no weight for", shape, "- skipped")
            continue
        W = dequant_w(wname, headers)
        if W.shape[1] < k:
            W = np.pad(W, ((0, 0), (0, k - W.shape[1])))
        W = W[:, :k]
        y_ref = a @ W.T
        line = []
        for kind in "ABCD":
            deq, st = quant(a, kind)
            st["a_recon"] = float(np.linalg.norm(deq - a) / max(np.linalg.norm(a), 1e-12))
            y = deq @ W.T
            num = np.linalg.norm(y - y_ref)
            den = max(np.linalg.norm(y_ref), 1e-12)
            rel = num / den
            bref = np.asarray(y_ref.astype(ml_dtypes.bfloat16), dtype=np.float64)
            barr = np.asarray(y.astype(ml_dtypes.bfloat16), dtype=np.float64)
            flip = float((bref != barr).mean())
            scale = float(np.abs(barr - bref).max() / max(np.abs(bref).max(), 1e-12))
            g = agg.setdefault((kind, shape), [])
            g.append((rel, flip, scale, st))
            line.append(f"{kind}:aerr={st['a_recon']:.2e},out={rel:.2e},flip={flip*100:.1f}%")
        print(f"  k={k:6d} rows={rows:4d} W={shape} {wname.split('.')[-3]:12s} " + " ".join(line))

    print("\n=== aggregate by scheme (all captures) ===")
    print(f"{'scheme':6} {'rel_l2 mean':>12} {'rel_l2 max':>12} {'bf16 flip mean':>15} "
          f"{'out max rel':>12} {'subnormal codes':>16} {'zeroed codes':>13} {'row span p99':>13} {'fold d>8 rows':>14}")
    label = {"A": "A g128-f32", "B": "B tok-f32", "C": "C g32-E8M0", "D": "D g128-E8M0"}
    for kind in "ABCD":
        rows = [v for key, v in agg.items() if key[0] == kind for v in v]
        rel = np.array([r[0] for r in rows])
        flip = np.array([r[1] for r in rows])
        mx = np.array([r[2] for r in rows])
        sub = sum(r[3]["subnormal"] for r in rows)
        zer = sum(r[3]["zeroed"] for r in rows)
        nz = sum(r[3]["nz"] for r in rows)
        span = max(r[3]["span_p99"] for r in rows)
        fbad = sum(r[3]["fold_d_bad"] for r in rows)
        rows_all = sum(1 for r in rows for _ in [0])
        aerr = np.array([r[3]["a_recon"] for r in rows])
        print(f"{label[kind]:6} {rel.mean():12.3e} {rel.max():12.3e} {flip.mean()*100:14.2f}% "
              f"{mx.max():12.3e} {sub/nz*100:15.2f}% {zer/nz*100:12.2f}% {span:13d} {fbad:14d}"
              f"   a_recon mean={aerr.mean():.3e} max={aerr.max():.3e}")


if __name__ == "__main__":
    main()
