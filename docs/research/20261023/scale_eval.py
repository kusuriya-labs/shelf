"""PSQ5-CB28 の W32 scale 符号化候補の比較評価 (8bit / 16bit)。

候補:
  ue4m3_fold / ue4m3_unfold   Shelf ue4m3 codec (有効 0x00..0x7E, 2^-10..224, 1 byte)
  e8m0_fold  / e8m0_unfold    最寄り 2 的乗 (Quark to_e8m0_uint8 相当, 1 byte)
  ue4m4_fold                   独自: 符号なし E4M4 (bias 7, exp=15 予約, 1 byte)
  u5m3_fold                   独自: 符号なし E5M3 (bias 15, exp=31 予約, 1 byte)
  fp16_fold / fp16_unfold     IEEE half (2 byte)
  bf16_fold / bf16_unfold     bfloat16 (2 byte)

exact は scale を F32 のまま使う量子化誤差 (コードブック由来の下限)。
"""

import argparse
import glob
import os

import numpy as np
import torch
from safetensors import safe_open

CB28 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28], dtype=np.float64)
BLOCK_K = 32
CODE_BYTES = 20
CACHE_DIR = "/tmp/psq5/cache"
MODEL_DIR = ""
FULL = False


# ---------------------------------------------------------------- spec checks

def spec_checks() -> None:
    expected = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16]
    for m in range(16):
        raw = m + ((m >> 3) * (m & 0x07))
        assert raw == expected[m], (m, raw, expected[m])

    def e4m3_to_f32(raw: int) -> float:
        sign = -1.0 if (raw & 0x80) else 1.0
        exp = (raw >> 3) & 0xF
        man = raw & 0x7
        if exp == 0:
            return sign * man * (2.0 ** -9)
        if exp == 15 and man == 7:
            return float("nan")
        return sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7))

    for q5 in range(32):
        mag = q5 & 0x0F
        sign = (q5 & 0x10) << 3
        raw = (mag + ((mag >> 3) * (mag & 0x07))) | sign
        signed_cb = -CB28[mag] if (q5 & 0x10) else CB28[mag]
        assert abs(e4m3_to_f32(raw) * 512.0 - signed_cb) < 1e-9, (q5, raw, signed_cb)
    print("[spec] cb28_to_e4m3 と 512 fold 恒等式: OK (全32コード)", flush=True)


# ------------------------------------------------------------------- codecs

def ue4m3_decode(e: int) -> float:
    exp, man = e >> 3, e & 7
    if exp == 0:
        return man / 1024.0
    if exp >= 11:
        return float((8 + man) << (exp - 11))
    return (8 + man) / float(1 << (11 - exp))


def e8m0_decode(e: int) -> float:
    return 2.0 ** (e - 127)


def _ieee_like_decode(e: int, exp_bits: int, man_bits: int, bias: int) -> float:
    man_mask = (1 << man_bits) - 1
    exp = (e >> man_bits) & ((1 << exp_bits) - 1)
    man = e & man_mask
    if exp == 0:
        return man * (2.0 ** (1 - bias - man_bits))
    if exp == (1 << exp_bits) - 1:
        return float("nan")
    return (1.0 + man / float(1 << man_bits)) * (2.0 ** (exp - bias))


def ue4m4_decode(e: int) -> float:
    return _ieee_like_decode(e, 4, 4, 7)


def u5m3_decode(e: int) -> float:
    return _ieee_like_decode(e, 5, 3, 15)


class LutScale:
    """8bit LUT 最寄り符号化 (値昇順 = バイト昇順、tie は下側)。"""

    def __init__(self, decode, valid, fold: float):
        bytes_ = np.array([b for b in range(256) if valid(b)], dtype=np.int64)
        vals = np.array([decode(int(b)) for b in bytes_], dtype=np.float64)
        order = np.argsort(vals, kind="stable")
        self.bytes = bytes_[order]
        self.vals = vals[order]
        self.fold = fold
        self.scale_bytes = 1
        self.pos_first = int(np.argmax(self.vals > 0))
        self.max_val = float(self.vals[-1])
        self.lut = np.zeros(256, dtype=np.float64)
        self.lut[self.bytes] = self.vals

    def encode(self, target: np.ndarray) -> np.ndarray:
        t = np.asarray(target, dtype=np.float64)
        out = np.zeros(t.shape, dtype=np.int64)
        pos = t > 0
        if pos.any():
            tt = t[pos]
            j = np.clip(np.searchsorted(self.vals, tt, side="left"), 0, len(self.vals) - 1)
            lo = np.clip(j - 1, 0, len(self.vals) - 1)
            use_lo = np.abs(self.vals[lo] - tt) <= np.abs(self.vals[j] - tt)
            best = np.where(use_lo, lo, j)
            below = tt < self.vals[self.pos_first]
            best = np.where(below, self.pos_first, best)
            out[pos] = self.bytes[best]
        return out

    def apply(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = s * self.fold
        b = self.encode(v)
        out = self.lut[b] / self.fold
        sat = (v > self.max_val * (1 - 1e-9)) & (s > 0)
        zero = (self.lut[b] <= 0.0) & (s > 0)
        return out, sat, zero


class F16Scale:
    """2byte 浮動小数点 scale。"""

    def __init__(self, kind: str, fold: float):
        self.kind = kind
        self.fold = fold
        self.scale_bytes = 2
        self.max_val = 65504.0 if kind == "fp16" else float(torch.finfo(torch.bfloat16).max)

    def apply(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = s * self.fold
        sat = (v > self.max_val) & (s > 0)
        v = np.minimum(v, self.max_val)
        t = torch.from_numpy(v.astype(np.float32))
        q = (t.to(torch.float16) if self.kind == "fp16" else t.to(torch.bfloat16)).to(torch.float32).numpy()
        out = q.astype(np.float64) / self.fold
        zero = (out <= 0.0) & (s > 0)
        return out, sat, zero


def make_candidates() -> list:
    ue4m3 = LutScale(ue4m3_decode, lambda b: b <= 0x7E, 512.0)
    e8m0 = LutScale(e8m0_decode, lambda b: True, 512.0)
    ue4m4 = LutScale(ue4m4_decode, lambda b: (b >> 4) != 0xF, 512.0)
    u5m3 = LutScale(u5m3_decode, lambda b: (b >> 3) != 0x1F, 512.0)
    ue4m3_u = LutScale(ue4m3_decode, lambda b: b <= 0x7E, 1.0)
    e8m0_u = LutScale(e8m0_decode, lambda b: True, 1.0)
    return [
        ("ue4m3_fold", ue4m3),
        ("ue4m3_unfold", ue4m3_u),
        ("e8m0_fold", e8m0),
        ("e8m0_unfold", e8m0_u),
        ("ue4m4_fold", ue4m4),
        ("u5m3_fold", u5m3),
        ("fp16_fold", F16Scale("fp16", 512.0)),
        ("fp16_unfold", F16Scale("fp16", 1.0)),
        ("bf16_fold", F16Scale("bf16", 512.0)),
        ("bf16_unfold", F16Scale("bf16", 1.0)),
    ]


# -------------------------------------------------------------- quantizer

def nearest_cb28(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    j = np.clip(np.searchsorted(CB28, a, side="left"), 0, 15)
    lo = np.clip(j - 1, 0, 15)
    use_lo = np.abs(a - CB28[lo]) <= np.abs(a - CB28[j])
    return np.where(use_lo, lo, j).astype(np.int8)


def encode_codes(w: np.ndarray, scale: np.ndarray):
    idx = nearest_cb28(w / scale[..., None])
    recon = np.where(w < 0, -CB28[idx], CB28[idx]) * scale[..., None]
    return idx, recon


# ------------------------------------------------------------------- load

def load_tensors() -> list[tuple[str, np.ndarray]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    full = FULL
    fp8_dir = MODEL_DIR
    picks = [f"layers-{i}" for i in range(0, 66, 8)] if full else ["layers-0", "layers-13", "layers-33"]
    per_shard = 6 if full else 3
    want = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "gate_proj.weight", "up_proj.weight", "down_proj.weight",
            "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight"]
    out: list[tuple[str, np.ndarray]] = []
    for stem in picks:
        path = f"{fp8_dir}/{stem}.safetensors"
        if not glob.glob(path):
            continue
        taken = 0
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
            for name in sorted(keys):
                if not any(name.endswith(w) for w in want):
                    continue
                sl = f.get_slice(name)
                if len(sl.get_shape()) != 2:
                    continue
                short = name.replace("model.language_model.layers.", "L").replace(".weight", "")
                label = f"{stem}/{short}"
                cache = os.path.join(CACHE_DIR, label.replace("/", "_") + ".npy")
                if os.path.exists(cache):
                    out.append((label, np.load(cache)))
                    taken += 1
                    if taken >= per_shard:
                        break
                    continue
                if sl.get_dtype() == "F8_E4M3":
                    sname = name + "_scale_inv"
                    if sname not in keys:
                        continue
                    w = f.get_tensor(name).to(torch.float32).numpy().astype(np.float32)
                    s = f.get_tensor(sname).to(torch.float32).numpy()
                    r = np.minimum(np.arange(w.shape[0])[:, None] // 128, s.shape[0] - 1)
                    c = np.minimum(np.arange(w.shape[1])[None, :] // 128, s.shape[1] - 1)
                    w = (w * s[r, c]).astype(np.float32)
                elif sl.get_dtype() == "BF16":
                    w = f.get_tensor(name).to(torch.float32).numpy().astype(np.float32)
                else:
                    continue
                np.save(cache, w)
                out.append((label, w))
                taken += 1
                if taken >= per_shard:
                    break
    return out


# ------------------------------------------------------------------- main

def main() -> None:
    global CACHE_DIR, MODEL_DIR, FULL
    parser = argparse.ArgumentParser(description="PSQ5-CB28 scale encoding evaluation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    CACHE_DIR = args.cache_dir
    FULL = args.full
    spec_checks()
    tensors = load_tensors()
    print(f"[data] {len(tensors)} tensors ({'full' if FULL else 'quick'})", flush=True)

    cands = make_candidates()
    stat = {name: dict(rel=[], flip=0.0, elems=0.0, blocks=0.0, num=0.0, den=0.0,
                       zero=0.0, zero_e=0.0, tot_e=0.0, sat=0.0) for name, _ in cands}
    base = dict(num=0.0, den=0.0)

    for ti, (name, w) in enumerate(tensors):
        n_rows, k = w.shape
        pad = (-k) % BLOCK_K
        if pad:
            w = np.pad(w, ((0, 0), (0, pad)))
        nb = w.shape[1] // BLOCK_K
        b = w.reshape(n_rows, nb, BLOCK_K).astype(np.float32)
        s_true = (np.abs(b).max(axis=2) / 28.0).astype(np.float32)
        energy = float((b * b).sum(dtype=np.float64))
        nz = s_true > 0

        idx_ex, recon_ex = encode_codes(b, s_true)
        d = b - recon_ex
        base["num"] += float((d * d).sum(dtype=np.float64))
        base["den"] += energy

        for cname, codec in cands:
            a = stat[cname]
            s_enc, sat_m, zero_m = codec.apply(s_true)
            valid = nz & (s_enc > 0)
            if valid.any():
                a["rel"].append(np.abs(s_enc[valid] - s_true[valid]) / s_true[valid])
            a["zero"] += float(zero_m.sum())
            a["sat"] += float(sat_m.sum())
            a["tot_e"] += energy
            if zero_m.any():
                bz = b[zero_m]
                a["zero_e"] += float((bz * bz).sum(dtype=np.float64))
            idx_q, recon_q = encode_codes(b, s_enc)
            d = b - recon_q
            a["num"] += float((d * d).sum(dtype=np.float64))
            a["den"] += energy
            a["flip"] += float((idx_q != idx_ex).sum())
            a["elems"] += float(idx_q.size)
            a["blocks"] += float(s_true.size)
        print(f"    [{ti + 1}/{len(tensors)}] {name} done", flush=True)

    exact = (base["num"] / base["den"]) ** 0.5
    print(f"\n=== scale 符号化 候補比較 (Qwen3.8-27B-FP8, W32, scale=amax/28) ===", flush=True)
    print(f"{'candidate':15s} {'B':>2s} {'bpw':>5s} {'mean':>9s} {'p99':>8s} {'max':>8s} "
          f"{'>10%':>7s} {'>50%':>7s} {'zero%':>7s} {'sat%':>6s} {'flip%':>7s} "
          f"{'relFrob':>9s} {'/exact':>7s}")
    print(f"{'exact(F32)':15s} {'-':>2s} {'5.00':>5s} {'-':>9s} {'-':>8s} {'-':>8s} "
          f"{'-':>7s} {'-':>7s} {'-':>7s} {'-':>6s} {'-':>7s} {exact:9.5f} {'1.000':>7s}")
    for name, codec in cands:
        a = stat[name]
        rel = np.concatenate(a["rel"]) if a["rel"] else np.zeros(1)
        rq = (a["num"] / a["den"]) ** 0.5
        bpw = (CODE_BYTES + codec.scale_bytes) * 8 / 32
        print(f"{name:15s} {codec.scale_bytes:2d} {bpw:5.2f} {float(rel.mean()):9.5f} "
              f"{float(np.percentile(rel, 99)):8.4f} {float(rel.max()):8.4f} "
              f"{100.0 * float((rel > 0.10).mean()):7.3f} {100.0 * float((rel > 0.50).mean()):7.3f} "
              f"{100.0 * a['zero'] / a['blocks']:7.4f} {100.0 * a['sat'] / a['blocks']:6.3f} "
              f"{100.0 * a['flip'] / a['elems']:7.4f} {rq:9.5f} {rq / exact:7.4f}")
    print(f"\nblocks = {stat['ue4m3_fold']['blocks']:.0f} / exact relFrob = {exact:.6f}", flush=True)


if __name__ == "__main__":
    main()
