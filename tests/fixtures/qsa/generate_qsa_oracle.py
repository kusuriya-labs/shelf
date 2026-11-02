import argparse
import os
import struct
import zlib

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN_G = 0x9E3779B97F4A7C15
MIX_C1 = 0xBF58476D1CE4E5B9
MIX_C2 = 0x94D049BB133111EB

SEED_INDEXER = 0x5153415F49445831
SEED_SPARSE = 0x5153415F53415454

IDX_T = 2053
IDX_HEADS = 4
IDX_HEAD_DIM = 128
IDX_RATIO = 4
IDX_TOPK = 512
IDX_ROTARY = 64
IDX_THETA = 10000000.0
IDX_EPS = 1e-6
IDX_CRITICAL = [0, 1, 2, 3, 4, 5, 6, 7, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052]

SP_T = 4096
SP_Q_HEADS = 24
SP_KV_HEADS = 2
SP_HEAD_DIM = 256
SP_ROTARY = 64
SP_THETA = 10000000.0
SP_EPS = 1e-6
SP_PAGE_TOKENS = 16
SP_NUM_PAGES = 512
SP_CRITICAL = [0, 1, 2, 3, 7, 8, 100, 2047, 2048, 4092, 4093, 4094, 4095]
SP_MASK_STRIDE = 8
SP_MASK_KEEP = 2


def splitmix_u64(seed, n):
    states = (
        np.uint64(seed)
        + (np.arange(n, dtype=np.uint64) + np.uint64(1)) * np.uint64(GOLDEN_G)
    ) & np.uint64(MASK64)
    z = states
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(MIX_C1)) & np.uint64(MASK64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(MIX_C2)) & np.uint64(MASK64)
    return z ^ (z >> np.uint64(31))


def splitmix_f32(seed, n):
    u = (splitmix_u64(seed, n) >> np.uint64(32)).astype(np.uint32)
    return u.astype(np.float32) * np.float32(2.0 / 4294967296.0) - np.float32(1.0)


def f32_to_bf16_bits(f):
    u = f.astype(np.float32).view(np.uint32).astype(np.uint32).copy()
    exp = u & np.uint32(0x7F800000)
    mant = u & np.uint32(0x007FFFFF)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    rounded = (u + bias) >> np.uint32(16)
    nan = np.uint32(0x7FC0)
    special = exp == np.uint32(0x7F800000)
    out = np.where(special, np.where(mant != np.uint32(0), nan, u >> np.uint32(16)), rounded)
    return out.astype(np.uint32)


def f32_to_bf16(f):
    return f32_to_bf16_bits(f).astype(np.uint16)


def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


def bf16_bits_to_f32(bits):
    return ((bits.astype(np.uint32) << np.uint32(16)).view(np.float32)).astype(np.float32)


def bf16_quant(f):
    bits = f32_to_bf16(f)
    return ((bits.astype(np.uint32) << np.uint32(16)).view(np.float32)).astype(np.float32)


def rmsnorm_one_plus(x, weight, eps, group):
    shape = x.shape
    g = x.reshape(-1, group).astype(np.float32)
    mean_sq = np.mean(g * g, axis=-1, keepdims=True).astype(np.float32)
    inv = np.reciprocal(np.sqrt(mean_sq + np.float32(eps))).astype(np.float32)
    w = (np.float32(1.0) + weight.astype(np.float32)).astype(np.float32)
    y = (g * inv) * np.broadcast_to(w, g.shape)
    return y.reshape(shape).astype(np.float32)


def rope_partial(x, positions, head_dim, rotary, theta):
    half = rotary // 2
    idx = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / (float(theta) ** (idx / float(half)))
    angle = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    bshape = [angle.shape[0]] + [1] * (x.ndim - 2) + [angle.shape[1]]
    cs = np.cos(angle).astype(np.float32).reshape(bshape)
    sn = np.sin(angle).astype(np.float32).reshape(bshape)
    y = x.astype(np.float32).copy()
    x0 = y[..., 0:half].copy()
    x1 = y[..., half:rotary].copy()
    y[..., 0:half] = x0 * cs - x1 * sn
    y[..., half:rotary] = x1 * cs + x0 * sn
    return y


def rope_rows(x, positions, head_dim, rotary, theta):
    rows = x.shape[0]
    heads = x.shape[1] // head_dim
    view = x.reshape(rows, heads, head_dim)
    out = rope_partial(view, positions, head_dim, rotary, theta)
    return out.reshape(rows, -1)


def indexer_oracle(q_raw, key_cache, q_weight, k_weight, positions):
    q_raw = bf16_bits_to_f32(q_raw)
    key_cache = bf16_bits_to_f32(key_cache)
    q_weight = bf16_bits_to_f32(q_weight)
    k_weight = bf16_bits_to_f32(k_weight)
    t = q_raw.shape[0]
    qn = rmsnorm_one_plus(q_raw.reshape(t, IDX_HEADS, IDX_HEAD_DIM), q_weight, IDX_EPS,
                          IDX_HEAD_DIM)
    q = rope_rows(qn.reshape(t, -1), positions, IDX_HEAD_DIM, IDX_ROTARY, IDX_THETA)
    q = bf16_quant(q)
    q = q.reshape(t, IDX_HEADS, IDX_HEAD_DIM)

    n_blocks = t // IDX_RATIO
    pooled = key_cache.reshape(-1, IDX_HEAD_DIM)[: n_blocks * IDX_RATIO].reshape(
        n_blocks, IDX_RATIO, IDX_HEAD_DIM
    ).astype(np.float32).mean(axis=1)
    pooled = pooled.astype(np.float32)
    kn = rmsnorm_one_plus(pooled, k_weight, IDX_EPS, IDX_HEAD_DIM)
    block_pos = positions[: n_blocks * IDX_RATIO : IDX_RATIO]
    bk = rope_partial(kn, block_pos, IDX_HEAD_DIM, IDX_ROTARY, IDX_THETA)
    bk = bf16_quant(bk)

    scale = np.float32(1.0 / np.sqrt(float(IDX_HEAD_DIM)))
    scores = np.zeros((t, n_blocks), dtype=np.float32)
    mask = np.zeros((t, t), dtype=np.uint8)
    selected_blocks = []
    counts = []
    for i in range(t):
        ncb = (i + 1) // IDX_RATIO
        if ncb > 0:
            dots = np.einsum("hd,bd->hb", q[i], bk[:ncb], optimize=True)
            relu = np.maximum(dots, np.float32(0.0))
            row = (relu.sum(axis=0) * scale).astype(np.float32)
            scores[i, :ncb] = row
            ids = np.arange(ncb, dtype=np.int64)
            order = np.lexsort((ids, -row))
            k = min(IDX_TOPK, ncb)
            sel = order[:k].astype(np.uint32)
        else:
            k = 0
            sel = np.zeros((0,), dtype=np.uint32)
        tail = np.arange(ncb * IDX_RATIO, i + 1, dtype=np.int64)
        toks = (sel.astype(np.int64)[:, None] * IDX_RATIO +
                np.arange(IDX_RATIO, dtype=np.int64)[None, :]).reshape(-1)
        mask[i, toks] = 1
        mask[i, tail] = 1
        selected_blocks.append(sel)
        counts.append(k)
    return {
        "scores": scores,
        "mask": mask,
        "selected_blocks": selected_blocks,
        "counts": np.array(counts, dtype=np.uint32),
    }


def sparse_prepare(query_raw, k_raw, v_raw, q_weight, k_weight, positions):
    query_raw = bf16_bits_to_f32(query_raw)
    k_raw = bf16_bits_to_f32(k_raw)
    v_raw = bf16_bits_to_f32(v_raw)
    q_weight = bf16_bits_to_f32(q_weight)
    k_weight = bf16_bits_to_f32(k_weight)
    t = query_raw.shape[0]
    qn = rmsnorm_one_plus(query_raw.reshape(t, SP_Q_HEADS, SP_HEAD_DIM), q_weight, SP_EPS,
                          SP_HEAD_DIM)
    q = rope_rows(qn.reshape(t, -1), positions, SP_HEAD_DIM, SP_ROTARY, SP_THETA)
    q = bf16_quant(q).reshape(t, SP_Q_HEADS, SP_HEAD_DIM)

    kn = rmsnorm_one_plus(k_raw.reshape(t, SP_KV_HEADS, SP_HEAD_DIM), k_weight, SP_EPS,
                          SP_HEAD_DIM)
    k = rope_rows(kn.reshape(t, -1), positions, SP_HEAD_DIM, SP_ROTARY, SP_THETA)
    k = bf16_quant(k).reshape(t, SP_KV_HEADS, SP_HEAD_DIM)
    v = bf16_quant(v_raw)
    v = v.reshape(t, SP_KV_HEADS, SP_HEAD_DIM)
    return q, k, v


def dense_mask(t):
    m = np.zeros((t, t), dtype=np.uint8)
    for i in range(t):
        m[i, : i + 1] = 1
    return m


def rule_mask(t):
    m = np.zeros((t, t), dtype=np.uint8)
    for i in range(t):
        ncb = (i + 1) // IDX_RATIO
        keep = [b for b in range(ncb) if (b % SP_MASK_STRIDE) < SP_MASK_KEEP]
        toks = (np.array(keep, dtype=np.int64)[:, None] * IDX_RATIO +
                np.arange(IDX_RATIO, dtype=np.int64)[None, :]).reshape(-1)
        if toks.size:
            m[i, toks] = 1
        m[i, ncb * IDX_RATIO : i + 1] = 1
    return m


def sparse_oracle_rows(q, k, v, mask, rows):
    scale = np.float32(1.0 / np.sqrt(float(SP_HEAD_DIM)))
    group = SP_Q_HEADS // SP_KV_HEADS
    out = {}
    for i in rows:
        sel = np.flatnonzero(mask[i])
        acc = np.zeros((SP_Q_HEADS, SP_HEAD_DIM), dtype=np.float32)
        for h in range(SP_Q_HEADS):
            kv = h // group
            ks = k[sel, kv, :]
            logits = (q[i, h, :][None, :] * ks).sum(axis=-1).astype(np.float32) * scale
            logits = logits - logits.max()
            w = np.exp(logits)
            w = (w / w.sum()).astype(np.float32)
            acc[h] = (w[:, None] * v[sel, kv, :]).sum(axis=0)
        out[i] = acc.reshape(-1)
    return out


def write_indexer_golden(path, oracle, inputs_crc):
    with open(path, "wb") as f:
        f.write(b"PSQSAIDX")
        f.write(struct.pack("<IIQ", 1, IDX_T, SEED_INDEXER))
        f.write(struct.pack("<Q", inputs_crc))
        f.write(struct.pack("<IIII", IDX_HEADS, IDX_HEAD_DIM, IDX_RATIO, IDX_TOPK))
        f.write(struct.pack("<II", IDX_ROTARY, len(IDX_CRITICAL)))
        f.write(struct.pack("<II", 0, 0))
        f.write(struct.pack("<fff", IDX_THETA, IDX_EPS, 1.0 / np.sqrt(float(IDX_HEAD_DIM))))
        f.write(np.array(IDX_CRITICAL, dtype=np.uint32).tobytes())
        for row in IDX_CRITICAL:
            ncb = (row + 1) // IDX_RATIO
            k = min(IDX_TOPK, ncb)
            sel = oracle["selected_blocks"][row]
            f.write(struct.pack("<III", row, ncb, k))
            f.write(oracle["scores"][row, :ncb].astype("<f4").tobytes())
            f.write(sel.astype("<u4").tobytes())
            f.write(oracle["mask"][row].tobytes())


def write_sparse_golden(path, outputs, inputs_crc, rows):
    with open(path, "wb") as f:
        f.write(b"PSQSASPT")
        f.write(struct.pack("<IIQ", 1, SP_T, SEED_SPARSE))
        f.write(struct.pack("<Q", inputs_crc))
        f.write(struct.pack("<IIII", SP_Q_HEADS, SP_KV_HEADS, SP_HEAD_DIM, SP_ROTARY))
        f.write(struct.pack("<IIII", SP_PAGE_TOKENS, SP_NUM_PAGES, SP_MASK_STRIDE, SP_MASK_KEEP))
        f.write(struct.pack("<II", len(rows), 6144))
        f.write(struct.pack("<fff", SP_THETA, SP_EPS, 1.0 / np.sqrt(float(SP_HEAD_DIM))))
        f.write(np.array(rows, dtype=np.uint32).tobytes())
        for row in rows:
            f.write(struct.pack("<I", row))
            f.write(outputs[row].astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    positions = np.arange(IDX_T, dtype=np.uint32)
    q_raw = f32_to_bf16(splitmix_f32(SEED_INDEXER, IDX_T * IDX_HEADS * IDX_HEAD_DIM))
    key_cache = f32_to_bf16(
        splitmix_f32(SEED_INDEXER ^ np.uint64(1), IDX_T * IDX_HEAD_DIM)
    )
    q_weight = f32_to_bf16(splitmix_f32(SEED_INDEXER ^ np.uint64(2), IDX_HEAD_DIM))
    k_weight = f32_to_bf16(splitmix_f32(SEED_INDEXER ^ np.uint64(3), IDX_HEAD_DIM))
    payload = b"".join(
        a.tobytes()
        for a in (q_raw, key_cache, q_weight, k_weight)
    )
    idx_crc = crc32(payload)
    oracle = indexer_oracle(
        q_raw.reshape(IDX_T, IDX_HEADS, IDX_HEAD_DIM),
        key_cache.reshape(IDX_T, IDX_HEAD_DIM),
        q_weight,
        k_weight,
        positions,
    )
    write_indexer_golden(os.path.join(args.out_dir, "qsa_indexer_golden.bin"), oracle, idx_crc)

    s_positions = np.arange(SP_T, dtype=np.uint32)
    s_count = SP_T * (SP_Q_HEADS * SP_HEAD_DIM + 2 * SP_KV_HEADS * SP_HEAD_DIM)
    s_all = f32_to_bf16(splitmix_f32(SEED_SPARSE, s_count))
    q_features = SP_Q_HEADS * SP_HEAD_DIM
    kv_features = SP_KV_HEADS * SP_HEAD_DIM
    query_raw = s_all[: SP_T * q_features].reshape(SP_T, q_features)
    k_raw = s_all[SP_T * q_features : SP_T * (q_features + kv_features)].reshape(SP_T, kv_features)
    v_raw = s_all[SP_T * (q_features + kv_features):].reshape(SP_T, kv_features)
    sp_qw = f32_to_bf16(splitmix_f32(SEED_SPARSE ^ np.uint64(1), SP_HEAD_DIM))
    sp_kw = f32_to_bf16(splitmix_f32(SEED_SPARSE ^ np.uint64(2), SP_HEAD_DIM))
    sp_payload = b"".join(a.tobytes() for a in (query_raw, k_raw, v_raw, sp_qw, sp_kw))
    sp_crc = crc32(sp_payload)

    q, k, v = sparse_prepare(query_raw, k_raw, v_raw, sp_qw, sp_kw, s_positions)
    mask = rule_mask(SP_T)
    outputs = sparse_oracle_rows(q, k, v, mask, SP_CRITICAL)
    write_sparse_golden(
        os.path.join(args.out_dir, "qsa_sparse_attention_golden.bin"), outputs, sp_crc,
        SP_CRITICAL,
    )
    print("indexer inputs crc32=%08x rows=%d" % (idx_crc, IDX_T))
    print("sparse inputs crc32=%08x rows=%d" % (sp_crc, SP_T))
    print("sparse selected fraction=%.4f" % (mask.sum(axis=1).mean() / float(SP_T)))


if __name__ == "__main__":
    main()
