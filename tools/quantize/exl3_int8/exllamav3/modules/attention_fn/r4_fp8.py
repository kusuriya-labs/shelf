"""Round 4: minimal-diff FP8 / INT8 prefill from the tuned fp16 Triton kernel.

Goal is FASTER PREFILL (memory saving is secondary). Rounds 1-3 failed on
kernel engineering, not the concept: per-32-block K means + q.mu bias, per
block-per-channel V rescales of the whole accumulator, per-row P scales and
per-tile K/V re-quantization inside the loop. This variant removes all of
that from the inner loop.

The kernels here are exact clones of the tuned fp16 prefill kernel
(``triton_paged._paged_attn_prefill_kernel`` / ``_paged_attn_prefill_inner``
+ combine) with the minimum possible diff:

* Q/K/V tiles load already-quantized values (FP8 E4M3 everywhere for
  Variant A; INT8 Q/K + FP8 V/P for Variant B). Quantization happens once
  per call in :func:`r4_quantize` (torch, outside the kernel).
* QK scale is a per-query-row fused factor (sq[row] * sk * sm_scale, in the
  log2 domain the fp16 kernel already uses): the SAME single broadcast
  multiply as the fp16 ``* qk_scale_log2e``, now with a row vector.
* K smoothing (Variant B) uses one FROZEN mean per (kv-head, channel) for
  the whole call, so ``q.mu`` is constant per row and softmax-invariant:
  NO bias term at all.
* V uses one FROZEN per-(kv-head, channel) scale applied ONCE to the output
  tile at the end (block-invariant, so exact under online rescaling).
* P uses a FIXED scale 1/448 (P in [0, 1]): one convert, no reduction, and
  the inverse is folded into the final normalization.

Inner loop vs fp16: one fp8/int8 dot (instead of fp16), the same single
scale multiply, unchanged online softmax, one ``p -> fp8`` convert, one
fp8 PV dot. Nothing else.

Status: prototype (opt-in, benchmarks only). No dispatch wiring yet.
Prefill-only, causal or non-causal, no window/softcap/sinks/quant-cache/
append path (K/V caches arrive pre-quantized; Q is quantized in the
pre-pass). Head dim <= 512 (power-of-two tile, same as fp16).
"""

import math

import torch

import triton
import triton.language as tl

FP8_MAX = 448.0
_I8_MAX = 127.0
_LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Pre-pass quantization (torch, once per call, outside the kernel)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def r4_quantize(q, k_full, v_full, sm_scale, mode="fp8"):
    """Quantize one prefill call.

    q:      (B, Tq, Hq, D) fp16.
    k_full: (B, Tk, Hkv, D) fp16 full attended K window (contiguous).
    v_full: (B, Tk, Hkv, D) fp16 full attended V window (contiguous).
    mode:   "fp8" (Variant A, all-FP8) or "int8" (Variant B, INT8 QK).

    Returns a dict with fp8/int8 q/k/v plus:
      row_log2e: (B, Tq, Hq) fp32 fused per-row QK factor in log2 domain
                 (sq[row] * sk[head] * sm_scale * log2(e)).
      v_scale:   (Hkv, D) fp32 frozen per-channel V scale.
      mu:        (Hkv, D) fp32 frozen K mean (int8 mode only, else None).
    """
    assert mode in ("fp8", "int8")
    assert q.is_cuda and k_full.is_cuda and v_full.is_cuda
    bsz, tq, hq, dim = q.shape
    tk, hkv = k_full.shape[1], k_full.shape[2]
    assert hq % hkv == 0
    qf = q.float()
    kf = k_full.float()
    vf = v_full.float()

    # Q: per-row scale (one absmax per (b, t, h) over D), computed once.
    qmax = 127.0 if mode == "int8" else 448.0
    sq = qf.abs().amax(dim=-1).clamp_min(1e-12) / qmax          # (B,Tq,Hq)
    q_q = (qf / sq[..., None]).to(
        torch.int8 if mode == "int8" else torch.float8_e4m3fn)

    # K: frozen mean (int8 mode) + one scalar scale per kv head over the
    # whole window. The mean is identical for all keys in the call, so the
    # q.mu correction is softmax-invariant and dropped entirely.
    if mode == "int8":
        mu = kf.mean(dim=(0, 1))                                # (Hkv,D)
        kc = kf - mu[None, None, :, :]
    else:
        mu = None
        kc = kf
    kmax = 127.0 if mode == "int8" else 448.0
    sk = kc.abs().amax(dim=(0, 1, 3)).clamp_min(1e-12) / kmax  # (Hkv,)
    k_q = (kc / sk[None, None, :, None]).to(
        torch.int8 if mode == "int8" else torch.float8_e4m3fn)

    # V: frozen per-(head, channel) scale over the whole window.
    sv = vf.abs().amax(dim=(0, 1)).clamp_min(1e-12) / FP8_MAX  # (Hkv,D)
    v_q = (vf / sv[None, None, :, :]).to(torch.float8_e4m3fn)

    # GQA: the kv-head scalar fans out to its sibling q heads.
    sk_q = sk.repeat_interleave(hq // hkv)                        # (Hq,)
    row_log2e = (sq * sk_q[None, None, :] * sm_scale * _LOG2E).to(torch.float32)
    return {"q_q": q_q, "k_q": k_q, "v_q": v_q,
            "row_log2e": row_log2e.contiguous(),
            "v_scale": sv.to(torch.float32).contiguous(),
            "mu": mu.to(torch.float32) if mu is not None else None}


@torch.inference_mode()
def r4_to_paged(quant, page_size=256):
    """Fold the quantized full-window K/V into paged fp8/int8 caches.

    Returns (k_cache, v_cache, block_table, cache_seqlens) with
    cache_seqlens = Tk (no append; the kernel is a pure reader).
    """
    k_q, v_q = quant["k_q"], quant["v_q"]
    bsz, tk, hkv, dim = k_q.shape
    assert tk % page_size == 0
    pages = tk // page_size
    k_cache = k_q.view(bsz * pages, page_size, hkv, dim)
    v_cache = v_q.view(bsz * pages, page_size, hkv, dim)
    block_table = torch.arange(bsz * pages, dtype=torch.int32,
                               device=k_q.device).view(bsz, pages)
    cache_seqlens = torch.full((bsz,), tk, dtype=torch.int32,
                               device=k_q.device)
    return k_cache, v_cache, block_table, cache_seqlens


# ---------------------------------------------------------------------------
# Torch reference (exact scheme math, fp32, no blocking artifacts)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def sdpa_r4_ref(q, k_full, v_full, causal=True, sm_scale=None, mode="fp8"):
    """Block-free reference: fake-quantize with the same scales as
    :func:`r4_quantize`, then fp32 SDPA with fixed-scale FP8 P."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    quant = r4_quantize(q, k_full, v_full, sm_scale, mode)
    # Dequantize with the exact scales from r4_quantize (recomputed here so
    # the reference stays valid even if the quant helper changes order).
    qmax = 127.0 if mode == "int8" else 448.0
    sq = q.float().abs().amax(dim=-1).clamp_min(1e-12) / qmax
    qf = quant["q_q"].float() * sq[..., None]
    kmax = 127.0 if mode == "int8" else 448.0
    if mode == "int8":
        mu = k_full.float().mean(dim=(0, 1))
        sk = (k_full.float() - mu[None, None, :, :]).abs().amax(
            dim=(0, 1, 3)).clamp_min(1e-12) / kmax
        kf = quant["k_q"].float() * sk[None, None, :, None] + mu[None, None, :, :]
    else:
        sk = k_full.float().abs().amax(dim=(0, 1, 3)).clamp_min(1e-12) / kmax
        kf = quant["k_q"].float() * sk[None, None, :, None]
    sv = v_full.float().abs().amax(dim=(0, 1)).clamp_min(1e-12) / FP8_MAX
    vf = quant["v_q"].float() * sv[None, None, :, :]
    return _sdpa_fp32(qf, kf, vf, causal, sm_scale)


# ---------------------------------------------------------------------------
# Triton kernels: exact clones of triton_paged._paged_attn_prefill_inner /
# _paged_attn_prefill_kernel / _paged_attn_prefill_combine_kernel, minus the
# paths the prototype declines (windows, softcap, sinks, quant-cache, NEW_KV,
# wide index). Diffs vs fp16 are marked with [R4].
# ---------------------------------------------------------------------------

@triton.jit
def _r4_prefill_inner(
    q_tile, acc, m, l,
    row_sc,  # [R4] (BLOCK_M,) fused per-row QK factor, log2 domain (value)
    k_cache, v_cache, block_table_b,
    kv_head, offs_n_base, n_start, n_end,
    q_abs, valid_row,
    total_k_len,
    n_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    QK_MODE: tl.constexpr,  # [R4] 0 = fp8 QK (Variant A), 1 = int8 QK (Variant B)
    CAUSAL: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One pass over kv tiles [n_start, n_end). MASKED=False tiles are fully
    inside the causal region for every row (maskless fast path, as fp16)."""
    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + offs_n_base
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        if MASKED:
            phys = tl.load(block_table_b + page, mask=offs_n < n_end, other=0)
        else:
            phys = tl.load(block_table_b + page)

        # [R4] tiles arrive pre-quantized; no mean/scale reduction, no cvt.
        k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        if HD_PAD > head_dim:
            if MASKED:
                k_tile = tl.load(k_ptrs, mask=(offs_n[None, :] < n_end) & d_mask[:, None], other=0.0)
            else:
                k_tile = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)
        elif MASKED:
            k_tile = tl.load(k_ptrs, mask=offs_n[None, :] < n_end, other=0.0)
        else:
            k_tile = tl.load(k_ptrs)

        # [R4] fp8/int8 dot at native rate; the SAME single scale multiply as
        # fp16 (now a fused per-row factor instead of a scalar).
        if QK_MODE == 0:
            scores = tl.dot(q_tile, k_tile) * row_sc[:, None]
        else:
            scores = tl.dot(q_tile, k_tile).to(tl.float32) * row_sc[:, None]

        if MASKED:
            valid = valid_row[:, None] & (offs_n[None, :] < n_end)
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= q_abs[:, None])
            scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        if MASKED:
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        else:
            m_exp = m_new
        p = tl.exp2(scores - m_exp[:, None])
        if MASKED:
            p = tl.where(valid, p, 0.0)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp2(m - m_exp))
        else:
            alpha = tl.exp2(m - m_exp)
        l = l * alpha + tl.sum(p, axis=1)

        v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
        if HD_PAD > head_dim:
            if MASKED:
                v_tile = tl.load(v_ptrs, mask=(offs_n[:, None] < n_end) & d_mask[None, :], other=0.0)
            else:
                v_tile = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)
        elif MASKED:
            v_tile = tl.load(v_ptrs, mask=offs_n[:, None] < n_end, other=0.0)
        else:
            v_tile = tl.load(v_ptrs)
        # [R4] fixed-scale FP8 P (one convert, no reduction) + fp8 PV dot.
        pq = (p * 448.0).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(pq, v_tile)
        m = m_new
    return acc, m, l


@torch.inference_mode()
def _sdpa_fp32(qf, kf, vf, causal, sm_scale):
    bsz, tq, hq, dim = qf.shape
    tk, hkv = kf.shape[1], kf.shape[2]
    group = hq // hkv
    out = torch.zeros_like(qf)
    for b in range(bsz):
        for h in range(hq):
            kvh = h // group
            s = (qf[b, :, h, :] @ kf[b, :, kvh, :].t()) * sm_scale
            if causal:
                q_abs = (tk - tq) + torch.arange(tq, device=qf.device)
                vis = torch.arange(tk, device=qf.device)[None, :] <= q_abs[:, None]
                s = torch.where(vis, s, torch.tensor(float("-inf"), device=qf.device))
            p = torch.softmax(s.float(), dim=1)
            pq = (p * FP8_MAX).to(torch.float8_e4m3fn).float() / FP8_MAX
            out[b, :, h, :] = pq @ vf[b, :, kvh, :]
    return out.half()


@triton.jit
def _r4_prefill_kernel(
    q,               # [R4] pre-quantized (fp8/int8) (B, Tq, Hq, D)
    k_cache, v_cache,  # [R4] pre-quantized paged caches
    block_table,
    cache_seqlens,
    row_log2e,       # [R4] (B, Tq, Hq) fp32 fused per-row factor
    v_scale,         # [R4] (Hkv, D) fp32 frozen per-channel V scale
    out,
    partial_o,
    partial_ml,
    num_splits,          # runtime
    IS_SPLIT: tl.constexpr,
    q_len,               # runtime
    kv_append_len,       # runtime (0: pure reader over the full window)
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    num_pages_per_seq,   # runtime
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    QK_MODE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FA2-style prefill over the paged quantized cache (clone of the fp16
    kernel body; [R4] marks every diff)."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    split = tl.program_id(2)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads
    group_size = n_q_heads // n_kv_heads
    kv_head = q_head // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    valid_row = offs_m < q_len

    # [R4] Q arrives pre-quantized; dtype follows the tensor. One extra
    # per-program (not per-block) row-scale load outside the loop.
    q_ptrs = q + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=valid_row[:, None] & d_mask[None, :], other=0.0)
    row_sc = tl.load(row_log2e + (batch * q_len + offs_m) * n_q_heads + q_head,
                     mask=valid_row, other=0.0)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + offs_m

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HD_PAD), tl.float32)
    offs_n_base = tl.arange(0, BLOCK_N)
    block_table_b = block_table + batch * num_pages_per_seq

    q_abs_min = total_k_len - q_len + pid_m * BLOCK_M
    if CAUSAL:
        n_hi = tl.minimum(q_abs_min + BLOCK_M, total_k_len)
    else:
        n_hi = total_k_len
    n_lo = 0
    if IS_SPLIT:
        span = tl.cdiv(tl.cdiv(n_hi - n_lo, num_splits), BLOCK_N) * BLOCK_N
        s_lo = n_lo + split * span
        s_hi = tl.minimum(s_lo + span, n_hi)
    else:
        s_lo = n_lo
        s_hi = n_hi

    if CAUSAL:
        n_full = tl.maximum(((q_abs_min + 1) // BLOCK_N) * BLOCK_N, 0)
        acc, m, l = _r4_prefill_inner(
            q_tile, acc, m, l, row_sc,
            k_cache, v_cache, block_table_b, kv_head,
            offs_n_base, s_lo, tl.minimum(n_full, s_hi), q_abs, valid_row,
            total_k_len,
            n_kv_heads, page_size, head_dim, HD_PAD, QK_MODE, CAUSAL,
            False, BLOCK_N,
        )
        acc, m, l = _r4_prefill_inner(
            q_tile, acc, m, l, row_sc,
            k_cache, v_cache, block_table_b, kv_head,
            offs_n_base, tl.maximum(n_full, s_lo), s_hi, q_abs, valid_row,
            total_k_len,
            n_kv_heads, page_size, head_dim, HD_PAD, QK_MODE, CAUSAL,
            True, BLOCK_N,
        )
    else:
        acc, m, l = _r4_prefill_inner(
            q_tile, acc, m, l, row_sc,
            k_cache, v_cache, block_table_b, kv_head,
            offs_n_base, s_lo, s_hi, q_abs, valid_row,
            total_k_len,
            n_kv_heads, page_size, head_dim, HD_PAD, QK_MODE, CAUSAL,
            True, BLOCK_N,
        )

    if IS_SPLIT:
        pid_lin = (pid_m * tl.num_programs(1) + bh) * num_splits + split
        po_base = pid_lin * BLOCK_M * HD_PAD
        tl.store(partial_o + po_base + tl.arange(0, BLOCK_M)[:, None] * HD_PAD + offs_d[None, :], acc)
        ml_base = pid_lin * BLOCK_M * 2
        tl.store(partial_ml + ml_base + tl.arange(0, BLOCK_M) * 2, m)
        tl.store(partial_ml + ml_base + tl.arange(0, BLOCK_M) * 2 + 1, l)
    else:
        # [R4] frozen per-channel V scale + fixed P scale folded in once.
        vs = tl.load(v_scale + kv_head * head_dim + offs_d, mask=d_mask, other=1.0)
        out_tile = acc * vs[None, :] / tl.where(l[:, None] == 0.0, 1.0, l[:, None] * 448.0)
        out_ptrs = out + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
        tl.store(out_ptrs, out_tile, mask=valid_row[:, None] & d_mask[None, :])


@triton.jit
def _r4_prefill_combine_kernel(
    partial_o,
    partial_ml,
    v_scale,
    out,
    num_splits,          # runtime
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    q_len,               # runtime
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads
    kv_head = q_head // (n_q_heads // n_kv_heads)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    rows = tl.arange(0, BLOCK_M)
    pid_lin = ((pid_m * tl.num_programs(1) + bh) * num_splits)

    m_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for sp in range(num_splits):
        m_s = tl.load(partial_ml + (pid_lin + sp) * BLOCK_M * 2 + rows * 2)
        m_max = tl.maximum(m_max, m_s)
    m_safe = tl.where(m_max == -float("inf"), 0.0, m_max)

    l_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HD_PAD), tl.float32)
    for sp in range(num_splits):
        ml_base = (pid_lin + sp) * BLOCK_M * 2
        m_s = tl.load(partial_ml + ml_base + rows * 2)
        l_s = tl.load(partial_ml + ml_base + rows * 2 + 1)
        w = tl.where(m_s == -float("inf"), 0.0, tl.exp2(m_s - m_safe))
        o_s = tl.load(partial_o + (pid_lin + sp) * BLOCK_M * HD_PAD + rows[:, None] * HD_PAD + offs_d[None, :])
        acc += o_s * w[:, None]
        l_sum += l_s * w

    # [R4] same folded output as the non-split path.
    vs = tl.load(v_scale + kv_head * head_dim + offs_d, mask=d_mask, other=1.0)
    out_tile = acc * vs[None, :] / tl.where(l_sum[:, None] == 0.0, 1.0, l_sum[:, None] * 448.0)
    out_ptrs = out + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out_tile, mask=(offs_m[:, None] < q_len) & d_mask[None, :])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

_decode_sm_count = {}


def r4_prefill(
    q_q: torch.Tensor,       # pre-quantized Q (B, Tq, Hq, D) fp8/int8
    k_cache, v_cache,        # pre-quantized paged caches (P, S, Hkv, D)
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    row_log2e: torch.Tensor,  # (B, Tq, Hq) fp32
    v_scale: torch.Tensor,    # (Hkv, D) fp32
    causal: bool = True,
    mode: str = "fp8",
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
    extra_launch: dict | None = None,  # AMD knobs: kpack, matrix_instr_nonkdim,
                                       # waves_per_eu, schedule_hint, ...
) -> torch.Tensor:
    """Prefill over the pre-quantized paged cache (pure reader).

    Tile/split selection mirrors the fp16 kernel exactly so A/B numbers
    compare the dots, not the tiling.
    """
    assert mode in ("fp8", "int8")
    bsz, q_len, n_q_heads, head_dim = q_q.shape
    _, page_size, n_kv_heads, cache_dim = k_cache.shape
    assert cache_dim == head_dim and v_cache.shape == k_cache.shape
    assert n_q_heads % n_kv_heads == 0
    assert row_log2e.shape == (bsz, q_len, n_q_heads)
    assert v_scale.shape == (n_kv_heads, head_dim)
    hd_pad = triton.next_power_of_2(head_dim)
    assert hd_pad == head_dim, "prototype needs pow2 head_dim"
    qk_mode = 0 if mode == "fp8" else 1

    if out is None:
        out = torch.empty((bsz, q_len, n_q_heads, head_dim),
                          dtype=torch.float16, device=q_q.device)

    # Same tile table as paged_attn_triton_prefill on ROCm.
    if hd_pad <= 128:
        cfg = (128, 32, 8, 2)
    elif hd_pad <= 256:
        cfg = (64, 32, 8, 2)
    else:
        cfg = (32, 16, 4, 2)
    block_m = block_m or cfg[0]
    block_n = block_n or cfg[1]
    num_warps = num_warps or cfg[2]
    num_stages = num_stages or cfg[3]
    extra_launch = dict(extra_launch or {})

    num_pages_per_seq = block_table.shape[1]
    q_blocks = triton.cdiv(q_len, block_m)
    programs = q_blocks * bsz * n_q_heads

    if num_splits is None:
        bound_kv = num_pages_per_seq * page_size
        if max_kv_len is not None:
            bound_kv = min(bound_kv, max_kv_len)
        dev = q_q.device.index
        if dev not in _decode_sm_count:
            _decode_sm_count[dev] = torch.cuda.get_device_properties(
                q_q.device).multi_processor_count
        sms = _decode_sm_count[dev]
        num_splits = 1
        if bound_kv >= 8192 and programs:
            best = None
            for cand in (1, 2, 3, 4, 5, 6, 8):
                cost = math.ceil(programs * cand / sms) / cand + 0.03 * (cand - 1)
                if best is None or cost < best[0]:
                    best = (cost, cand)
            num_splits = best[1]
            max_partial_bytes = 128 * 1024 * 1024
            max_splits = max(1, max_partial_bytes // (programs * block_m * hd_pad * 4))
            num_splits = min(num_splits, max_splits)

    if num_splits > 1:
        partial_o = torch.empty(programs * num_splits * block_m * hd_pad,
                                dtype=torch.float32, device=q_q.device)
        partial_ml = torch.empty(programs * num_splits * block_m * 2,
                                 dtype=torch.float32, device=q_q.device)
    else:
        partial_o = q_q  # unused
        partial_ml = q_q  # unused

    assert bsz * q_len * n_q_heads * head_dim < 1 << 31
    assert q_blocks * bsz * n_q_heads * num_splits * block_m * hd_pad < 1 << 31

    with torch.cuda.device(q_q.device):
        grid = (q_blocks, bsz * n_q_heads, num_splits)
        _r4_prefill_kernel[grid](
            q_q, k_cache, v_cache, block_table, cache_seqlens,
            row_log2e, v_scale, out,
            partial_o, partial_ml,
            num_splits, num_splits > 1,
            q_len, 0, n_q_heads, n_kv_heads,
            num_pages_per_seq, page_size, head_dim, hd_pad,
            qk_mode, bool(causal), block_m, block_n,
            num_warps=num_warps, num_stages=num_stages, **extra_launch,
        )
        if num_splits > 1:
            _r4_prefill_combine_kernel[(q_blocks, bsz * n_q_heads)](
                partial_o, partial_ml, v_scale, out,
                num_splits, n_q_heads, n_kv_heads, head_dim, hd_pad,
                q_len, block_m,
                num_warps=8, num_stages=1,
            )
    return out


@torch.inference_mode()
def r4_prefill_auto(q, k_full, v_full, causal=True, sm_scale=None,
                    mode="fp8", **kw):
    """Convenience: quantize (pre-pass) + kernel. For accuracy tests; the
    bench calls :func:`r4_quantize` / :func:`r4_prefill` separately so the
    quantize pass is timed apart from the kernel."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    quant = r4_quantize(q, k_full, v_full, sm_scale, mode)
    k_cache, v_cache, bt, cs = r4_to_paged(quant)
    return r4_prefill(quant["q_q"], k_cache, v_cache, bt, cs,
                      quant["row_log2e"], quant["v_scale"],
                      causal=causal, mode=mode, **kw)
