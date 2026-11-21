"""Prefill attention reading the CacheLayer_i8f8 packed KV cache directly.

Round 2 (design (b)): K is stored pre-quantized as INT8 (per-32-token block
mean + scale) and V as FP8 E4M3 (+ per-block-per-channel scale), written once
when tokens enter the cache. The inner loop is therefore minimal:

- Q quantized once per call (per-row absmax / 127, same as round 1),
- QK^T as an INT8 WMMA dot + scale + q.mu bias, fp32 online softmax,
- P row-quantized to FP8, PV as an FP8 WMMA dot, fp32 accumulation.

No per-tile K/V quantization, reductions, or sanitize-over-proofing inside
the loop -- that work moved to ``update_kv_direct`` (amortized once per
token). The <32 staged tail rows are handled with the exact round-1 block
math (imported from ``int8fp8``), so full-block numerics match round 1.

Enabled by the same ``EXL3_ATTN_INT8FP8=1`` switch, but only for layers whose
cache is a ``CacheLayer_i8f8`` (wired in ``dispatch.py``). Prefill only
(``q_len > 16``); decode and every other declined shape fall back to the
fp16 dequant path in ``dispatch.py``. The ``nosmooth`` / ``per-tensor-k``
ablations are round-1-only (the cached format always carries per-block mu
and scales) and decline here; ``fp16pv`` is honored (PV in fp16).
"""

import math
import os

import torch

import triton
import triton.language as tl

from .common import AttnArgs
from .int8fp8 import (
    _i8f8_block,
    _quant_i8,
    _has_fp8_arch,
    int8fp8_ablate,
    int8fp8_enabled,
)


# Call counters so evals can assert the new path actually ran (round-1 lesson:
# silent fallback produces fake-good numbers).
CALLS = {"cached": 0}


def _debug() -> bool:
    return os.environ.get("EXL3_I8F8_DEBUG", "0") != "0"


@triton.jit
def _i8f8c_prefill_kernel(
    q, kq, ksc, kmu, vq, vsc, kst, vst, slen,
    block_table, cache_seqlens, out,
    q_len, kv_append_len,
    n_q_heads: tl.constexpr, n_kv_heads: tl.constexpr,
    num_pages_per_seq,
    page_size: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr,
    QB: tl.constexpr, NB: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    PV_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads
    kv_head = q_head // (n_q_heads // n_kv_heads)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    valid_row = offs_m < q_len

    q_ptrs = q + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    # Keep Q in fp16 (like the fp16 kernel): a persistent fp32 copy would cost
    # BLOCK_M*HD_PAD registers per thread and spill. Convert per block below.
    qf = tl.load(q_ptrs, mask=valid_row[:, None] & d_mask[None, :], other=0.0)
    sq = tl.maximum(tl.max(tl.abs(qf), axis=1), 1e-12).to(tl.float32) / 127.0
    qi8 = _quant_i8(qf.to(tl.float32), 1.0 / sq[:, None])

    past_len = tl.load(cache_seqlens + batch)
    total_k_len = past_len + kv_append_len
    q_abs = past_len + offs_m

    m = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HD_PAD), tl.float32)

    block_table_b = block_table + batch * num_pages_per_seq
    offs_n = tl.arange(0, BLOCK_N)

    # Staged tail: total = t_full + r with t_full 32-aligned.
    tail_phys = tl.load(block_table_b + (total_k_len - 1) // page_size,
                        mask=total_k_len > 0, other=0)
    r = tl.load(slen + tail_phys, mask=total_k_len > 0, other=0)
    t_full = total_k_len - r

    q_abs_min = past_len + pid_m * BLOCK_M
    if CAUSAL:
        n_hi = tl.minimum(q_abs_min + BLOCK_M, total_k_len)
    else:
        n_hi = total_k_len
    n_full = (tl.minimum(q_abs_min, n_hi) // BLOCK_N) * BLOCK_N
    n_full = tl.maximum(tl.minimum(n_full, t_full), 0)

    # -- Full quantized blocks [0, n_full): no masking, no reductions. --
    for n0 in range(0, n_full, BLOCK_N):
        pg = n0 // page_size
        sub = (n0 - pg * page_size) // QB
        phys = tl.load(block_table_b + pg)
        k_ptrs = (kq + (((phys * page_size + (n0 - pg * page_size) + offs_n[None, :])
                         * n_kv_heads + kv_head) * head_dim + offs_d[:, None]))
        k_tile = tl.load(k_ptrs, mask=d_mask[:, None], other=0)
        ks = tl.load(ksc + (phys * NB + sub) * n_kv_heads + kv_head)
        mu = tl.load(kmu + (((phys * NB + sub) * n_kv_heads + kv_head) * head_dim
                            + offs_d), mask=d_mask, other=0.0).to(tl.float32)
        dot = tl.dot(qi8, k_tile).to(tl.float32)
        scores = dot * (sq[:, None] * ks * scale)
        scores = scores + tl.sum(qf * mu[None, :], axis=1)[:, None] * scale

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        l_new = l * alpha + tl.sum(p, axis=1)

        v_ptrs = (vq + (((phys * page_size + (n0 - pg * page_size) + offs_n[:, None])
                         * n_kv_heads + kv_head) * head_dim + offs_d[None, :]))
        vu = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)
        vs = tl.load(vsc + (((phys * NB + sub) * n_kv_heads + kv_head) * head_dim
                            + offs_d), mask=d_mask, other=0.0).to(tl.float32)
        if PV_MODE == 0:
            vf = vu.to(tl.float16) * vs.to(tl.float16)[None, :]
            pv = tl.dot(p.to(tl.float16), vf)
            acc = acc * alpha[:, None] + pv
        elif PV_MODE == 1:
            sp1 = tl.maximum(tl.max(p, axis=1), 1e-12) / 127.0
            pq1 = _quant_i8(p, 1.0 / sp1[:, None])
            vv_f = vu.to(tl.float32) * vs[None, :]
            sv1 = tl.maximum(tl.max(tl.abs(vv_f), axis=0), 1e-12) / 127.0
            vq1 = _quant_i8(vv_f, 1.0 / sv1[None, :])
            pv32 = tl.dot(pq1, vq1).to(tl.float32) * (sp1[:, None] * sv1[None, :])
            acc = acc * alpha[:, None] + pv32
        else:
            sp = tl.maximum(tl.max(p, axis=1), 1e-12) / 448.0
            pq = (p / sp[:, None]).to(tl.float8e4nv)
            vv = vu
            pv = tl.dot(pq, vv).to(tl.float32) * (sp[:, None] * vs[None, :])
            acc = acc * alpha[:, None] + pv
        m = m_new
        l = l_new

    # -- Masked quantized blocks [n_full, n_hi): tiles are whole 32-blocks and
    # only the causal edge needs score masking -- same tile loads as the full
    # loop, no vector gathers (those crashed LLVM's SimplifyCFG on gfx1030).
    # The bound is n_hi (not t_full): tiles above a query tile's causal
    # window are never visited, which is what makes causal half the work.
    n_mask_end = tl.minimum(n_hi, t_full)
    for n0 in range(n_full, n_mask_end, BLOCK_N):
        pg = n0 // page_size
        sub = (n0 - pg * page_size) // QB
        phys = tl.load(block_table_b + pg)
        k_ptrs = (kq + (((phys * page_size + (n0 - pg * page_size) + offs_n[None, :])
                         * n_kv_heads + kv_head) * head_dim + offs_d[:, None]))
        k_tile = tl.load(k_ptrs, mask=d_mask[:, None], other=0)
        ks = tl.load(ksc + (phys * NB + sub) * n_kv_heads + kv_head)
        mu = tl.load(kmu + (((phys * NB + sub) * n_kv_heads + kv_head) * head_dim
                            + offs_d), mask=d_mask, other=0.0).to(tl.float32)
        dot = tl.dot(qi8, k_tile).to(tl.float32)
        scores = dot * (sq[:, None] * ks * scale)
        scores = scores + tl.sum(qf * mu[None, :], axis=1)[:, None] * scale
        if CAUSAL:
            ok = (n0 + offs_n[None, :]) <= q_abs[:, None]
        else:
            ok = tl.full((BLOCK_M, BLOCK_N), True, tl.int1)
        scores = tl.where(ok, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(ok, p, 0.0)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        l_new = l * alpha + tl.sum(p, axis=1)
        v_ptrs = (vq + (((phys * page_size + (n0 - pg * page_size) + offs_n[:, None])
                         * n_kv_heads + kv_head) * head_dim + offs_d[None, :]))
        vu = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)
        vs = tl.load(vsc + (((phys * NB + sub) * n_kv_heads + kv_head) * head_dim
                            + offs_d), mask=d_mask, other=0.0).to(tl.float32)
        if PV_MODE == 0:
            vf = vu.to(tl.float16) * vs.to(tl.float16)[None, :]
            pv = tl.dot(p.to(tl.float16), vf)
            acc = acc * alpha[:, None] + pv
        elif PV_MODE == 1:
            sp1 = tl.maximum(tl.max(p, axis=1), 1e-12) / 127.0
            pq1 = _quant_i8(p, 1.0 / sp1[:, None])
            vv_f = vu.to(tl.float32) * vs[None, :]
            sv1 = tl.maximum(tl.max(tl.abs(vv_f), axis=0), 1e-12) / 127.0
            vq1 = _quant_i8(vv_f, 1.0 / sv1[None, :])
            pv32 = tl.dot(pq1, vq1).to(tl.float32) * (sp1[:, None] * sv1[None, :])
            acc = acc * alpha[:, None] + pv32
        else:
            sp = tl.maximum(tl.max(p, axis=1), 1e-12) / 448.0
            pq = (p / sp[:, None]).to(tl.float8e4nv)
            vv = vu
            pv = tl.dot(pq, vv).to(tl.float32) * (sp[:, None] * vs[None, :])
            acc = acc * alpha[:, None] + pv
        m = m_new
        l = l_new

    # -- Staged tail [t_full, total): fp16 rows, exact round-1 block math. --
    if r > 0:
        n0 = t_full
        st_offs = tl.arange(0, BLOCK_N)
        s_valid = st_offs < r
        k_ptrs = (kst + ((tail_phys * QB + st_offs[None, :])
                         * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=s_valid[None, :] & d_mask[:, None], other=0.0)
        v_ptrs = (vst + ((tail_phys * QB + st_offs[:, None])
                         * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=s_valid[:, None] & d_mask[None, :], other=0.0)
        m, l, acc = _i8f8_block(
            qf, qi8, sq, 0.0,
            k_tile, v_tile, s_valid, r.to(tl.float32),
            m, l, acc, q_abs, n0,
            scale, CAUSAL, True, PV_MODE, True, False,
            BLOCK_M, BLOCK_N, HD_PAD,
        )

    out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    out_ptrs = out + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out_tile, mask=valid_row[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# Wrapper + dispatch entry point
# ---------------------------------------------------------------------------

def paged_attn_triton_prefill_i8f8_cached(
    q: torch.Tensor,
    kv_append_len: int,
    packed: tuple,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    ablate: frozenset | None = None,
    pv_mode: int | None = None,
) -> torch.Tensor:
    """Prefill attention over the CacheLayer_i8f8 packed cache (pure reader).

    The new ``kv_append_len`` rows were already quantized into the cache by
    ``update_kv_direct``; ``cache_seqlens`` still holds the pre-append past
    lengths, so the attended window is ``past + kv_append_len``.
    """
    from ...cache.i8f8 import QB, NB

    ablate = frozenset(ablate if ablate is not None else int8fp8_ablate())
    (kq, ksc, kmu, vq, vsc, kst, vst, slen) = packed
    if q.dtype != torch.float16 or kq.dtype != torch.int8 or vq.dtype != torch.uint8:
        raise ValueError("i8f8 cached prefill requires fp16 q, int8 K, uint8 V")
    bsz, q_len, n_q_heads, head_dim = q.shape
    if q_len != kv_append_len:
        raise ValueError(f"q_len ({q_len}) != kv_append_len ({kv_append_len})")
    if head_dim > 512:
        raise ValueError("i8f8 cached prefill supports head_dim <= 512")
    if n_q_heads % kq.shape[2] != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    n_kv_heads = kq.shape[2]
    if vq.shape != kq.shape or kmu.shape[2:] != (n_kv_heads, head_dim):
        raise ValueError("packed cache / q dimension mismatch")
    if ksc.shape != (kq.shape[0], NB, n_kv_heads) or kq.shape[1] != 256:
        raise ValueError("packed cache page/block geometry mismatch")

    if out is None:
        out = torch.empty_like(q)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    hd_pad = triton.next_power_of_2(head_dim)

    if pv_mode is None:
        if "fp16pv" in ablate:
            pv_mode = 0
        elif _has_fp8_arch(q.device):
            pv_mode = 2
        else:
            pv_mode = 1

    block_n = QB
    assert block_n == 32
    # BLOCK_M is capped for register pressure (the fp32 accumulator alone is
    # BM*HD_PAD registers/thread) as well as 64 KB LDS (gfx1030 and gfx1201).
    if hd_pad <= 128:
        block_m, num_warps = 64, 8
    elif hd_pad <= 256:
        block_m, num_warps = 32, 8
    else:
        block_m, num_warps = 32, 4

    # The uint8 cache holds raw E4M3 bits; pass an fp8 view so the kernel
    # loads real fp8 values (no bitcast/conversion in the loop).
    vq_fp8 = vq.view(torch.float8_e4m3fn)

    def launch(pv: int):
        with torch.cuda.device(q.device):
            grid = (triton.cdiv(q_len, block_m), bsz * n_q_heads)
            _i8f8c_prefill_kernel[grid](
                q, kq, ksc, kmu, vq_fp8, vsc, kst, vst, slen,
                block_table, cache_seqlens, out,
                q_len, kv_append_len,
                n_q_heads, n_kv_heads, block_table.shape[1],
                256, head_dim, hd_pad, QB, NB, float(softmax_scale),
                bool(causal), pv, block_m, block_n,
                num_warps=num_warps, num_stages=2,
            )

    # On gfx1030 the real-fp8 kernel cannot codegen; fall back to the proxy.
    try:
        launch(pv_mode)
    except Exception:
        if pv_mode == 2:
            launch(1)
        else:
            raise
    return out


def fn_triton_paged_attn_prefill_i8f8_cached(args: AttnArgs) -> torch.Tensor | None:
    """Dispatch entry: only fires when EXL3_ATTN_INT8FP8=1 and the call carries
    a CacheLayer_i8f8 packed cache with a supported prefill shape. Anything
    else returns None so dispatch falls back to the fp16 dequant path."""
    if not int8fp8_enabled():
        return None
    if args.i8f8_cache is None or args.q_cache is not None:
        return None
    ablate = int8fp8_ablate()
    if "nosmooth" in ablate or "per-tensor-k" in ablate:
        # Round-1-only ablations: the cached format always carries per-block
        # mu and scales, so there is nothing to ablate here.
        return None
    wl, wr = args.get_window_size()
    if (
        args.is_varlen() or
        args.q_len <= 16 or
        args.dim > 512 or
        args.q.dtype != torch.float16 or
        args.non_causal_spans or
        args.softcap or
        args.sinks is not None or
        wl != -1 or wr != -1
    ):
        return None
    CALLS["cached"] += 1
    if _debug():
        print(f"[i8f8-cached] call #{CALLS['cached']}: bsz={args.bsz} "
              f"q_len={args.q_len} kv_len={args.kv_len} hq={args.num_q_heads} "
              f"hkv={args.num_kv_heads} dim={args.dim} causal={args.causal}",
              flush=True)
    return paged_attn_triton_prefill_i8f8_cached(
        q=args.q,
        kv_append_len=args.kv_len,
        packed=args.i8f8_cache,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
    )
