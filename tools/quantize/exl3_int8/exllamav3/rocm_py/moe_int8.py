# SPDX-License-Identifier: MIT
"""Fused INT8 weight quantize for the grouped MoE path (RDNA2 W8A8 work).

Single-pass Triton kernel: fp16 slab / per-expert fp32 step -> round ->
clamp -> int8 slab. Replaces a multi-pass torch chain (fp16->fp32 temp,
broadcast divide, round, clamp, convert) that runs at ~150 GB/s effective
on this ROCm build (16.7ms for a 128x2048x768 slab).

Numerical contract: bitwise identical to the per-expert prototype's
``(w.float() / s0_python_float).round().clamp(-128, 127).to(torch.int8)``.
The divisor MUST stay a scalar: tensor division selects a different kernel
whose last-ulp behavior differs on .5-boundary elements (found via expert
43's up weights). Each Triton program handles one (expert, chunk) tile and
loads its expert's step as a scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rint_half_even(x):
    # Round-half-to-even matching torch.round (banker's). tl.math has no
    # rint on this backend, and half-up (+0.5 floor) differs on exact .5,
    # which real grid-exact weights do hit (see expert-43 saga).
    fx = tl.floor(x)
    d = x - fx
    is_even = (fx * 0.5) == tl.floor(fx * 0.5)
    q = tl.where(d < 0.5, fx, tl.where(d > 0.5, fx + 1.0, tl.where(is_even, fx, fx + 1.0)))
    # Non-finite inputs never occur here (finite fp16 / positive fp32), but
    # keep torch.round semantics if they do (round(inf) = inf -> clamps).
    q = tl.where(x == float("inf"), float("inf"), q)
    q = tl.where(x == float("-inf"), float("-inf"), q)
    return q


@triton.jit
def _w8quant_kernel(
    WFP,
    S0,
    W8,
    KN,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    cid = tl.program_id(1)
    offs = cid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < KN
    # Reciprocal-multiply, NOT true division: torch's scalar division on
    # this build lowers to x * (1/s) (verified: scalar-div == explicit
    # recip-mul bitwise, while tensor division uses true fdiv and differs
    # on .5-boundary elements). Match the prototype's kernel selection.
    r = 1.0 / tl.load(S0 + b)
    x = tl.load(WFP + b * KN + offs, mask).to(tl.float32)
    q = _rint_half_even(x * r)
    q = tl.clamp(q, -128, 127).to(tl.int8)
    tl.store(W8 + b * KN + offs, q, mask)


def quant_w8_slab(wfp: torch.Tensor, s0: torch.Tensor, w8: torch.Tensor) -> None:
    """wfp [B, K, N] half, s0 [B] fp32 CUDA, w8 [B, K, N] int8 out."""
    assert wfp.is_cuda and w8.is_cuda and s0.is_cuda
    B, K, N = wfp.shape
    assert w8.shape == (B, K, N) and s0.numel() == B
    assert wfp.is_contiguous() and w8.is_contiguous() and s0.is_contiguous()
    KN = K * N
    BLOCK = 4096
    grid = (B, triton.cdiv(KN, BLOCK))
    _w8quant_kernel[grid](wfp, s0, w8, KN, BLOCK)


@triton.jit
def _butterflies_moe(value, columns, GROUPS: tl.constexpr):
    # Local copy of the repo Hadamard butterflies (rocm_py/moe_prefill.py).
    for shift in tl.static_range(7):
        step = 1 << shift
        indices = tl.broadcast_to(columns[None, :] ^ step, (GROUPS, 128))
        peer = tl.gather(value, indices, axis=1)
        value = tl.where((columns[None, :] & step) != 0, -value, value) + peer
    return value


@triton.jit
def _gu_epilog_kernel(
    CG, CU, SA, S0G, S0U, SVG, SVU, IDS, G, U,
    ROWS, CMAX, COLS: tl.constexpr, GROUPS: tl.constexpr = 4,
):
    # Fused gate/up epilogue + output Hadamard. One (row, 128-chunk) per
    # program group row. Replaces: int32->float, xsa, xs0, to-half, had_batch
    # (5 passes) with a single pass. Association matches the torch sequence:
    # ((C * sa) * s0) in fp32, butterflies, x 0.088388347648, x sign.
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (COLS // 128)
    chunk = transform % (COLS // 128)
    c = tl.arange(0, 128)
    valid = row < ROWS
    # S0G/S0U are per-entry (one scale per slab entry, matching the s0g_b /
    # s0u_b vectors built from experts_b); SVG/SVU are per-expert tables.
    # Indexing S0 by expert id reads out of bounds whenever an expert id
    # exceeds the entry count (found via factorial128: g-vs-p maxabs=449).
    entry = row // CMAX
    expert = tl.load(IDS + entry, valid, 0)
    sa = tl.load(SA + row, valid, 0.0)
    s0g = tl.load(S0G + entry, valid, 0.0)
    s0u = tl.load(S0U + entry, valid, 0.0)
    off = row[:, None] * COLS + chunk[:, None] * 128 + c[None, :]
    m = valid[:, None]
    g = tl.load(CG + off, m, 0).to(tl.float32)
    u = tl.load(CU + off, m, 0).to(tl.float32)
    g = (g * sa[:, None]) * s0g[:, None]
    u = (u * sa[:, None]) * s0u[:, None]
    g = _butterflies_moe(g, c, GROUPS)
    u = _butterflies_moe(u, c, GROUPS)
    g = g * 0.088388347648
    u = u * 0.088388347648
    sg = tl.load(SVG + expert[:, None] * COLS + chunk[:, None] * 128 + c[None, :], m, 0).to(tl.float32)
    su = tl.load(SVU + expert[:, None] * COLS + chunk[:, None] * 128 + c[None, :], m, 0).to(tl.float32)
    g = (g * sg).to(G.dtype.element_ty)
    u = (u * su).to(U.dtype.element_ty)
    tl.store(G + off, g, m)
    tl.store(U + off, u, m)


def gu_epilog_had(cg, cu, sa, s0g, s0u, svg, svu, ids, cmax, g, u):
    """cg/cu [R,N] int32, sa [R,1] fp32, s0g/s0u [B] fp32,
    svg/svu [E,N] half, ids [B] int64. g/u [R,N] half out."""
    R, N = cg.shape
    assert N % 128 == 0
    GROUPS = 4
    grid = (triton.cdiv(R * (N // 128), GROUPS),)
    _gu_epilog_kernel[grid](cg, cu, sa, s0g, s0u, svg, svu, ids, g, u,
                            R, cmax, N, GROUPS, num_warps=4,
                            enable_fp_fusion=False)


@triton.jit
def _dquant_kernel(
    A, SAD, XD,
    ROWS, COLS: tl.constexpr, BLOCK: tl.constexpr = 128,
):
    # Fused activation amax + quantize (true division, half-even).
    # Replaces amax / div / round / clamp / convert (5 passes) with 2.
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    amax = 0.0
    for c0 in tl.static_range(0, COLS, BLOCK):
        cols = c0 + offs
        m = cols < COLS
        v = tl.load(A + r * COLS + cols, m, 0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), 0))
    sad = tl.maximum(amax / 127.0, 1e-12)
    tl.store(SAD + r, sad)
    for c0 in tl.static_range(0, COLS, BLOCK):
        cols = c0 + offs
        m = cols < COLS
        v = tl.load(A + r * COLS + cols, m, 0.0).to(tl.float32)
        q = _rint_half_even(v / sad)
        q = tl.clamp(q, -128, 127).to(tl.int8)
        tl.store(XD + r * COLS + cols, q, m)


def dquant_rowwise(a: torch.Tensor, sad: torch.Tensor, xd: torch.Tensor) -> None:
    """a [R,N] half, sad [R,1] fp32 out, xd [R,N] int8 out."""
    R, N = a.shape
    assert sad.shape == (R, 1) and xd.shape == (R, N)
    assert a.is_contiguous() and xd.is_contiguous() and sad.is_contiguous()
    _dquant_kernel[(R,)](a, sad, xd, R, N)


@triton.jit
def _d_epilog_kernel(
    CD, SAD, S0D, SVD, IDS, D,
    ROWS, CMAX, COLS: tl.constexpr, GROUPS: tl.constexpr = 4,
):
    # Fused down epilogue + output Hadamard. Replaces int32->float, xsad,
    # xs0, to-half, had_batch (5 passes) with a single pass.
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (COLS // 128)
    chunk = transform % (COLS // 128)
    c = tl.arange(0, 128)
    valid = row < ROWS
    # S0D is per-entry (see _gu_epilog_kernel note); SVD is per-expert.
    entry = row // CMAX
    expert = tl.load(IDS + entry, valid, 0)
    sad = tl.load(SAD + row, valid, 0.0)
    s0d = tl.load(S0D + entry, valid, 0.0)
    off = row[:, None] * COLS + chunk[:, None] * 128 + c[None, :]
    m = valid[:, None]
    v = tl.load(CD + off, m, 0).to(tl.float32)
    v = (v * sad[:, None]) * s0d[:, None]
    v = _butterflies_moe(v, c, GROUPS)
    v = v * 0.088388347648
    sv = tl.load(SVD + expert[:, None] * COLS + chunk[:, None] * 128 + c[None, :], m, 0).to(tl.float32)
    v = (v * sv).to(D.dtype.element_ty)
    tl.store(D + off, v, m)


def d_epilog_had(cd, sad, s0d, svd, ids, cmax, d):
    """cd [R,N] int32, sad [R,1] fp32, s0d [B] fp32, svd [E,N] half,
    ids [B] int64. d [R,N] half out."""
    R, N = cd.shape
    assert N % 128 == 0
    GROUPS = 4
    grid = (triton.cdiv(R * (N // 128), GROUPS),)
    _d_epilog_kernel[grid](cd, sad, s0d, svd, ids, d,
                           R, cmax, N, GROUPS, num_warps=4,
                           enable_fp_fusion=False)
