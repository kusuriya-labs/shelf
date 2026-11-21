# SPDX-License-Identifier: MIT
"""Fused Triton helpers for the dense W8A8 prefill path (RDNA2 C2 work).

Reuses the proven MoE kernels where the contract matches bitwise:
- activation quant: ``dquant_rowwise`` (moe_int8) == torch
  ``(xh.float()/sa).round().clamp(-128,127).to(int8)`` with
  ``sa = amax/127`` (true division, round-half-even).
- weight quant: ``quant_w8_slab`` (moe_int8, B=1) == torch
  ``(w.float()/s0).round().clamp(-128,127).to(int8)`` for s0 models.

The dense epilogue below fuses dequant + output Hadamard + svh scaling in
fp32 (one pass). It is NOT bitwise identical to the torch sequence
(dequant-round-to-half, fp16 Hadamard, fp16 svh scale) — it keeps full
precision until the final store, so expect small (tolerance-level)
differences, verified by tail-maxabs and KLD, not by bitwise equality.
"""

import torch
import triton
import triton.language as tl

from .moe_int8 import _butterflies_moe, dquant_rowwise, quant_w8_slab

__all__ = ["dquant_rowwise", "quant_w8_slab", "dense_epilog_had"]


@triton.jit
def _dense_epilog_kernel(
    C, SA, WS, SVH, Y,
    ROWS, CN: tl.constexpr, SN: tl.constexpr, YSTRIDE: tl.constexpr,
    YOFF: tl.constexpr, GROUPS: tl.constexpr = 4,
):
    # One (row, 128-chunk) group per program. Replaces: int32->float,
    # x sa, x ws, to-half, had_r_128 (5 passes) with a single pass.
    # Association matches torch ((C * sa) * ws); the Hadamard scale
    # 1/sqrt(128) and post-scale SVH match had_r_128(x, y, None, svh, 1.0).
    # C is compact [R,CN]; Y is [R,*] with row stride YSTRIDE and column
    # base YOFF; SVH is the compact [SN] column slice.
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (SN // 128)
    chunk = transform % (SN // 128)
    c = tl.arange(0, 128)
    valid = row < ROWS
    sa = tl.load(SA + row, valid, 0.0)
    ws = tl.load(WS)
    off_c = row[:, None] * CN + chunk[:, None] * 128 + c[None, :]
    off_y = row[:, None] * YSTRIDE + YOFF + chunk[:, None] * 128 + c[None, :]
    m = valid[:, None]
    v = tl.load(C + off_c, m, 0).to(tl.float32)
    v = (v * sa[:, None]) * ws
    v = _butterflies_moe(v, c, GROUPS)
    v = v * 0.088388347648
    sv = tl.load(SVH + chunk[:, None] * 128 + c[None, :]).to(tl.float32)
    v = (v * sv).to(Y.dtype.element_ty)
    tl.store(Y + off_y, v, m)


def dense_epilog_had(c, sa, ws, svh_slice, y, y_stride, y_off=0):
    """c [R,SN] compact int32, sa [R,1] fp32, ws scalar fp32 CUDA,
    svh_slice [SN] contiguous half, y [R,Nfull] out (strided view ok).
    y_stride = full row stride in elements, y_off = column base."""
    R, SN = c.shape
    assert c.is_cuda and sa.is_cuda and svh_slice.is_cuda and y.is_cuda
    assert SN % 128 == 0 and svh_slice.numel() == SN
    assert svh_slice.is_contiguous() and c.is_contiguous()
    GROUPS = 4
    grid = (triton.cdiv(R * (SN // 128), GROUPS),)
    _dense_epilog_kernel[grid](c, sa, ws, svh_slice, y, R, SN, SN, y_stride,
                               y_off, GROUPS, num_warps=4,
                               enable_fp_fusion=False)
