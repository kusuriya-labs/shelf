"""
Forward-only chunked linear-attention kernels vendored from flash-linear-attention
(https://github.com/fla-org/flash-linear-attention), v0.5.2, MIT license: see LICENSE in this
directory. Only the Triton forward paths exllamav3 uses for prefill are included (the gated delta
rule, KDA and simple GLA chunk recurrences), without autograd, backward kernels, context
parallelism, in-kernel gate activation or the fla backend dispatch. The kernel sources are
verbatim apart from import paths; the three entry points below replace fla's autograd wrappers.

Inputs are `[B, T, H, K]` / `[B, T, HV, V]`, equal-length sequences only (no cu_seqlens).
"""
from __future__ import annotations
import torch

from .utils import input_guard
from .utils import device_lacks_bf16_dot
from .l2norm import l2norm_fwd
from .cumsum import chunk_local_cumsum
from .gdn_chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_h import chunk_fwd_h
from .kda_chunk_intra import chunk_kda_fwd_intra
from .gla_chunk import chunk_gla_fwd_o_gk

# 1 / ln(2) as fla defines it (best fp32 approximation); gates are pre-scaled so the kernels use exp2
RCP_LN2 = 1.4426950216


def _promote_bf16_operands(q, k, v, beta, promote: bool) -> tuple:
    """
    gfx103x (RDNA2) compatibility step for chunk_gated_delta_rule: RDNA2 has no BF16 dot
    instruction, so a Triton tl.dot on BF16 operands lowers to %llvm.amdgcn.fdot2.bf16.bf16,
    fails LLVM selection and aborts the whole process (exit 134) while compiling the chunk
    kernels below. On those parts the BF16 operands are run in FP32 instead: cast exactly the
    tensors that are BF16 (a FP32 beta is already the target dtype), never FP16 -- downcasting
    BF16 through FP16 would cut dynamic range. The casts are exact, and the internal tensors
    (A, w, u, h, v_new) all inherit their dtype from these operands, so the whole chunk path
    compiles as FP32. Nothing else is an operand here: the cumsum of the gates outputs FP32 by
    contract whatever comes in, and the recurrent state is allocated FP32 in chunk_delta_h, so
    gates and state keep their original FP32 semantics untouched.

    Returns (q, k, v, beta, out_dtype) with out_dtype the ORIGINAL input dtype, which the caller
    casts the result back to so callers always see what they passed in. `promote` is the cached
    per-device probe's answer, taken as an argument to keep this a pure, testable cast step:
    promote=False returns the inputs unchanged (the no-op every other platform and every FP16 /
    FP32 call sees).
    """
    out_dtype = q.dtype
    if not promote:
        return q, k, v, beta, out_dtype
    return (
        q.float() if q.dtype == torch.bfloat16 else q,
        k.float() if k.dtype == torch.bfloat16 else k,
        v.float() if v.dtype == torch.bfloat16 else v,
        beta.float() if beta.dtype == torch.bfloat16 else beta,
        out_dtype,
    )


@input_guard
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked gated delta rule (Gated DeltaNet prefill).

    q, k: [B, T, H, K]; v: [B, T, HV, V] (HV a multiple of H); g: [B, T, HV] log-space decay;
    beta: [B, T, HV] post-sigmoid; initial_state: [B, HV, K, V] fp32 or None.
    Returns (o [B, T, HV, V] in the original input dtype, final_state [B, HV, K, V] fp32 or None).
    On AMD RDNA2 (gfx103x), which cannot compile BF16 tl.dot, BF16 operands run the chunk path in
    FP32 and the output is cast back; everywhere else the dtypes pass through unchanged.
    """
    H, HV = q.shape[2], v.shape[2]
    assert q.shape[2] == k.shape[2], "q and k must have the same number of heads"
    assert HV % H == 0, f"num_v_heads ({HV}) must be a multiple of num_heads ({H})"
    assert chunk_size in (16, 32, 64), f"chunk_size must be 16, 32 or 64, got {chunk_size}"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # see _promote_bf16_operands: narrow RDNA2-only workaround, probe cached per device index
    q, k, v, beta, out_dtype = _promote_bf16_operands(
        q, k, v, beta, q.dtype == torch.bfloat16 and device_lacks_bf16_dot(q.device)
    )
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    # WY representation: fused kkt + solve_tril + recompute_w_u. u is the new v
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k = k, v = v, g = g, beta = beta, chunk_size = chunk_size)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k = k, w = w, u = u, g = g,
        initial_state = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
    )
    o = chunk_fwd_o(q = q, k = k, v = v_new, h = h, g = g, scale = scale, chunk_size = chunk_size)
    return o.to(out_dtype), final_state


@input_guard
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked Kimi Delta Attention (per-channel decay) prefill.

    q, k: [B, T, H, K] with K <= 256; v: [B, T, HV, V]; g: [B, T, HV, K] log-space decay;
    beta: [B, T, HV] post-sigmoid; initial_state: [B, HV, K, V] fp32 or None.
    Returns (o [B, T, HV, V] in q's dtype, final_state [B, HV, K, V] fp32 or None).
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got {q.shape} vs {k.shape}"
    assert K <= 256, f"KDA supports key head dims up to 256, got {K}"
    assert HV % H == 0, f"num_v_heads ({HV}) must be a multiple of num_heads ({H})"
    assert g.shape == (B, T, HV, K), f"g must have shape {[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape {[B, T, HV]}, got {list(beta.shape)}"
    assert chunk_size in (32, 64), f"chunk_size must be 32 or 64, got {chunk_size}"
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be float32"
    if scale is None:
        scale = K ** -0.5
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q = q, k = k, v = v, gk = g, beta = beta, scale = scale, chunk_size = chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k = kg, w = w, u = u, gk = g,
        initial_state = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
    )
    o = chunk_gla_fwd_o_gk(q = q, v = v_new, g = g, A = Aqk, h = h, scale = scale, chunk_size = chunk_size)
    return o.to(q.dtype), final_state


@input_guard
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked simple GLA (head-wise scalar decay, no delta correction), used for the Mamba2 SSD
    prefill.

    q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H] log-space decay or None;
    initial_state: [B, H, K, V] or None.
    Returns (o [B, T, H, V] in q's dtype, final_state [B, H, K, V] or None).
    """
    assert chunk_size & (chunk_size - 1) == 0, f"chunk_size must be a power of 2, got {chunk_size}"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    h, ht = chunk_fwd_h(
        k = k, v = v, g = g,
        h0 = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
        states_in_fp32 = False,
    )
    o = chunk_fwd_o(q = q, k = k, v = v, g = g, h = h, scale = scale, chunk_size = chunk_size)
    return o.to(q.dtype), ht


__all__ = ["chunk_gated_delta_rule", "chunk_kda", "chunk_simple_gla"]
