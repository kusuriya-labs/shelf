"""Opt-in SageAttention2-style prefill attention: INT8 QK^T, FP8 (E4M3) PV.

Off by default. Enabled with ``EXL3_ATTN_INT8FP8=1`` (read dynamically on every
dispatch so the same process can A/B the fp16 baseline against the new mode).

Design (option (a) from the task: on-the-fly, no cache format change):

- The fp16 paged KV cache (and the existing quantized KV cache path) is
  untouched. Per attention call the kernel loads fp16 Q/K/V tiles and
  quantizes them on the fly:
  - Q: per query-row symmetric INT8 (absmax / 127).
  - K: per K-block (BLOCK_N tokens) mean smoothing + symmetric INT8. Softmax
    invariance (``q.mu`` is constant across keys in a row) is restored with an
    explicit per-row ``q.mu`` bias added back after the INT8 dot, so a single
    global K mean -- which is unstable as a cache appends -- is never needed.
    Each block carries its own mean and scale.
  - P (attention weights in [0, 1]): per query-row FP8 E4M3 (scale 448).
  - V: per-channel (per head-dim column over the K block) FP8 E4M3.
  - Accumulation stays in fp32 (never fp16: the FP8 PV range would collapse).
- Prefill only (``q_len > 16``). Decode keeps the existing kernels; the
  dispatch function declines decode shapes and they fall back silently.
- ``tl.dot`` with int8 inputs accumulates int32; the fp8 dot accumulates
  fp32.

Portability:

- ``PV_MODE 2`` uses real ``tl.float8e4nv`` operands (meant for gfx1201 /
  RDNA4 WMMA). ``PV_MODE 1`` is a V620 (gfx1030, no FP8/WMMA) fallback that
  keeps INT8 QK but uses symmetric-INT8 PV as a mechanics proxy -- accuracy
  claims for FP8 PV always come from the torch reference below, which uses
  the exact E4M3 grid via ``torch.float8_e4m3fn`` round-trips.
- ``PV_MODE 0`` keeps P@V in fp16 (ablation ``fp16pv`` and accuracy upper
  bound).

Ablations (``EXL3_ATTN_INT8FP8_ABLATE``, comma separated):

- ``nosmooth``: skip K mean smoothing (mu = 0).
- ``per-tensor-k``: one K scale per kv head for the whole attended window
  instead of per-block scales.
- ``fp16pv``: PV in fp16 instead of FP8 E4M3.
"""

import math
import os

import torch

import triton
import triton.language as tl

from .common import AttnArgs


FP8_E4M3_MAX = 448.0
_I8_MAX = 127.0


def int8fp8_enabled() -> bool:
    """Dynamic switch so one process can interleave baseline/new-mode runs."""
    return os.environ.get("EXL3_ATTN_INT8FP8", "0") != "0"


def int8fp8_ablate() -> frozenset:
    raw = os.environ.get("EXL3_ATTN_INT8FP8_ABLATE", "")
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


def _has_fp8_arch(device: torch.device) -> bool:
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName
    except Exception:
        return False
    # gfx11+ has FP8; the WMMA-bearing target we care about is gfx1201
    import re
    m = re.search(r"gfx(\d+)", arch or "")
    return bool(m and int(m.group(1)) >= 1100)


# ---------------------------------------------------------------------------
# Torch reference (accuracy vehicle; exact E4M3 grid, block algorithm matches
# the Triton kernel so kernel-vs-reference deltas isolate codegen/rounding)
# ---------------------------------------------------------------------------

def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, (x + 0.5).floor(), (x - 0.5).ceil())


def fake_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round-trip through the E4M3 grid, returned as dequantized fp32."""
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


@torch.inference_mode()
def sdpa_int8fp8_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
    ablate: frozenset | None = None,
    block_m: int = 64,
    block_n: int = 32,
) -> torch.Tensor:
    """Blockwise Sage-style attention in fp32 torch ops.

    q/k/v: (B, Tq/Tk, Hq/Hkv, D) fp16, contiguous. Returns fp16 (B, Tq, Hq, D).
    k/v cover the full attended KV (cache window + appended rows already
    concatenated by the caller).
    """
    ablate = frozenset(ablate or ())
    assert q.is_cuda and k.is_cuda and v.is_cuda
    bsz, tq, hq, dim = q.shape
    tk, hkv = k.shape[1], k.shape[2]
    assert q.shape[0] == k.shape[0] == v.shape[0] and hq % hkv == 0
    assert k.shape[1] == v.shape[1] and k.shape[3] == v.shape[3] == dim
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim)
    group = hq // hkv

    qf = q.float()
    out = torch.zeros((bsz, tq, hq, dim), dtype=torch.float32, device=q.device)
    for b in range(bsz):
        for h in range(hq):
            kvh = h // group
            kf = k[b, :, kvh, :].float()          # (Tk, D)
            vf = v[b, :, kvh, :].float()          # (Tk, D)
            if "per-tensor-k" in ablate:
                if "nosmooth" in ablate:
                    gscale = kf.abs().amax().clamp_min(1e-12) / _I8_MAX
                    gmu = None
                else:
                    gmu = kf.mean(dim=0)
                    gscale = (kf - gmu).abs().amax().clamp_min(1e-12) / _I8_MAX
            else:
                gscale, gmu = None, None
            for m0 in range(0, tq, block_m):
                m1 = min(m0 + block_m, tq)
                qq = qf[b, m0:m1, h, :]           # (M, D)
                sq = qq.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _I8_MAX
                qi = (_round_half_away(qq / sq)).clamp(-128, 127)
                m = torch.full((m1 - m0,), float("-inf"), device=q.device)
                l = torch.zeros((m1 - m0,), device=q.device)
                acc = torch.zeros((m1 - m0, dim), device=q.device)
                q_abs = (tk - tq) + torch.arange(m0, m1, device=q.device)
                n_hi_full = tk
                for n0 in range(0, n_hi_full, block_n):
                    n1 = min(n0 + block_n, n_hi_full)
                    kk = kf[n0:n1, :]             # (N, D)
                    vv = vf[n0:n1, :]
                    nloc = torch.arange(n0, n1, device=q.device)
                    if causal:
                        vis = nloc[None, :] <= q_abs[:, None]
                    else:
                        vis = torch.ones((m1 - m0, n1 - n0), dtype=torch.bool, device=q.device)
                    if not bool(vis.any()):
                        continue
                    if "nosmooth" in ablate:
                        mu = torch.zeros((dim,), device=q.device)
                        kq_in = kk
                        sk = gscale if gscale is not None else kq_in.abs().amax().clamp_min(1e-12) / _I8_MAX
                    else:
                        if gmu is not None:
                            mu = gmu
                        else:
                            mu = kk.mean(dim=0)
                        kq_in = kk - mu
                        sk = gscale if gscale is not None else kq_in.abs().amax().clamp_min(1e-12) / _I8_MAX
                    ki = _round_half_away(kq_in / sk).clamp(-128, 127)
                    s = (qi @ ki.t()).float() * (sq * sk * sm_scale)
                    if "nosmooth" not in ablate:
                        s = s + ((qq * mu).sum(dim=1, keepdim=True) * sm_scale)
                    s = torch.where(vis, s, torch.tensor(float("-inf"), device=q.device))
                    m_new = torch.maximum(m, s.amax(dim=1))
                    p = torch.exp(s - m_new[:, None])
                    p = torch.where(vis, p, torch.zeros((), device=q.device))
                    alpha = torch.where(m == float("-inf"), torch.zeros((), device=q.device), torch.exp(m - m_new))
                    l = l * alpha + p.sum(dim=1)
                    if "fp16pv" in ablate:
                        acc = acc * alpha[:, None] + (p.half().float() @ vv.half().float())
                    else:
                        sp = p.amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
                        pq = fake_e4m3(p, sp)
                        sv = vv.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
                        vq = fake_e4m3(vv, sv)
                        acc = acc * alpha[:, None] + (pq @ vq)
                    m = m_new
                out[b, m0:m1, h, :] = acc / l.clamp_min(1e-30)[:, None]
    return out.half()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _quant_i8(x, inv_scale):
    # Round-half-away-from-zero then clamp to the int8 lane (truncation after
    # the +/-0.5 shift implements the rounding).
    r = tl.where(x >= 0.0, x * inv_scale + 0.5, x * inv_scale - 0.5)
    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
    return r.to(tl.int8)


@triton.jit
def _i8f8_block(
    qf, qi8, sq, gscale,
    k_tile, v_tile, valid_b, n_valid,
    m, l, acc,
    q_abs_row, n0,
    scale, CAUSAL: tl.constexpr,
    MASKED: tl.constexpr,
    PV_MODE: tl.constexpr, SMOOTH: tl.constexpr, USE_GSCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HD_PAD: tl.constexpr,
):
    """One K-block [n0, n0+BLOCK_N): INT8 QK, online softmax, FP8/int8/fp16 PV."""
    valid_f = valid_b.to(tl.float32)
    kf = k_tile.to(tl.float32)
    if SMOOTH:
        mu_b = tl.sum(kf * valid_f[None, :], axis=1) / n_valid
    else:
        mu_b = tl.zeros((HD_PAD,), tl.float32)
    if SMOOTH:
        ks = kf - mu_b[:, None]
    else:
        ks = kf
    sk_b = tl.maximum(tl.max(tl.abs(ks * valid_f[None, :])), 1e-12) / 127.0
    if USE_GSCALE:
        sk_use = tl.where(gscale > 0.0, gscale, sk_b)
    else:
        sk_use = sk_b
    kq = _quant_i8(ks, 1.0 / sk_use)

    dot = tl.dot(qi8, kq).to(tl.float32)
    scores = dot * (sq[:, None] * sk_use * scale)
    if SMOOTH:
        qmu = tl.sum(qf * mu_b[None, :], axis=1) * scale
        scores = scores + qmu[:, None]

    if MASKED:
        offs_n = n0 + tl.arange(0, BLOCK_N)
        valid = valid_b[None, :]
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs_row[:, None])
        scores = tl.where(valid, scores, float("-inf"))
    else:
        valid = tl.full((BLOCK_M, BLOCK_N), True, tl.int1)

    m_new = tl.maximum(m, tl.max(scores, axis=1))
    p = tl.exp(scores - m_new[:, None])
    if MASKED:
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
    else:
        alpha = tl.exp(m - m_new)
    l_new = l * alpha + tl.sum(p, axis=1)

    vf = v_tile.to(tl.float32)
    if PV_MODE == 0:
        pv = tl.dot(p.to(tl.float16), v_tile)
        acc = acc * alpha[:, None] + pv
    elif PV_MODE == 1:
        # V620 proxy: symmetric INT8 PV (mechanics only; FP8 accuracy comes
        # from the torch reference on the exact E4M3 grid).
        sp = tl.maximum(tl.max(p, axis=1), 1e-12) / 127.0
        pq = _quant_i8(p, 1.0 / sp[:, None])
        sv = tl.maximum(tl.max(tl.abs(vf), axis=0), 1e-12) / 127.0
        vq = _quant_i8(vf, 1.0 / sv[None, :])
        pv32 = tl.dot(pq, vq).to(tl.float32) * (sp[:, None] * sv[None, :])
        acc = acc * alpha[:, None] + pv32
    else:
        # Real FP8 E4M3 PV (gfx1201 path).
        sp = tl.maximum(tl.max(p, axis=1), 1e-12) / 448.0
        pq = (p / sp[:, None]).to(tl.float8e4nv)
        sv = tl.maximum(tl.max(tl.abs(vf), axis=0), 1e-12) / 448.0
        vq = (vf / sv[None, :]).to(tl.float8e4nv)
        pv = tl.dot(pq, vq).to(tl.float32) * (sp[:, None] * sv[None, :])
        acc = acc * alpha[:, None] + pv
    return m_new, l_new, acc


@triton.jit
def _i8f8_prefill_kernel(
    q, k_cache, v_cache, block_table, cache_seqlens, out,
    k_gscale,
    q_len, kv_append_len,
    n_q_heads: tl.constexpr, n_kv_heads: tl.constexpr,
    num_pages_per_seq,
    page_size: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    PV_MODE: tl.constexpr, SMOOTH: tl.constexpr, USE_GSCALE: tl.constexpr,
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
    qf = tl.load(q_ptrs, mask=valid_row[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    sq = tl.maximum(tl.max(tl.abs(qf), axis=1), 1e-12) / 127.0
    qi8 = _quant_i8(qf, 1.0 / sq[:, None])

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + offs_m
    if USE_GSCALE:
        gscale = tl.load(k_gscale + kv_head)
    else:
        gscale = 0.0

    m = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HD_PAD), tl.float32)

    block_table_b = block_table + batch * num_pages_per_seq
    offs_n_base = tl.arange(0, BLOCK_N)

    q_abs_min = total_k_len - q_len + pid_m * BLOCK_M
    if CAUSAL:
        n_hi = tl.minimum(q_abs_min + BLOCK_M, total_k_len)
    else:
        n_hi = total_k_len
    n_full = (tl.minimum(q_abs_min, n_hi) // BLOCK_N) * BLOCK_N
    n_full = tl.maximum(n_full, 0)

    for n0 in range(0, n_full, BLOCK_N):
        offs_n = n0 + offs_n_base
        phys = tl.load(block_table_b + offs_n // page_size, mask=offs_n < total_k_len, other=0)
        page_off = offs_n - (offs_n // page_size) * page_size
        k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=(offs_n[None, :] < total_k_len) & d_mask[:, None], other=0.0)
        v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=(offs_n[:, None] < total_k_len) & d_mask[None, :], other=0.0)
        vn = tl.full((BLOCK_N,), True, tl.int1)
        m, l, acc = _i8f8_block(
            qf, qi8, sq, gscale,
            k_tile, v_tile, vn, float(BLOCK_N),
            m, l, acc, q_abs, n0,
            scale, CAUSAL, False, PV_MODE, SMOOTH, USE_GSCALE, BLOCK_M, BLOCK_N, HD_PAD,
        )

    for n0 in range(n_full, n_hi, BLOCK_N):
        offs_n = n0 + offs_n_base
        in_range = offs_n < n_hi
        phys = tl.load(block_table_b + offs_n // page_size, mask=in_range, other=0)
        page_off = offs_n - (offs_n // page_size) * page_size
        k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=in_range[None, :] & d_mask[:, None], other=0.0)
        v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=in_range[:, None] & d_mask[None, :], other=0.0)
        vn = in_range
        n_valid = tl.sum(vn.to(tl.float32))
        n_valid = tl.maximum(n_valid, 1.0)
        m, l, acc = _i8f8_block(
            qf, qi8, sq, gscale,
            k_tile, v_tile, vn, n_valid,
            m, l, acc, q_abs, n0,
            scale, CAUSAL, True, PV_MODE, SMOOTH, USE_GSCALE, BLOCK_M, BLOCK_N, HD_PAD,
        )

    out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    out_ptrs = out + ((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out_tile, mask=valid_row[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# Wrapper + dispatch entry point
# ---------------------------------------------------------------------------

def _window_gscale(k_cache, block_table, cache_seqlens, kv_append_len, n_kv_heads, head_dim, page_size, smooth: bool):
    """One scale per kv head over the whole attended window (per-tensor-k
    ablation only; the fast path uses per-block scales inside the kernel)."""
    bsz = block_table.shape[0]
    scales = []
    for b in range(bsz):
        past = int(cache_seqlens[b].item())
        total = past + kv_append_len
        pages = block_table[b].tolist()
        rows = []
        for pos in range(total):
            pg, off = pos // page_size, pos % page_size
            rows.append(k_cache[pages[pg], off].reshape(-1))
        win = torch.stack(rows, dim=0).float()      # (total, kvh*hd)
        win = win.view(total, n_kv_heads, head_dim)
        if smooth:
            mu = win.mean(dim=0, keepdim=True)
            win = win - mu
        scales.append(win.abs().amax(dim=(0, 2)).clamp_min(1e-12) / _I8_MAX)
    return torch.stack(scales, dim=0).to(torch.float32)  # (bsz, n_kv_heads)


def paged_attn_triton_prefill_int8fp8(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    ablate: frozenset | None = None,
    pv_mode: int | None = None,
) -> torch.Tensor:
    """Prefill attention over the paged fp16 cache with INT8 QK / FP8 PV.

    ``k``/``v`` (bsz, append, kvh, hd) are appended to the cache first with
    the shared paged-update kernel, then the whole window is attended. The
    cache tensors themselves are only read (fp16) -- no format change.
    """
    from .triton_paged import _paged_kv_update_kernel

    ablate = frozenset(ablate if ablate is not None else int8fp8_ablate())
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        raise ValueError("int8fp8 prefill requires fp16 q/k_cache")
    bsz, q_len, n_q_heads, head_dim = q.shape
    if head_dim > 512:
        raise ValueError("int8fp8 prefill supports head_dim <= 512")
    if n_q_heads % k_cache.shape[2] != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    n_kv_heads = k_cache.shape[2]
    _, page_size, _, cache_dim = k_cache.shape
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("cache/head dimension mismatch")

    kv_append_len = 0
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        kv_append_len = k.shape[1]

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
    smooth = "nosmooth" not in ablate
    use_gscale = "per-tensor-k" in ablate

    if k is not None and kv_append_len:
        update_block_d = triton.next_power_of_2(head_dim)
        with torch.cuda.device(q.device):
            _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                k, v, k_cache, v_cache, block_table, cache_seqlens,
                block_table.shape[1], kv_append_len, n_kv_heads, page_size, head_dim, update_block_d,
                num_warps=2, num_stages=3,
            )

    if use_gscale:
        gscale = _window_gscale(k_cache, block_table, cache_seqlens, kv_append_len,
                                n_kv_heads, head_dim, page_size, smooth)
        # kernel takes one scale row per program batch: flatten (bsz, n_kv_heads)
        gscale_ptr = gscale.contiguous()
    else:
        gscale_ptr = torch.zeros((max(bsz, 1), max(n_kv_heads, 1)), dtype=torch.float32, device=q.device)

    if hd_pad <= 128:
        block_m, block_n, num_warps = 128, 32, 8
    elif hd_pad <= 256:
        block_m, block_n, num_warps = 64, 32, 8
    else:
        block_m, block_n, num_warps = 32, 16, 4

    # gscale is (bsz, n_kv_heads); pass a per-batch row via strided indexing:
    # simplest is one launch per batch only when USE_GSCALE, else the pointer
    # is a dummy. Keep a single-graph fast path for the default (no gscale).
    def launch(pv: int):
        with torch.cuda.device(q.device):
            if use_gscale:
                for b in range(bsz):
                    grid = (triton.cdiv(q_len, block_m), n_q_heads)
                    # slice trick: offset the gscale pointer by batch row
                    _i8f8_prefill_kernel[grid](
                        q[b:b + 1], k_cache, v_cache,
                        block_table[b:b + 1], cache_seqlens[b:b + 1], out[b:b + 1],
                        gscale_ptr[b:b + 1].contiguous(),
                        q_len, kv_append_len,
                        n_q_heads, n_kv_heads, block_table.shape[1],
                        page_size, head_dim, hd_pad, float(softmax_scale),
                        bool(causal), pv, smooth, True, block_m, block_n,
                        num_warps=num_warps, num_stages=2,
                    )
            else:
                grid = (triton.cdiv(q_len, block_m), bsz * n_q_heads)
                _i8f8_prefill_kernel[grid](
                    q, k_cache, v_cache, block_table, cache_seqlens, out,
                    gscale_ptr[:1, :].contiguous(),
                    q_len, kv_append_len,
                    n_q_heads, n_kv_heads, block_table.shape[1],
                    page_size, head_dim, hd_pad, float(softmax_scale),
                    bool(causal), pv, smooth, False, block_m, block_n,
                    num_warps=num_warps, num_stages=2,
                )

    # On gfx1030 the real-fp8 kernel cannot codegen; fall back to the proxy.
    # (The per-batch gscale path above indexes q/out per batch; the kernel
    #  treats batch 0 of the sliced view, with block_table/cache_seqlens
    #  sliced to match.)
    try:
        launch(pv_mode)
    except Exception:
        if pv_mode == 2:
            launch(1)
        else:
            raise
    return out


def _nocache_to_temp_pages(args: AttnArgs):
    """Fold a cache-less (bsz, kv, kvh, hd) K/V pair into a temporary paged
    cache so the same kernel serves the no-cache prefill path. The copy is
    the only overhead vs the direct form; decode-length queries (q_len <= 16)
    never reach here."""
    page_size = 256
    bsz, kv_len = args.k.shape[0], args.k.shape[1]
    n_kv_heads = args.k.shape[2]
    pages = (kv_len + page_size - 1) // page_size
    k_cache = torch.empty((bsz * pages, page_size, n_kv_heads, args.dim),
                          dtype=torch.float16, device=args.q.device)
    v_cache = torch.empty_like(k_cache)
    k_cache.view(bsz, pages * page_size, n_kv_heads, args.dim)[:, :kv_len].copy_(args.k)
    v_cache.view(bsz, pages * page_size, n_kv_heads, args.dim)[:, :kv_len].copy_(args.v)
    block_table = (torch.arange(bsz * pages, dtype=torch.int32, device=args.q.device)
                   .view(bsz, pages))
    cache_seqlens = torch.full((bsz,), kv_len, dtype=torch.int32, device=args.q.device)
    return k_cache, v_cache, block_table, cache_seqlens


def fn_triton_paged_attn_prefill_int8fp8(args: AttnArgs) -> torch.Tensor | None:
    """Dispatch entry: only fires when EXL3_ATTN_INT8FP8=1 and the shape is a
    supported fp16 prefill (paged-cache or cache-less). Anything else returns
    None so the default fp16 path (and the quantized-cache path) is
    bit-for-bit unchanged."""
    if not int8fp8_enabled():
        return None
    wl, wr = args.get_window_size()
    if (
        args.is_varlen() or
        args.q_cache is not None or
        args.q_len <= 16 or
        args.dim > 512 or
        args.q.dtype != torch.float16 or
        args.non_causal_spans or
        args.softcap or
        args.sinks is not None or
        wl != -1 or wr != -1
    ):
        return None
    if args.has_kv_cache():
        if args.k_cache.dtype != torch.float16:
            return None
        return paged_attn_triton_prefill_int8fp8(
            q=args.q,
            k=args.k,
            v=args.v,
            k_cache=args.k_cache,
            v_cache=args.v_cache,
            block_table=args.block_table,
            cache_seqlens=args.cache_seqlens,
            causal=args.causal,
            softmax_scale=args.sm_scale,
        )
    # Cache-less prefill (e.g. model.forward bulk): fold K/V into temporary
    # pages and run the same kernel. q/k/v must be contiguous fp16.
    if (args.k is None or args.v is None or args.k.dtype != torch.float16 or
            args.v.dtype != torch.float16 or not args.k.is_contiguous() or
            not args.v.is_contiguous() or not args.q.is_contiguous()):
        return None
    k_cache, v_cache, block_table, cache_seqlens = _nocache_to_temp_pages(args)
    return paged_attn_triton_prefill_int8fp8(
        q=args.q,
        k=None,
        v=None,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
    )
