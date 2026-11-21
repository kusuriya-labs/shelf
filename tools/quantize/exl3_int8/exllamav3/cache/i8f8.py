"""Opt-in quantized KV cache layer for INT8-QK / FP8-PV attention (design (b)).

Format (per layer, ``P`` pages of ``PAGE_SIZE`` = 256 tokens, ``QB`` = 32
tokens per quantization block, ``NB`` = 8 blocks per page, ``H`` kv heads,
``D`` head dim):

- ``k_q`` int8   (P, 256, H, D): K smoothed by per-block mean, absmax / 127.
- ``k_sc`` fp32  (P, 8, H):      per-block K scale.
- ``k_mu`` fp16  (P, 8, H, D):   per-block K mean (softmax invariance is
  restored with an explicit q.mu bias, same as the round-1 on-the-fly mode).
- ``v_q`` uint8  (P, 256, H, D): V as raw FP8 E4M3 bits (per-block-per-channel
  scaled, absmax / 448).
- ``v_sc`` fp16  (P, 8, H, D):   per-block-per-channel V scale.
- ``k_stage`` / ``v_stage`` fp16 (P, 32, H, D) + ``stage_len`` int32 (P,):
  the not-yet-quantized tail rows of each page.

Write path (``update_kv_direct``): tokens arrive in arbitrary chunks, but a
32-token block is only quantized once all 32 rows are known. The most recent
partial block of each page is therefore kept in fp16 ``*_stage`` buffers; on
append the staged rows are concatenated with the new rows (all fp16, exact),
complete 32-blocks are quantized in one vectorized torch pass, and the
remainder (< 32 rows) goes back to the stage. Re-quantizing the staged head
costs at most 31 rows per append and is amortized away. Blocks are
page-local (256 % 32 == 0) and block-aligned, so an append never needs values
outside [old_tail - 31, new_end). Partially shared pages (prompt-cache
``copy_page``) stay exact as long as shared prefixes hold identical values,
which is how the paged cache shares pages.

Read paths:

- The cached prefill kernel (``attention_fn/i8f8_cached.py``) reads the
  packed tensors directly through ``get_qkv`` -- no per-tile K/V
  re-quantization. The staged tail (< 32 rows) is handled inside the kernel
  with the same on-the-fly math as round 1, so numerics match round 1
  exactly for full blocks.
- ``get_kv`` dequantizes to fp16 (full pages + stage overlay). Used for
  decode and every declined shape: correct, but with a full-size temporary
  like the quant-cache level-2 path -- the fast path is the cached kernel.

The default fp16 path and the existing quantized cache (``quant.py``) are
untouched; this layer is only selected explicitly
(``Cache(..., layer_type=CacheLayer_i8f8)`` or ``EXL3_CACHE_I8F8=1``).
"""

from __future__ import annotations
from typing_extensions import override
import torch
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention
    from ..model import Model, Config
import numpy as np


QB = 32          # tokens per quantization block (== kernel BLOCK_N)
NB = PAGE_SIZE // QB   # 8 blocks per page

FP8_MAX = 448.0
_I8_MAX = 127.0


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, (x + 0.5).floor(), (x - 0.5).ceil())


def quantize_blocks(kf: torch.Tensor, vf: torch.Tensor):
    # NOTE: plain torch ops on purpose (no Triton): portable across gfx1030 /
    # gfx1201, and the cost is amortized over the prefill (memory-bound once
    # per token, vs per-tile re-quantization in round 1).
    """Quantize ``(Bf, 32, H, D)`` fp16 blocks. Returns
    ``(kq int8, ksc fp32 (Bf,H), kmu fp16 (Bf,H,D), vq uint8 bits, vsc fp16)``.
    Mirrors the round-1 (``int8fp8``) block math exactly: per-block mean
    smoothing, symmetric INT8 K, per-channel E4M3 V, round-half-away."""
    assert kf.shape == vf.shape and kf.shape[1] == QB
    kf32 = kf.float()
    vf32 = vf.float()
    mu = kf32.mean(dim=1)                                  # (Bf,H,D)
    kc = kf32 - mu[:, None]
    sk = kc.abs().amax(dim=(1, 3)).clamp_min(1e-12) / _I8_MAX   # (Bf,H)
    kq = _round_half_away(kc / sk[:, None, :, None]).clamp(-128, 127).to(torch.int8)
    sv = vf32.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX     # (Bf,H,D)
    vq = (vf32 / sv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return kq, sk.to(torch.float32), mu.to(torch.float16), vq, sv.to(torch.float16)


class CacheLayer_i8f8(CacheLayer):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."

        self.shape = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE, attention.num_kv_heads, attention.head_dim)
            if attention else None
        )
        self.num_pages = self.shape[0] if attention else 0
        self.k_q = None
        self.k_sc = None
        self.k_mu = None
        self.v_q = None
        self.v_sc = None
        self.k_stage = None
        self.v_stage = None
        self.stage_len = None
        self.device = None

    @override
    def alloc(self, device: torch.device):
        self.device = device
        if not self.shape:
            return
        P, S, H, D = self.shape
        self.k_q = torch.zeros((P, S, H, D), dtype=torch.int8, device=device)
        self.k_sc = torch.zeros((P, NB, H), dtype=torch.float32, device=device)
        self.k_mu = torch.zeros((P, NB, H, D), dtype=torch.float16, device=device)
        self.v_q = torch.zeros((P, S, H, D), dtype=torch.uint8, device=device)
        self.v_sc = torch.zeros((P, NB, H, D), dtype=torch.float16, device=device)
        self.k_stage = torch.zeros((P, QB, H, D), dtype=torch.float16, device=device)
        self.v_stage = torch.zeros((P, QB, H, D), dtype=torch.float16, device=device)
        self.stage_len = torch.zeros((P,), dtype=torch.int32, device=device)

    @override
    def free(self):
        self.device = None
        self.k_q = None
        self.k_sc = None
        self.k_mu = None
        self.v_q = None
        self.v_sc = None
        self.k_stage = None
        self.v_stage = None
        self.stage_len = None

    def get_qkv(self):
        # Packed tensors for the cached prefill kernel (pure reader; the
        # write already happened in update_kv_direct, quant-direct style).
        return (self.k_q, self.k_sc, self.k_mu, self.v_q, self.v_sc,
                self.k_stage, self.v_stage, self.stage_len)

    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # Quantize-on-write, exact: the staged tail (fp16) + new rows are
        # requantized from fp16, so appending never compounds quantization
        # error. Blocks are page-local and globally 32-aligned, therefore an
        # append only ever touches the head page's stage plus complete
        # blocks. torch ops (no Triton): portable, amortized cost.
        if length <= 0:
            return
        with torch.no_grad():
            bsz = k.shape[0]
            H = k.shape[2]
            for b in range(bsz):
                S = int(cache_seqlens[b].item())
                kb = k[b, :length].contiguous()
                vb = v[b, :length].contiguous()
                phys_h = int(block_table[b, S // PAGE_SIZE].item())
                s = int(self.stage_len[phys_h].item())
                assert 0 <= s < QB
                bufk = torch.cat([self.k_stage[phys_h, :s].float(), kb.float()], dim=0)
                bufv = torch.cat([self.v_stage[phys_h, :s].float(), vb.float()], dim=0)
                n_full = (s + length) // QB
                if n_full > 0:
                    q = bufk[:n_full * QB].to(torch.float16).view(n_full, QB, H, -1)
                    qv = bufv[:n_full * QB].to(torch.float16).view(n_full, QB, H, -1)
                    kq, ksc, kmu, vqb, vsc = quantize_blocks(q, qv)
                    for i in range(n_full):
                        g = S - s + i * QB
                        phys = int(block_table[b, g // PAGE_SIZE].item())
                        sub = (g % PAGE_SIZE) // QB
                        self.k_q[phys, sub * QB:(sub + 1) * QB] = kq[i]
                        self.k_sc[phys, sub] = ksc[i]
                        self.k_mu[phys, sub] = kmu[i]
                        self.v_q[phys, sub * QB:(sub + 1) * QB] = vqb[i]
                        self.v_sc[phys, sub] = vsc[i]
                r = (s + length) % QB
                phys_t = int(block_table[b, (S + length - 1) // PAGE_SIZE].item())
                if r > 0:
                    self.k_stage[phys_t, :r] = bufk[n_full * QB:].to(torch.float16)
                    self.v_stage[phys_t, :r] = bufv[n_full * QB:].to(torch.float16)
                    assert n_full * QB + r == s + length
                self.stage_len[phys_t] = r
                if phys_t != phys_h:
                    self.stage_len[phys_h] = 0
                    # Middle pages (multi-page appends) are fully covered by
                    # complete blocks above, so any stale stage there is dead.
                    pg0, pg1 = S // PAGE_SIZE, (S + length - 1) // PAGE_SIZE
                    for pg in range(pg0 + 1, pg1):
                        self.stage_len[int(block_table[b, pg].item())] = 0

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1):
        # Decode / declined-shape fallback: dequantize to fp16 (full pages +
        # stage overlay). Correct but with a full-size temporary, like the
        # quant-cache level-2 path; the fast path is the cached kernel.
        with torch.no_grad():
            P, S, H, D = self.shape
            sc = self.k_sc.repeat_interleave(QB, dim=1).unsqueeze(-1)   # (P,S,H,1)
            mu = self.k_mu.repeat_interleave(QB, dim=1)                 # (P,S,H,D)
            k = self.k_q.float() * sc + mu.float()
            v = self.v_q.view(torch.float8_e4m3fn).float() * self.v_sc.repeat_interleave(QB, dim=1).float()
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            bsz = cache_seqlens.shape[0]
            for b in range(bsz):
                T = int(cache_seqlens[b].item())
                if T <= 0:
                    continue
                pg = (T - 1) // PAGE_SIZE
                phys = int(block_table[b, pg].item())
                r = int(self.stage_len[phys].item())
                fill = T - pg * PAGE_SIZE
                n = min(r, fill)
                if n > 0:
                    k[phys, fill - n:fill] = self.k_stage[phys, :n]
                    v[phys, fill - n:fill] = self.v_stage[phys, :n]
            return k, v

    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # No-op: every write goes through update_kv_direct (quantize-on-write)
        # before the attention call, quant-direct style.
        pass

    @override
    def copy_page(self, source: CacheLayer_i8f8, from_page: int, to_page: int, num_tokens: int):
        assert self.shape == source.shape
        n_blk = num_tokens // QB
        rem = num_tokens % QB
        self.k_q[to_page, :num_tokens] = source.k_q[from_page, :num_tokens]
        self.v_q[to_page, :num_tokens] = source.v_q[from_page, :num_tokens]
        if n_blk > 0:
            self.k_sc[to_page, :n_blk] = source.k_sc[from_page, :n_blk]
            self.k_mu[to_page, :n_blk] = source.k_mu[from_page, :n_blk]
            self.v_sc[to_page, :n_blk] = source.v_sc[from_page, :n_blk]
        if rem > 0:
            self.k_stage[to_page, :rem] = source.k_stage[from_page, :rem]
            self.v_stage[to_page, :rem] = source.v_stage[from_page, :rem]
        self.stage_len[to_page] = rem

    @override
    def get_tensors(self):
        return [self.k_q, self.k_sc, self.k_mu, self.v_q, self.v_sc,
                self.k_stage, self.v_stage, self.stage_len]

    @override
    def storage_size(self):
        P, S, H, D = self.shape
        return int(
            P * S * H * D * 1 +              # k_q int8
            P * NB * H * 4 +                 # k_sc fp32
            P * NB * H * D * 2 +             # k_mu fp16
            P * S * H * D * 1 +              # v_q fp8
            P * NB * H * D * 2 +             # v_sc fp16
            2 * P * QB * H * D * 2 +         # stages fp16
            P * 4                            # stage_len int32
        )

    @override
    def overhead_size(self):
        return 2 * int(np.prod(self.shape[2:])) * torch.half.itemsize

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_i8f8,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }