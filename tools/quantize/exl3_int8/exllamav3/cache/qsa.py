from __future__ import annotations
from typing_extensions import override
import numpy as np
import os
import torch
from .cache import CacheLayer
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant
from ..constants import PAGE_SIZE


class QSAPlanes:
    """
    FP16 QSA side planes: pooled keys for every block and a small raw ring per
    page for unfinished pools / speculative rollback. New projections feed
    pooling directly before the ring is updated. Completed historical pools
    retain no full raw-key history. PAGE_SIZE is divisible by the pool and ring
    widths, so page sharing, defragmentation and CPU paging move both planes.
    EXL3_QSA_FULL_RAW=1 restores the original full raw plane for comparisons.
    """

    def _init_planes(self, attention, max_num_tokens: int, raw_history: int = 0, raw_rows: int | None = None):
        idx = attention.qsa_indexer
        assert idx is not None
        self.index_head_dim = idx.head_dim
        self.compress_ratio = idx.compress_ratio
        assert PAGE_SIZE % self.compress_ratio == 0
        num_pages = max_num_tokens // PAGE_SIZE
        if raw_rows is None:
            # Keep the unfinished pool plus the speculative rollback frontier.
            # Full pages are reused only at pool boundaries; completed pools
            # never need their raw keys again. The legacy plane is an A/B switch.
            need = max(16, raw_history + self.compress_ratio)
            raw_rows = (PAGE_SIZE if os.environ.get("EXL3_QSA_FULL_RAW") == "1"
                        else min(PAGE_SIZE, 1 << (need - 1).bit_length()))
        assert 0 < raw_rows <= PAGE_SIZE and PAGE_SIZE % raw_rows == 0
        assert raw_rows % self.compress_ratio == 0
        self.raw_rows = raw_rows
        self.raw_k_shape = (num_pages, raw_rows, self.index_head_dim)
        self.pooled_shape = (num_pages, PAGE_SIZE // self.compress_ratio, self.index_head_dim)
        self.raw_k = None
        self.pooled = None

    @override
    def alloc(self, device: torch.device):
        super().alloc(device)
        self.raw_k = torch.zeros(self.raw_k_shape, dtype = torch.half, device = device)
        self.pooled = torch.zeros(self.pooled_shape, dtype = torch.half, device = device)

    @override
    def free(self):
        super().free()
        self.raw_k = None
        self.pooled = None

    @override
    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        assert self.raw_rows == source.raw_rows, "QSA page copies need matching raw layouts"
        assert self.raw_rows == PAGE_SIZE or num_tokens % self.compress_ratio == 0, \
            "Compact QSA partial prefix copies must end at a pool boundary"
        super().copy_page(source, from_page, to_page, num_tokens)
        if self.raw_rows == PAGE_SIZE:
            self.raw_k[to_page, :num_tokens].copy_(source.raw_k[from_page, :num_tokens], non_blocking = True)
        else:
            self.raw_k[to_page].copy_(source.raw_k[from_page], non_blocking = True)
        nb = (num_tokens + self.compress_ratio - 1) // self.compress_ratio
        self.pooled[to_page, :nb].copy_(source.pooled[from_page, :nb], non_blocking = True)

    @override
    def get_tensors(self):
        return super().get_tensors() + [self.raw_k, self.pooled]

    @override
    def storage_size(self):
        return super().storage_size() + \
            (np.prod(self.raw_k_shape) + np.prod(self.pooled_shape)) * torch.half.itemsize


class CacheLayer_qsa(QSAPlanes, CacheLayer_fp16):
    """fp16 KV cache layer with the QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        raw_history: int = 0,
        raw_rows: int | None = None,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        self._init_planes(attention, max_num_tokens, raw_history, raw_rows)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "raw_rows": self.raw_rows,
            }
        }


class CacheLayer_qsa_quant(QSAPlanes, CacheLayer_quant):
    """Quantized KV cache layer (CacheLayer_quant packing, read online by the dense and the
    gathered sparse attention kernels) with the fp16 QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
        compand_a: float = 0.0,
        raw_history: int = 0,
        raw_rows: int | None = None,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a)
        self._init_planes(attention, max_num_tokens, raw_history, raw_rows)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa_quant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "raw_rows": self.raw_rows,
            }
        }
