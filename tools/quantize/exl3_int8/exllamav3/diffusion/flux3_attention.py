"""Query-sliced exact FLUX.3 attention for GPUs without a fused SDPA kernel."""

import torch
from torch.nn import functional as F


class Flux3QuerySlicedAttnProcessor:
    """Bound the score matrix's query dimension, preserving the full key set.

    Uses the pinned Diffusers projection/RoPE helpers and the same FP32 QK norm
    recipe. There is no key-window approximation or change to softmax's domain.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, query_chunk_size=1024):
        if query_chunk_size < 1:
            raise ValueError("Attention query chunk must be positive")
        self.query_chunk_size = query_chunk_size

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux3 import _get_qkv_projections, apply_rotary_emb

        if self._parallel_config is not None:
            raise ValueError("Query-sliced attention does not support context parallelism")
        projections = _get_qkv_projections(attn, hidden_states)
        query, key, value = [p.unflatten(-1, (attn.heads, -1)) for p in projections]
        query, key = attn.norm_q(query.float()), attn.norm_k(key.float())
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query = query.to(value.dtype).transpose(1, 2)
        key = key.to(value.dtype).transpose(1, 2)
        value = value.transpose(1, 2)
        result = torch.empty_like(query)
        for start in range(0, query.shape[-2], self.query_chunk_size):
            end = min(start + self.query_chunk_size, query.shape[-2])
            mask = attention_mask
            if mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            result[..., start:end, :] = F.scaled_dot_product_attention(
                query[..., start:end, :], key, value, attn_mask=mask, dropout_p=0.0,
            )
        result = result.transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](result))


def configure_flux3_attention(model, device, query_chunk_size=None):
    """Use native attention by default; bound queries automatically on gfx10."""
    if query_chunk_size is None:
        properties = torch.cuda.get_device_properties(device)
        query_chunk_size = 1024 if torch.version.hip and properties.gcnArchName.startswith("gfx10") else 0
    if query_chunk_size < 0:
        raise ValueError("Attention query chunk must be nonnegative")
    if query_chunk_size:
        model.set_attn_processor(Flux3QuerySlicedAttnProcessor(query_chunk_size))
    else:
        from diffusers.models.transformers.transformer_flux3 import Flux3AttnProcessor

        model.set_attn_processor(Flux3AttnProcessor())
    return query_chunk_size
