"""Local expert-id mapping for MoE routing.

Global router picks are remapped to this module's local expert slice, with
out-of-range picks folded into a trailing sentinel bucket (id == num_ex).
The sentinel bucket is counted by ``torch.bincount`` but never iterated, so
invalid picks contribute exact zeros to the routed sum.
"""
from __future__ import annotations
import torch


def localize_expert_ids(
    flat_expert_global: torch.Tensor,
    num_local_experts: int | None,
    num_experts: int,
    routing_first: int | None,
) -> torch.Tensor:
    """Map flattened global expert picks to local ids + sentinel bucket."""
    num_ex = num_local_experts or num_experts
    if num_local_experts == num_experts:
        flat_expert_local = flat_expert_global
    else:
        flat_expert_local = flat_expert_global - routing_first
    # Out-of-range picks (negative or >= num_ex) fold into the trailing
    # sentinel bucket, where they contribute exact zeros downstream. The
    # replicated branch used to pass them through, which crashed
    # torch.bincount on negative ids (KLD dump, en_prose_129).
    valid = (flat_expert_local >= 0) & (flat_expert_local < num_ex)
    return torch.where(
        valid, flat_expert_local, torch.full_like(flat_expert_local, num_ex))
