"""Calibrated conversion of a diffusion Linear using the existing EXL3 encoder."""

import torch
from torch import nn

from .linear import EXL3DiffusionLinear


@torch.inference_mode()
def quantize_linear(layer: nn.Linear, samples: torch.Tensor, *, bits=4, device="cuda:0",
                    seed=0, workspace_bytes=256 * 1024**2):
    """Convert one layer from representative inputs, preserving its BF16 bias.

    Samples must come from the layer's real input across denoising timesteps.
    The caller owns collecting/saving calibration provenance. The returned
    statistics are local reconstruction diagnostics, not a generation-quality
    evaluation. This function does not retain the original layer or samples.
    """
    from ..modules.quant.exl3_lib.quantize import quantize_exl3

    if not isinstance(layer, nn.Linear):
        raise TypeError("Expected torch.nn.Linear")
    k, n = layer.in_features, layer.out_features
    if k % 128 or n % 128 or min(k, n) == 0:
        raise ValueError("Only positive 128-aligned Linear dimensions are supported")
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("EXL3 bits must be an integer from 1 through 8")
    if samples.ndim != 2 or samples.shape[1] != k or samples.shape[0] == 0:
        raise ValueError("Calibration samples must have nonzero rows and in_features columns")
    if not torch.isfinite(samples).all() or not torch.isfinite(layer.weight).all():
        raise ValueError("Nonfinite weights or calibration activations")
    device = torch.device(device)
    h = torch.zeros(k, k, dtype=torch.float32, device=device)
    for chunk in samples.split(512):
        x = chunk.to(device=device, dtype=torch.float32)
        h.addmm_(x.T, x)
    if not torch.isfinite(h).all() or h.diagonal().mean().item() <= 1e-20:
        raise ValueError("Calibration has no usable finite activation energy")
    h_data = {
        "H": h, "count": samples.shape[0], "num_total": samples.numel(),
        "finalized": False, "first_key": "diffusion_linear", "device": device,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
    }
    quant_args = {"K": bits, "devices": [device], "seed": seed, "apply_out_scales": None}
    _, proxy_error, tensors = quantize_exl3(
        layer.weight.detach().T.to(device=device, dtype=torch.float32).contiguous(),
        h_data, quant_args, return_weight_q=False,
    )
    if h_data["q_fallback"]:
        raise RuntimeError("EXL3 quantizer fell back to uncalibrated conversion")
    if not torch.isfinite(tensors["suh"]).all() or not torch.isfinite(tensors["svh"]).all():
        raise RuntimeError("EXL3 scale is outside representable FP16 range")
    bias = None if layer.bias is None else layer.bias.detach().to(device).clone()
    converted = EXL3DiffusionLinear(k, n, tensors["trellis"], tensors["suh"], tensors["svh"],
                                   bias, workspace_bytes=workspace_bytes)
    stats = {
        "bits": bits, "in_features": k, "out_features": n,
        "calibration_rows": samples.shape[0], "proxy_error": float(proxy_error),
        "source_weight_bytes": layer.weight.numel() * layer.weight.element_size(),
        "quantized_state_bytes": sum(t.numel() * t.element_size() for t in converted.state_dict().values()),
    }
    return converted, stats
