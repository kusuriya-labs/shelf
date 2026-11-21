"""
LoRA (Low-Rank Adaptation) support for ExLlamaV3.

Loads PEFT-format LoRA adapters and applies them at runtime without
modifying base model weights. Multiple adapters can be loaded
simultaneously; all loaded adapters are applied during forward pass.

Usage::

    lora = LoRA.from_directory(model, "/path/to/peft-adapter")
    # All generation now includes this adapter's contribution
    response = generator.generate(prompt = "Hello", ...)
    # Unload to revert to base model
    lora.unload()

Compatible with adapters trained via PEFT/Unsloth on the full-precision
base model. LoRA weights are applied on top of the dequantized output
of each target linear layer.
"""

from __future__ import annotations
import os
import json
import math
import re
import torch
from safetensors.torch import load_file as safe_load_file
from ..modules.linear import Linear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model


class LoRA:
    """
    LoRA adapter loaded from PEFT format.

    Stores pre-transposed and pre-scaled A/B weight matrices on target
    Linear modules. During forward pass, Linear.forward() computes:
    ``output += input @ A @ B`` for each loaded adapter.
    """

    @staticmethod
    def from_directory(
            model: Model,
            directory: str,
            lora_scaling: float = 1.0,
            strict: bool = False,
            dtype: torch.dtype = torch.float16,
    ) -> LoRA:
        """
        Load LoRA adapter from a PEFT directory.

        :param model:
            Loaded ExLlamaV3 model instance.

        :param directory:
            Path to directory containing adapter_config.json and
            adapter_model.safetensors (or .bin).

        :param lora_scaling:
            Additional scaling factor applied on top of alpha/r.
        """
        config_path = os.path.join(directory, "adapter_config.json")
        weights_st = os.path.join(directory, "adapter_model.safetensors")
        weights_bin = os.path.join(directory, "adapter_model.bin")

        if os.path.exists(weights_st):
            return LoRA(model, config_path, weights_st, lora_scaling, strict=strict, dtype=dtype)
        if os.path.exists(weights_bin):
            return LoRA(model, config_path, weights_bin, lora_scaling, strict=strict, dtype=dtype)
        raise FileNotFoundError(f"No LoRA adapter found in {directory}")

    @torch.inference_mode()
    def __init__(
            self,
            model: Model,
            config_path: str,
            weights_path: str,
            lora_scaling: float = 1.0,
            *,
            strict: bool = False,
            dtype: torch.dtype = torch.float16,
    ):
        self.target_modules = {}
        self.name = os.path.basename(os.path.dirname(config_path))
        self.enabled = True

        # Read adapter config
        with open(config_path, encoding="utf8") as f:
            config = json.load(f)

        self.lora_r = config["r"]
        self.lora_alpha = float(config["lora_alpha"])

        effective_alpha = self.lora_alpha
        if config.get("use_rslora", False):
            effective_alpha *= math.sqrt(self.lora_r)

        self.lora_scaling = lora_scaling * effective_alpha / self.lora_r

        if config.get("fan_in_fan_out", False):
            raise ValueError("fan_in_fan_out mode is not supported")
        if any(config.get(k) for k in ("use_dora", "lora_bias", "modules_to_save", "rank_pattern", "alpha_pattern")):
            raise ValueError("DoRA, LoRA bias, saved modules and per-module rank/alpha are not supported")
        if getattr(model, "loaded_tp", False):
            raise ValueError("Runtime LoRA requires a single-process model (single GPU or layer split)")

        # Build modules dict if needed
        if model.modules_dict is None:
            model.modules_dict = {m.key: m for m in model}

        # Load weights
        if weights_path.endswith(".safetensors"):
            raw_tensors = safe_load_file(weights_path, device="cpu")
        else:
            raw_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)

        pairs = {}
        skipped_keys = []
        for key, tensor in raw_tensors.items():
            path, half = self._parse_key(key)
            if path is None:
                skipped_keys.append(key)
                continue
            halves = pairs.setdefault(self._canonical_path(path), {})
            if half in halves:
                raise ValueError(f"Duplicate LoRA half: {key}")
            halves[half] = tensor

        # Validate and stage every pair before modifying any module. The VL backbone
        # adds language_model to HF keys; local MLP slices represent portions of a
        # single PEFT projection, not unsupported tensor-parallel ranks.
        staged = []
        for path, halves in pairs.items():
            if set(halves) != {"lora_A", "lora_B"}:
                raise ValueError(f"Incomplete LoRA pair: {path}")
            a, b = halves["lora_A"], halves["lora_B"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"Invalid LoRA shapes: {path}")
            targets = [m for m in model.modules_dict.values() if isinstance(m, Linear)
                       and self._canonical_path(m.alt_key or m.key) == path]
            if not targets:
                skipped_keys.append(path)
                continue
            for target in targets:
                if target.device is None:
                    raise ValueError(f"Load the model before its LoRA: {target.key}")
                if a.shape[1] != target.full_in_features or b.shape[0] != target.full_out_features:
                    # Full dimensions may contain EXL3 zero padding.
                    if not (a.shape[1] <= target.full_in_features and b.shape[0] <= target.full_out_features
                            and a.shape[1] >= target.first_in_feature + target.in_features_unpadded
                            and b.shape[0] >= target.first_out_feature + target.out_features_unpadded):
                        raise ValueError(f"LoRA dimensions do not match {target.key}")
                sa = a[:, target.first_in_feature:target.first_in_feature + target.in_features_unpadded].T
                sb = b[target.first_out_feature:target.first_out_feature + target.out_features_unpadded].T
                sa = torch.nn.functional.pad(sa.to(dtype), (0, 0, 0, target.in_features - sa.shape[0]))
                sb = torch.nn.functional.pad(sb.to(dtype), (0, target.out_features - sb.shape[1]))
                sa = sa.contiguous().to(target.device)
                sb = (sb * self.lora_scaling).contiguous().to(target.device)
                if not bool(torch.isfinite(sa).all()) or not bool(torch.isfinite(sb).all()):
                    raise ValueError(f"Non-finite LoRA weights: {target.key}")
                staged.append((target, sa, sb))
        if strict and skipped_keys:
            raise ValueError(f"Unmatched LoRA weights: {skipped_keys[:8]}")
        if not staged:
            raise ValueError("No matching LoRA projections")
        for target, a, b in staged:
            target.lora_a_tensors[self] = a
            target.lora_b_tensors[self] = b
            self.target_modules[target.key] = target
        loaded = 2 * len(staged)
        self.skipped_keys = skipped_keys

        print(
            f" -- LoRA '{self.name}': loaded {loaded} tensors "
            f"(r={self.lora_r}, alpha={self.lora_alpha:.0f}, "
            f"scaling={self.lora_scaling:.4f})"
        )
        if skipped_keys:
            print(
                f" -- LoRA '{self.name}': skipped {len(skipped_keys)} "
                f"unmatched keys"
            )

    @staticmethod
    def _canonical_path(path: str) -> str:
        path = re.sub(r"\.slice\.\d+$", "", path).replace(".language_model.", ".")
        while path.startswith("base_model."):
            path = path[len("base_model."):]
        while path.startswith("model.model."):
            path = path[len("model."):]
        if path.startswith("model.lm_head"):
            path = path[len("model."):]
        return path

    @staticmethod
    def _parse_key(key: str) -> tuple[str | None, str | None]:
        """
        Parse PEFT tensor key to (full_path, lora_half).

        Returns the full dotted path before lora_A/lora_B and the half
        name. The caller matches this path against model modules by
        suffix, so any PEFT key prefix format is handled automatically.

            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            -> ("base_model.model.model.layers.0.self_attn.q_proj", "lora_A")
        """
        parts = key.split(".")
        for j, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                return ".".join(parts[:j]), p
        return None, None

    def unload(self):
        """Remove this adapter's tensors from all target modules."""
        for target in self.target_modules.values():
            target.lora_a_tensors.pop(self, None)
            target.lora_b_tensors.pop(self, None)

        self.target_modules = {}
