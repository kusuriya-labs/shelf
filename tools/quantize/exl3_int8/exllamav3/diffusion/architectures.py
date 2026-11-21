"""Explicit transformer families and EXL3 block selection policies."""

import torch


BLOCK_PREFIXES = {
    "QwenImage21Transformer2DModel": ("transformer_blocks.",),
    "Flux3Transformer2DModel": ("mode_blocks.", "txt_mode_blocks.", "single_transformer_blocks."),
}


def transformer_name(config):
    name = config.get("_class_name", "QwenImage21Transformer2DModel")
    if name not in BLOCK_PREFIXES:
        raise ValueError(f"Unsupported EXL3 diffusion transformer: {name}")
    return name


def transformer_class(config):
    import diffusers

    return getattr(diffusers, transformer_name(config))


def block_linears(model):
    # Also works for small nn.Module graphs used by the calibration tests.
    prefixes = BLOCK_PREFIXES.get(type(model).__name__, ("transformer_blocks.",))
    return {name: layer for name, layer in model.named_modules()
            if name.startswith(prefixes) and isinstance(layer, torch.nn.Linear)
            and layer.in_features % 128 == 0 and layer.out_features % 128 == 0}
