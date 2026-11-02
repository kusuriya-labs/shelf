import fnmatch

import torch
import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark_psq5.artifact import build_manifest, stream_keys, write_artifact
from quark_psq5.codebook import quantize_weight
from quark_psq5.scale_codec import make_scale_codec
from quark.torch.algorithm.processor import BaseAlgoProcessor

logger = ScreenLogger(__name__)

BUFFER_NAMES = ("psq5_magnitude", "psq5_sign", "psq5_scale")


def selected(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(fnmatch.fnmatch(name, pattern) for pattern in include):
        return False
    return not any(fnmatch.fnmatch(name, pattern) for pattern in exclude)


class PSQ5CB28Processor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, quant_algo_config, calib_data=None) -> None:
        self.model = model
        self.config = quant_algo_config
        self.calib_data = calib_data
        self.processed: list[str] = []
        self.skipped: list[str] = []

    @torch.no_grad()
    def apply(self) -> None:
        codec = make_scale_codec(self.config.scale_format, self.config.scale_fold)
        entries: dict = {}
        streams: dict = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not selected(name, self.config.include, self.config.exclude):
                self.skipped.append(name)
                continue
            result = quantize_weight(module.weight.detach(), codec, self.config.block_k)
            self._attach(module, result)
            weight = module.weight
            weight.copy_(result["reconstruction"].to(device=weight.device, dtype=weight.dtype))
            keys = stream_keys(name)
            for field in ("magnitude", "sign", "scale"):
                streams[keys[field]] = result[field]
            entries[name] = {
                "weight_shape": list(result["shape"]),
                "block_k": result["block_k"],
                "source_dtype": str(weight.dtype).replace("torch.", ""),
            }
            self.processed.append(name)
        if not self.processed:
            logger.warning("[psq5_cb28] no nn.Linear matched include/exclude")
        else:
            logger.info(f"[psq5_cb28] quantized {len(self.processed)} linears")
        if self.config.output_path:
            write_artifact(self.config.output_path, streams,
                           build_manifest(entries, self.config, codec))
            logger.info(f"[psq5_cb28] wrote artifact {self.config.output_path}")

    @staticmethod
    def _attach(module: nn.Module, result: dict) -> None:
        tensors = (result["magnitude"], result["sign"], result["scale"])
        for buffer_name, tensor in zip(BUFFER_NAMES, tensors):
            if buffer_name in module._buffers:
                del module._buffers[buffer_name]
            module.register_buffer(buffer_name, tensor.to(module.weight.device), persistent=False)
