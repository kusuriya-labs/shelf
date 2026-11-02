import sys
from pathlib import Path

_VENDOR_QUARK = Path(__file__).resolve().parents[2] / "vendor" / "quark"
if _VENDOR_QUARK.is_dir() and str(_VENDOR_QUARK) not in sys.path:
    sys.path.insert(0, str(_VENDOR_QUARK))

from quark_psq5.artifact import read_artifact, reconstruct, write_artifact
from quark_psq5.codebook import dequantize_weight, quantize_weight
from quark_psq5.config import PSQ5CB28Config
from quark_psq5.processor import PSQ5CB28Processor
from quark_psq5.scale_codec import make_scale_codec
from quark.torch.algorithm import ALGORITHM_REGISTRY, QuarkAlgorithm

PSQ5CB28_ALGORITHM = QuarkAlgorithm(
    name="psq5_cb28",
    algo_config=PSQ5CB28Config,
    algo_processor=PSQ5CB28Processor,
)

ALGORITHM_REGISTRY.register(PSQ5CB28_ALGORITHM)

__all__ = [
    "PSQ5CB28_ALGORITHM",
    "PSQ5CB28Config",
    "PSQ5CB28Processor",
    "dequantize_weight",
    "make_scale_codec",
    "quantize_weight",
    "read_artifact",
    "reconstruct",
    "write_artifact",
]
