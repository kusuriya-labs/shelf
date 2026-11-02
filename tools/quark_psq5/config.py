from dataclasses import dataclass, field

from quark.torch.quantization.config.config import AlgoConfig


@dataclass
class PSQ5CB28Config(AlgoConfig):
    name: str = "psq5_cb28"
    block_k: int = 32
    scale_format: str = "ue4m4"
    scale_fold: float = 512.0
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    output_path: str = ""
