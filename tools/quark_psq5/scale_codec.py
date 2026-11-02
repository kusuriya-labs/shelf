import torch

UE4M3_FOLD = 512.0
UE4M3_MAX = 0x7E
UE4M4_MAX = 0xEF


def ue4m3_value(byte: int) -> float:
    exponent = byte >> 3
    mantissa = byte & 7
    if exponent == 0:
        return mantissa / 1024.0
    if exponent >= 11:
        return float((8 + mantissa) << (exponent - 11))
    return (8 + mantissa) / float(1 << (11 - exponent))


def ue4m4_value(byte: int) -> float:
    exponent = byte >> 4
    mantissa = byte & 0xF
    if exponent == 0:
        return mantissa / 1024.0
    if exponent == 15:
        return float("nan")
    return (1.0 + mantissa / 16.0) * (2.0 ** (exponent - 7))


def ue4m3_table() -> torch.Tensor:
    return torch.tensor([ue4m3_value(b) for b in range(UE4M3_MAX + 1)], dtype=torch.float32)


def ue4m4_table() -> torch.Tensor:
    return torch.tensor([ue4m4_value(b) for b in range(UE4M4_MAX + 1)], dtype=torch.float32)


class ByteScaleCodec:
    name = ""
    storage_dtype = torch.uint8
    storage_bytes = 1

    def __init__(self, name: str, values: torch.Tensor, fold: float) -> None:
        self.name = name
        self.fold = fold
        self._values = values
        self._positive = values[1:]

    def encode(self, scale: torch.Tensor) -> torch.Tensor:
        target = scale.to(torch.float32) * self.fold
        codes = torch.zeros(target.shape, dtype=torch.int64, device=target.device)
        positive = target > 0
        if positive.any():
            values = self._positive
            upper = torch.searchsorted(values, target[positive].contiguous()).clamp(max=len(values) - 1)
            lower = (upper - 1).clamp(min=0)
            head = target[positive]
            take_lower = (head - values[lower]).abs() <= (head - values[upper]).abs()
            codes[positive] = torch.where(take_lower, lower, upper) + 1
        return codes.to(torch.uint8)

    def decode(self, stored: torch.Tensor) -> torch.Tensor:
        table = torch.zeros(256, dtype=torch.float32, device=stored.device)
        table[: len(self._values)] = self._values.to(stored.device)
        return table[stored.to(torch.int64)] / self.fold


class UE4M3Codec(ByteScaleCodec):
    def __init__(self, fold: float = UE4M3_FOLD) -> None:
        super().__init__("ue4m3", ue4m3_table(), fold)


class UE4M4Codec(ByteScaleCodec):
    def __init__(self, fold: float = UE4M3_FOLD) -> None:
        super().__init__("ue4m4", ue4m4_table(), fold)


class FP16Codec:
    name = "fp16"
    storage_dtype = torch.float16
    storage_bytes = 2
    max_value = 65504.0

    def __init__(self, fold: float = UE4M3_FOLD) -> None:
        self.fold = fold

    def encode(self, scale: torch.Tensor) -> torch.Tensor:
        target = torch.minimum(scale.to(torch.float32) * self.fold,
                               torch.tensor(self.max_value, dtype=torch.float32))
        return target.to(torch.float16)

    def decode(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.to(torch.float32) / self.fold


class BF16Codec:
    name = "bf16"
    storage_dtype = torch.bfloat16
    storage_bytes = 2

    def __init__(self, fold: float = UE4M3_FOLD) -> None:
        self.fold = fold

    def encode(self, scale: torch.Tensor) -> torch.Tensor:
        return (scale.to(torch.float32) * self.fold).to(torch.bfloat16)

    def decode(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.to(torch.float32) / self.fold


def make_scale_codec(name: str, fold: float):
    key = name.lower()
    if key == "ue4m3":
        return UE4M3Codec(fold)
    if key == "ue4m4":
        return UE4M4Codec(fold)
    if key == "fp16":
        return FP16Codec(fold)
    if key == "bf16":
        return BF16Codec(fold)
    raise ValueError(f"unsupported PSQ5 scale format: {name}")
