import torch

CB28 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28)
CB28_MAX = 28.0
BLOCK_K = 32
MAGNITUDE_BYTES_PER_BLOCK = 16
SIGN_BYTES_PER_BLOCK = 4


def cb28_to_e4m3(q5: torch.Tensor) -> torch.Tensor:
    magnitude = q5 & 0x0F
    sign = (q5 & 0x10) << 3
    mantissa_field = magnitude + ((magnitude >> 3) * (magnitude & 0x07))
    return mantissa_field | sign


def cb28_table(reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(CB28, device=reference.device, dtype=reference.dtype)


def nearest_index(absolute: torch.Tensor) -> torch.Tensor:
    table = cb28_table(absolute)
    upper = torch.searchsorted(table, absolute.contiguous()).clamp(max=len(CB28) - 1)
    lower = (upper - 1).clamp(min=0)
    take_lower = (absolute - table[lower]).abs() <= (absolute - table[upper]).abs()
    return torch.where(take_lower, lower, upper)


def pack_magnitude(index: torch.Tensor) -> torch.Tensor:
    even = index[..., 0::2] & 0x0F
    odd = index[..., 1::2] & 0x0F
    return (even | (odd << 4)).to(torch.uint8)


def unpack_magnitude(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    return torch.stack((low, high), dim=-1).flatten(-2)


def pack_sign(sign: torch.Tensor) -> torch.Tensor:
    bits = sign.reshape(*sign.shape[:-1], 4, 8).to(torch.int32)
    weights = 1 << torch.arange(8, device=sign.device, dtype=torch.int32)
    return (bits * weights).sum(-1).to(torch.uint8)


def unpack_sign(packed: torch.Tensor) -> torch.Tensor:
    weights = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits = ((packed.to(torch.int32).unsqueeze(-1) >> weights) & 1) != 0
    return bits.flatten(-2)




def quantize_weight(weight: torch.Tensor, scale_codec, block_k: int = BLOCK_K) -> dict:
    rows, columns = weight.shape
    padding = (-columns) % block_k
    work = weight.to(torch.float32)
    if padding:
        work = torch.nn.functional.pad(work, (0, padding))
    blocks = work.reshape(rows, -1, block_k)
    scale = blocks.abs().amax(dim=-1) / CB28_MAX
    stored_scale = scale_codec.encode(scale)
    restored_scale = scale_codec.decode(stored_scale)
    expanded = restored_scale.unsqueeze(-1)
    active = expanded > 0
    normalized = torch.where(active, blocks / expanded.clamp(min=torch.finfo(torch.float32).tiny),
                             torch.zeros_like(blocks))
    index = nearest_index(normalized.abs())
    sign = blocks < 0
    table = cb28_table(blocks)
    reconstruction = torch.where(sign, -table[index], table[index]) * expanded
    return {
        "magnitude": pack_magnitude(index),
        "sign": pack_sign(sign),
        "scale": stored_scale,
        "reconstruction": reconstruction.reshape(rows, -1)[:, :columns],
        "shape": (rows, columns),
        "block_k": block_k,
    }


def dequantize_weight(magnitude: torch.Tensor, sign: torch.Tensor, stored_scale: torch.Tensor,
                      scale_codec, rows: int, columns: int, block_k: int = BLOCK_K) -> torch.Tensor:
    index = unpack_magnitude(magnitude)
    sign_bits = unpack_sign(sign)
    expanded = scale_codec.decode(stored_scale).unsqueeze(-1)
    table = cb28_table(expanded)
    values = torch.where(sign_bits, -table[index], table[index]) * expanded
    return values.reshape(rows, -1)[:, :columns]
