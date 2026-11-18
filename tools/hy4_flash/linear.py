import ctypes
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


def hadamard(x, width):
    values = x.reshape(-1, width)
    stride = 1
    while stride < width:
        a, b = values.reshape(-1, 2, stride).unbind(1)
        values = torch.stack((a + b, a - b), dim=1)
        stride *= 2
    return values.reshape(x.shape) * width ** -0.5


@lru_cache(maxsize=2)
def decode_table(bits):
    t = np.arange(256, dtype=np.uint32)
    bit = ((t + 1) * bits + 256 * bits - 16) % (256 * bits)
    states = np.arange(65536, dtype=np.uint32) * np.uint32(0xDCD12D)
    byte_sum = sum((states >> shift) & 255 for shift in (0, 8, 16, 24)).astype(np.int32)
    order = np.argsort((t // 8 % 16) * 16 + (t // 128) * 8 + (t % 8))
    return ((byte_sum - 509) >> 2).astype(np.int8), bit[order]


def unpack(tiles):
    kt, nt, halfwords = tiles.shape
    bits = halfwords // 16
    lut, bit = decode_table(bits)
    words = tiles.numpy().view(np.uint32)
    result = np.empty((nt, 16, kt, 16), dtype=np.int8)
    for first in range(0, kt, 4):
        part = words[first:first + 4]
        top = part[:, :, bit // 32].astype(np.uint64) << 32
        pair = top | part[:, :, (bit // 32 + 1) % (8 * bits)]
        codes = lut[(pair >> (48 - bit % 32)) & 65535]
        result[:, :, first:first + len(part)] = codes.reshape(len(part), nt, 16, 16).transpose(1, 2, 0, 3)
    return torch.from_numpy(result.reshape(nt * 16, kt * 16))


class Projection:
    def __init__(self, trellis, suh, svh, marker, width):
        k, n, size = trellis.shape
        self.bits = size // 16
        if (marker != 0xDCD12D or self.bits not in (4, 6) or size != self.bits * 16
                or k * 16 != suh.numel() or n * 16 != svh.numel()
                or k % 8 or n % 8 or not 0 < width <= n * 16 or n * 16 - width >= 128
                or suh.dtype != torch.float16 or svh.dtype != torch.float16 or trellis.dtype != torch.int16):
            raise ValueError("mul24 行列の形状・型・乗数が不正です")
        self.trellis, self.suh, self.svh, self.width = trellis, suh, svh, width

    def input_codes(self, x):
        k, n = self.suh.numel(), self.svh.numel()
        padded = F.pad(x.float(), (0, k - x.shape[-1]))
        rotated = hadamard(padded * self.suh, 128).reshape(-1, k // 128, 128)
        scales = rotated.abs().amax(-1, keepdim=True) / 127
        codes = (rotated / torch.where(scales == 0, 1, scales)).round()
        return (codes * scales).reshape(-1, k)

    def cpu(self, x, fp32=False):
        k, n = self.suh.numel(), self.svh.numel()
        a = self.input_codes(x)
        partial = torch.empty(len(a), n)
        for first in range(0, n, 512):
            w = unpack(self.trellis[:, first // 16:(first + 512) // 16].contiguous())
            partial[:, first:first + 512] = a @ w.float().T
        y = hadamard(partial * (4 * 887 / 131072), 128) * self.svh
        return y[:, :self.width] if fp32 else y[:, :self.width].bfloat16()


class HipLinear:
    def __init__(self, library, wmma=False):
        self.wmma = wmma
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.shelf_hy4_linear
        self.launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 6 + [ctypes.c_void_p, ctypes.c_int]
        self.launch.restype = ctypes.c_int

    def __call__(self, w, x):
        if x.dtype != torch.bfloat16 or x.device != w.trellis.device or x.ndim != 2:
            raise ValueError("線形入力の型・形状・device が不正です")
        rows, width = x.shape
        k, n = w.suh.numel(), w.svh.numel()
        with torch.cuda.device(x.device):
            x = x.contiguous()
            codes = torch.empty(rows, k, device=x.device, dtype=torch.int8)
            scales = torch.empty(rows, k // 128, device=x.device, dtype=torch.float32)
            partial = torch.empty(rows, n, device=x.device, dtype=torch.float32)
            y = torch.empty(rows, w.width, device=x.device, dtype=torch.bfloat16)
            tensors = (x, w.suh, w.svh, w.trellis, codes, scales, partial, y)
            result = self.launch(*(t.data_ptr() for t in tensors), rows, width, w.width, k, n,
                                 w.bits, torch.cuda.current_stream().cuda_stream, int(self.wmma))
            if result:
                raise RuntimeError(f"Hy4 HIP 線形を起動できません: {result}")
            return y
