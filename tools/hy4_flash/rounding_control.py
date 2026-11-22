from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

import decoder
from linear import Projection, hadamard, unpack


def wave_sum(x):
    while x.size(-1) > 1:
        left, right = x.chunk(2, dim=-1)
        x = left + right
    return x.squeeze(-1)


def block_sum(x):
    return wave_sum(wave_sum(x.reshape(*x.shape[:-1], 8, 32)))


def lanes(x):
    return F.pad(x.float(), (0, -x.size(-1) % 256)).reshape(*x.shape[:-1], -1, 256)


def inverse_norm(x, eps):
    vectors = lanes(x)
    acc = torch.zeros_like(vectors[..., 0, :])
    for v in vectors.unbind(-2):
        acc.addcmul_(v, v)
    return (block_sum(acc).unsqueeze(-1) / x.size(-1) + eps).rsqrt()


def normalized(x, weight=None, eps=1e-5):
    value = (x.float() * inverse_norm(x, eps)).to(x.dtype)
    return value if weight is None else value * weight


def collapse(x, fn, scale, base):
    flat = x.flatten(1)
    acc = torch.zeros(len(x), len(fn), 256)
    for v, weight in zip(lanes(flat).unbind(-2), lanes(fn).unbind(-2)):
        acc.addcmul_(v[:, None], weight[None])
    mix = (block_sum(acc) * inverse_norm(flat, 1e-5)).reshape(len(x), -1, 4)
    logits = mix * scale.float().reshape(1, -1, 1) + base.float().reshape(1, -1, 4)
    denom = 1 + (-logits).exp()
    pre = 1 / denom[:, 0] + 1e-6
    out = torch.zeros_like(x[:, 0], dtype=torch.float32)
    for stream in range(4):
        out += pre[:, stream, None] * x[:, stream].float()
    return out.to(x.dtype), 2 / denom[:, 1] + 1e-6 if denom.size(1) > 1 else None


def projection(matrix, x, fp32=False):
    k, n = matrix.suh.numel(), matrix.svh.numel()
    groups = k // 128
    padded = F.pad(x.float(), (0, k - x.size(1)))
    rotated = hadamard(padded * matrix.suh, 128).reshape(len(x), groups, 128)
    scales = rotated.abs().amax(-1, keepdim=True) / 127
    codes = (rotated / torch.where(scales == 0, 1, scales)).round().clamp(-128, 127).transpose(0, 1).contiguous()
    partial = torch.empty(len(x), n)
    for first in range(0, n, 512):
        w = unpack(matrix.trellis[:, first//16:(first+512)//16].contiguous())
        tiles = w.float().reshape(len(w), groups, 128).permute(1, 2, 0)
        if k == n == 1536:
            left = codes.reshape(groups, len(x), 8, 16).permute(0, 2, 1, 3)
            dots = left @ tiles.reshape(groups, 8, 16, len(w))
            slices = torch.zeros(8, len(x), len(w))
            for g in range(groups):
                slices.addcmul_(dots[g], scales[:, g])
        else:
            dots = torch.bmm(codes, tiles)
            splits = 1
            if not fp32:
                while splits < 16 and n//64 * splits < 128 and groups % (2*splits) == 0:
                    splits *= 2
            slices = torch.zeros(splits, len(x), len(w))
            for g in range(groups):
                slices[g // (groups//splits)].addcmul_(dots[g], scales[:, g])
        acc = slices[0].clone()
        for part in slices[1:]:
            acc += part
        partial[:, first:first+len(w)] = acc
    result = hadamard(partial * (4*887/131072), 128) * matrix.svh
    return result[:, :matrix.width] if fp32 else result[:, :matrix.width].bfloat16()


@contextmanager
def rounding_control():
    linear = F.linear

    def router(x, w, bias=None):
        if w.shape != (512, 3072) or x.dtype != torch.float32:
            return linear(x, w, bias)
        acc = torch.zeros(len(x), 512, 32)
        for a, b in zip(x.reshape(-1, 96, 32).unbind(1), w.reshape(512, 96, 32).unbind(1)):
            acc.addcmul_(a[:, None], b[None])
        return wave_sum(acc)

    def rms_op(x, shape, weight=None, eps=None):
        return normalized(x, weight, torch.finfo(torch.float32).eps if eps is None else eps)

    with ExitStack() as stack:
        for module, name, fn in ((Projection, "cpu", projection), (decoder, "rms", normalized),
                                 (decoder, "collapse", collapse), (F, "linear", router), (F, "rms_norm", rms_op)):
            stack.enter_context(patch.object(module, name, fn))
        yield
