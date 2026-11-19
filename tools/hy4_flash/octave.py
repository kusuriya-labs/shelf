import math
from functools import lru_cache

import torch

from linear import hadamard


@lru_cache(maxsize=4)
def levels(device):
    v = [-2.5 + 5 * (i + 0.5) / 16 for i in range(16)]
    for _ in range(4000):
        limits = [-12] + [(a + b) / 2 for a, b in zip(v, v[1:])] + [12]
        prob = [math.erfc(-x / math.sqrt(2)) / 2 for x in limits]
        pdf = [math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) for x in limits]
        new = [(pdf[i] - pdf[i+1]) / (prob[i+1] - prob[i]) for i in range(16)]
        if max(abs(a - b) for a, b in zip(v, new)) < 1e-13:
            break
        v = new
    return torch.tensor(v, dtype=torch.float32, device=device)


@lru_cache(maxsize=16)
def signs(width, salt, device):
    x = (torch.arange(width, dtype=torch.int64, device=device) * 2654435761 + salt * 40503 + 12345) & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
    return 1.0 - 2.0 * ((x ^ (x >> 16)) & 1)


def encode(x, width, salt):
    v = hadamard(x.float().reshape(-1, width) * signs(width, salt, x.device), width)
    energy = v.square().sum(-1, keepdim=True)
    table = levels(x.device)
    codes = torch.bucketize(v * (energy / width).clamp_min(1e-30).rsqrt(),
                            (table[1:] + table[:-1]) / 2)
    dot = (v * table[codes]).sum(-1, keepdim=True)
    scale = torch.where(dot > 0, energy / dot, 0).half()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).byte()
    return torch.cat((packed, scale.view(torch.uint8)), dim=-1)


def decode(stored, width, salt):
    packed = stored[..., :width // 2]
    code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    scale = stored[..., -2:].contiguous().view(torch.float16)
    v = levels(stored.device)[code.long()] * scale
    return (hadamard(v, width) * signs(width, salt, stored.device)).bfloat16()


class Cache:
    def __init__(self, capacity, sliding, device):
        self.capacity = capacity
        self.sliding = sliding
        self.slots = min(518, capacity) if sliding else capacity
        self.kv = torch.empty(self.slots, 294, device=device, dtype=torch.uint8)
        if not sliding:
            self.groups = torch.empty(capacity // 4, 128, device=device, dtype=torch.bfloat16)
            self.pending = torch.empty(12, 128, device=device, dtype=torch.bfloat16)
        self.end = 0

    def append(self, latent, rotary):
        if self.end >= self.capacity:
            raise ValueError("KV の予約長を超えました")
        record = torch.cat((encode(latent, 256, 29).reshape(-1), encode(rotary, 64, 11).reshape(-1)))
        self.kv[self.end % self.slots] = record
        self.end += 1

    def get(self, positions):
        packed = self.kv[positions % self.slots]
        latent = decode(packed[:, :260].reshape(-1, 130), 256, 29).reshape(-1, 512)
        return latent, decode(packed[:, 260:], 64, 11)

    def rewind(self, end):
        if not self.end - 6 <= end <= self.end or end < 0:
            raise ValueError("DFly の巻き戻し位置が不正です")
        self.end = end


def selected_tokens(group_scores, position):
    device = group_scores.device
    if position < 2048:
        return torch.arange(position + 1, device=device)
    complete, partial = divmod(position, 4)
    groups = group_scores[:complete].topk(511).indices
    past = (groups[:, None] * 4 + torch.arange(4, device=device)).flatten()
    return torch.cat((past, complete * 4 + torch.arange(partial + 1, device=device)))
