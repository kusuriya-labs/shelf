import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from linear import Projection


def read_json(path):
    return json.loads(path.read_bytes())


class Weights:
    def __init__(self, model, engram, work=None):
        config = read_json(model / "config.json")
        self.config = config.get("text_config", config)
        cfg = self.config
        if (cfg["model_type"] != "hy_v4_flash" or cfg["num_hidden_layers"] != 42
                or cfg["n_routed_experts"] != 512 or cfg["hc_mult"] != 4):
            raise ValueError("対象の Hy4-flash checkpoint が必要です")
        self.index = read_json(model / "model.safetensors.index.json")["weight_map"]
        self.model, self.engram, self.work, self.device = model, engram, work, "cpu"
        self.shapes = {}
        self.local = {}
        if work is None:
            q = cfg["quantization_config"]
            if q["mul1_multiplier"] != 0xDCD12D or q["bits"] not in (3.5,4) or q["head_bits"] != 6:
                raise ValueError("本体 3.5/4 bpw、head 6 bit の mul24 が必要です")
            if not (model / "conversion_report.json").is_file():
                raise ValueError("本体 checkpoint の組立が完了していません")
            self.shapes.update({name[:-7]: item["original_shape"] for name, item in q["linear_storage"].items()})
        else:
            for marker in sorted(work.glob("layer-*/linears/linear-*.json")):
                info = read_json(marker)
                key = info["source"].removesuffix(".weight")
                self.shapes[key] = info["original_shape"]
                self.local[key] = marker.with_suffix(".safetensors")
        self.owners = None
        self.devices = ["cpu"]

    def require_layer(self, layer):
        if self.work is None:
            return
        root = self.work / f"layer-{layer:02d}"
        entries = read_json(root / "capture/capture.json")["entries"]
        missing = [name for name in entries if name.removesuffix(".weight") not in self.shapes]
        if missing or layer > read_json(self.work / "progress.json")["last_layer"]:
            raise ValueError(f"層 {layer} の量子化が未完了です")

    def source(self, key):
        return self.model / self.index[key]

    @lru_cache(maxsize=8192)
    def raw(self, key):
        with safe_open(self.source(key), framework="pt", device="cpu") as f:
            return f.get_tensor(key).to(self.device)

    def tensor(self, name, key, file, device):
        with safe_open(file or self.source(name + "." + key), framework="pt", device="cpu") as f:
            return f.get_tensor(key if file and ".experts." in name else name + "." + key).to(device)

    def projection(self, name, device):
        file = self.local.get(name)
        width = 1536
        if ".experts." in name:
            if self.work:
                parts = name.split(".")
                file = self.work / f"layer-{int(parts[2]):02d}/experts" / (
                    f"expert-{int(parts[5]):04d}-{parts[6].removesuffix('_proj')}.safetensors")
        else:
            width = self.shapes[name][1]
        parts = [self.tensor(name, suffix, file, device) for suffix in ("trellis", "suh", "svh")]
        marker = self.tensor(name, "mul1", file, "cpu").item()
        return Projection(*parts, marker, width)

    def linear(self, name, x):
        if name not in self.shapes and ".experts." not in name:
            weight = self.raw(name + ".weight")
            return F.linear(x.to(weight.dtype), weight)
        return self.projection(name, x.device).cpu(x, fp32=name == "lm_head")

    @lru_cache(maxsize=4)
    def engram_table(self, layer):
        prefix = f"model.layers.{layer}.engram_layer."
        key = prefix + "multi_head_embedding.embedding.weight"
        sizes = np.array(self.config["engram_layer_head_vocab_sizes"][str(layer + 1)], dtype=np.int64).flatten()
        with safe_open(self.engram / (key + ".fp8.safetensors"), framework="pt", device="cpu") as f:
            codes, scales = f.get_tensor(key).view(torch.uint8), f.get_tensor(key + ".scale")
        if tuple(codes.shape) != (int(sizes.sum()), 96) or tuple(scales.shape) != (len(codes), 1):
            raise ValueError("Engram の符号とスケールの形状が不正です")
        lut = self.raw(prefix + "lookup_table").cpu().numpy()
        bound = (2**63-1) // (int(lut.max()) + 1) // 2
        rng = np.random.default_rng(self.config["engram_seed"] + 10007 * (layer + 1))
        mult = rng.integers(0, bound, 3, dtype=np.int64) * 2 + 1
        return codes, scales, sizes, lut, mult

    def lookup(self, layer, history, ids):
        codes, scales, sizes, lut, mult = self.engram_table(layer)
        pad = self.config["engram_pad_id"]
        seq = np.array(([pad, pad] + list(history))[-2:] + list(ids), dtype=np.int64)
        seq = np.where(seq < len(lut), seq, pad)
        compressed = lut[seq].astype(np.int64)
        pair = compressed[2:] * mult[0] ^ compressed[1:-1] * mult[1]
        triple = pair ^ compressed[:-2] * mult[2]
        mixed = np.stack((pair, triple), -1).repeat(8, axis=-1)
        rows = torch.from_numpy(mixed % sizes + sizes.cumsum() - sizes)
        return (codes[rows].view(torch.float8_e4m3fn).bfloat16() * scales[rows]).to(self.device)
