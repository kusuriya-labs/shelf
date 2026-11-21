import argparse
import json
import os
import struct
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


def encode(rows):
    if not torch.isfinite(rows).all():
        raise ValueError("非有限の Engram 重みがあります")
    scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    bits = (rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return bits, scale


def convert(source, key, shape, directory, chunk=65536):
    target = directory / (key + ".fp8.safetensors")
    if target.exists():
        raise FileExistsError(target)
    rows, width = shape
    n = rows * width
    header = {
        key + ".scale": {"dtype": "BF16", "shape": [rows, 1], "data_offsets": [0, 2 * rows]},
        key: {"dtype": "F8_E4M3", "shape": shape, "data_offsets": [2 * rows, 2 * rows + n]},
    }
    payload = json.dumps(header, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 8)
    start = 8 + len(payload)
    with safe_open(source, framework="pt", device="cpu") as src:
        table = src.get_slice(key)
        if table.get_shape() != shape or table.get_dtype() != "BF16":
            raise ValueError(f"Engram の形状または型が異なります: {key}")
        with tempfile.NamedTemporaryFile(dir=directory) as out:
            out.write(struct.pack("<Q", len(payload)) + payload)
            for offset in range(0, rows, chunk):
                codes, scale = encode(table[offset:min(rows, offset + chunk)])
                out.seek(start + 2 * offset)
                out.write(scale.view(torch.uint16).numpy().astype("<u2", copy=False))
                out.seek(start + 2 * rows + width * offset)
                out.write(codes.view(torch.uint8).numpy())
            out.flush()
            os.fsync(out.fileno())
            os.link(out.name, target)
    return target


def main():
    parser = argparse.ArgumentParser(description="Hy4-flash の Engram をホスト参照用 FP8 表に変換する")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, choices=[2, 12, 22, 32], action="append")
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    if text["model_type"] != "hy_v4_flash" or not text["engram_fp8_embedding"]:
        raise ValueError("FP8 Engram を使う Hy4-flash が必要です")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for layer in args.layer or text["engram_layer_ids"]:
        key = f"model.layers.{layer - 1}.engram_layer.multi_head_embedding.embedding.weight"
        sizes = text["engram_layer_head_vocab_sizes"][str(layer)]
        shape = [sum(map(sum, sizes)), text["engram_n_embed_per_ngram"] // text["engram_n_head_per_ngram"]]
        print(convert(args.model / index[key], key, shape, args.output), flush=True)


if __name__ == "__main__":
    main()
