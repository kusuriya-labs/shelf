import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from checkpoint import Weights
from decoder import DecoderLayer


def fixture(directory,w,layer,ids,x,expected):
    directory.mkdir(parents=True,exist_ok=True)
    prefix = f"model.layers.{layer}."
    shards = defaultdict(dict)
    shapes = {k:v for k,v in w.shapes.items() if k.startswith(prefix)}
    for name in w.index:
        if not name.startswith(prefix) or ".mlp.experts." in name:
            continue
        projection = name.removesuffix(".weight")
        if projection in shapes:
            file = w.local[projection]
            for key in ("suh","svh","trellis","mul1"):
                tensor = projection + "." + key
                shards[str(file)][tensor]=tensor
        elif name.endswith("multi_head_embedding.embedding.weight"):
            for key in (name,name+".scale"):
                shards[str(w.engram/(name+".fp8.safetensors"))][key]=key
        else:
            shards[str(w.source(name))][name]=name
    if layer:
        for e in range(512):
            for kind in ("gate","up","down"):
                file=w.work/f"layer-{layer:02d}/experts/expert-{e:04d}-{kind}.safetensors"
                for key in ("suh","svh","trellis"):
                    shards[str(file)][key]=prefix+f"mlp.experts.{e}.{kind}_proj."+key
    data=dict(layer=layer,files=shards,shapes=shapes,ids=ids,
              moduli=w.config["engram_layer_head_vocab_sizes"])
    (directory/"fixture.json").write_text(json.dumps(data))
    save_file(dict(input=x.contiguous(),expected=expected.bfloat16().contiguous()),directory/"rows.safetensors")


def main():
    parser = argparse.ArgumentParser(description="校正時の同一入力で量子化した 1 層の出力を比較する")
    parser.add_argument("model", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("engram", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    i, n = args.layer, args.tokens
    w = Weights(args.model, args.engram, args.work)
    if not 0 <= i < 42 or not 0 < n <= 512:
        raise ValueError("層番号または token 数が範囲外です")
    base = args.work / f"layer-{i:02d}"
    with safe_open(base / "capture/next.safetensors", framework="pt") as f:
        ids = f.get_slice("input_ids")[0, :n].tolist()
        expected = f.get_slice("hidden")[0, :n].float()
    if not i:
        x = w.raw("model.embed_tokens.weight")[ids][:, None].expand(-1, 4, -1)
    else:
        with safe_open(args.work / f"layer-{i-1:02d}/capture/next.safetensors", framework="pt") as f:
            x = f.get_slice("hidden")[0, :n]
    layer = DecoderLayer(w, i, n)
    result = layer(x, [], ids).float()
    if args.fixture:
        fixture(args.fixture,w,i,ids,x,result)
    error = (result - expected).norm() / expected.norm()
    print(f"層 {i} token 数 {len(ids)} 相対 L2 {error.item():.6f}", flush=True)
    if not error.isfinite() or error >= 0.3:
        raise ValueError("BF16 校正出力からの差が大きすぎます")


if __name__ == "__main__":
    main()
