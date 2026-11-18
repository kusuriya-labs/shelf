import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from checkpoint import Weights, read_json
from decoder import DecoderLayer, collapse, rms


def main():
    p = argparse.ArgumentParser(description="先頭から連続した量子化層を CPU で照合し、途中状態を保存する")
    for name in ("model", "work", "engram", "result"):
        p.add_argument(name, type=Path)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--placement", type=Path)
    p.add_argument("--body", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if not 0 < args.tokens <= 128:
        raise ValueError("校正列の先頭 1〜128 token を指定してください")
    w = Weights(args.model,args.engram,args.work)
    if args.placement:
        w.owners = read_json(args.placement)
        w.devices = ["cpu"] * 3
    args.result.mkdir(parents=True,exist_ok=True)
    with safe_open(args.work/"layer-00/capture/next.safetensors", framework="pt") as f:
        ids = f.get_slice("input_ids")[0,:args.tokens].tolist()
    signature = dict(model=str(args.model.resolve()),work=str(args.work.resolve()),engram=str(args.engram.resolve()),
                     settings=read_json(args.work/"settings.json"),ids=ids,owners=w.owners)
    settings = args.result/"input.json"
    if settings.exists():
        if read_json(settings) != signature:
            raise ValueError("再開する CPU 照合の入力が異なります")
    else:
        settings.write_text(json.dumps(signature))
    x = w.raw("model.embed_tokens.weight")[ids][:,None].expand(-1,4,-1)
    end = read_json(args.work/"progress.json")["last_layer"] + 1
    for i in range(end):
        path = args.result/f"layer-{i:02d}.safetensors"
        if path.exists():
            x = load_file(path)["hidden"]
            if x.shape != (len(ids),4,3072) or not x.isfinite().all():
                raise ValueError(f"保存状態が不正です: {path}")
            continue
        x = DecoderLayer(w,i,len(ids))(x,[],ids)
        with safe_open(args.work/f"layer-{i:02d}/capture/next.safetensors",framework="pt") as f:
            expected = f.get_slice("hidden")[0,:len(ids)].float()
        error = (x.float()-expected).norm()/expected.norm()
        if not error.isfinite():
            raise ValueError(f"層 {i} の連続照合で非有限値が発生しました")
        tmp = path.with_suffix(".tmp")
        save_file({"hidden":x.contiguous(),"relative_l2":error},tmp)
        tmp.replace(path)
        print("連続照合",i,error.item(),flush=True)
    if args.body and end == 42:
        body = Weights(args.body,args.engram)
        prefix = "model.hc_head.hc_head_"
        h = collapse(x,*(body.raw(prefix+k) for k in ("fn","scale","base")))[0]
        logits = body.linear("lm_head",rms(h,body.raw("model.norm.weight")))
        save_file({"logits":logits,"input_ids":torch.tensor(ids)},args.result/"logits.safetensors")
        with safe_open(args.work/"layer-41/capture/next.safetensors",framework="pt") as f:
            original_streams = f.get_slice("hidden")[0,:len(ids)]
        original_hidden = collapse(original_streams,*(w.raw(prefix+k) for k in ("fn","scale","base")))[0]
        original_hidden = rms(original_hidden,w.raw("model.norm.weight")).float()
        head = w.raw("lm_head.weight")
        original_logits = torch.cat([original_hidden @ block.float().T for block in head.split(2048)],dim=-1)
        log_p = original_logits.log_softmax(-1)
        log_q = logits.float().log_softmax(-1)
        metrics = {
            "relative_l2":((logits-original_logits).norm()/original_logits.norm()).item(),
            "kl_bf16_to_quant":(log_p.exp()*(log_p-log_q)).sum(-1).tolist(),
            "bf16_top1":original_logits.argmax(-1).tolist(),
            "quant_top1":logits.argmax(-1).tolist(),
        }
        (args.result/"head_comparison.json").write_text(json.dumps(metrics,allow_nan=False,indent=2))
        print("head 照合",metrics,flush=True)


if __name__ == "__main__":
    main()
