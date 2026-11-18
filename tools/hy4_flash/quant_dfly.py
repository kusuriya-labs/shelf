import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from decoder import collapse, rms, rotate
from octave import decode, encode

SOURCES = (25, 29, 33, 37, 41)
PREFIX = "model.dfly_layers."


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def groups(cols, anchors):
    ends = torch.linspace(0, cols - 7, anchors).round().long().unique()
    if cols < 7 or len(ends) != anchors:
        raise ValueError("校正列の長さに対して anchor が多すぎます")
    return ends.unsqueeze(-1) + torch.arange(1, 7)


class Calibration:
    def __init__(self, args):
        self.args = args
        self.index = json.loads((args.original / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.device = torch.device("cuda:0")
        self.h = {}
        self.samples = {}

    @lru_cache(maxsize=64)
    def weight(self, key):
        with safe_open(self.args.original / self.index[key], framework="pt") as f:
            return f.get_tensor(key).to(self.device)

    def linear(self, name, x):
        v = x.reshape(-1, x.size(-1)).float()
        if name not in self.h:
            self.h[name] = torch.zeros(v.size(1), v.size(1), device=v.device)
        self.h[name].addmm_(v.T, v)
        self.samples[name] = self.samples.get(name, 0) + len(v)
        return F.linear(x, self.weight(name + ".weight"))

    def norm(self, name, x):
        return rms(x, self.weight(name + ".weight"))

    def capture(self, layer, row):
        file = self.args.work / f"layer-{layer:02d}/capture/next.safetensors"
        with safe_open(file, framework="pt") as f:
            return f.get_slice("input_ids")[row].long().to(self.device), f.get_slice("hidden")[row].to(self.device)

    def source_kv(self, source, row):
        _, hidden = self.capture(source - 1, row)
        p = f"model.layers.{source}."
        h = p + "hc_attn_layer.hc_pre.hc_"
        x = collapse(hidden, *(self.weight(h + k) for k in ("fn", "scale", "base")))[0]
        x = self.norm(p + "input_layernorm", x)
        a = p + "self_attn."
        kv = F.linear(x, self.weight(a + "kv_a_proj_with_mqa.weight"))
        return self.latents(a, kv, torch.arange(len(x), device=x.device))

    def latents(self, p, kv, pos):
        latent = self.norm(p + "kv_a_layernorm", kv[..., :512])
        rotary = rotate(kv[..., 512:], pos, 1e7)
        return (decode(encode(latent, 256, 29), 256, 29).reshape(-1, 512),
                decode(encode(rotary, 64, 11), 64, 11))

    def fusion(self, row, blocks):
        ids, hidden = self.capture(41, row)
        p = "model.hc_head.hc_head_"
        h = collapse(hidden[blocks[:, 0] - 1], *(self.weight(p+k) for k in ("fn","scale","base")))[0]
        h = self.norm("model.norm", h)
        tokens = torch.full_like(blocks, 120009)
        tokens[:, 0] = ids[blocks[:, 0]]
        x = self.weight("model.embed_tokens.weight")[tokens]
        p = PREFIX + "mtp_fusion."
        inp = torch.cat((self.norm(p+"enorm", x[:, 0]), self.norm(p+"hnorm", h)), -1)
        x[:, 0] += self.weight(p+"gate") * self.linear(p+"eh_proj", inp)
        return x

    def layer(self, layer, row, blocks, hidden):
        p = PREFIX + str(layer) + "."
        x = hidden.flatten(0, 1)
        z = self.norm(p+"input_layernorm", x)
        a = p+"self_attn."
        qres = self.norm(a+"q_a_layernorm", self.linear(a+"q_a_proj", z))
        query = rotate(self.linear(a+"q_b_proj", qres).reshape(-1,32,256), blocks.flatten(), 1e7)
        kv = self.linear(a+"kv_a_proj_with_mqa", z)
        latent, rotary = self.latents(a, kv, blocks.flatten())
        past, pos = self.source_kv(SOURCES[layer], row)
        history = self.linear(a+"kv_b_proj", past).reshape(-1,32,448)
        local = self.linear(a+"kv_b_proj", latent).reshape(len(blocks),6,32,448)
        outputs = []
        for i, anchor in enumerate(blocks[:,0].tolist()):
            keys = torch.cat((history[:anchor], local[i]))
            kr = torch.cat((pos[:anchor], rotary[i*6:i*6+6]))
            q = query[i*6:i*6+6]
            scores = torch.einsum("shd,thd->sht",q[:,:,:192].float(),keys[:,:,:192].float())
            scores += torch.einsum("shd,td->sht",q[:,:,192:].float(),kr.float())
            mask = torch.arange(len(keys),device=x.device)[None,:] > blocks[i,:,None]
            scores = (scores * 0.0625).masked_fill(mask[:,None], -torch.inf)
            sink = self.weight(a+"learnable_sink_param").expand(6,-1).unsqueeze(-1)
            probs = torch.cat((scores,sink),-1).softmax(-1)[:,:,:-1].to(x.dtype)
            outputs.append(torch.einsum("sht,thd->shd",probs,keys[:,:,192:]))
        y = torch.cat(outputs).reshape(len(x),-1) * self.linear(a+"linear_gate",z).sigmoid()
        x = x + self.norm(a+"post_proj_norm", self.linear(a+"o_proj", y))
        y = self.norm(p+"post_attention_layernorm",x)
        m = p+"mlp."
        result = x + self.linear(m+"down_proj",F.silu(self.linear(m+"gate_proj",y))*self.linear(m+"up_proj",y))
        return result.reshape_as(hidden)

    def correction(self,row,blocks,hidden):
        x = self.norm(PREFIX+"final_norm", hidden[:,1:].reshape(-1,3072))
        ids, _ = self.capture(41,row)
        prev = self.weight("model.embed_tokens.weight")[ids[blocks[:,:-1]].flatten()]
        p = PREFIX+"hidden_correction."
        u = torch.cat((self.norm(p+"hidden_norm",x),self.norm(p+"embed_norm",prev)),-1)
        return x + self.linear(p+"down_proj",F.silu(self.linear(p+"gate_proj",u))*self.linear(p+"up_proj",u))

    def quantize(self, name, number):
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

        h = self.h.pop(name)
        w = self.weight(name+".weight").T.float().contiguous()
        if not torch.isfinite(h).all() or not h.diagonal().gt(0).any():
            raise ValueError("DFly 校正 Hessian が不正です: "+name)
        asym = (h - h.T).abs().max() / h.diagonal().max()
        if asym.item() > 1e-5:
            raise ValueError("DFly 校正 Hessian が非対称です: "+name)
        shape = list(w.shape)
        k, n = shape
        if k % 128 or n % 128:
            w = F.pad(w, (0, -n % 128, 0, -k % 128))
            h = F.pad(h, (0, -k % 128, 0, -k % 128))
        count = self.samples.pop(name)
        data = dict(H=h,first_key=name,count=count,finalized=False,num_total=count*k,
                    inf_nan=torch.zeros(2,dtype=torch.long,device=h.device),device=h.device)
        opts = dict(seed=number,K=4,devices=[h.device],device_ratios=None,apply_out_scales=True,mul1=True)
        _, _, packed = quantize_exl3(w,data,opts,False)
        if opts["q_fallback"] or packed["mul1"].item()!=0xDCD12D:
            raise ValueError("DFly の量子化に失敗しました: "+name)
        return {name+"."+k:v.cpu().contiguous() for k,v in packed.items()}, shape


@torch.inference_mode()
def run(args):
    if not json.loads((args.work/"progress.json").read_bytes()).get("head_complete"):
        raise ValueError("本体の完了と mul24 拡張が必要です")
    body = args.body / "conversion_report.json"
    report = json.loads(body.read_bytes())
    orig_index = args.original / "model.safetensors.index.json"
    settings = args.work / "settings.json"
    if (report["head_bits"] != 6 or report["decoder_code_bpw"] != 4 or
            report["assembly_signature"]["settings_sha256"] != digest(settings) or
            json.loads(settings.read_bytes())["index_sha256"] != digest(orig_index)):
        raise ValueError("原本・校正作業・本体の署名が一致しません")
    from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult
    if codebook_mul1_mult != 0xDCD12D:
        raise ValueError("mul24 拡張が必要です")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    cal = Calibration(args)
    with safe_open(args.work/"layer-41/capture/next.safetensors",framework="pt") as f:
        rows,cols = f.get_slice("input_ids").get_shape()
    if not 0 < args.rows <= rows or cols>2048 or not 0<args.anchors<=cols-6:
        raise ValueError("校正の行数または文脈長が範囲外です")
    block = groups(cols,args.anchors).to(cal.device)
    signature = dict(body=digest(body), original_index=digest(orig_index), rows=args.rows,
                     anchors=args.anchors, work=str(args.work.resolve()), algorithm=1)
    scratch = args.output.with_name(args.output.name + ".work")
    scratch.mkdir(parents=True,exist_ok=True)
    with (scratch / "convert.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            old = json.loads((args.output/"conversion_report.json").read_bytes())
            if old.get("signature") == signature and all(
                digest(args.output/k) == v for k,v in old["shard_sha256"].items()
            ):
                return
            raise FileExistsError(args.output)
        settings_path = scratch/"settings.json"
        if settings_path.exists() and json.loads(settings_path.read_bytes()) != signature:
            raise ValueError("既存の DFly 作業領域の入力が異なります")
        atomic_json(settings_path, signature)
        staging=scratch/"dfly"
        staging.mkdir(exist_ok=True)
        names, shapes, checksums = {}, {}, {}
        states = []
        for stage in range(-1,6):
            marker = scratch/f"stage-{stage}.json"
            hidden_path = scratch/f"hidden-{stage}.safetensors"
            if marker.exists():
                saved = json.loads(marker.read_bytes())
                if any(digest(staging/file) != expected for file,expected in saved["checksums"].items()):
                    raise ValueError(f"保存済み DFly 層が破損しています: {stage}")
                if stage < 5:
                    if digest(hidden_path) != saved["hidden_sha256"]:
                        raise ValueError("保存済み校正 hidden が破損しています")
                    states = list(load_file(hidden_path)["hidden"].unbind())
                names.update(saved["names"])
                shapes.update(saved["shapes"])
                checksums.update(saved["checksums"])
                continue
            stage_shapes, stage_files, stage_names = {}, {}, {}
            for row in range(args.rows):
                if stage==-1:
                    states.append(cal.fusion(row,block).cpu())
                elif stage==5:
                    out=cal.correction(row,block,states[row].to(cal.device))
                    if not torch.isfinite(out).all():
                        raise ValueError("hidden correction の校正出力が非有限です")
                else:
                    states[row]=cal.layer(stage,row,block,states[row].to(cal.device)).cpu()
            for name in sorted(cal.h):
                packed, stage_shapes[name] = cal.quantize(name,len(shapes)+len(stage_shapes))
                file=name+".safetensors"
                save_file(packed,staging/file)
                stage_names.update({k:file for k in packed})
                stage_files[file]=digest(staging/file)
            names.update(stage_names)
            shapes.update(stage_shapes)
            checksums.update(stage_files)
            hidden_hash = None
            if stage < 5:
                save_file({"hidden": torch.stack(states)},hidden_path)
                hidden_hash = digest(hidden_path)
            atomic_json(marker,dict(names=stage_names,shapes=stage_shapes,checksums=stage_files,hidden_sha256=hidden_hash))
            cal.weight.cache_clear()
            print("DFly の校正・量子化が完了した段階",stage,flush=True)
        raw={}
        for key in cal.index:
            if key.startswith(PREFIX) and key.removesuffix(".weight") not in shapes:
                with safe_open(args.original/cal.index[key],framework="pt") as f:
                    value=f.get_tensor(key)
                    if value.ndim==2 and not key.startswith(PREFIX+"confidence_head."):
                        raise ValueError("量子化対象が不足しています: "+key)
                    raw[key]=value
        save_file(raw,staging/"raw.safetensors")
        names.update({k:"raw.safetensors" for k in raw})
        checksums["raw.safetensors"]=digest(staging/"raw.safetensors")
        atomic_json(staging/"model.safetensors.index.json",dict(weight_map=names))
        checksums["model.safetensors.index.json"]=digest(staging/"model.safetensors.index.json")
        atomic_json(staging/"conversion_report.json",dict(method="dfly",bits=4,linear_shapes=shapes,
                    signature=signature,body_assembly_signature=report["assembly_signature"],body_report_sha256=digest(body),rows=args.rows,anchors=args.anchors,shard_sha256=checksums))
        staging.rename(args.output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="本体完了後に DFly を校正して EXL3-INT8 に量子化する")
    for name in ("original","body","work","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--rows",type=int,default=128)
    parser.add_argument("--anchors",type=int,default=16)
    run(parser.parse_args())
