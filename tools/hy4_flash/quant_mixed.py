import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def layout(args, report):
    width = 1536
    elements = 3 * width * width
    budget = round(report["decoder_quantized_weights"] / (2 * elements))
    base, extra = divmod(budget, 41)
    selected = {}
    for layer in range(1, 42):
        folder = args.work / f"layer-{layer:02d}/experts"
        counts = [read(folder / f"expert-{i:04d}.json")["counts"]["routed_tokens"] for i in range(512)]
        take = base + int(layer <= extra)
        selected[str(layer)] = sorted(sorted(range(512), key=lambda e: (counts[e], e))[:take])
    return selected


def convert_rank(rank, args, selected, signature, staging, scratch):
    from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult, quantize_exl3
    from expert_hessian import expert_hessians

    if codebook_mul1_mult != 0xDCD12D:
        raise ValueError("mul24 拡張が必要です")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    index = read(args.original / "model.safetensors.index.json")["weight_map"]
    packed_index = read(args.body / "model.safetensors.index.json")["weight_map"]
    body_hash = read(args.body / "conversion_report.json")["shard_sha256"]
    for layer in range(1, 42):
        prefix = f"model.layers.{layer}.mlp.experts."
        gate_key, down_key = prefix+"gate_up_proj", prefix+"down_proj"
        cap = args.work / f"layer-{layer:02d}/capture"
        routing = cap / "experts.safetensors"
        cap_hash = read(cap / "capture.json")["sha256"][routing.name]
        if digest(routing) != cap_hash:
            raise ValueError(f"校正入力の SHA-256 が異なります: {layer}")
        inputs = load_file(routing)
        shards = defaultdict(list)
        for e in selected[str(layer)]:
            name = packed_index[prefix+f"{e}.gate_proj.trellis"]
            shards[name].append(e)
        for number, (name, ids) in enumerate(sorted(shards.items())):
            if number % 2 != rank:
                continue
            marker = scratch / (name+".json")
            expected = dict(parent=signature, source=body_hash[name], calibration=cap_hash, experts=ids)
            target = staging/name
            if marker.exists():
                saved = read(marker)
                if saved["input"] != expected or digest(target) != saved["sha256"]:
                    raise ValueError(f"保存済みの shard が不正です: {name}")
                continue
            if digest(args.body/name) != body_hash[name]:
                raise ValueError(f"4 bit 元 shard の SHA-256 が異なります: {name}")
            tensors = load_file(args.body/name)
            errors, fallback = {}, []
            hessians = expert_hessians(inputs["latent"], inputs["choices"], args.original/index[gate_key],
                                       gate_key, ids, device, min_tokens=32)
            with safe_open(args.original/index[gate_key], framework="pt") as gates, \
                 safe_open(args.original/index[down_key], framework="pt") as downs:
                for e, gh, dh, counts in hessians:
                    if counts["fallback"]:
                        fallback.append(e)
                    def calibration(hessian):
                        count = counts["hessian_tokens"]
                        return dict(H=hessian, first_key=prefix+str(e), count=count, finalized=False,
                                    num_total=count*hessian.shape[0], inf_nan=torch.zeros(2, dtype=torch.long, device=device),
                                    device=device)
                    gate_h, down_h = calibration(gh), calibration(dh)
                    gate, up = gates.get_slice(gate_key)[e:e+1][0].chunk(2, 0)
                    down = downs.get_slice(down_key)[e:e+1][0]
                    for ordinal, (kind, weight, h) in enumerate((("gate",gate,gate_h),("up",up,gate_h),("down",down,down_h))):
                        key = prefix+f"{e}.{kind}_proj"
                        opts = dict(seed=layer*10000+e*3+ordinal, K=3, devices=[device], device_ratios=None,
                                    apply_out_scales=True, mul1=True)
                        _, err, packed = quantize_exl3(weight.T.float().contiguous(), h, opts, False)
                        if (opts["q_fallback"] or packed["mul1"].item()!=0xDCD12D or
                                tuple(packed["trellis"].shape)!=(96,96,48) or not math.isfinite(err)):
                            raise ValueError(f"3 bit 射影の変換が不正です: {key}")
                        for suffix, value in packed.items():
                            tensor = key+"."+suffix
                            if packed_index[tensor]!=name:
                                raise ValueError(f"射影の shard が分割されています: {key}")
                            tensors[tensor] = value.cpu().contiguous()
                        errors[key] = float(err)
            tmp = target.with_suffix(".tmp")
            save_file(tensors,tmp,metadata={"format":"pt"})
            os.replace(tmp,target)
            write(marker,dict(input=expected,sha256=digest(target),proxy_errors=errors,fallback=fallback))
            print(f"V620 {rank} 層 {layer} {name} の 3 bit expert {len(ids)} 本を保存",flush=True)
        del inputs


def assemble(args, selected, signature, staging, scratch, report):
    config = read(args.body/"config.json")
    q = config["quantization_config"]
    storage = q["tensor_storage"]
    original_index = read(args.body/"model.safetensors.index.json")["weight_map"]
    by_file = defaultdict(list)
    for key, file in original_index.items():
        by_file[file].append(key)
    weight_map, hashes = {}, {}
    total = fallback = 0
    for file, names in sorted(by_file.items()):
        engram = [n for n in names if n.endswith("multi_head_embedding.embedding.weight")]
        if engram:
            if len(names)!=1:
                raise ValueError("Engram が専用 shard ではありません")
            key = engram[0]
            file = key+".fp8.safetensors"
            source = args.engram/file
            wanted = {key,key+".scale"}
        else:
            source = args.body/file
            wanted = set(names)
        dest = staging/file
        marker = scratch/(file+".json")
        if marker.exists():
            meta = read(marker)
            expected = meta["sha256"]
            fallback += len(meta["fallback"])
        else:
            expected = digest(source) if engram else report["shard_sha256"][file]
            if not dest.exists():
                tmp = dest.with_suffix(".copying")
                subprocess.run(["cp","--reflink=auto","--",source,tmp],check=True)
                os.replace(tmp,dest)
        hashes[file] = digest(dest)
        if hashes[file]!=expected:
            raise ValueError(f"組立後の SHA-256 が一致しません: {file}")
        with dest.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q",stream.read(8))[0]))
        header.pop("__metadata__",None)
        if set(header)!=wanted:
            raise ValueError(f"組立時に tensor 名が一致しません: {file}")
        for key,spec in header.items():
            length = spec["data_offsets"][1]-spec["data_offsets"][0]
            total += length
            weight_map[key] = file
            if engram:
                module = engram[0].removesuffix(".weight")
                storage[module]["stored_tensors"][key] = dict(shape=spec["shape"],n_bytes=length,
                    dtype="torch.bfloat16" if key.endswith(".scale") else "torch.float8_e4m3fn")
            elif key.endswith(".trellis") and ".mlp.experts." in key:
                parts = key.split(".")
                bits = 3 if int(parts[5]) in selected[parts[2]] else 4
                if spec["shape"] != [96,96,16*bits]:
                    raise ValueError(f"指定 bpw と実 trellis が一致しません: {key}")
                module = key.removesuffix(".trellis")
                storage[module]["bits_per_weight"] = bits
                storage[module]["stored_tensors"][key].update(shape=spec["shape"],n_bytes=length)
    count = sum(map(len,selected.values()))
    elements = report["decoder_quantized_weights"]
    reduced = count*3*1536*1536
    q.update(bits=3.5,engram_dtype="F8_E4M3",mixed_expert_plan="mixed_plan.json")
    write(staging/"mixed_plan.json",selected)
    write(staging/"config.json",config)
    write(staging/"quantization_config.json",q)
    write(staging/"model.safetensors.index.json",dict(metadata=dict(total_size=total),weight_map=weight_map))
    for name in ("LICENSE","tokenizer.json","tokenizer_config.json","generation_config.json","chat_template.jinja"):
        if (args.body/name).is_file():
            shutil.copy2(args.body/name,staging/name)
    out=dict(report)
    out.update(tensor_bytes=total,shards=len(hashes),shard_sha256=hashes,decoder_code_bpw=4-reduced/elements,
               decoder_quantized_stored_bpw=report["decoder_quantized_stored_bpw"]-reduced/elements,
               decoder_linear_stored_bpw=report["decoder_linear_stored_bpw"]-reduced/(elements+report["decoder_unquantized_linear_weights"]),
               mixed_3bit_experts=count,mixed_4bit_experts=41*512-count,mixed_3bit_calibration_fallbacks=fallback,
               dfly_calibration_signature=report["assembly_signature"],assembly_signature=signature)
    out.pop("elapsed_wall_seconds_since_work_start",None)
    out.pop("elapsed_scope",None)
    write(staging/"conversion_report.json",out)
    staging.rename(args.output)


def main():
    p = argparse.ArgumentParser(description="4 bit 検証後、利用頻度の低い expert を 3 bit にして平均 3.5 bpw にする")
    for name in ("original","body","engram","work","output"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    report=read(args.body/"conversion_report.json")
    settings=read(args.work/"settings.json")
    if (report["decoder_code_bpw"]!=4 or report["head_bits"]!=6 or
            digest(args.work/"settings.json")!=report["assembly_signature"]["settings_sha256"] or
            digest(args.original/"model.safetensors.index.json")!=settings["index_sha256"]):
        raise ValueError("4 bit 校正 work、本体、原本を照合できません")
    selected=layout(args,report)
    signature=dict(body_report_sha256=digest(args.body/"conversion_report.json"),
                   selection_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
                   method="frequency-mixed-mul24-v1")
    scratch=args.output.with_name(args.output.name+".work")
    scratch.mkdir(parents=True,exist_ok=True)
    with (scratch/"convert.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.output.exists():
            if read(args.output/"conversion_report.json")["assembly_signature"]!=signature:
                raise FileExistsError(args.output)
            for file,expected in read(args.output/"conversion_report.json")["shard_sha256"].items():
                if digest(args.output/file)!=expected:raise ValueError(f"既存 shard の破損: {file}")
            return
        saved=scratch/"settings.json"
        if saved.exists() and read(saved)!=signature:
            raise ValueError("再開時の入力または配置が異なります")
        write(saved,signature)
        staging=scratch/"checkpoint"
        staging.mkdir(exist_ok=True)
        if torch.cuda.device_count()!=2:
            raise ValueError("排他した V620 2 台が必要です")
        ctx=multiprocessing.get_context("spawn")
        jobs=[ctx.Process(target=convert_rank,args=(r,args,selected,signature,staging,scratch)) for r in range(2)]
        for proc in jobs:proc.start()
        try:
            while any(proc.is_alive() for proc in jobs):
                if any(proc.exitcode not in (None,0) for proc in jobs):
                    raise RuntimeError("3 bit 変換 worker に失敗しました")
                time.sleep(2)
            if any(proc.exitcode!=0 for proc in jobs):raise RuntimeError("3 bit 変換が未完了です")
        finally:
            for proc in jobs:
                if proc.is_alive():proc.terminate()
                proc.join()
        assemble(args,selected,signature,staging,scratch,report)


if __name__=="__main__":
    main()
