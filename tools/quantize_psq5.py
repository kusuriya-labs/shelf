#!/usr/bin/env python3
"""PSQ5-CB28 量子化 runner。

AMD Quark の algorithm registry に登録された psq5_cb28 processor を使い、
safetensors モデルの Linear weight を W32 block + 8bit scale の 5bit へ量子化する。

単一 `model.safetensors` と sharded (`model.safetensors.index.json`) の両方を受け付け、
sharded では shard 単位で並列処理し、同じ shard 構成で出力する。
"""

import argparse
import dataclasses
import fnmatch
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import quark_psq5
from quark_psq5.artifact import build_manifest, read_artifact, reconstruct, stream_keys, write_manifest
from quark_psq5.codebook import dequantize_weight, quantize_weight
from quark_psq5.config import PSQ5CB28Config
from quark_psq5.processor import BUFFER_NAMES, PSQ5CB28Processor
from quark_psq5.scale_codec import make_scale_codec

DEFAULT_INCLUDE = [
    "*q_proj.weight",
    "*k_proj.weight",
    "*v_proj.weight",
    "*o_proj.weight",
    "*gate_proj.weight",
    "*up_proj.weight",
    "*down_proj.weight",
    "*in_proj_qkv.weight",
    "*in_proj_z.weight",
    "*linear_attn.out_proj.weight",
    "*kernel_projection.weight",
    "fc.weight",
    "*hidden_projection.weight",
]
DEFAULT_EXCLUDE = ["*codebook*", "mtp.*", "model.visual.*"]
COPY_EXCLUDE = {"model.safetensors", "crc32.txt", "model.safetensors.index.json"}
SOURCE_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "model.safetensors.json"
DEFAULT_CHUNK_ROWS = 8192


def load_specs(source: Path) -> tuple[list[tuple[str, int, int, str]], list[tuple[str, str]]]:
    linears: list[tuple[str, int, int, str]] = []
    passthrough: list[tuple[str, str]] = []
    with safe_open(str(source), "pt") as handle:
        for name in sorted(handle.keys()):
            slice_ = handle.get_slice(name)
            shape = slice_.get_shape()
            dtype = slice_.get_dtype()
            if dtype in SOURCE_DTYPES and len(shape) == 2:
                linears.append((name, shape[0], shape[1], dtype))
            else:
                passthrough.append((name, dtype))
    return linears, passthrough


def select(linears, include, exclude):
    kept = []
    dropped = []
    for spec in linears:
        name = spec[0]
        included = not include or any(fnmatch.fnmatch(name, pattern) for pattern in include)
        excluded = any(fnmatch.fnmatch(name, pattern) for pattern in exclude)
        (kept if included and not excluded else dropped).append(spec)
    return kept, dropped


def quantize(source: Path, output_dir: Path, config: PSQ5CB28Config) -> Path:
    linears, passthrough = load_specs(source)
    linears, dropped = select(linears, config.include, config.exclude)
    codec = make_scale_codec(config.scale_format, config.scale_fold)
    if not linears:
        raise SystemExit("量子化対象の Linear がありません (include/exclude を確認してください)")

    container = nn.ModuleList()
    with safe_open(str(source), "pt") as handle:
        for name, rows, columns, _dtype in linears:
            linear = nn.Linear(columns, rows, bias=False)
            linear.weight = nn.Parameter(handle.get_tensor(name), requires_grad=False)
            container.append(linear)
    names = [spec[0] for spec in linears]

    processor_config = dataclasses.replace(config, include=["*"], exclude=[])
    processor = PSQ5CB28Processor(container, processor_config, None)
    processor.apply()

    streams: dict = {}
    entries: dict = {}
    for index in sorted(int(entry) for entry in processor.processed):
        module = container[index]
        name = names[index]
        keys = stream_keys(name)
        for field, buffer_name in zip(("magnitude", "sign", "scale"), BUFFER_NAMES):
            streams[keys[field]] = module._buffers[buffer_name].detach().cpu()
        entries[name] = {
            "weight_shape": [module.out_features, module.in_features],
            "block_k": config.block_k,
            "source_dtype": linears[index][3].lower(),
        }

    passthrough_tensors: dict = {}
    with safe_open(str(source), "pt") as handle:
        for name, _dtype in passthrough + [(spec[0], spec[3]) for spec in dropped]:
            passthrough_tensors[name] = handle.get_tensor(name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_weights = output_dir / "model.safetensors"
    save_file({**streams, **passthrough_tensors}, str(output_weights))

    manifest = build_manifest(entries, config, codec)
    manifest["source_model"] = source.parent.name
    manifest["quantized_tensors"] = len(entries)
    manifest["passthrough_tensors"] = len(passthrough_tensors)
    manifest["pre_quant_optimization"] = "none"
    write_manifest(str(output_weights) + ".json", manifest)

    for item in sorted(source.parent.iterdir()):
        if item.name in COPY_EXCLUDE or item.is_dir():
            continue
        shutil.copy2(item, output_dir / item.name)

    total_source_bytes = sum(
        rows * columns * SOURCE_DTYPES[dtype].itemsize for _n, rows, columns, dtype in linears)
    quantized_bytes = sum(value.numel() for value in streams.values())
    print(f"quantized {len(entries)} linears, passthrough {len(passthrough_tensors)} tensors")
    print(f"scale_format={codec.name} fold={codec.fold} bpw={manifest['bits_per_weight']:.3f}")
    print(f"linear storage: {total_source_bytes} -> {quantized_bytes} bytes")
    print(f"output: {output_weights}")
    return output_weights


def psq5_streams(weight: torch.Tensor, codec, block_k: int, chunk_rows: int) -> dict:
    rows, columns = weight.shape
    blocks = (columns + block_k - 1) // block_k
    magnitude = torch.empty((rows, blocks, 16), dtype=torch.uint8)
    sign = torch.empty((rows, blocks, 4), dtype=torch.uint8)
    scale = torch.empty((rows, blocks), dtype=torch.uint8)
    for start in range(0, rows, chunk_rows):
        result = quantize_weight(weight[start:start + chunk_rows], codec, block_k)
        stop = start + result["magnitude"].shape[0]
        magnitude[start:stop] = result["magnitude"]
        sign[start:stop] = result["sign"]
        scale[start:stop] = result["scale"]
    return {"magnitude": magnitude, "sign": sign, "scale": scale}


def _convert_shard(job: tuple) -> dict:
    (source, destination, fragment_path, include, exclude,
     block_k, scale_format, scale_fold, chunk_rows, threads) = job
    torch.set_num_threads(max(1, threads))
    codec = make_scale_codec(scale_format, scale_fold)
    started = time.time()
    fragment = {"shard": os.path.basename(str(destination)), "tensors": {},
                "passthrough": 0, "passthrough_names": [],
                "bytes": {"psq5": 0, "bf16": 0}}
    tensors: dict = {}
    with safe_open(str(source), "pt") as handle:
        for name in handle.keys():
            weight = handle.get_tensor(name)
            usable = (weight.dim() == 2 and weight.dtype in SOURCE_DTYPES.values())
            wanted = (not include or any(fnmatch.fnmatch(name, p) for p in include)) and \
                     not any(fnmatch.fnmatch(name, p) for p in exclude)
            if usable and wanted:
                streams = psq5_streams(weight, codec, block_k, chunk_rows)
                keys = stream_keys(name)
                for field, tensor in streams.items():
                    tensors[keys[field]] = tensor
                    fragment["bytes"]["psq5"] += tensor.numel()
                fragment["tensors"][name] = {
                    "weight_shape": [weight.shape[0], weight.shape[1]],
                    "block_k": block_k,
                    "source_dtype": str(weight.dtype).replace("torch.", ""),
                    "streams": keys,
                    "shard": fragment["shard"],
                }
                del weight, streams
            else:
                tensors[name] = weight
                fragment["bytes"]["bf16"] += weight.numel() * weight.element_size()
                fragment["passthrough"] += 1
                fragment["passthrough_names"].append(name)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(destination))
    del tensors
    tmp = Path(str(fragment_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(fragment, handle, sort_keys=True)
        handle.write("\n")
    tmp.replace(fragment_path)
    return {"shard": fragment["shard"], "quantized": len(fragment["tensors"]),
            "passthrough": fragment["passthrough"],
            "seconds": round(time.time() - started, 3)}


def plan_sharded(model_dir: Path, include: list, exclude: list) -> dict:
    index = json.loads((model_dir / INDEX_NAME).read_text())
    shards = sorted(set(index["weight_map"].values()))
    stats = {"quantized": 0, "source_bytes": 0, "output_bytes": 0, "passthrough": 0,
             "passthrough_bytes": 0, "shards": len(shards)}
    itemsize = {"BF16": 2, "F16": 2, "F32": 4}
    for shard in shards:
        with safe_open(str(model_dir / shard), "pt") as handle:
            for name in handle.keys():
                slice_ = handle.get_slice(name)
                shape = slice_.get_shape()
                dtype = slice_.get_dtype()
                usable = dtype in SOURCE_DTYPES and len(shape) == 2
                wanted = (not include or any(fnmatch.fnmatch(name, p) for p in include)) and \
                         not any(fnmatch.fnmatch(name, p) for p in exclude)
                if usable and wanted:
                    rows, columns = shape
                    blocks = (columns + 31) // 32
                    stats["quantized"] += 1
                    stats["source_bytes"] += rows * columns * itemsize[dtype]
                    stats["output_bytes"] += rows * blocks * 21
                else:
                    stats["passthrough"] += 1
                    n = 1
                    for d in shape:
                        n *= d
                    stats["passthrough_bytes"] += n * itemsize.get(dtype, 2)
    return stats


def quantize_sharded(model_dir: Path, output_dir: Path, config: PSQ5CB28Config,
                     jobs: int, chunk_rows: int, overwrite: bool) -> Path:
    index = json.loads((model_dir / INDEX_NAME).read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir = output_dir / ".quantize_fragments"
    fragment_dir.mkdir(exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // max(1, jobs))
    pending = []
    for shard in shard_names:
        destination = output_dir / shard
        fragment = fragment_dir / f"{shard}.json"
        if not overwrite and destination.exists() and fragment.exists():
            print(f"skip {shard} (exists)")
            continue
        pending.append((str(model_dir / shard), str(destination), str(fragment),
                        config.include, config.exclude, config.block_k,
                        config.scale_format, config.scale_fold, chunk_rows, threads))
    started = time.time()
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("fork")) as pool:
        futures = [pool.submit(_convert_shard, job) for job in pending]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{done}/{len(pending)}] {result['shard']} {result['seconds']:.1f}s "
                  f"quantized={result['quantized']} passthrough={result['passthrough']}", flush=True)
    print(f"convert done in {time.time() - started:.1f}s")

    codec = make_scale_codec(config.scale_format, config.scale_fold)
    entries: dict = {}
    weight_map: dict = {}
    bytes_by_kind = {"psq5": 0, "bf16": 0}
    passthrough = 0
    shards_out = []
    for fragment_path in sorted(fragment_dir.glob("*.json")):
        fragment = json.loads(fragment_path.read_text())
        shards_out.append(fragment["shard"])
        for name, entry in fragment["tensors"].items():
            entries[name] = entry
            for key in entry["streams"].values():
                weight_map[key] = fragment["shard"]
        for name in fragment["passthrough_names"]:
            weight_map[name] = fragment["shard"]
        passthrough += fragment["passthrough"]
        for kind, value in fragment["bytes"].items():
            bytes_by_kind[kind] += value
    manifest = build_manifest(entries, config, codec)
    manifest["source_model"] = model_dir.name
    manifest["shards"] = shards_out
    manifest["quantized_tensors"] = len(entries)
    manifest["passthrough_tensors"] = passthrough
    manifest["bytes"] = bytes_by_kind
    manifest["pre_quant_optimization"] = "none"
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index_out = {
        "metadata": {"format": manifest["format"], "source_model": model_dir.name,
                     "total_size": sum(bytes_by_kind.values())},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_dir / INDEX_NAME).write_text(json.dumps(index_out, indent=2, sort_keys=True) + "\n")
    for item in sorted(model_dir.iterdir()):
        if item.name in COPY_EXCLUDE or item.name.endswith(".safetensors") or item.is_dir():
            continue
        shutil.copy2(item, output_dir / item.name)
    shutil.rmtree(fragment_dir)
    print(f"manifest: {len(entries)} psq5 tensors, {passthrough} passthrough, "
          f"scale_format={codec.name}, bpw={manifest['bits_per_weight']:.3f}")
    print(f"output: {output_dir}")
    return output_dir


def verify(output_weights: Path, source_weights: Path, seed: int = 1234) -> None:
    streams, manifest = read_artifact(str(output_weights))
    generator = torch.Generator().manual_seed(seed)
    exact = True
    weight_error_sum = 0.0
    weight_energy = 0.0
    worst_projection = 0.0
    worst_name = ""
    codec = make_scale_codec(manifest["scale_format"], manifest["scale_fold"])
    for name, entry in sorted(manifest["tensors"].items()):
        rebuilt = reconstruct(streams, manifest, name)
        with safe_open(str(source_weights), "pt") as handle:
            original = handle.get_tensor(name)
        expected = quantize_weight(original, codec, entry["block_k"])["reconstruction"].to(original.dtype)
        exact = exact and torch.equal(rebuilt.to(expected.dtype), expected)

        columns = entry["weight_shape"][1]
        original32 = original.to(torch.float32)
        inputs = torch.randn((8, columns), generator=generator, dtype=torch.float32)
        reference = inputs @ original32.t()
        candidate = inputs @ rebuilt.t()
        projection = float((reference - candidate).norm() / reference.norm())
        if projection > worst_projection:
            worst_projection, worst_name = projection, name

        weight_error_sum += float((original32 - rebuilt).pow(2).sum(dtype=torch.float64))
        weight_energy += float(original32.pow(2).sum(dtype=torch.float64))
        del inputs, reference, candidate, original, expected, original32

    if not manifest["tensors"]:
        print("artifact round trip exact: n/a (no quantized tensor)")
        return
    relative_weight_error = (weight_error_sum / weight_energy) ** 0.5
    print(f"artifact round trip exact: {exact}")
    print(f"relative weight error vs source: {relative_weight_error:.6f}")
    print(f"worst projection relative error: {worst_projection:.6f} ({worst_name})")


def verify_sharded(output_dir: Path, model_dir: Path, config: PSQ5CB28Config,
                   sample: int, chunk_rows: int) -> None:
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
    index = json.loads((output_dir / INDEX_NAME).read_text())["weight_map"]
    codec = make_scale_codec(config.scale_format, config.scale_fold)
    missing = []
    for shard in manifest["shards"]:
        with safe_open(str(output_dir / shard), "pt") as handle:
            keys = set(handle.keys())
        expected = {key for key, value in index.items() if value == shard}
        missing.extend(sorted(expected - keys))
    print(f"structural: shards={len(manifest['shards'])} keys={len(index)} missing={len(missing)}")

    names = sorted(manifest["tensors"])
    if sample and sample < len(names):
        step = max(1, len(names) // sample)
        names = names[::step][:sample]
    exact = True
    error_sum = 0.0
    energy_sum = 0.0
    for name in names:
        entry = manifest["tensors"][name]
        streams = entry["streams"]
        with safe_open(str(output_dir / entry["shard"]), "pt") as handle:
            magnitude = handle.get_tensor(streams["magnitude"])
            sign = handle.get_tensor(streams["sign"])
            scale = handle.get_tensor(streams["scale"])
        with safe_open(str(model_dir / entry["shard"]), "pt") as handle:
            source = handle.get_tensor(name)
        rows, columns = entry["weight_shape"]
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            rebuilt = dequantize_weight(magnitude[start:stop], sign[start:stop],
                                        scale[start:stop], codec, stop - start,
                                        columns, entry["block_k"])
            expected = quantize_weight(source[start:stop], codec, entry["block_k"])["reconstruction"]
            exact = exact and torch.equal(rebuilt.to(expected.dtype), expected)
            original = source[start:stop].to(torch.float32)
            error_sum += float((original - rebuilt).pow(2).sum(dtype=torch.float64))
            energy_sum += float(original.pow(2).sum(dtype=torch.float64))
        del magnitude, sign, scale, source
    relative = (error_sum / energy_sum) ** 0.5 if energy_sum else 0.0
    print(f"verified {len(names)}/{len(manifest['tensors'])} psq5 tensors: "
          f"bit_exact={exact} rel_weight_error={relative:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PSQ5-CB28 量子化")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale-format", default="ue4m4", choices=["ue4m4", "ue4m3", "fp16", "bf16"])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--verify-sample", type=int, default=8, help="0 で全数を検証")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = PSQ5CB28Config(
        scale_format=args.scale_format,
        include=args.include or DEFAULT_INCLUDE,
        exclude=list(args.exclude) + DEFAULT_EXCLUDE,
    )
    sharded = (model_dir / INDEX_NAME).exists()

    if args.dry_run:
        if sharded:
            stats = plan_sharded(model_dir, config.include, config.exclude)
            print(f"shards={stats['shards']} quantized={stats['quantized']} "
                  f"source={stats['source_bytes'] / 1e9:.3f} GB "
                  f"-> psq5={stats['output_bytes'] / 1e9:.3f} GB")
            print(f"passthrough={stats['passthrough']} "
                  f"({stats['passthrough_bytes'] / 1e9:.3f} GB)")
            print(f"total output: {(stats['output_bytes'] + stats['passthrough_bytes']) / 1e9:.3f} GB")
        else:
            linears, _ = load_specs(model_dir / "model.safetensors")
            kept, _ = select(linears, config.include, config.exclude)
            print(f"single file: {len(kept)} quantized linears")
        return 0

    if sharded:
        quantize_sharded(model_dir, output_dir, config, args.jobs, args.chunk_rows, args.overwrite)
        if not args.no_verify:
            verify_sharded(output_dir, model_dir, config, args.verify_sample, args.chunk_rows)
        return 0

    source = model_dir / "model.safetensors"
    if not source.exists():
        raise SystemExit(f"model.safetensors が見つかりません: {source}")
    output_weights = quantize(source, output_dir, config)
    if not args.no_verify:
        verify(output_weights, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
