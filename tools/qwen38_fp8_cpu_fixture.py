#!/usr/bin/env python3
"""Qwen3.8-27B-FP8 の CPU f64 参照 fixture を生成する。

HF の FP8 checkpoint を F32 (FP8 code * BF16 scale は F32 で厳密) または
BF16 へ dequantize し、外部の qwen35-f64-oracle で f64 forward を実行する。
oracle は本 repository に vendor しない。docs/developer/testing_f64_oracle.md
を参照。

例:
  tools/qwen38_fp8_cpu_fixture.py \
      --model-dir models/Qwen3.8-27B-FP8 \
      --oracle-dir ../references/qwen35-f64-oracle \
      --ids 814,20139,303,799,13901,3069,279,12515,7701,6105,2261,279,1834,321,2438,506,41564,13 \
      --max-new-tokens 64 --hidden-tokens 16 \
      --out-json tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle.json \
      --out-hidden tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle_hidden.safetensors \
      --out-hidden-generated tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle_hidden_generated.safetensors
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_fp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    q64 = q.to(torch.float64)
    s64 = s.to(torch.float64)
    rows, cols = q64.shape[-2:]
    srows, scols = s64.shape[-2:]
    if rows % srows or cols % scols:
        raise SystemExit(f"weight {tuple(q.shape)} / scale {tuple(s.shape)} mismatch")
    bm, bn = rows // srows, cols // scols
    return (q64.reshape(srows, bm, scols, bn) * s64[:, None, :, None]).reshape(rows, cols)


def dequantize_model(src: Path, dst: Path, dtype: torch.dtype) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    shutil.copy2(src / "config.json", tmp / "config.json")
    shards = sorted(set(index["weight_map"].values()))
    t0 = time.time()
    for i, shard in enumerate(shards):
        print(f"dequant {i + 1}/{len(shards)} {shard}", flush=True)
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            out = {}
            for key in keys:
                t = f.get_tensor(key)
                if t.dtype == torch.float8_e4m3fn:
                    out[key] = dequant_fp8(t, f.get_tensor(key + "_scale_inv")).to(dtype)
                else:
                    out[key] = t.to(dtype)
            save_file(out, str(tmp / shard))
    tmp.rename(dst)
    print(f"dequantized {src} -> {dst} in {time.time() - t0:.1f}s", flush=True)


def run_oracle(oracle: Path, model: Path, ids, out_dir: Path, threads: int,
               dump_hidden: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(oracle), "--model", str(model), "--ids", " ".join(map(str, ids)),
           "--out", str(out_dir), "--threads", str(threads)]
    if dump_hidden:
        cmd.append("--dump-hidden")
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(run.stdout[-4000:] + run.stderr[-4000:])


def last_logits(path: Path, vocab: int) -> np.ndarray:
    a = np.memmap(path, dtype="<f8", mode="r")
    if a.size % vocab:
        raise SystemExit(f"bad logits size {a.size}")
    rows = a.size // vocab
    return np.asarray(a[(rows - 1) * vocab:rows * vocab])


def generate(oracle: Path, model: Path, ids, n: int, threads: int) -> dict:
    vocab = int(json.loads((model / "config.json").read_text()).get("vocab_size")
                or json.loads((model / "config.json").read_text())["text_config"]["vocab_size"])
    work = Path(tempfile.mkdtemp(prefix="shelf_oracle_gen_"))
    view = list(ids)
    generated, steps = [], []
    try:
        for _ in range(n):
            run_oracle(oracle, model, view, work, threads)
            logits = last_logits(work / "logits.bin", vocab)
            order = np.argsort(logits)[::-1][:5]
            steps.append({
                "step": len(generated),
                "token_id": int(order[0]),
                "top5": [[int(t), float(logits[t])] for t in order],
                "margin": float(logits[order[0]] - logits[order[1]]),
            })
            generated.append(int(order[0]))
            view.append(int(order[0]))
            print(f"token {len(generated)}/{n}: {generated[-1]}", flush=True)
            (work / "logits.bin").unlink()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"generated_ids": generated, "steps": steps}


def package_hidden(hidden_dir: Path, start: int, positions: int, hidden_size: int,
                   num_layers: int, out_path: Path, source_tokens: int) -> dict:
    tensors = {}
    for i in range(num_layers + 1):
        a = np.fromfile(hidden_dir / f"hidden_{i:02d}.bin", dtype="<f8")
        if a.size % hidden_size:
            raise SystemExit(f"hidden_{i:02d}: bad size {a.size}")
        a = a.reshape(-1, hidden_size)[start:start + positions]
        if a.shape[0] != positions:
            raise SystemExit(f"hidden_{i:02d}: only {a.shape[0]} positions")
        tensors[f"hidden_{i:02d}"] = torch.from_numpy(a.astype(np.float32))
    save_file(tensors, str(out_path))
    return {
        "file": out_path.name,
        "dtype": "float32",
        "shape": [positions, hidden_size],
        "positions": positions,
        "start_position": start,
        "source_prefill_tokens": source_tokens,
        "layout": {
            "hidden_00": "embedding output = decoder layer 0 の入力",
            "hidden_01..hidden_{:02d}".format(num_layers - 1):
                "decoder layer 0..{} の残差出力 = 次の layer の入力".format(num_layers - 2),
            "hidden_{:02d}".format(num_layers): "final norm output",
        },
        "note": "qwen35-f64-oracle は final norm を hidden_{:02d} に書くため、"
                "layer {} の残差出力は個別の key を持たない".format(
                    num_layers, num_layers - 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--oracle-dir", default="../references/qwen35-f64-oracle")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--hidden-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--work-dir", default=".cache/qwen38-27b-fp8-f32")
    ap.add_argument("--generation-dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--hidden-scope", choices=["prefill", "generated", "both"], default="both")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-hidden", required=True)
    ap.add_argument("--out-hidden-generated", default=None)
    a = ap.parse_args()

    model_dir = Path(a.model_dir)
    oracle = Path(a.oracle_dir) / "qwen35-oracle"
    if not oracle.exists():
        raise SystemExit(f"oracle binary not found: {oracle}")
    ids = [int(x) for x in a.ids.replace(",", " ").split()]

    f32_dir = Path(a.work_dir)
    dequantize_model(model_dir, f32_dir, torch.float32)

    gen_dir = f32_dir
    if a.generation_dtype == "bf16":
        gen_dir = f32_dir.with_name(f32_dir.name.replace("f32", "bf16"))
        dequantize_model(model_dir, gen_dir, torch.bfloat16)

    cfg = json.loads((model_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)

    gen = generate(oracle, gen_dir, ids, a.max_new_tokens, a.threads)

    hidden_states = {}
    if a.hidden_scope in ("prefill", "both"):
        hidden_dir = Path(tempfile.mkdtemp(prefix="shelf_oracle_hidden_"))
        try:
            run_oracle(oracle, f32_dir, ids, hidden_dir, a.threads, dump_hidden=True)
            hidden_states["prefill"] = package_hidden(
                hidden_dir, 0, a.hidden_tokens, tc["hidden_size"],
                tc["num_hidden_layers"], Path(a.out_hidden), len(ids))
        finally:
            shutil.rmtree(hidden_dir, ignore_errors=True)

    if a.hidden_scope in ("generated", "both"):
        if not a.out_hidden_generated:
            raise SystemExit("--out-hidden-generated is required for --hidden-scope generated/both")
        prefix = ids + gen["generated_ids"][:a.hidden_tokens]
        hidden_dir = Path(tempfile.mkdtemp(prefix="shelf_oracle_hidden_gen_"))
        try:
            run_oracle(oracle, f32_dir, prefix, hidden_dir, a.threads, dump_hidden=True)
            meta = package_hidden(
                hidden_dir, len(ids), a.hidden_tokens, tc["hidden_size"],
                tc["num_hidden_layers"], Path(a.out_hidden_generated), len(prefix))
            meta["generated_token_indices"] = list(range(a.hidden_tokens))
            hidden_states["generated"] = meta
        finally:
            shutil.rmtree(hidden_dir, ignore_errors=True)

    try:
        from tokenizers import Tokenizer
        decoded = Tokenizer.from_file(str(model_dir / "tokenizer.json")).decode(
            gen["generated_ids"])
    except Exception:
        decoded = None
    fixture = {
        "format": "shelf-qwen38-27b-fp8-cpu-oracle-v1",
        "model": {
            "name": "Qwen3.8-27B-FP8",
            "path": str(model_dir),
            "quantization": {
                "quant_method": (cfg.get("quantization_config") or {}).get("quant_method"),
                "fmt": (cfg.get("quantization_config") or {}).get("fmt"),
                "weight_block_size": (cfg.get("quantization_config") or {}).get("weight_block_size"),
            },
            "geometry": {
                "hidden_size": tc["hidden_size"],
                "num_hidden_layers": tc["num_hidden_layers"],
                "num_attention_heads": tc["num_attention_heads"],
                "num_key_value_heads": tc["num_key_value_heads"],
                "head_dim": tc["head_dim"],
                "vocab_size": tc["vocab_size"],
            },
        },
        "oracle": {
            "implementation": "qwen35-f64-oracle",
            "dtype": "f64",
            "weights": "FP8 E4M3 x BF16 scale を F32 へ厳密 dequantize",
            "generation_weights": a.generation_dtype,
            "hidden_weights": "f32",
            "threads": a.threads,
        },
        "case": {
            "prompt": a.prompt,
            "input_ids": ids,
            "max_new_tokens": a.max_new_tokens,
            "generated_ids": gen["generated_ids"],
            "decoded": decoded,
            "steps": gen["steps"],
        },
        "hidden_states": hidden_states,
    }
    out_json = Path(a.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {a.out_json}, {a.out_hidden}"
          + (f", {a.out_hidden_generated}" if a.out_hidden_generated else ""))


if __name__ == "__main__":
    main()
