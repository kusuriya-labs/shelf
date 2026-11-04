#!/usr/bin/env python3
"""GDN gate 契約を固定する tiny fixture を生成する。

合成 checkpoint（2 層: 層 0 = GDN、層 1 = full attention、`output_gate_type` を
**宣言しない**）を作り、外部の qwen35-f64-oracle で f64 forward を実行して
expected を commit できる形に書き出す。oracle は本 repository に vendor しない
（docs/developer/testing_f64_oracle.md 参照）。

例:
  tools/qwen35_gdn_tiny_fixture.py \
      --out-dir tests/fixtures/qwen35/qwen35_gdn_tiny \
      --oracle-dir ../references/qwen35-f64-oracle

生成物は test_qwen35_gdn_gate_reference が読む。checkpoint 自体は deterministic
（seed 固定・torch 非依存）なので、oracle が無くても checkpoint 側は再生成できる。
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

LAYERS = 2
VOCAB = 512
HIDDEN = 512
INTER = 512
KEY_HEADS = 2
KEY_HEAD_DIM = 128
VALUE_HEADS = 4
VALUE_HEAD_DIM = 128
CONV_KERNEL = 4
ATTN_HEADS = 2
ATTN_KV_HEADS = 1
ATTN_HEAD_DIM = 128
IDS = [331, 154, 404, 49, 74, 96, 374, 59, 219, 38, 88, 444, 428, 71, 246, 92]
SEED = 7


def bf16_bits(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x8000) >> 16).astype(np.uint16)


def write_safetensors(path: Path, tensors: dict) -> None:
    hdr, off, blobs = {}, 0, []
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes()
        hdr[name] = {"dtype": dtype, "shape": list(arr.shape),
                     "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    js = json.dumps({**hdr, "__metadata__": {"format": "np"}})
    js += " " * ((8 - len(js) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js.encode())
        for b in blobs:
            f.write(b)


def build_checkpoint(out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    qk = KEY_HEADS * KEY_HEAD_DIM
    value = VALUE_HEADS * VALUE_HEAD_DIM
    conv_dim = 2 * qk + value

    tensors = {}

    def add(name, shape, kind="mat"):
        n = int(np.prod(shape))
        if kind == "mat":
            v = rng.normal(0.0, 0.05, size=n)
        elif kind == "norm":
            v = rng.uniform(-0.5, 0.5, size=n)
        elif kind == "gdn_norm":
            v = rng.uniform(0.5, 1.5, size=n)
        elif kind == "dt_bias":
            v = rng.uniform(0.5, 1.5, size=n)
        elif kind == "a_log":
            tensors[name] = ("F32", np.log(rng.uniform(0.5, 8.0, size=n)).astype(np.float32).reshape(shape))
            return
        tensors[name] = ("BF16", bf16_bits(v).reshape(shape))

    add("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN])
    add("lm_head.weight", [VOCAB, HIDDEN])
    add("model.language_model.norm.weight", [HIDDEN], "norm")
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}."
        add(p + "input_layernorm.weight", [HIDDEN], "norm")
        add(p + "post_attention_layernorm.weight", [HIDDEN], "norm")
        add(p + "mlp.gate_proj.weight", [INTER, HIDDEN])
        add(p + "mlp.up_proj.weight", [INTER, HIDDEN])
        add(p + "mlp.down_proj.weight", [HIDDEN, INTER])
        if layer % 2 == 0:
            add(p + "linear_attn.in_proj_qkv.weight", [conv_dim, HIDDEN])
            add(p + "linear_attn.in_proj_z.weight", [value, HIDDEN])
            add(p + "linear_attn.in_proj_a.weight", [VALUE_HEADS, HIDDEN])
            add(p + "linear_attn.in_proj_b.weight", [VALUE_HEADS, HIDDEN])
            add(p + "linear_attn.conv1d.weight", [conv_dim, 1, CONV_KERNEL])
            add(p + "linear_attn.out_proj.weight", [HIDDEN, value])
            add(p + "linear_attn.norm.weight", [VALUE_HEAD_DIM], "gdn_norm")
            add(p + "linear_attn.dt_bias", [VALUE_HEADS], "dt_bias")
            add(p + "linear_attn.A_log", [VALUE_HEADS], "a_log")
        else:
            add(p + "self_attn.q_proj.weight", [2 * ATTN_HEADS * ATTN_HEAD_DIM, HIDDEN])
            add(p + "self_attn.k_proj.weight", [ATTN_KV_HEADS * ATTN_HEAD_DIM, HIDDEN])
            add(p + "self_attn.v_proj.weight", [ATTN_KV_HEADS * ATTN_HEAD_DIM, HIDDEN])
            add(p + "self_attn.o_proj.weight", [HIDDEN, ATTN_HEADS * ATTN_HEAD_DIM])
            add(p + "self_attn.q_norm.weight", [ATTN_HEAD_DIM], "norm")
            add(p + "self_attn.k_norm.weight", [ATTN_HEAD_DIM], "norm")

    write_safetensors(out_dir / "model.safetensors", tensors)
    # output_gate_type は意図的に宣言しない（参照実装はこの key を読まない）。
    config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": LAYERS,
            "layer_types": ["linear_attention", "full_attention"] * (LAYERS // 2),
            "full_attention_interval": 2,
            "num_attention_heads": ATTN_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS,
            "head_dim": ATTN_HEAD_DIM,
            "linear_num_key_heads": KEY_HEADS,
            "linear_key_head_dim": KEY_HEAD_DIM,
            "linear_num_value_heads": VALUE_HEADS,
            "linear_value_head_dim": VALUE_HEAD_DIM,
            "linear_conv_kernel_dim": CONV_KERNEL,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"checkpoint: {out_dir} tensors={len(tensors)} hidden={HIDDEN} "
          f"conv_dim={conv_dim} value={value}")


def run_oracle(oracle: Path, model: Path, out_dir: Path, threads: int) -> dict:
    r = subprocess.run([str(oracle), "--model", str(model),
                        "--ids", ",".join(str(i) for i in IDS),
                        "--out", str(out_dir), "--dump-hidden",
                        "--threads", str(threads)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"oracle failed: {r.stderr[-800:]}")
    meta = json.loads((out_dir / "meta.json").read_text())
    logits = np.fromfile(out_dir / "logits.bin", dtype=np.float64).reshape(meta["T"], meta["V"])
    return {"meta": meta, "logits": logits, "dir": out_dir}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/fixtures/qwen35/qwen35_gdn_tiny")
    ap.add_argument("--oracle-dir", default="../references/qwen35-f64-oracle")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    build_checkpoint(out_dir)

    oracle = Path(a.oracle_dir) / "qwen35-oracle"
    if not oracle.exists():
        raise SystemExit(f"oracle binary not found: {oracle}")
    work = Path("/tmp/shelf_gdn_tiny_oracle")
    if work.exists():
        shutil.rmtree(work)
    res = run_oracle(oracle, out_dir, work, a.threads)
    meta, logits = res["meta"], res["logits"]

    hidden = {}
    for index in range(meta["n_hidden"]):
        arr = np.fromfile(work / f"hidden_{index:02d}.bin", dtype=np.float64)
        hidden[f"hidden_{index:02d}"] = arr.reshape(meta["T"], HIDDEN)
    write_safetensors(out_dir / "oracle_hidden.safetensors",
                      {k: ("F64", v) for k, v in hidden.items()})

    top = int(logits[-1].argmax())
    ordered = np.sort(logits[-1])
    margin = float(ordered[-1] - ordered[-2])
    fixture = {
        "format": "shelf-qwen35-gdn-tiny-oracle-v1",
        "note": "合成 Qwen3.5 checkpoint（output_gate_type 非宣言）の f64 参照。"
                "層 0 = GDN。hidden_00 = embedding、hidden_01 = 層 0 の残差出力。",
        "case": {
            "input_ids": IDS,
            "prompt_last_argmax": top,
            "prompt_margin_last": margin,
        },
        "model": {
            "geometry": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": LAYERS,
                "linear_num_key_heads": KEY_HEADS,
                "linear_key_head_dim": KEY_HEAD_DIM,
                "linear_num_value_heads": VALUE_HEADS,
                "linear_value_head_dim": VALUE_HEAD_DIM,
                "linear_conv_kernel_dim": CONV_KERNEL,
                "vocab_size": VOCAB,
            },
            "seed": SEED,
            "declares_output_gate_type": False,
        },
        "oracle": {"implementation": "qwen35-f64-oracle", "mode": meta["mode"],
                   "gdn": meta["gdn"], "dtype": meta["dtype"]},
        "layout": {"hidden_00": "embedding", "hidden_01": "decoder layer 0 の残差出力",
                   "hidden_02": "final norm output"},
    }
    (out_dir / "oracle.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out_dir / 'oracle.json'} (last-position argmax={top}, top1-top2={margin:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
