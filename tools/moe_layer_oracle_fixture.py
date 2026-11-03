#!/usr/bin/env python3
"""Qwen3.5 MoE 1 layer の f64 参照 fixture を生成する。

referee は HF transformers の `Qwen3_5MoeSparseMoeBlock` を float64 で実行したものである。
docs/research/20261103/qwen36_35b_a3b_moe_structure.md §6 に写した順序
(softmax f32 → top-k 確率降順・同値は小 index → top-k 合計で renormalize →
expert 昇順に index_add → 最後に sigmoid(shared_expert_gate) * shared) は
この HF 実装そのものから取っている。

expert 重みは Shelf の MXFP4 codec と同じ丸め規則 (E8M0 = 最近べき乗 2 / 4、
E2M1 は round-nearest-even) で量子化し、**量子化後の値**を HF に入れて参照を作る。
これで fixture と MXFP4 payload の対応が固定される。量子化前の BF16 重みも併せて
保存するので、テスト側は自前 codec が同じ payload を再現することを確認できる。

例:
  python3 tools/moe_layer_oracle_fixture.py \
      --out-dir tests/fixtures/qwen35moe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

GROUP = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e8m0_scale_byte(amax: np.float32) -> int:
    """Shelf `e8m0_scale_byte` と同一の bit 操作 (RNE power-of-two / 4)。"""
    if np.isnan(amax):
        return 255
    if amax == 0.0:
        return 1
    bits = np.float32(amax).view(np.uint32)
    biased = int(((bits + np.uint32(1 << 21)) >> np.uint32(23)) & np.uint32(0xFF))
    if biased >= 255:
        return 254
    exponent = biased - 127 - 2
    exponent = max(-127, min(127, exponent))
    return exponent + 127


def e2m1_encode(value: np.float32) -> int:
    """Shelf `e2m1_encode_u8` と同一の bit 操作 (round nearest even)。"""
    bits = np.float32(value).view(np.uint32)
    sign = int(bits & np.uint32(0x80000000))
    exponent = int((bits >> np.uint32(23)) & np.uint32(0xFF))
    mantissa = int(bits & np.uint32(0x007FFFFF))
    if exponent < 127:
        shift = 126 - exponent
        mantissa = 0 if shift >= 32 else ((0x400000 | (mantissa >> 1)) >> shift)
    exponent = (126 if exponent < 126 else exponent) - 126
    code = (((exponent << 2) | (mantissa >> 21)) + 1) >> 1
    if code > 7:
        code = 7
    return ((sign >> 28) | code) & 0xFF


def quantize_mxfp4(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[rows, cols] float32 -> codes [rows, cols/2] u8, scales [rows, cols/32] u8。"""
    rows, cols = matrix.shape
    if cols % GROUP:
        raise SystemExit(f"cols {cols} is not a multiple of {GROUP}")
    codes = np.zeros((rows, cols // 2), dtype=np.uint8)
    scales = np.zeros((rows, cols // GROUP), dtype=np.uint8)
    for row in range(rows):
        for group in range(cols // GROUP):
            begin = group * GROUP
            block = matrix[row, begin:begin + GROUP]
            byte = e8m0_scale_byte(np.float32(np.max(np.abs(block))))
            scales[row, group] = byte
            scale = np.float32(2.0 ** (int(byte) - 127)) if byte != 0 else np.float32(2.0 ** -127)
            inv = np.float32(1.0) / scale if scale > 0 else np.float32(0.0)
            for col in range(begin, begin + GROUP):
                code = e2m1_encode(np.float32(matrix[row, col]) * inv)
                if (col & 1) == 0:
                    codes[row, col // 2] = (int(codes[row, col // 2]) & 0xF0) | code
                else:
                    codes[row, col // 2] = (int(codes[row, col // 2]) & 0x0F) | (code << 4)
    return codes, scales


def dequantize_mxfp4(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, half = codes.shape
    out = np.zeros((rows, half * 2), dtype=np.float64)
    for row in range(rows):
        for group in range(scales.shape[1]):
            byte = int(scales[row, group])
            scale = 2.0 ** (byte - 127) if byte != 0 else 2.0 ** -127
            for col in range(group * GROUP, (group + 1) * GROUP):
                nib = codes[row, col // 2]
                code = int(nib & 0x0F) if (col & 1) == 0 else int(nib >> 4)
                value = float(E2M1[code & 7])
                if code & 8:
                    value = -value
                out[row, col] = value * scale
    return out


def build_config_json(hidden: int, inter: int, shared_inter: int, experts: int, top_k: int,
                      layers: int, tokens: int) -> str:
    return json.dumps(
        {
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 256,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_experts": experts,
                "num_experts_per_tok": top_k,
                "moe_intermediate_size": inter,
                "shared_expert_intermediate_size": shared_inter,
                "hidden_act": "silu",
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": hidden // 4,
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention"] * layers,
                "max_position_embeddings": 1024,
            },
            "quantization_config": {
                "quant_method": "quark",
                "version": "0.13",
                "global_quant_config": {
                    "weight": {
                        "dtype": "fp4",
                        "qscheme": "per_group",
                        "group_size": GROUP,
                        "is_dynamic": False,
                        "scale_format": "e8m0",
                        "ch_axis": -1,
                    },
                    "input_tensors": {
                        "dtype": "fp4",
                        "qscheme": "per_group",
                        "group_size": GROUP,
                        "is_dynamic": True,
                        "scale_format": "e8m0",
                    },
                },
                "export": {"weight_format": "real_quantized", "pack_method": "reorder"},
            },
            "shelf_moe_fixture": {
                "tokens": tokens,
                "hidden": hidden,
                "moe_intermediate_size": inter,
                "shared_expert_intermediate_size": shared_inter,
                "num_experts": experts,
                "num_experts_per_tok": top_k,
                "seed": 20261104,
                "expert_weight_scale": 0.08,
                "referee": "transformers Qwen3_5MoeSparseMoeBlock in float64",
            },
        },
        indent=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="tests/fixtures/qwen35moe")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inter", type=int, default=128)
    ap.add_argument("--shared-inter", dest="shared_inter", type=int, default=128)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", dest="top_k", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    if args.hidden % 128 or args.inter % 128:
        raise SystemExit("hidden and moe_intermediate_size must be multiples of 128 (W4A8 K group)")
    if (2 * args.inter) % 16 or args.hidden % 16:
        raise SystemExit("expert output rows must be multiples of 16 (native row tile)")

    from safetensors.torch import save_file

    torch.manual_seed(20261104)
    rng = np.random.default_rng(20261104)
    H, I, SH, E, K, T = (args.hidden, args.inter, args.shared_inter, args.experts,
                         args.top_k, args.tokens)

    def bf16(rows: int, cols: int, scale: float) -> torch.Tensor:
        values = torch.randn(rows, cols, dtype=torch.float32) * scale
        return values.to(torch.bfloat16)

    router_w = bf16(E, H, 0.12)
    shared_gate_w = bf16(1, H, 0.2)
    shared = {
        "shared_expert.gate_proj.weight": bf16(SH, H, 0.08),
        "shared_expert.up_proj.weight": bf16(SH, H, 0.08),
        "shared_expert.down_proj.weight": bf16(H, SH, 0.08),
    }
    xn = bf16(T, H, 0.5)

    prefix = "model.language_model.layers.0.mlp."
    checkpoint: dict[str, torch.Tensor] = {
        prefix + "gate.weight": router_w,
        prefix + "shared_expert_gate.weight": shared_gate_w,
    }
    for name, tensor in shared.items():
        checkpoint[prefix + name] = tensor

    quantized: dict[str, np.ndarray] = {}
    reference: dict[str, torch.Tensor] = {}
    for e in range(E):
        for proj, rows, cols in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            key = prefix + f"experts.{e}.{proj}.weight"
            original = bf16(rows, cols, 0.08)
            reference[key] = original
            original_f32 = original.to(torch.float32).numpy()
            codes, scales = quantize_mxfp4(original_f32)
            quantized[key] = codes
            quantized[key + "_scale"] = scales
            checkpoint[key] = torch.from_numpy(codes).contiguous()
            checkpoint[key + "_scale"] = torch.from_numpy(scales).contiguous()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "config.json").write_text(
        build_config_json(H, I, SH, E, K, 1, T) + "\n", encoding="utf-8")

    # referee: HF MoE block in float64, on the dequantized MXFP4 expert weights.
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    cfg = Qwen3_5MoeConfig(hidden_size=H, num_experts=E, num_experts_per_tok=K,
                           moe_intermediate_size=I, shared_expert_intermediate_size=SH,
                           hidden_act="silu")
    block = Qwen3_5MoeSparseMoeBlock(cfg).to(torch.float64).eval()
    with torch.no_grad():
        block.gate.weight.copy_(router_w.to(torch.float64))
        block.shared_expert_gate.weight.copy_(shared_gate_w.to(torch.float64))
        block.shared_expert.gate_proj.weight.copy_(
            shared["shared_expert.gate_proj.weight"].to(torch.float64))
        block.shared_expert.up_proj.weight.copy_(
            shared["shared_expert.up_proj.weight"].to(torch.float64))
        block.shared_expert.down_proj.weight.copy_(
            shared["shared_expert.down_proj.weight"].to(torch.float64))
        gate_up = torch.zeros(E, 2 * I, H, dtype=torch.float64)
        down = torch.zeros(E, H, I, dtype=torch.float64)
        for e in range(E):
            gate = dequantize_mxfp4(quantized[prefix + f"experts.{e}.gate_proj.weight"],
                                    quantized[prefix + f"experts.{e}.gate_proj.weight_scale"])
            up = dequantize_mxfp4(quantized[prefix + f"experts.{e}.up_proj.weight"],
                                  quantized[prefix + f"experts.{e}.up_proj.weight_scale"])
            dn = dequantize_mxfp4(quantized[prefix + f"experts.{e}.down_proj.weight"],
                                  quantized[prefix + f"experts.{e}.down_proj.weight_scale"])
            gate_up[e, :I, :] = torch.from_numpy(gate)
            gate_up[e, I:, :] = torch.from_numpy(up)
            down[e] = torch.from_numpy(dn)
        block.experts.gate_up_proj.copy_(gate_up)
        block.experts.down_proj.copy_(down)

        x64 = xn.to(torch.float64).unsqueeze(0)
        hidden_states = x64.view(-1, H)
        shared_out = block.shared_expert(hidden_states)
        _, routing_weights, selected_experts = block.gate(hidden_states)
        expert_out = block.experts(hidden_states, selected_experts, routing_weights)
        # The gate linear output is stored both raw (what a BF16 linear produces, which the
        # sigmoid elementwise kernel consumes) and activated (what HF adds).
        gate_logit = block.shared_expert_gate(hidden_states)
        gate_sigmoid = torch.sigmoid(gate_logit)
        combined = expert_out + gate_sigmoid * shared_out
        # The op-by-op decomposition above must be what the module itself does, so the
        # fixture cannot drift from HF semantics without failing here.
        whole = block(x64)
        if not torch.equal(whole.view(-1, H), combined):
            raise SystemExit("the op-by-op reference differs from Qwen3_5MoeSparseMoeBlock.forward")

    oracle = {
        "xn": xn,
        "expected_routed": expert_out.to(torch.float32),
        "shared_expert_out": shared_out.to(torch.float32),
        "shared_gate_sigmoid": gate_sigmoid.squeeze(-1).to(torch.float32),
        "shared_gate_logit": gate_logit.squeeze(-1).to(torch.bfloat16),
        "hf_selected_experts": selected_experts.to(torch.int32),
        "hf_routing_weights": routing_weights.to(torch.float32),
        "expected_total": combined.to(torch.float32),
    }
    # Only expert 0 keeps its pre-quantization BF16 weight: it is enough for the test to
    # check that the Shelf codec reproduces the fixture payload, and keeps the fixture small.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        oracle[f"reference_bf16.experts.0.{proj}.weight"] = reference[
            prefix + f"experts.0.{proj}.weight"]
    save_file(oracle, str(out_dir / "oracle.safetensors"), metadata={"format": "pt"})

    spread = max(
        int(np.max(v)) - int(np.min(v)) for k, v in quantized.items() if k.endswith("_scale")
    )
    print(f"fixture written to {out_dir}")
    print(f"tokens {T} hidden {H} moe_inter {I} shared {SH} experts {E} top_k {K}")
    print(f"E8M0 byte spread over the fixture: {spread}")
    print("routing weights (first token):",
          routing_weights[0].tolist(), "experts:", selected_experts[0].tolist())


if __name__ == "__main__":
    main()
