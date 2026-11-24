# HunYuanDenseV1（Hy-MT3-1.8B / Hy-MT3-7B）R9700 実装・検証記録（2026-11-24）

- **実験 ID**: EXP-20261124-HYMT3
- **結論**: **ACCEPT**（Hy-MT3-1.8B / 7B の BF16 および EXL3-INT8 4 bpw を `shelf` 本体へ統合し、単体テスト・PyTorch 参照比較・KLD・サーバー E2E・速度計測のすべてを通過）
- **実行環境**: AMD Radeon AI PRO R9700 (`gfx1201`, `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`), ROCm 10.2 (`/opt/rocm/core-10.2`), `flock /tmp/r9700.lock` 排他制御
- **ビルド構成**: `Release`, `CMAKE_HIP_ARCHITECTURES=gfx1201`, `ROCm 10.2`（`SHELF_BUILD_TESTS=ON`, baseline commit `882b2078`）

## 1. アーキテクチャ統合と判明した不具合修正

1. **Post-RoPE QK RMSNorm & 標準 RMSNorm**
   - `self_attn.q_proj` / `self_attn.k_proj` の出力へ先に `RoPENode`（`bf16_intermediates = true`）を適用し、その後で `query_layernorm` / `key_layernorm`（`OFFSET_BF16_SMALL_GROUP`, `group_size = 128`, `weight_offset = 0.0f`）を適用した。
2. **NTK-Alpha 静的 RoPE**
   - `rope_scaling = {"type": "dynamic", "alpha": 1000.0}` に対し `EffectiveRopeProfile::Type::NtkAlpha` を実装し、`Executor` 初期化で $\theta_{\text{eff}} = \theta \cdot \alpha^{d/(d-2)}$ から float32 周波数テーブルを構築する。
3. **128 境界パディング済み EXL3 `lm_head` と `Exl3` / `BF16` 出力行ストライド整合**
   - EXL3 版では `config.json` に `tie_word_embeddings: true` が残ったまま 6 bpw の `lm_head`（1.8B: `120832`, 7B: `128256`）が独立に存在するため、存在時は独立重みを優先ロードし、`SamplingNode` に有効 `vocab_size`（1.8B: `120818`, 7B: `128167`）を渡してパディング領域の選択を防いだ。
   - **修正した既存不整合**: `program.cpp` が `WORKSPACE` 値に 16 バイト境界行ストライド（`120818 -> 120820`, `128167 -> 128168`）を付与した後に `EXTERNAL_OUTPUT` へ変更する一方、`Executor::logits` はパディング無し連続ストライドで確保されていたため、非 4 倍数語彙の BF16 複数行 prefill（`--dump-prefill`）で行ズレが発生し、`sampling_dispatch.hip` でも `row_stride != feature_count` による correctness fallback が起きていた。`Executor::logits` を 16 バイト境界ストライド（`{MO, logits_width}`, `{logits_stride, 1}`）で確保し、`sampling_dispatch.hip` を `row_stride < feature_count` に緩和し、`ComputeSession::copy_logits` / `shelf-bench quality` を `strides_data()[0]` 基準の `hipMemcpy2D` に統一したことで、BF16・EXL3 ともフォールバック 0 件かつ高一致となった。
4. **BF16 / Server selector**
   - `kv_append_selector.cpp` に 7B 向けの `(BF16, kv_heads=8, head_dim=128)` ルール、`paged_attention_selector.cpp` に `(16, 4, 128)`・`(32, 8, 128)`、`linear_selector.cpp` / `elementwise_selector.cpp` / `sampling_selector.cpp` に 1.8B・7B 形状を追加した。
   - `shelf_server.py` / `chat_service.py` に非 reasoning 翻訳モデル用の `supports_reasoning=False` を導入し、生成テキストが `reasoning_content` へ誤分離される問題を防止した。

## 2. 品質・参照一致結果

### PyTorch HF (`HunYuanDenseV1ForCausalLM`, BF16 CPU) との比較（`--dump-prefill` 全位置, FP64 KL）

| モデル | 候補形式 | 評価位置数 | Mean KLD | P95 KLD | Max KLD | Top-1 一致率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hy-MT3-1.8B | Shelf BF16 | 53 | 0.001273 | 0.006080 | 0.016354 | 98.11% |
| Hy-MT3-1.8B | Shelf EXL3-4bpw | 53 | 0.112006 | 0.508044 | 2.428096 | 94.34% |
| Hy-MT3-7B | Shelf BF16 | 51 | 0.000540 | 0.001200 | 0.007984 | 98.04% |
| Hy-MT3-7B | Shelf EXL3-4bpw | 51 | 0.020482 | 0.060941 | 0.127560 | 94.12% |

### `shelf-bench quality`（BF16 基準 vs EXL3-INT8 4bpw, `kv-cache-dtype=bf16`）

| モデル | context / positions | `QUALITY_KL_MEAN` | `QUALITY_TOP1_AGREEMENT` | `QUALITY_TOP5_OVERLAP` |
| --- | --- | ---: | ---: | ---: |
| Hy-MT3-1.8B | 24 / 29 | 0.018801 | 1.000000 | 0.917241 |
| Hy-MT3-7B | 24 / 27 | 0.022355 | 0.888889 | 0.955556 |

## 3. R9700 性能測定結果

条件: `shelf-bench tg --mode greedy --context 128 --tokens 128 --warmup 16 --device 0`、`shelf-bench pp --mode forward --prompt-tokens 512 --runs 5 --warmup 2 --device 0`、`kv-cache-dtype=bf16`、R9700 単体。

| モデル | 形式 | Decode (`tg` mean) | Decode (`tg` p50) | Prefill (`pp512` median) |
| --- | --- | ---: | ---: | ---: |
| Hy-MT3-1.8B | BF16 | 118.15 tok/s | 118.89 tok/s | 10152.92 tok/s |
| Hy-MT3-1.8B | EXL3-4bpw | 178.57 tok/s | 186.75 tok/s | 7588.33 tok/s |
| Hy-MT3-7B | BF16 | 37.64 tok/s | 37.74 tok/s | 3955.68 tok/s |
| Hy-MT3-7B | EXL3-4bpw | 89.55 tok/s | 91.77 tok/s | 3441.24 tok/s |

## 4. `shelf-server` E2E 翻訳検証（通常応答 & SSE ストリーミング）

- **1.8B BF16 & EXL3-4bpw**（両者で出力完全一致、`finish_reason="stop"`、`reasoning_content` なし）:
  - En→Ja: `光学センサーは、静かな谷全体において正確な波長データを記録した。`
  - Ja→En: `In the quiet valley, the researchers continued to record data until sunset.`
- **7B EXL3-4bpw** (`finish_reason="stop"`, `reasoning_content` なし):
  - En→Ja: `光センサーは、静かな谷全体にわたって正確な波長データを記録した。`
  - Ja→En: `In a quiet valley, the researchers continued recording data until sunset.`
- **7B BF16** (`finish_reason="stop"`, `reasoning_content` なし):
  - En→Ja: `光センサーは、静かな谷全体にわたって正確な波長データを記録した。`
  - Ja→En: `In the quiet valley, the researchers continued recording data until sunset.`
