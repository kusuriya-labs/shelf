# Qwen3.8-Flash-Next モデル・tensor manifest 調査

- ID: QWEN4EXP-G0-INVENTORY
- 実施日: 2026-10-24
- 対象 checkpoint: `/opt/zen/work/models/Qwen3.8-Flash-Next`（BF16）
- 参照 revision: HF `de4b8e4d43b917e7706784d8bb445c9af86a3540`
  （`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/refs/main` と
  全 `.metadata` が一致）
- 手法: `model.safetensors.index.json` と各 shard の safetensors ヘッダのみを解析。
  重み本文はロードしていない。

## 1. checkpoint 全体

| 項目 | 値 |
| --- | --- |
| `model_type` | `qwen4_exp`（text 側 `qwen4_exp_text`） |
| `architectures` | `["Qwen4ExpForConditionalGeneration"]` |
| shard 数 | 131（`model-00001-of-00131` … `model-00131-of-00131`） |
| index `total_size` | 359,999,963,128 B（約 335 GiB） |
| shard 実 bytes 合計 | 360,000,192,888 B |
| tensor 総数 | 1,658 |
| dtype census | BF16: 1,655 / I64: 3 |
| quantization_config | **なし（BF16 checkpoint）** |
| MTP tensor | 31（`mtp.*`） |
| vision tensor | 333（`model.visual.*`、本 Gate の対象外） |

`docs/developer/qwen4exp.md` §12 の「tensor 総数 1658 (1294 required + 31 MTP + 333 vision)」と
一致する。I64 が 3 tensor 実在する点も同 doc の記載と一致した。

## 2. text_config の実値

`config.json` の `text_config` 全項目を実 checkpoint から取得した。
`docs/developer/qwen4exp.md` §1.1 との差分は**なし**（全項目一致）。

補足: `config.json` には `rope_parameters` に `mrope_section: [11,11,10]`、
`mrope_interleaved: true`、`rope_type: default` が入っている。
`bos_token_id` / `eos_token_id` は 248044、`pad_token_id` は null。

`layer_types` は 48 件で `full_attention` が index `3,7,11,…,47`（12 件）、
`linear_attention` が残り 36 件。実 tensor の存在層と一致する:

| pattern | 実在 layer index（0-indexed） |
| --- | --- |
| `self_attn.*`（QSA） | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47（12 層） |
| `linear_attn.*`（GDN） | 0,1,2,4,5,6,8,…,46（36 層） |
| `ple.*` | 1 のみ |
| `mlp.experts.*` | 0..47 全 48 層 |

## 3. PLE

| tensor | dtype | shape | shard | 備考 |
| --- | --- | --- | --- | --- |
| `…ple_embedding.layer_multipliers` | **I64** | `[3]` | `model-00005` | 24 B |
| `…ple_embedding.ngram_heads_vocab_sizes` | **I64** | `[16]` | `model-00037` | 128 B |
| `…ple_embedding.ngram_heads_offsets` | **I64** | `[16]` | `model-00037` | 128 B |
| `…ngram_embedding.shard_0.weight` … `shard_127.weight` | BF16 | `[2500012, 160]` | shard_0=`model-00005` ほか | 各 800,003,840 B |

- 128 shard 合計 102,400,491,520 B（約 95.4 GiB）。
- row 幅は `160 * 2 = 320 bytes/row`。1 token 参照 16 row = 5,120 bytes。
  `docs/developer/qwen4exp.md` §9.1 および手順書 §7.2 の contract と一致。
- `shard_0` の data offsets は `begin=1755878424 / end=2555882264`
  （`model-00005-of-00131.safetensors`、`data_base_offset=2400`）。
  → 絶対 file offset = `2400 + 1755878424` から row が並ぶ。
- PLE の small tensor（`ple.key_proj` / `value_proj` / `conv1d` / `norm_*`）は
  すべて `model-00037` に在る。

**G1 への含意**: `SType::I64` が無いと PLE の 3 tensor を読めない。
実 checkpoint で I64 が必須であることが確定した。

## 4. fused expert

| tensor | dtype | shape | 例 |
| --- | --- | --- | --- |
| `layers.N.mlp.experts.gate_up_proj` | BF16 | `[512, 1280, 2560]` | layer0 = `model-00002`、3,355,443,200 B |
| `layers.N.mlp.experts.down_proj` | BF16 | `[512, 2560, 640]` | layer0 = `model-00003`、1,677,721,600 B |
| `layers.N.mlp.gate.weight`（router） | BF16 | `[512, 2560]` | `model-00003` |
| `layers.N.mlp.shared_expert.gate/up/down_proj.weight` | BF16 | `[640,2560]` / `[2560,640]` | |
| `layers.N.mlp.shared_expert_gate.weight` | BF16 | `[1, 2560]` | |

- per-expert の論理形状は `gate_up_proj[e] = [1280, 2560]`、`down_proj[e] = [2560, 640]`。
  手順書 §8 の記載と一致。
- **FP8 tensor / scale tensor は存在しない**。FP8 checkpoint
  `Qwen3.8-Flash-Next-FP8` はローカルに無い（§6 参照）。

## 5. その他主要 tensor（実 shape）

| tensor | dtype | shape |
| --- | --- | --- |
| `embed_tokens.weight` / `lm_head.weight` | BF16 | `[248320, 2560]`（tie しない） |
| `hyper_connection_mixer.hc_norm.weight` | BF16 | `[10240]` |
| `hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | `[320, 10240]` |
| `hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | `[10240, 320]` |
| `layers.N.{attn,mlp}_hyper_connection.hc_norm.weight` | BF16 | `[10240]` |
| `layers.N.*_hyper_connection.block_inject_weight.weight` | BF16 | `[4, 10240]` |
| `layers.N.linear_attn.in_proj_qkv.weight` | BF16 | `[10240, 2560]` |
| `layers.N.linear_attn.conv1d.weight` | BF16 | `[10240, 1, 4]` |
| `layers.N.linear_attn.out_proj.weight` | BF16 | `[2560, 6144]` |
| `layers.N.linear_attn.norm.weight` | BF16 | `[128]` |
| `layers.N.linear_attn.A_log` / `dt_bias` | BF16 | `[48]` |
| `layers.N.self_attn.q_proj.weight` | BF16 | `[12288, 2560]` |
| `layers.N.self_attn.k_proj.weight` / `v_proj` | BF16 | `[512, 2560]` |
| `layers.N.self_attn.o_proj.weight` | BF16 | `[2560, 6144]` |
| `layers.N.self_attn.q_norm` / `k_norm.weight` | BF16 | `[256]` |
| `layers.N.self_attn.indexer.index_qk_proj.weight` | BF16 | `[640, 2560]` |
| `layers.N.self_attn.indexer.{q,k}_layernorm.weight` | BF16 | `[128]` |
| `layers.N.ple.key_proj.weight` | BF16 | `[10240, 2560]` |
| `layers.N.ple.value_proj.weight` | BF16 | `[2560, 2560]` |
| `layers.N.ple.norm_{key,query,conv}.weight` | BF16 | `[10240]` |

`docs/developer/qwen4exp.md` §12 の表と食い違いは検出されなかった。

## 6. 未取得・未確定

- **`Qwen3.8-Flash-Next-FP8` はローカルに存在しない**。
  したがって FP8 の `quantization_config`（quant_method / fmt / activation_scheme /
  weight_block_size）と scale tensor 名・shape・dtype は**未確定**。
  推測で埋めない。既存の `Qwen3.8-27B-FP8` の実値は
  `quant_method=fp8, fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128,128]`
  であり、これを Flash-Next-FP8 に外挿してはならない。
- Hugging Face Transformers `qwen4_exp` の参照 commit は未固定。
  インストール済み `transformers 5.14.1` には `models/qwen4_exp` が**存在しない**。
  参照実装の取得は別途行う（§2.2 の revision 固定義務が未達の項目）。

---

## 7. 手順書 §2.2 の固定成果物

### 固定できたもの

| 成果物 | 値 |
| --- | --- |
| PhaseShift HEAD（作業開始時） | `9825d10896fea4f4d9736f918617c41322ae53b2` |
| `config.json` SHA256 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| `model.safetensors.index.json` SHA256 | `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de` |
| `generation_config.json` SHA256 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| HF model revision (BF16) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| ROCm / HIP / ROCk | ROCm 10.0.0（pip SDK）/ HIP 7.15.26333 / ROCk 7.1.3.31500000 |
| hipFile | `libhipfile.so.0.4.0`（`hipFileGetVersion` 実測 0.4.0） |
| GPU / OS | AMD Radeon AI PRO R9700 (gfx1201) ×4、Linux kernel 7.0.0-38-generic |
| 試験に使用した tensor（PLE I64） | `layer_multipliers` `[3]` / `ngram_heads_vocab_sizes` `[16]` / `ngram_heads_offsets` `[16]`（実値は §3 に記載） |
| 試験に使用した tensor（PLE table） | `ngram_embedding.shard_{0..127}.weight` BF16 `[2500012,160]`（`data_base_offset` 実測: `model-00005` = 2400、`model-00037` = 2776） |
| 試験に使用した tensor（fused expert） | `gate_up_proj` BF16 `[512,1280,2560]` / `down_proj` `[512,2560,640]` |
| oracle 生成環境 | Python 3.12.3 / PyTorch `2.13.0+rocm10.0.0` / `safetensors 0.8.0` |

### 未達（引き継ぎ）

| 成果物 | 状態 |
| --- | --- |
| `Qwen/Qwen3.8-Flash-Next-FP8` の revision と `quantization_config` | **ローカルに checkpoint が無く取得不能** |
| Hugging Face Transformers `modeling_qwen4_exp.py` / `configuration_qwen4_exp.py` の commit ID | **未固定**（インストール済み `transformers 5.14.1` に `models/qwen4_exp` が存在しない） |
