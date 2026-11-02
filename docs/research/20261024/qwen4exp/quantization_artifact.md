# Qwen3.8-Flash-Next-Quark-AWQ-PSQ の生成

- ID: QWEN4EXP-QUANT-01
- 実施日: 2026-10-24
- 目的: [quantization_profile.md](quantization_profile.md) のプロファイルに従い
  Qwen3.8-Flash-Next の mixed-precision artifact を生成する。
- 入力: `/opt/zen/work/models/Qwen3.8-Flash-Next`（BF16、131 shard、1658 tensor、
  359,999,963,128 B）
- 出力: `/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ`
- 判定: **ACCEPT**（artifact として成立。runtime 接続と精度測定は範囲外）

## 1. 方法

shard 単位の file-to-file 変換を行う専用 runner を新規に作った。
この runner はリポジトリに保持せず、Git history に残す。
tensor 名から次の 4 分類を決める。

| 分類 | 対象 | 形式 |
| --- | --- | --- |
| `psq5` | `layers.N.mlp.experts.gate_up_proj` / `down_proj` | PSQ5-CB28、`ue4m4`、block_k 32 |
| `fp8_block` | `layers.N.linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight` | FP8 E4M3、K128N128、BF16 `weight_scale_inv` |
| `fp8_row` | `layers.1.ple...ngram_embedding.shard_N.weight` | FP8 E4M3、行単位 BF16 `weight_scale` |
| `bf16` | 上記以外（MTP / vision / QSA / GR / shared expert / embed / lm_head） | BF16 のままコピー |

- fused expert の 3D tensor は expert 行方向に `[E*I, H]` へ flatten して量子化し、
  manifest に元の `weight_shape` と `rows` / `columns` を保持する。W32 block は行内で
  閉じるため、flatten しても量子化結果は変わらない。
- 変換は 8192 行単位の chunk で行い、shard を 8 worker で並列処理する。
  検証（再構成の比較）は同じ runner の `--verify` で実施した。
- 出力 safetensors は入力と同じ shard 名で書き、`model.safetensors.index.json` を
  再生成する。全 tensor（passthrough を含む）が weight_map に載る。
- `model.safetensors.json` が artifact manifest である。`verification.json` は
  生成時の検証記録である。

## 2. 結果（実測）

### 容量

| 分類 | tensor 数 | 入力 | 出力 |
| --- | ---: | ---: | ---: |
| PSQ5-CB28 | 96 | 241.592 GB | 79.272 GB |
| FP8 block 128x128 | 108 | 4.152 GB | 2.076 GB |
| FP8 行単位 | 128 | 102.400 GB | 51.840 GB |
| BF16 | 1326 | 11.855 GB | 11.855 GB |
| **合計** | 1658 | **359.999 GB** | **145.044 GB** |

- PSQ5 は manifest の `bits_per_weight = 5.25`。
- 出力 shard ファイルの実 bytes 合計は 145,044,514,568 B（index の `total_size` と一致）。
- 出力の dtype 内訳: BF16 1559 / F8_E4M3 236 / U8 288 / I64 3。
  U8 は PSQ5 の 3 stream（96 x 3）、F8_E4M3 は GDN 108 + PLE 128 である。

### 実行時間

- 変換: **1,417.5 s**（8 worker、2 thread/worker）
- 検証込みの合計: **1,974.3 s**
- 変換中の peak は worker あたり数 GB であり、187 GB RAM（available 163 GB）で余裕がある。
- 作業ディスクは出力 145 GB を使用した。

### 検証（`verification.json`）

| 検査 | 対象 | 結果 |
| --- | --- | --- |
| 構造 | 131 shard / 2086 key | missing 0、extra 0 |
| PSQ5 再構成 | 96 tensor から 4 を抽出 | **bit exact = True**、rel error 平均 0.049665（最大 0.049682） |
| FP8 block 復元 | 全 108 tensor | rel error 平均 0.026431（最大 0.026513） |
| FP8 行単位 復元 | 128 tensor から 8 を抽出 | rel error 平均 0.025922（最大 0.025927） |

独立の spot check（runner の verify とは別コード）:

- PLE shard_0 の row 12345: rel 0.02591
- GDN layer 0 `in_proj_qkv` の block (0,0): rel 0.02630
- PSQ5 layer 0 `gate_up_proj` の rows 0:512: rel 0.04951、shape `(512, 2560)`

FP8 block の rel 0.0264 は HF `Qwen3.8-27B-FP8` の同一層で測った 0.0265 と整合する。
PSQ5 は [PSQ5-20261023-03](../../20261023/psq5_cb28_quark_plugin.md) の 27B 実測
0.049693 と同水準である。

## 3. AWQ を適用しなかった理由

計画は Quark の `algo_config` に `AWQConfig` と `PSQ5CB28Config` を並べ、
AWQ（オフラインリカバリ）→ PSQ5 の順に実行するものだった。本環境では適用できない。

確認した事実:

1. **HF クラスが無い** — `transformers 5.14.1` で
   `from transformers import Qwen4ExpForConditionalGeneration` は ImportError になる（実測）。
   vendor した Quark の `model_preparation.py` も `qwen4_exp` 分岐でこのクラスを要求し、
   コメントに「transformers 5.16 以降でしか到達しない」とある。
2. **RAM / VRAM が足りない** — BF16 checkpoint は 180.0B パラメータ = 360 GB。
   本機は RAM 187 GB（available 163 GB）と VRAM 4 x 32 GB = 128 GB で、
   合計しても 360 GB に届かない。AWQ の calibration forward はモデル全体を
   保持する必要があり、swap も無い。
3. **GDN fast path が無い** — `flash_linear_attention` / `causal_conv1d` は未導入で、
   forward は torch の slow path になる。180B MoE の calibration は非現実的である。
4. ネットワークは利用可能（HF / PyPI へ到達できる）が、transformers を上げても
   2 と 3 は解消しない。

そのため `pre_quant_optimization = "none"` とし、artifact 名だけを指示どおり
`Qwen3.8-Flash-Next-Quark-AWQ-PSQ` とした。将来 AWQ を適用する条件は次のとおり。

- 360 GB 以上を保持できる host RAM、または layer streaming 型の calibration forward
- `transformers >= 5.16`（`Qwen4ExpForConditionalGeneration`）
- calibration corpus と KLD / Top1 の評価手段

## 4. artifact の構成

```text
Qwen3.8-Flash-Next-Quark-AWQ-PSQ/
  model-00001-of-00131.safetensors ...   # 入力と同じ shard 構成
  model.safetensors.index.json           # 全 key の weight_map
  model.safetensors.json                 # 量子化 manifest（PSQ5 / FP8 / passthrough）
  verification.json                      # 生成時検証の記録
  config.json / tokenizer ほか           # 入力からコピー
```

manifest の各セクション:

- `tensors` — PSQ5。`weight_shape`（3D を含む）/ `rows` / `columns` / `block_k` /
  `streams`（magnitude, sign, scale の key）/ `shard`。
- `fp8_block.tensors` — `codes_key`（`...weight`、F8_E4M3）/
  `scale_key`（`...weight_scale_inv`、BF16、K128N128）/ `block_size`。
- `fp8_row.tensors` — `codes_key` / `scale_key`（`...weight_scale`、BF16、行単位）。
- `passthrough_tensors` / `quantized_tensors` / `bytes` / `notes`。

復元契約は `weight = code * scale`。FP8 の scale は BF16 で格納し、
[quantization.md](../../../developer/quantization.md) の HF 経路と同じく F32 へ widen して使う。

## 5. 未実施・制約

- PhaseShift runtime はこの artifact をロードできない。現行 encoder は BF16 / FP8
  K128N128 のみで、PSQ5、FP8 行単位 scale、3D fused expert のいずれも未対応である。
- タスク精度（KLD / Top1）と GPU decode の subnormal 取り扱いは未測定。
- PLE の CPU オフロード実装（1 token あたり 16 row gather）は runtime 側の未実装項目。
- 元の BF16 checkpoint は変更していない。
