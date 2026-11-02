# Qwen3.8-Flash-Next 推奨量子化プロファイル

- ID: QWEN4EXP-PROFILE-01
- 実施日: 2026-10-24
- 状態: 採用。artifact の生成結果は
  [quantization_artifact.md](quantization_artifact.md) に記録する。
- 対象 checkpoint: `/opt/zen/work/models/Qwen3.8-Flash-Next`（BF16、131 shard）
- 正本: 構造と精度契約は [docs/developer/qwen4exp.md](../../../developer/qwen4exp.md)、
  PSQ5-CB28 の仕様は [docs/quant/PSQ5-CB28.md](../../../quant/PSQ5-CB28.md)、
  Quark plugin は [tools/quark_psq5/](../../../../tools/quark_psq5/)。

## 結論

Routed Expert は **PSQ5-CB28**、PLE（N-gram Embedding）は **FP8**、
それ以外は重要度に応じて **BF16** と **FP8** を使い分ける構成を採用する。

Qwen3.8-Flash-Next では、従来の Transformer 以上に、量子化誤差を避けたい部分が明確である。

- **Gated Residual（GR）**: 4 本の Residual Stream の読み書きを制御するため BF16 を優先。
- **QSA Indexer**: Attention で参照するトークンを選択するため BF16 を優先。
- **GDN の Gate・状態更新関連**: 誤差が状態として蓄積する可能性があるため BF16 を優先。
- **PLE N-gram Embedding**: 51.2B パラメータを占めるため、FP8 化のメモリ削減効果が極めて大きい。

## 1. モデル構造の確認

checkpoint の safetensors ヘッダから実測した値である。

| 項目 | Qwen3.8-Flash-Next |
| --- | --- |
| Main Model | 125.74B |
| PLE N-gram Embedding | 51.20B |
| MTP | 2.61B（追加） |
| Vision Encoder | 0.45B |
| checkpoint 合計 | 180.00B |
| Active Parameters | 約 6B/token |
| Decoder | 48 層 |
| Gated DeltaNet | 36 層 |
| Qwen Sparse Attention | 12 層 |
| Routed Experts | 512 個/層 |
| Active Experts | 10 個 + Shared Expert |
| Hidden Size | 2560 |
| Residual Streams | 4 本（`hc_count = 4`） |
| Context | 262,144 tokens |

Decoder の繰り返し構造:

```text
GDN -> MoE -> GDN -> MoE -> GDN -> MoE -> QSA -> MoE
```

4 層 × 12 セット = 48 層。各層に Gated Residual の制御を含む。

PSQ5-CB28 は 32 weights ごとに 5bit コードと 8bit scale を格納するため **5.25 bpw** である。
scale の 8bit 符号化は [docs/research/20261023/psq5_cb28_quark_plugin.md](../../20261023/psq5_cb28_quark_plugin.md)
で比較し、`ue4m4`（fold 512）を既定として採用した。

## 2. 推奨量子化プロファイル

| コンポーネント | 推奨形式 | 理由 |
| --- | --- | --- |
| Routed Expert `gate/up/down` | PSQ5-CB28 | 最大のパラメータ群 |
| Shared Expert | BF16 | 全トークンで利用、容量も小さい |
| MoE Router | BF16 | Expert 選択を保護 |
| Shared Expert Gate | BF16 | 出力混合の制御 |
| GDN `in_proj_qkv` | FP8 | 大きい行列演算 |
| GDN `in_proj_z` | FP8 | 大きい行列演算 |
| GDN `out_proj` | FP8 | 大きい行列演算 |
| GDN `in_proj_a/b` | BF16 | 状態更新の制御 |
| GDN Conv1D / Norm | BF16 | 小容量・状態への影響 |
| QSA `q/k/v/o_proj` | BF16 | まず Attention 品質を保護 |
| QSA Indexer | BF16 | 検索対象の選択精度 |
| QSA Q/K Norm | BF16 | Attention score の安定性 |
| Gated Residual / HyperConnection | BF16 | 全層への誤差伝播を抑制 |
| PLE N-gram Table | FP8（行単位 scale） | 約 51.2B の容量削減 |
| PLE Key/Value Projection | BF16 | N-gram 情報の融合 |
| PLE Conv / Norm | BF16 | 小容量・融合演算 |
| Token Embedding | BF16 | 入力表現を維持 |
| LM Head | BF16 | Logits の精度を保護 |
| MTP | BF16（初期） | Draft の採択率を保護 |
| Vision Encoder | BF16（初期） | 画像特徴抽出を保護 |

QSA を BF16、GDN の主要 Projection を FP8 に分けた理由は、QSA が選択済み
コンテキストへの Attention を担当し、GDN が長い履歴の圧縮を担当するためである。
両者の量子化誤差の影響は同じではない。

FP8 は checkpoint と runtime の既存契約に合わせ、weight を `float8_e4m3fn`
（OCP E4M3）、GDN Projection の scale を K128N128 の BF16 `weight_scale_inv` とする。
PLE table は行（160 要素）単位の BF16 scale とする。

## 3. 特に注意したい 4 つのコンポーネント

### ① Gated Residual: BF16 固定

`attn_hyper_connection`、`mlp_hyper_connection`、`hyper_connection_mixer` が対象である。

通常の Residual 接続と違い、4 本の Stream から読み出す情報と各 Stream に書き込む情報を
動的に制御する。`input_mix_weight_down`、`input_mix_weight_up`、`block_inject_weight`
などが該当する。

これらを FP8 にするメリットは比較的小さく、影響が複数層へ広がるリスクがある。
まず BF16 に固定すべきである。

### ② QSA Indexer: BF16 固定

QSA の Indexer は、長いコンテキストから参照すべき micro-block を選択する。

ここで誤った選択が発生すると、後段 Attention の計算精度を上げても、本来参照すべき
情報を取り戻せない。そのため Indexer の `index_qk_proj`、`q_layernorm`、
`k_layernorm` は BF16 を維持する。

Indexer の重み精度と Indexer KV Cache の精度は別設定である。外部実装には Indexer KV を
FP8 にする構成もあるが、本 repository では未検証であり、長文での検索一致率を確認して
から適用する。

### ③ GDN: Projection だけ FP8 化

| 分類 | 形式 | 対象 |
| --- | --- | --- |
| GDN Projection | FP8 | `in_proj_qkv`、`in_proj_z`、`out_proj` |
| GDN Control | BF16 | `in_proj_a`、`in_proj_b`、`conv1d`、Norm、`A_log`、`dt_bias` |

GDN のリカレント状態は量子化対象外とする。参照実装の config は
`mamba_ssm_dtype = float32` だが、Shelf の実装方針は FP16 である
（[qwen4exp.md](../../../developer/qwen4exp.md) §5.0）。いずれも重み量子化とは別の話であり、
このプロファイルでは state の dtype を変更しない。

### ④ PLE N-gram Embedding: FP8 を最優先で検証

このモデルでは、PLE の FP8 化は GDN や QSA の FP8 化よりメモリ削減効果が大きい。

PLE の実テーブルは 320,001,536 行 × 160 要素である。

| 形式 | テーブル本体 |
| --- | ---: |
| BF16 | 102.4 GB |
| FP8（scale 除く） | 51.2 GB |
| FP8 + 行単位 BF16 scale | 51.84 GB |

推奨は FP8 E4M3 + 行単位 scale である。各行に BF16 scale を 1 個追加すると scale は
約 0.64 GB になる。この方式なら各 N-gram 行の振幅差に対応できる。
CPU オフロードと復元処理は runtime 側の実装であり、本プロファイルの範囲外である。

## 4. メモリ使用量の目安

Routed Expert のパラメータ数は構造から次のように求められる。

```text
48 layers x 512 experts x (2*640*2560 + 640*2560)
  = 48 x 512 x 3 x 2560 x 640
  = 120.80B
```

PSQ5-CB28 の 5.25 bpw を適用すると、`120.80B x 5.25 / 8 = 79.3 GB` である。

| コンポーネント | 概算容量 |
| --- | ---: |
| Routed Experts: PSQ5 | 79.3 GB |
| PLE: FP8 + BF16 行 scale | 51.8 GB |
| GDN 主要 Projection: FP8 | 2.1 GB |
| その他: BF16（Embedding / LM Head / QSA / GR / Shared Expert / Router / MTP / Vision） | 11.9 GB |
| **合計** | **145.0 GB** |

実測値と生成方法は [quantization_artifact.md](quantization_artifact.md) を参照する。
KV Cache、GDN State、workspace などの実行時メモリは別途見積もる必要がある。

## 5. 採用方針と未検証事項

採用する組み合わせ:

```text
PSQ5-CB28 (ue4m4) + PLE FP8 行 scale + GDN 主要 Projection FP8
  + 制御系 BF16 + QSA BF16 + MTP/Vision BF16
```

AWQ はオフラインリカバリとして計画したが、本環境では適用できない。
理由と代替の扱いは [quantization_artifact.md](quantization_artifact.md) §3 に記録する。

未検証事項:

- PSQ5-CB28 のタスク精度（KLD / Top1）と GPU decode の subnormal 取り扱い。
  重量空間の形式間比較とタスク精度の試行は
  [psq5_accuracy_probe.md](psq5_accuracy_probe.md) に記録している
  （重量誤差は 5.0%、同 bpw の一様 INT5 より 1.4〜6% 大きい。タスク精度は未確立）。
- PLE 行単位 FP8 の精度影響（テーブル行の振幅分布に対する実測）。
- runtime 接続。現行 Shelf は PSQ5、FP8 行単位 scale、3D fused expert の
  encoder を持たない（[quantization.md](../../../developer/quantization.md)）。
