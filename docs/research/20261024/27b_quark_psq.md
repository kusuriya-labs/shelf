# Qwen3.8-27B の PSQ5 / AWQ→PSQ5 artifact 生成

- ID:
  - **PSQ5-20261024-01** — 27B BF16 から PSQ5 artifact（プロファイル指定）
  - **PSQ5-20261024-02** — AWQ（オフラインリカバリ）→ PSQ5 の逐次適用
- 実施日: 2026-10-24
- 入力: `/opt/zen/work/models/Qwen3.8-27B`（BF16、18 shard、1,199 tensor、55,562,855,904 B）
- 関連: [PSQ5-CB28 仕様](../quant/PSQ5-CB28.md)、
  [AWQ 判定の経緯](20261023/psq5_cb28_quark_plugin.md)、
  [PSQ5 精度の試行](qwen4exp/psq5_accuracy_probe.md)

## 1. 適用プロファイル

| コンポーネント | 量子化 |
| --- | --- |
| FFN gate/up/down_proj（64 層 × 3） | PSQ5 |
| GDN in_proj_qkv / in_proj_z / out_proj（48 層 × 3） | PSQ5 |
| Attention q/k/v/o_proj（16 層 × 4） | PSQ5 |
| GDN in_proj_a / in_proj_b | BF16 |
| GDN A_log / dt_bias、Norm、Conv1D | BF16 |
| Embedding / LM Head、Vision、MTP | BF16 |

対象は **400 tensor**、残り 799 tensor（799 個、含む I64 3 tensor）は BF16 passthrough。

## 2. PSQ5-20261024-01: PSQ5 artifact

`tools/quantize_psq5.py` を sharded 入力入力対応に拡張して生成した
（shard 単位の並列処理、manifest + index、採様/全数検証）。

```bash
python3 tools/quantize_psq5.py \
  --model-dir models/Qwen3.8-27B \
  --output-dir models/Qwen3.8-27B-Quark-PSQ \
  --jobs 8 --verify-sample 0
```

### 結果（実測）

| 項目 | 値 |
| --- | --- |
| 変換時間 | 338.3 s（8 worker） |
| manifest | 400 psq5 / 799 passthrough / bpw 5.250 |
| 構造検査 | 18 shard、1,999 key、missing 0 |
| 全数検証 | **400/400 bit exact**、rel weight error 0.049644 |
| 容量 | 約 23 GB（psq5 15.965 GB + bf16 6.909 GB） |

**経緯:** artifact は 15:14 時点までディスク上に存在したが、15:47 の確認で
`models/Qwen3.8-27B-Quark-PSQ` が存在しない（削除）。本セッションのプロセスによる
削除ではなく、他者（ユーザー操作）によるものと判断する。移動先も見当たらない。
再生成は同コマンドで約 25 分。

## 3. PSQ5-20261024-02: AWQ → PSQ5

### 結果（実測）

| 項目 | 値 |
| --- | --- |
| AWQ 実行時間 | 2,962.5 s（64 層、約 46 s/層） |
| 中間产物 | `models/Qwen3.8-27B-AWQ-bf16`（52 GB、1,184 tensor 保存） |
| PSQ5 変換時間 | 338.8 s |
| 構造検査 | 18 shard、1,999 key、missing 0 |
| 全数検証 | **400/400 bit exact**、rel weight error 0.049611 |
| 成果物 | `models/Qwen3.8-27B-Quark-AWQ-PSQ`（22 GB） |

AWQ で実際に重みが変わったかの抽査（rel diff、BF16 原本との比較）:

| tensor | rel diff |
| --- | ---: |
| layers.0.mlp.gate_proj.weight | 0.69362 |
| layers.0.mlp.up_proj.weight | 0.43299 |
| layers.0.mlp.down_proj.weight | 0.56146 |
| layers.3.mlp.gate_proj.weight | 0.59119 |
| layers.0.post_attention_layernorm.weight | 8.62146 |
| layers.3.post_attention_layernorm.weight | 10.00826 |
| layers.0.input_layernorm.weight（対象外） | 0（同一） |
| layers.3.self_attn.q_proj.weight（対象外） | 0（同一） |
| embed_tokens.weight（対象外） | 0（同一） |

qwen3_5 テンプレートの AWQ は MLP の 2 group
（`post_attention_layernorm → gate/up`、`up → down`）のみが対象で、
attention / norm（prev_op 以外）は変更されない。

### 再現

```bash
python3 tools/awq_qwen35.py --samples 16 --length 64 \
  --devices 0,1,2,3 --max-memory-per-device 13.4GiB
python3 tools/quantize_psq5.py \
  --model-dir models/Qwen3.8-27B-AWQ-bf16 \
  --output-dir models/Qwen3.8-27B-Quark-AWQ-PSQ \
  --jobs 8 --verify-sample 0
```

## 4. AWQ 実行で判明した事項

いずれも試行錯誤の結果、実測で確認したもの。

1. **plain `nn.Linear` に対する AWQ は無言 no-op になる。**
   `get_named_quant_linears` は Quark の `QuantLinear` のみを対象にするため、
   module replacement を経由しないと `input_feat` が空になり、
   scale 検索も clip 検索も実行されない（**例外は出ない**）。
   `ModelQuantizer.quantize_model`（＝`_prepare_model` の module replacement を含む）を
   経由する必要がある。
2. **`AwqProcessor._apply_quant` は無効化した。** AWQ の step 4 は
   fake-quant を重みへ書き戻す処理で、後段に PSQ5 がある本 workflow では不要。
   無効化により重みは smoothing / clipping のみが適用された状態で保存される。
   （ベンダーの `vendor/quark` ファイル自体は無改変。subclass / monkeypatch のみ。）
3. **`from_pretrained(device_map="auto")` は GPU 配置後に segfault する。**
   traceback は `modeling_qwen3_5._init_weights` → `_initialize_missing_keys`
   （missing keys は 0）。CPU ロード → `accelerate.dispatch_model` で回避した。
4. **`no_split_module_classes=model._no_split_modules` の指定必須。**
   省略すると層の途中で device が割れ、`residual + hidden_states` で
   cuda:0 と cuda:1 の不一致エラーになる。
5. **GPU 予算は dispatch と AWQ ループの両方を考慮する。**
   per-device 上限が model 本体だけを計算すると `lm_head` が cpu offload 指定になり
   dispatch が失敗する（本件は cpu/disk エントリを最終 GPU へフォールバックさせて解消）。
   4 GPU × 13.4 GiB で全層が GPU に収まり、AWQ ループも 64 層回り切った。
   なお plain-linear 版の試行ではレイヤー処理ごとに約 1 GiB が
   GPU0/1 上に積み上がり OOM した（module replacement 後は解消）。
6. **calibration データは少なかった。** BetterBench corpus + モデル README から
   16 連続 × 64 token = 1,024 token。AWQ の scale 統計としては限定的で、
   コーパスが大きければ N を上げるのが望ましい（メモリ内訳は 5. 参照）。

## 5. 制約・未実施

- タスク精度（PPL / KLD / Top1）は未測定。
  重み空間のノイズ感度を含む注意点は [psq5_accuracy_probe.md](qwen4exp/psq5_accuracy_probe.md)。
- PSQ5 の CBR は実測で 5% 程度（FP8 の約 1.9 倍）。AWQ 適用前後で
  rel weight error はほぼ同じ（0.049644 / 0.049611）で、AWQ は
  重みのスケーリングであるため PSQ5 の誤差率を大きく変えていない。
- PhaseShift runtime は PSQ5 をロードできない
  （対応 format は BF16 / FP8 E4M3 のみ）。runtime 接続は範囲外。
