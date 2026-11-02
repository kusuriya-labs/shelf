# PSQ5-CB28 の Quark プラグイン実装と scale 符号化の採否

- 仕様の正本: [docs/quant/PSQ5-CB28.md](../../quant/PSQ5-CB28.md)
- 実験 ID:
  - **PSQ5-20261023-01** — W32 scale の符号化方式の比較（ACCEPT / REJECT 判定）
  - **PSQ5-20261023-02** — AMD Quark プラグイン実装と単体・統合検証
  - **PSQ5-20261023-03** — Qwen3.8-27B-DFlash2 の量子化 artifact 生成

## PSQ5-20261023-01: scale 符号化の比較

### 条件

- 対象: `models/Qwen3.8-27B-FP8`（HF FP8 checkpoint）を FP32 へデクオントした重み
- サンプル: `layers-{0,8,16,24,32,40,48,56}` の `mlp.{gate,up,down}_proj` と
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` = **48 tensor**
- W32 ブロック数: **95,682,560**
- `scale = amax / 28`（範囲ベース対称）、`fold = 512`、丸めは最寄り値（tie は下側）
- 再現:

```bash
python3 docs/research/20261023/scale_eval.py \
  --model-dir models/Qwen3.8-27B-FP8 \
  --cache-dir /tmp/psq5/cache --full
```

### 結果

`relFrob` は `||W - Ŵ||F / ||W||F`、`下限比` は scale を F32 そのまま使った場合（exact）との比。
`mean` / `p99` / `max` は scale の相対誤差 `|s' - s| / s`（ブロック単位）。

| 候補 | byte | bpw | mean | p99 | max | relFrob | 下限比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact（scale=F32、コードブック下限） | – | 5.00 | – | – | – | 0.04946 | 1.000 |
| ue4m3_fold | 1 | 5.25 | 0.02228 | 0.0555 | 5.1793 | 0.05032 | 1.0175 |
| ue4m3_unfold | 1 | 5.25 | 0.19547 | 0.7181 | 3162.8 | 0.08656 | 1.7501 |
| e8m0_fold | 1 | 5.25 | 0.16240 | 0.3299 | 0.3333 | 0.09285 | 1.8773 |
| e8m0_unfold | 1 | 5.25 | 0.16240 | 0.3299 | 0.3333 | 0.09285 | 1.8773 |
| **ue4m4_fold** | 1 | **5.25** | **0.01100** | **0.0286** | 5.1793 | **0.04943** | **0.9995** |
| u5m3_fold | 1 | 5.25 | 0.02227 | 0.0555 | 0.0588 | 0.05032 | 1.0175 |
| fp16_fold | 2 | 5.50 | 0.00015 | 0.0004 | 0.0004 | 0.04946 | 1.0000 |
| fp16_unfold | 2 | 5.50 | 0.00015 | 0.0004 | 0.0769 | 0.04946 | 1.0000 |
| bf16_fold | 2 | 5.50 | 0.00123 | 0.0034 | 0.0038 | 0.04947 | 1.0003 |
| bf16_unfold | 2 | 5.50 | 0.00123 | 0.0034 | 0.0038 | 0.04947 | 1.0003 |

補足:

- `ue4m3` / `ue4m4` の `max` が 5.1793 となるのは subnormal 域（絶対刻み `2^-10`）に
  落ちる極小ブロックのみで、該当は 0.001% 未満、重みエネルギーへの影響は relFrob に
  現れない（`>10%` 列で 0.001%）。`u5m3` と 16bit はこの裾が出ない。
- `ue4m4` の下限比が 1 をわずかに下回るのは、scale を最寄りに丸めた結果として
  `amax/28` より有利になるブロックが僅かにあるため。

### 判定

| 候補 | 判定 | 根拠 |
|---|---|---|
| **ue4m4** | **ACCEPT（既定）** | 1byte のまま scale 誤差が最小（下限比 0.9995、平均 1.10%）。5.25bpw を維持 |
| ue4m3 | ACCEPT（切替可能として実装） | Shelf 既存 codec をそのまま使える（下限比 1.0175） |
| fp16 / bf16 | REJECT（既定にしない） | 下限比は 1.0000 / 1.0003 と最良だが bpw が 5.25 → 5.50（+4.8%）となり、仕様の 8bit scale 目標に反する。`scale_format` として実装は維持 |
| e8m0 | **REJECT** | 下限比 1.8773。対数軸の丸めのため fold しても改善しない（fold / unfold が同値） |
| ue4m3_unfold | **REJECT** | 下限比 1.7501、max 3162.8。subnormal の絶対刻みに落ちる |
| u5m3 | REJECT（不採用） | 尾数 3bit のため精度は ue4m3 と実測一致（裾のみ改善）。ue4m4 の優位性を崩さない |

未検証の仮説: relFrob の差がタスク精度（KLD / Top1）にどう効くかは未測定。
GPU のネイティブ FP8 経路での subnormal 取り扱いも未実測（仕様書 §6 の観点）。

## PSQ5-20261023-02: プラグイン実装と検証

- 配置: `tools/quark_psq5/`。AMD Quark の `ALGORITHM_REGISTRY` に
  `psq5_cb28` として登録し、**Quark のコアファイルは無改変**
  （`QuarkAlgorithm` = Quark 公式の拡張機構）。
- runner: `tools/quantize_psq5.py`（safetensors 単位の file-to-file 量子化）
- scale 符号化: `ue4m4`（既定）/ `ue4m3` / `fp16` / `bf16` を `scale_format` で切替可能
- 出力: SoA 3 ストリーム（magnitude 16B / sign 4B / scale 1B 毎 W32）の safetensors と
  `model.safetensors.json` manifest
- 検証（すべて実測）:
  - `python3 -m pytest tools/quark_psq5/tests` → **28 passed**
    （仕様恒等式 全32コード、bitpack/ unpack、scale codec の誤差上限、
    量子化⇔復元一致、bpw=5.25、Quark 設定のデシリアライズ、
    `apply_advanced_quant_algo` 経由の dispatch、artifact round trip）
  - `ctest -R test_quark_psq5_plugin` → **Passed**（`cmake/tests.cmake` に登録、`cpu;optional`）
- 判定: **ACCEPT**

## PSQ5-20261023-03: Qwen3.8-27B-DFlash2 の量子化

- コマンド:

```bash
python3 tools/quantize_psq5.py \
  --model-dir models/Qwen3.8-27B-DFlash2 \
  --output-dir models/Qwen3.8-27B-DFlash2-Quark-AWQ-PSQ5
```

- 出力: `models/Qwen3.8-27B-DFlash2-Quark-AWQ-PSQ5/`
  - Linear **47 tensor 量子化**、非 Linear **34 tensor はそのまま**（norm / codebook / base_kernel）
  - linear weight: 3,593,994,240 → 1,179,279,360 bytes（BF16 の 16bpw → **5.25bpw**、-67.2%）
  - `model.safetensors` 全体: 3,848,817,896 → 1,434,115,856 bytes（-62.7%）
  - scale は `ue4m4`、fold 512、`bits_per_weight = 5.25`
- 検証（runner が同一実行内で実測）
  - artifact round trip exact: **True**（safetensors 復元値と量子化結果が bit 一致）
  - 重み相対誤差（元 BF16 に対して）: **0.049693**
  - ランダム入力での線形層射影誤差: 最大 **0.050848**
    （`layers.3.mlp_conv.kernel_projection.weight`）
- **AWQ 前処理は未適用**。manifest の `pre_quant_optimization = none`。
  理由: DFlash2 は HF 互換の Python モデルクラスを持たず、calibration forward を要する
  Quark の AWQ algorithm を実行できないため。ディレクトリ名は指示の表記に従った。
- Shelf runtime は **読込不可**（対応 format は BF16 / FP8 E4M3 のみ）。
- 判定: **ACCEPT**（artifact として成立。runtime 接続は範囲外）

## 再現条件

### 外部リポジトリ

- `amd/quark` — `release/0.13`（tag `v0.13`）、commit
  `313cb0b78ef364cb937598bc3431c46e28a226f4`、MIT
- 配置: `vendor/quark/`（upstream の `.git` を保持）。**この vendor はコミット対象外。**
  **`vendor/quark` を clone していないと PSQ5-CB28 などの一部のオリジナル量子化モデルは
  作成できない**（既存の BF16 / FP8 E4M3 は Quark が無くても扱える）。

```bash
git clone --depth 1 --branch release/0.13 https://github.com/amd/quark vendor/quark
```
- 非改変の確認: `git -C vendor/quark status --porcelain` と `git diff --stat` がともに空
  （追加は `quark/torch/kernel/hw_emulation/lib/libhw_emulation_gpu.so` のみ、upstream の
  `.gitignore` 対象）
- 本体（Shelf）側に patch は入れていない
- 補足: 親リポジトリから見れば nested `.git` なので `git add vendor/quark` は
  gitlink 扱いになる。取り扱い（submodule 化 / ignore）は未決定。

### 実行環境

- Python 3.12 / torch 2.13.0+rocm10.0.0（gfx1201）
- 追加で導入した pip パッケージ:
  `evaluate onnx onnxscript onnxslim plotly sentencepiece zstandard`
- Quark の HIP 拡張を初回 JIT ビルドする場合のみ次の環境変数が必要。
  本環境はビルド済み `.so` を
  `vendor/quark/quark/torch/kernel/hw_emulation/lib/libhw_emulation_gpu.so` に
  配置しているため env 不要。

```bash
export PYTORCH_ROCM_ARCH=gfx1201
export CPLUS_INCLUDE_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/include
export LIBRARY_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
```

### コマンド

```bash
python3 -m pytest tools/quark_psq5/tests
ctest --test-dir build -R test_quark_psq5_plugin
python3 tools/quantize_psq5.py --model-dir <dir> --output-dir <dir> [--scale-format ue4m4]
python3 docs/research/20261023/scale_eval.py --model-dir <dir> --cache-dir <dir> --full
```

## 未実施事項

- Shelf 本体の読込対応（manifest の encoding、codec、kernel）
- `ue4m4` の Shelf 側 codec（host / device）。既存は `ue4m3` のみで、
  `ue4m3` はバイアス 8、`ue4m4` はバイアス 7 という指数オフセット差がある
  （相対精度への影響はない）
- タスク精度の測定（KLD / Top1）と GPU decode の実測（subnormal flush の可否を含む）
- AWQ 前処理の適用（可否の確認結果は下記「次段階メモ」）

## 次段階メモ: AWQ（オフラインリカバリ）の適用可否

2026-10-23 に実装から確認した事実。**実施は未了**で、次回実行時の出発点として残す。

### 結論

| 対象 | AWQ → PSQ5 の適用 |
|---|---|
| `models/Qwen3.8-27B-FP8` | **機構上可能**（HF クラスあり、Quark の `qwen3_5` 対応あり） |
| `models/Qwen3.8-27B-DFlash2` | **現状不可**（HF 互換クラスが無く forward を作れない） |

### 根拠（すべて確認済み）

- transformers 5.14.1 に `Qwen3_5ForConditionalGeneration` が存在する
  （`model_type = qwen3_5`、`auto_map` なし＝remote code 不要）。
  auto クラスは **`AutoModelForImageTextToText`**。`AutoModelForCausalLM` は
  `model.*` のテキスト単体構造になってキー不一致になる。
- `init_empty_weights` でのキー照合: HF クラスの 1184 tensor がすべて checkpoint に一致
  （class 側だけのキーは 0）。checkpoint の残り 422 は `*.weight_scale_inv`
  （HF の fp8 読込経路が処理する block scale）。
- Quark の対応:
  - `quark/torch/quantization/config/template.py` の `"qwen3_5"` エントリ
    （kv / q パターン、`model.visual.*` などを exclude）
  - `quark/torch/utils/llm/model_preparation.py` の `model_type == "qwen3_5"` 分岐
  - **`AWQ_MAP["qwen3_5"]` が既定設定として存在**
    （`get_algo_config("awq", "qwen3_5")` → `AWQConfig`、scaling_layers は 2 群:
    `post_attention_layernorm → gate/up`、`up_proj → down_proj`。attention 側の
    smoothing は上流コメントいわくこのモデルでは不採用）
- 実行順: `QConfig(algo_config=[AWQConfig(...), PSQ5CB28Config(...)])` とすれば
  `apply_advanced_quant_algo` がリスト順に逐次実行する。
  両方とも `ALGORITHM_REGISTRY` 経由で解決することを確認済み
  （`psq5_cb28` は `get_supported_algorithm_types()` に含まれる）。

### 実行に必要なもの

1. **BF16 載せ替えディレクトリ** — `config.json` から `quantization_config`
   （`quant_method = fp8` / `weight_block_size = [128,128]` /
   `modules_to_not_convert = 882`）を外し、`float8_e4m3` → `bfloat16` で再保存する。
   理由: AWQ は重みへ乗算スケールを書き込むため、FP8 + block scale の上に
   乗せる経路は Quark の prequant routing 次第で未確認。
2. **calibration dataloader（実トークン）** — tokenizer と corpus は手元にある
   （`models/Qwen3.8-27B-FP8/tokenizer.json`、BetterBench corpus / quality corpus）。
3. **GPU と長時間実行** — linear attention の fast path が未導入のため
   torch の slow path で forward が走る
   （`flash-linear-attention` / `causal-conv1d` が未インストール）。
   これらを入れるか否かで calibration 所要が大きく変わる。
4. **精度検証** — PPL / KLD / Top1。現状計測済みなのは重み空間の誤差のみ。

### リスク・未検証

- FP8 のまま扱う経路（Quark の `keep_prequantized_layers` / exclude ルーティング）は未確認。
- AWQ の per-channel 重みスケールを掛けた結果に対して PSQ5 が W32 単位で
  scale を再計算する（実行順の設計は確認済みだが数値的な相互作用は未検証）。
- calibration 所要時間（slow path × 27B）は未計測。
- DFlash2 で AWQ を使うには自前 forward の実装が要る。
