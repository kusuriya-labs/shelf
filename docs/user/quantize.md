# EXL3-INT8 量子化と品質確認

`tools/quantize/shelf_quantize.py` は BF16 checkpoint を Shelf の **EXL3-INT8（mul24）** 形式に変換し、出力を自動検査する。元の checkpoint は変更しない。出力先と作業領域には十分な空き容量を用意する。

公開済みの EXL3-INT8（mul24）checkpoint（例: [kusuriya-labs/Qwen3.8-27B-EXL3-INT8-3.05bpw](https://huggingface.co/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-3.05bpw)）を利用する場合は、ダウンロードしたディレクトリをそのまま `--model-dir` へ渡せる。

## 形式契約（EXL3-INT8 mul24 と Ngram4 / Ngram5）

Shelf の EXL3-INT8 ランタイムおよび `tools/quantize/validate_exl3.py` が受理する形式は次のとおりである。

- **`config.json`**: `quantization_config.quant_method == "exl3"`（または `text_config.quantization_config`）。
- **線形層 tensor**: 各 `<prefix>` に `.trellis`（`I16 [k/16, n/16, 16*bits]`、K と N は 128 の倍数）、`.suh`（`F16 [k]`）、`.svh`（`F16 [n]`）、`.mul1`（`I32` scalar、値は **`0x00DCD12D`**）の 4 つを持つ。
- **非対応・拒否される形式**:
  - `.s0` tensor を含む checkpoint は使用できない（検証・ロードとも拒否）。
  - 旧 int8grid 形式、`.mcg` codebook、`0x00DCD12D`（mul24）以外の multiplier、および half bit rate（1.5 / 2.5 / 3.5 bpw）は拒否される。
- **N-gram table（`Ngram4` / `Ngram5`）**: PLE / N-gram 埋め込みテーブルを持つ対応モデル（Qwen3.8-Flash-Next 等）では、bundle root に `ngram_manifest.json`（`format: "ngram-q"`, `profile: "ngram4" | "ngram5" | "mixed"`）と対応 safetensors shard を同梱できる。仕様の正本は [../developer/ngram_quantization.md](../developer/ngram_quantization.md)。

## 準備

Linux、gfx1201 の R9700、ROCm、ROCm 対応 PyTorch を用意する。Python 3.10.11 以降と、`tools/quantize/exl3_int8/pyproject.toml` に記載された Python 依存パッケージが必要である。品質確認には `transformers` も必要である。[build.md](build.md) の手順で `shelf-bench` をビルドする。

使用する GPU を `rocm-smi --showproductname --showuniqueid` と `rocminfo` で確認し、他の GPU を除外する。以下の UUID は使用する R9700 の値に置き換える。ROCr と `rocm-smi` の番号は同じとは限らない。

```bash
export ROCR_VISIBLE_DEVICES=GPU-<R9700のUUID>
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
python tools/quantize/shelf_quantize.py build-ext
```

`build-ext` は gfx1201 用 HIP 拡張をビルドし、mul24 の multiplier（`0x00DCD12D`）を検査する。変換時も同じ検査を行う。

## 変換と検査

```bash
python tools/quantize/shelf_quantize.py convert \
  --model /path/to/model-bf16 --output /path/to/output \
  --work /path/to/work --bpw 4.0 --threads 8
```

既定の calibration データと seed 0 を使用する。必要に応じて `--calibration`、`--cal-rows`、`--cal-cols`、`--seed` を指定できる。指定した `--bpw` は主な重みの目標であり、すべての tensor が同じ bit 幅になるわけではない。変換後に形式検査（`validate_exl3.py`）が実行され、成功すると出力先に `quantize_report.json` が作られる。後から形式だけを検査するには次を実行する。

```bash
python tools/quantize/shelf_quantize.py validate /path/to/output
```

目安として、R9700 の `Qwen3.5-2B-BF16` を 4.0 bpw、既定 calibration、8 スレッドで変換した実測では、形式検査まで 16 分 40 秒、VRAM の観測最大約 9.6 GB、作業領域の観測最大約 15.6 GB、出力約 2.6 GB、ホストメモリの最大 RSS 約 33.4 GB だった。使用量は 2 秒間隔の観測値である。出力の整形時には出力先に一時的なコピーも作られるため、出力先にも余裕を持たせる。モデルや環境によって必要量と所要時間は変わる。

### Ngram4 / Ngram5 の同時変換

N-gram table を持つ対応 checkpoint では、変換コマンドに `--ngram ngram4` または `--ngram ngram5` を追加すると、table も量子化して同じ出力に含めて検査する（N-gram の対象 tensor がない checkpoint には指定しない）。事前に C 共有ライブラリをビルドしておく。

```bash
cmake --build build --target shelf_ngram_quant_c
```

`libshelf_ngram_quant_c.so` が自動検出されない場所にある場合は、環境変数 `SHELF_NGRAM_LIB` に絶対パスを指定する（[env_vars.md](env_vars.md#7-server量子化ツールテスト)）。table 単体の変換・検査には `tools/quantize/ngram_quantize.py` も直接利用できる。

## 品質確認

量子化後モデルや実行経路の品質は、**BF16 基準に対する平均 KLD（KL divergence）と top-1 一致率**で評価する。

### 1. 任意の BF16 モデルとの比較（`shelf_quantize.py quality`）

原本と変換後のモデルを同じ token corpus で比較する。以下の `--bench` は使用する `shelf-bench` のパスに置き換える。

```bash
python tools/quality_gate.py corpus --model-dir /path/to/model-bf16 --out /tmp/corpus.psktok
python tools/quantize/shelf_quantize.py quality \
  --reference /path/to/model-bf16 --model /path/to/output \
  --tokens /tmp/corpus.psktok --context 256 --token-count 128 \
  --bench ./build/shelf-bench
```

比較処理と出力項目の詳細は [bench.md](bench.md#quality) を参照する。

### 2. 固定基準 logits との比較（`tools/quality/kld-vs-bf16.sh`）

Qwen3.8-27B 向けに保存済みの BF16 基準 logits（`cases-v1` の主 8 case・2,632 位置）と突き合わせる場合は `tools/quality/kld-vs-bf16.sh` を使う。このラッパは `/tmp/r9700.lock` を取得して GPU を排他する（すでにロックを取得した環境内では `--already-locked` を付ける）。

```bash
# 既定の A 経路（hipBLASLt INT8 prefill）を測定
./tools/quality/kld-vs-bf16.sh A --bench ./build/shelf-bench --model /path/to/model

# B 経路（自前 WMMA prefill）を測定
./tools/quality/kld-vs-bf16.sh B --bench ./build/shelf-bench --model /path/to/model

# 2 つの測定 JSON を位置単位で対応比較（case bootstrap 付き）
python3 tools/quality/paired_kld.py docs/perf/quality-bf16/B.json docs/perf/quality-bf16/A.json
```

### KLD と top-1 一致率の見方

- **平均 KLD（nats）**: `KL(softmax(BF16) ‖ softmax(候補))`（温度 1・FP64）。**0 に近いほど** BF16 の確率分布に近い。
- **top-1 一致率**: BF16 と最上位予測 token が一致した位置の割合。**100% に近いほど**良い。
- 参考目安（Qwen3.8-27B、`cases-v1` 2,632 位置、BF16 KV、詳細は [../perf/exl3-int8-prefill-quality.md](../perf/exl3-int8-prefill-quality.md) および [../perf/quality-bf16/results.md](../perf/quality-bf16/results.md)）:
  - 4bpw: 平均 KLD 約 0.021、top-1 一致率 約 94.8%
  - 3.05bpw: 平均 KLD 約 0.059、top-1 一致率 約 91.2%
- corpus・測定位置数・KV dtype・prefill chunk 分割によって値は変動するため、比較時は必ず条件を揃える（測定原則は [../perf/methodology.md](../perf/methodology.md#10-モデル精度の比較) を参照）。
