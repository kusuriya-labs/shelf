# EXL3-INT8 量子化

`tools/quantize/shelf_quantize.py` は BF16 checkpoint を Shelf の EXL3-INT8 mul24 形式に変換し、出力を自動検査する。元の checkpoint は変更しない。出力先と作業領域には十分な空き容量を用意する。

## 準備

Linux、gfx1201 の R9700、ROCm、ROCm 対応 PyTorch を用意する。Python 3.10.11 以降と、`tools/quantize/exl3_int8/pyproject.toml` に記載された Python 依存パッケージが必要である。品質確認には `transformers` も必要である。[build.md](build.md) の手順で `shelf-bench` をビルドする。

使用する GPU を `rocm-smi --showproductname --showuniqueid` と `rocminfo` で確認し、他の GPU を除外する。以下の UUID は使用する R9700 の値に置き換える。ROCr と `rocm-smi` の番号は同じとは限らない。

```bash
export ROCR_VISIBLE_DEVICES=GPU-<R9700のUUID>
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
python tools/quantize/shelf_quantize.py build-ext
```

`build-ext` は gfx1201 用 HIP 拡張をビルドし、mul24 の multiplier を検査する。変換時も同じ検査を行う。

## 変換と検査

```bash
python tools/quantize/shelf_quantize.py convert \
  --model /path/to/model-bf16 --output /path/to/output \
  --work /path/to/work --bpw 4.0 --threads 8
```

既定の calibration データと seed 0 を使用する。必要に応じて `--calibration`、`--cal-rows`、`--cal-cols`、`--seed` を指定できる。指定した bpw は主な重みの目標であり、すべての tensor が同じ bit 幅になるわけではない。変換後に形式検査が実行され、成功すると出力先に `quantize_report.json` が作られる。後から検査するには次を実行する。

目安として、R9700 の `Qwen3.5-2B-BF16` を 4.0 bpw、既定 calibration、8 スレッドで変換した実測では、形式検査まで 16 分 40 秒、VRAM の観測最大約 9.6 GB、作業領域の観測最大約 15.6 GB、出力約 2.6 GB、ホストメモリの最大 RSS 約 33.4 GB だった。使用量は 2 秒間隔の観測値である。出力の整形時には出力先に一時的なコピーも作られるため、出力先にも余裕を持たせる。モデルや環境によって必要量と所要時間は変わる。

```bash
python tools/quantize/shelf_quantize.py validate /path/to/output
```

Ngram table を持つ対応 checkpoint では、変換コマンドに `--ngram ngram4` または `--ngram ngram5` を追加すると、table も量子化し同じ出力に含めて検査する。Ngram の対象 tensor がない checkpoint には指定しない。事前に `cmake --build build --target shelf_ngram_quant_c` を実行し、必要なら `SHELF_NGRAM_LIB` に生成された `libshelf_ngram_quant_c.so` の絶対パスを指定する。自動検出の制約は [Ngram 量子化](../developer/ngram_quantization.md) を参照する。

## 品質確認

原本と変換後のモデルを同じ token corpus で比較する。以下の `--bench` は使用する `shelf-bench` のパスに置き換える。

```bash
python tools/quality_gate.py corpus --model-dir /path/to/model-bf16 --out /tmp/corpus.psktok
python tools/quantize/shelf_quantize.py quality \
  --reference /path/to/model-bf16 --model /path/to/output \
  --tokens /tmp/corpus.psktok --context 256 --token-count 128 \
  --bench ./build/shelf-bench
```

出力される平均 KL と top-1 一致率を確認する。corpus と測定位置数は結果に影響する。比較処理と出力の詳細は [bench.md](bench.md#quality) を参照する。
