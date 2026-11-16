# BF16 small-M の SmallRows 実装

実験 ID: `SMALLM-BF16-1`

[bf16_small_m_reference.md](bf16_small_m_reference.md) の案 A（block-per-output-row）と
案 B（M=9..16 の WMMA）を BF16 に接続し、既存の `ExactRows` と比較した記録である。

## 1. 変更

### 1.1 Kernel

`src/shelf/kernels/optimized/linear/bf16.hip` に `SmallRows` を追加した。

- 1 block = 1 出力行。block は 256 thread（8 warp）。
- `kRows`（1..16）個の token accumulator を register に持つ。
- K は `off = i * 256 * 8 + tid * 8` の順に 8 BF16 ずつ読む。
  `load_bf16x4_masked` で K 末尾と未整列 pointer を処理する。
- 各 thread の accumulator を warp shuffle でまとめ、8 warp を LDS で足す。
- 行ごとの縮約順は `kRows` に依存しない。M=1 の呼び出しと M=16 の呼び出しで
  同じ行の出力は byte 一致する。

### 1.2 Selector

`src/shelf/inference/execution/selectors/linear_selector.cpp` の選択は次のとおり。

- `rows <= 16`、`out_features <= 192`、`allow_exact_rows`、
  `k % 8 == 0`、weight / input が 16 byte 整列、input row stride が 8 要素整列:
  `SmallRows`。
- `out_features > 192` の rows 1..8: `ExactRows` / `ExactRowsDot2`。
- `out_features > 192` の rows 9..16:
  - `out_features <= 2560`: `WmmaKPartition`
  - `out_features > 2560` かつ rows 9..10: `ExactRows`
  - `out_features > 2560` かつ rows 11..16: `Wmma`

### 1.3 その他

- `Bf16GemmConfigId::SmallRows` と `config.exact_rows`（行数）を追加。
- `PhysicalBf16Variant::SmallRows` と physical resolve の grid / workgroup を追加。
- `shelf-bench gemm --dtype bf16 --impl smallrows` を追加。
- `test_gemm_bf16_verify_exact` に SmallRows の M 不変性テストを追加。

## 2. 実測

R9700、`shelf-bench gemm --dtype bf16`、warmup 50 / samples 20 / launches 20。
p50 µs。`--impl exact` と `--impl smallrows` を交互に近い順で測った。

### 2.1 狭い出力

| shape | rows | ExactRows | SmallRows | 比 |
| --- | ---: | ---: | ---: | ---: |
| 16:2048 | 1 | 5.4 | 3.7 | 1.46x |
| 16:2048 | 8 | 9.7 | 4.9 | 1.98x |
| 16:2048 | 16 | 9.8 | 6.0 | 1.63x |
| 32:2048 | 1 | 5.4 | 3.6 | 1.50x |
| 32:2048 | 8 | 9.5 | 4.4 | 2.16x |
| 32:2048 | 16 | 8.9 | 5.7 | 1.56x |
| 48:5120 | 1 | 5.0 | 4.5 | 1.11x |
| 48:5120 | 8 | 9.9 | 6.9 | 1.43x |
| 48:5120 | 16 | 14.8 | 9.3 | 1.59x |
| 64:2048 | 8 | 6.7 | 5.0 | 1.34x |
| 128:2048 | 8 | 6.4 | 5.2 | 1.23x |
| 192:2048 | 8 | 6.6 | 5.7 | 1.16x |

### 2.2 広い出力

| shape | rows | ExactRows | SmallRows | 判定 |
| --- | ---: | ---: | ---: | --- |
| 256:2048 | 1 | 5.6 | 18.7 | ExactRows |
| 256:2048 | 8 | 10.0 | 18.7 | ExactRows |
| 640:2560 | 1 | 5.2 | 6.7 | ExactRows |
| 640:2560 | 8 | 9.8 | 15.3 | ExactRows |
| 1024:5120 | 1 | 8.2 | 10.3 | ExactRows |
| 1024:5120 | 8 | 17.1 | 28.3 | ExactRows |
| 2560:2560 | 8 | 24.9 | 46.7 | ExactRows |
| 5120:5120 | 8 | 57.7 | 127.5 | ExactRows |

`256:2048` は K を変えても約 19 µs の床になり、`240:2048` と `272:2048` は
4.2〜4.7 µs だった。N=256 の grid と 4 KiB の weight row stride の組合せによる
再現性のある異常なので、selector の上限は 192 に置いた。

### 2.3 auto の結果

| shape | rows | 変更前（ExactRows） | auto | 比 |
| --- | ---: | ---: | ---: | ---: |
| 16:2048 | 16 | 9.8 | 5.6 | 1.75x |
| 32:2048 | 16 | 8.9 | 5.7 | 1.56x |
| 48:5120 | 16 | 14.8 | 9.0 | 1.64x |
| 64:2048 | 8 | 6.7 | 5.0 | 1.34x |

M=9..16 の auto は 3 節に記す。

## 3. M=9..16 の接続

既存の `WmmaKPartition`（8-way K split）と `Wmma`（1 row tile）を
`--impl splitk` / `--impl wmma` で測った。p50 µs。

| shape | rows | ExactRows | WmmaKPartition | Wmma |
| --- | ---: | ---: | ---: | ---: |
| 1024:5120 | 16 | 29.1 | 16.1 | 72.0 |
| 2560:2560 | 16 | 33.6 | 16.6 | 37.8 |
| 5120:5120 | 12 | 91.9 | 140.6 | 69.7 |
| 5120:5120 | 16 | 130.8 | 145.3 | 70.2 |
| 256:2048 | 16 | 10.1 | 7.8 | 29.0 |

- out_features <= 2560 は `WmmaKPartition` が ExactRows より速い。
- out_features > 2560 は rows>=11 で `Wmma` が速い。rows=9〜10 は ExactRows が速い。
この切替は、M=1..8 の ExactRows と M=9..16 の split/Wmma で縮約木が変わる。
行ごとの縮約順は各 kernel の内部では row 数に依存しないが、8/9 の境界を
またぐ出力は bit 一致しない。MTP verify は selector が選んだ同じ帯の kernel を
使うため、同じ帯の中では再現性がある。この契約変更を明示したうえで接続した。

auto の実測（p50 µs、変更前は ExactRows）:

| shape | rows | 変更前 | auto | 比 |
| --- | ---: | ---: | ---: | ---: |
| 256:2048 | 16 | 10.1 | 7.1 | 1.42x |
| 1024:5120 | 12 | 23.7 | 15.0 | 1.58x |
| 1024:5120 | 16 | 29.1 | 14.7 | 1.98x |
| 2560:2560 | 12 | 28.0 | 15.7 | 1.78x |
| 2560:2560 | 16 | 33.6 | 16.0 | 2.10x |
| 5120:5120 | 12 | 91.9 | 72.2 | 1.27x |
| 5120:5120 | 16 | 130.8 | 70.5 | 1.86x |

## 4. 採否

- ACCEPT: `SmallRows` を rows<=16、out_features<=192 の narrow output に接続。
  GDN a|b（16/32）、48 幅の GDN 出力、64/128/192 の中間帯で 1.1〜2.2 倍。
- REJECT: out_features>=256 への `SmallRows`。N=256 の異常と、広い出力では
  ExactRows の warp-per-row が勝つ。
- ACCEPT: M=9..16 の `WmmaKPartition`（out_features <= 2560）と
  `Wmma`（out_features > 2560、rows >= 11）。M=1..8 との bit 一致は捨てる。
- 契約: 行ごとの縮約順は各 kernel の内部で row 数に依存しない。
  rows 1..8 と 9..16 は別 kernel なので、境界をまたぐ出力は一致しない。

## 5. 検証

- `test_qwen35_linear_selector`: SmallRows / ExactRows の選択を固定。
- `test_gemm_bf16_verify_exact`: N=16/48/192、rows=2/4/8/16 で
  SmallRows の multi-row 出力が single-row 出力と byte 一致することを固定。
- `test_dflash2_bf16_geometry`: 既存の geometry が pass。
- `ctest -R "bf16|linear_selector"` が pass（2026-11-16）。

## 6. 再現

```sh
HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype bf16 \
  --shapes 16:2048,32:2048,48:5120,64:2048,128:2048,192:2048 \
  --rows 1,2,4,8,16 --warmup 50 --samples 20 --launches 20 --device 0

HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype bf16 \
  --shapes 16:2048,48:5120,192:2048 --rows 1,8,16 \
  --warmup 50 --samples 20 --launches 20 --device 0 --impl smallrows

HIP_VISIBLE_DEVICES=1 ./build/tests/test_gemm_bf16_verify_exact
```
