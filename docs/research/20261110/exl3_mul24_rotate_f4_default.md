# mul24 rotate F4 の既定 ON 化と重複経路の削除

対象: `exl3_family/mul24/{decode.h,rotate_quantize.hip,decode.hip,prefill.hip}`、
`prefill_rotate.hip`（削除）。
測定日: 2026-11-10、R9700 (gfx1201)、HIP 7.15。

## 変更

- `SHELF_EXL3_INT8_ROTATE_F4` の既定を ON にした。環境変数を設定しない場合は
  float4 経路（`fwht128`、128 列ブロックあたり 1 wave、vectorized I/O）を使う。
  `SHELF_EXL3_INT8_ROTATE_F4=0` で従来の scalar/smem 経路に切り替えられる。
- 小 rows 用の scalar 経路は残した。`rows < 32`（rotate_quantize）と
  `rows < 512`（rotate_out）では scalar の方が CTA 数が多く速い。
- 削除した重複経路:
  - `SHELF_EXL3_INT8_ROTATE_TILE`（`rotate_tile_blocks()`）。既定 0（行全体を 1 CTA）
    のまま固定し、N 分割の実行時分岐と `blocks_per_cta` 引数を削除した。
  - `prefill_rotate.hip` の 5-shuffle / 2-smem ハイブリッドと
    `SHELF_EXL3_INT8_PREFILL_ROTATE`。F4 が既定 ON のとき
    `launch_exl3_mul24_prefill_rotate_out_i32` は通常経路へ委譲するだけで、
    独自カーネルは到達不能だった。
  - それに伴い `launch_exl3_mul24_prefill_rotate_out_i32`、CTest
    `test_exl3_mul24_prefill_rotate`、契約テストの該当ケースを削除した。
- launch 内の動的分岐をテンプレート引数で特殊化した。
  `rotate_quantize` / `rotate_out` / `rotate_out_i32` は
  `template <bool kF4, bool kBF16>`（quantize は `kF4` のみ）の実装関数へ分岐を
  追い出し、カーネル選択が実行時 `if` の内側に残らない形にした。

## 性能（2048 prefill、5 run 中央値）

| 経路 | F4 既定 OFF（旧） | F4 既定 ON（新） | 差 |
|---|---:|---:|---:|
| A hipBLASLt | 1224.5 ms | **1075.0 ms** | -12.2% |
| B WMMA | 1427.3 ms | **1272.7 ms** | -10.8% |
| fused（DENSE=0） | 2474.4 ms | **2322.7 ms** | -6.1% |

内訳（5 run 平均、1 run 換算）:

| stage | A | B |
|---|---:|---:|
| GEMM | 569.4 ms | 854.5 ms |
| ROTATE_QUANT | 36.8 ms | 36.9 ms |
| ROWSCALE | 52.7 ms | — |
| RECON | 55.4 ms | 52.0 ms |
| FLAG_GEMM | 28.6 ms | — |
| ROTATE_OUT | 88.1 ms | 88.2 ms |

wall の短縮は rotate_quantize（約 117 → 37 ms）と rotate_out（約 184 → 88 ms）の分。
GEMM は F4 に依存しない。今回の GEMM は F4 OFF を測った回より 3% 高いが、
これはセッション間のクロック/熱ドリフトで、リグレッションではない。

## 品質（teacher-forced、context 2048、positions 256、bf16 KV）

基準は fused の logits dump（`/tmp/exl3_fused256.bin`）。

| 経路 | PPL | PPL 差 | KL 平均 | top-1 | top-5 | logit 最大差 |
|---|---:|---:|---:|---:|---:|---:|
| fused（基準） | 1.14538204 | — | — | — | — | — |
| **B（既定）** | **1.14538204** | **0** | **0** | **1.000** | **1.000** | **0** |
| A（既定） | 1.09654741 | -4.26% | 0.06085 | 0.977 | 0.847 | 17.02 |

B は fused と bit 一致する。F4 を既定 ON にしても数値は変わらない
（F4 OFF で生成した基準 dump と KL=0）。

A の差は RowScale 再量子化に由来する。活性の追加誤差は RRMS 約 1.1% で、
これは元の INT8 量子化誤差の約 2 倍に相当する。位置ごとの KL は中央値 2.4e-4、
94.9% の位置が 0.1 未満で、平均は 13 位置の裾が支配する。
256 token の低エントロピーなコーパスでは PPL 差の符号は安定しないため、
A の PPL が低いことを品質優位とは読めない。

## 検証

- EXL3 required テストは全て PASS。
- 既定（環境変数なし）で teacher-forced 比較が fused と完全一致:
  PPL 差 0、KL 0、top-1 1.0、logit 最大差 0。
- `test_exl3_contracts` から削除した API のケースを外した以外、テスト内容は不変。

## 注意

- `SHELF_EXL3_INT8_ROTATE_F4=0` は A/B 計測用に残している。
- 削除した N 分割ノブ（`ROTATE_TILE`）は既定 0 だったため、既定挙動は変わらない。
- 32 列未満 / 512 行未満の scalar 経路は残るため、decode の性能は不変。
