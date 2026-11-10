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

## 性能（2048 prefill、3 run 中央値）

| 経路 | F4 既定 OFF（旧） | F4 既定 ON（新） | 差 |
|---|---:|---:|---:|
| A hipBLASLt | 1224.5 ms | **1069.8 ms** | -12.6% |
| B WMMA | 1427.3 ms | **1265.9 ms** | -11.3% |

wall の短縮は rotate_quantize と rotate_out の分。GEMM は変わらない。

## 検証

- EXL3 required テストは全て PASS。
- 既定（環境変数なし）で teacher-forced 比較が fused と完全一致:
  PPL 差 0、KL 0、top-1 1.0、logit 最大差 0。
- `test_exl3_contracts` から削除した API のケースを外した以外、テスト内容は不変。

## 注意

- `SHELF_EXL3_INT8_ROTATE_F4=0` は A/B 計測用に残している。
- 削除した N 分割ノブ（`ROTATE_TILE`）は既定 0 だったため、既定挙動は変わらない。
- 32 列未満 / 512 行未満の scalar 経路は残るため、decode の性能は不変。
