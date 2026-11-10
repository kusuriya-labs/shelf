# grouped EXL3 の K chunk ペア stage

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `src/shelf/kernels/optimized/linear/exl3_family/mul24/grouped.hip`。

## 結論

routed expert の grouped EXL3 で、K chunk を 2 つまとめて LDS に stage する
カーネルを追加した。barrier と xrot staging が半分になる。
gate/up（K=2560、`n_tiles=1`）だけに適用し、down（K=640、`n_tiles=2`）は
従来の 1 chunk stage のままにする。

8K prompt、両 rank 合計の grouped GPU 時間:

| 経路 | 変更前 | 変更後 |
|---|---:|---:|
| gate/up（`n_tiles=1`） | 1,155.4 ms | **978.8 ms** |
| down（`n_tiles=2`） | 432.6 ms | **390.6 ms** |
| grouped 合計 | 1,631.3 ms | **1,395.1 ms** |

E2E prefill（FlashNext 3.05 bpw、TP=2、fp8 KV、`--warmup 1 --runs 3`）:

| prompt | 変更前 | 変更後 |
|---|---:|---:|
| 8K | 2,827 ms / 2,902 tok/s | **2,682 ms / 3,054 tok/s** |
| 32K | 12,141 ms / 2,699 tok/s | **11,576 ms / 2,831 tok/s** |

## 実装

`gemm_mul24_grouped_kernel` に `kPairK` テンプレート引数を追加した。

- `kPairK=true` では xrot の pitch を 264（FP16）/ 272（int8）にし、2 chunk 分を
  1 回の `__syncthreads()` で stage する。
- `kPairK=false` では従来どおり 136 / 144。
- 起動時に `args.k / 128 >= 8` で選ぶ。gate/up は 20 chunk、down は 5 chunk。
- INT8 activation では chunk ごとに row scale が違うため、8 K step ごとに
  `acci` を fp32 へ加算して零リセットする。

K chunk 数が奇数の場合は、最後の 1 chunk を単独 stage する。

## 却下した形

最初に全 shape へペア化を適用したところ、down が 432.6 → 521.8 ms に悪化した。

- `n_tiles=2` の VGPR が 136 → 224、LDS が 25,088 → 33,280 B に増えた。
- `acci` の宣言位置を戻しても VGPR は変わらず、2 chunk 分を unroll した本体が
  原因だった。
- `kPairK` をコンパイル時特殊化にして down を 1 chunk のままにすると、
  down は 390.6 ms まで改善した。

## 検証

`test_exl3_mul24_grouped` に K=1024 / 1152 / 1280 / 1408 のケース
（`n_tiles=1/2`、偶奇の chunk 数、fuse / row_token）を追加した。
CPU reference との誤差は FP16 で最大 0.013、INT8 で 7.6e-05、未書き込み領域は 0。
