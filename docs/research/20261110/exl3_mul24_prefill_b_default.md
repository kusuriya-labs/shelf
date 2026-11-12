# mul24 dense prefill の B 経路を既定化

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。

## 結論

`SHELF_EXL3_MUL24_PREFILL_AB` の既定を A（hipBLASLt + RowScale）から
B（recon_wmma + 自作 iu8 WMMA）へ変更した。A は
`SHELF_EXL3_MUL24_PREFILL_AB=A` で明示した場合だけ使う。

理由は RowScale の追加誤差を避けること。速度では A が優位であり、B の既定化は
正しさを優先する選択である。

## 精度

`exl3_mul24_rowscale_accuracy.md` の実測では、RowScale は実モデル活性に
1.10% RRMS の追加誤差を与える（元の INT8 量子化誤差 0.54% の約 2.1 倍）。
end-to-end では KL 中央値 2.4e-4 だが、5.1% の位置で KL 0.1〜3.6 の裾を持つ。

B は per-128-K の activation scale をそのまま使い、行単位の再量子化をしない。
`test_exl3_mul24_linear` と実モデル teacher-forced で fused と KL=0 / logit 差 0 を
確認している。

## 性能

B の既定化は速度のためではない。dense 27B 4bpw、8K prompt、`--warmup 1 --runs 3`:

| 経路 | TP1 | TP2 |
|---|---:|---:|
| A（RowScale） | **4,026 ms / 2,035 tok/s** | **2,469 ms / 3,317 tok/s** |
| B（厳密） | 5,176 ms / 1,583 tok/s | 2,978 ms / 2,751 tok/s |

A は TP1 で 1.29 倍、TP2 で 1.21 倍速い。B は K128 ごとに scale + FMA を
適用するため GEMM 単体では不利で、A の約 1.5 倍が理論下限と見積もられている。
実測 1.21〜1.29 倍はその下限に近い。

FlashNext 3.05 bpw、32K prompt、TP2:

| 経路 | prefill | tok/s |
|---|---:|---:|
| A | 13.35 s | 2,454 |
| B（既定） | 13.23 s | 2,478 |

FlashNext では dense linear の比率が小さく、A/B の差は 1.5% 程度に収まる。

## 対象範囲

B が効くのは dense EXL3 linear（attention / GDN / shared expert）だけである。
routed expert は grouped fused kernel、router は F16 dense linear のまま。
decode は fused decode で、A/B の対象外。
