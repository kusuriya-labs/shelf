# FP8 PTPC の hipBLASLt 化（REJECT）

## 目的

`gemm_bf16` / `gemm_fp8_ptpc` の手書きカーネルを hipBLASLt に置き換える。
BF16 は ACCEPT 済み。ここでは FP8 per-token / per-channel（PTPC）の可否を判定する。

## 環境

- GPU: Radeon AI PRO R9700（gfx1201）× 4
- ROCm SDK: `/opt/zen/.venv/.../_rocm_sdk_devel`（HIP 7.15.26333、hipBLASLt 1.4 系）
- `HIPBLASLT_TENSILE_LIBPATH` 既定: `.../lib/hipblaslt/library/gfx1201`

## 実測した事実

### 1. scale attribute の正しい向き

`HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` には **ポインタ変数のアドレス**を渡す。

```cpp
hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                &a_scale, sizeof(a_scale));
```

ポインタ値そのもの（`buf = a_scale`）を渡すとライブラリは scale 要求として解釈せず、
scaled でない通常カーネルが返る。この場合 `hipblasLtMatmulAlgoGetHeuristic` は
`found=8` を返すが scale は適用されない。誤判定しやすいので注意する。

### 2. `OUTER_VEC_32F` が成立する組み合わせ

gfx1201 の solution DB で `SABV`（Scale A/B Vector）を持つ library は 1 個だけである。

```
TensileLibrary_F8F8_BF8_HA_Bias_SABV_SAV_UA_Type_F8B_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201
```

- 出力 D が **BF16** のときだけ `found>0`。D が F32 だと `found=0`。
- 選ばれる solution は
  `Cijk_Alik_Bljk_F8BS_..._SABV_SAV_..._MT128x128x32_MI16x16x1`
  などの汎用・参照系カーネルで、gfx1201 向けにチューニングされた FP8 library
  （`F8F8_SF8_..._Alik_Bljk`、3116 solution）とは別系統である。

### 3. operand の向き

BF16 経路と同じく **A=weight、B=activation** に置く。hipBLASLt は column-major 前提なので、
`weight[out_features, k]` を A に、`activation[rows, k]` を B にすると
`out_features × rows` が求まり、これが row-major の `[rows, out_features]` と一致する。
逆に置くと C が転置されて壊れる。

### 4. 数値

`M=1/8/64/128`、`K=2560/5120`、`N=4096` で CPU f64 参照と比較。

- `max_rel = 0.0036`（BF16 出力の丸め相当）
- `models/Qwen3.8-Flash-Next-Quark-MXFP4` の greedy 8 token で
  `GREEDY_TOKEN_SUM=819968`。手書きカーネルと一致する。

### 5. 性能

`models/Qwen3.8-Flash-Next-Quark-MXFP4`、TP4、pp2048（5 run median）。

| 構成 | gpu_ms p50 | tok/s |
| --- | ---: | ---: |
| BF16 のみ LT | 633.31 ms | 3233.81 |
| BF16 + FP8 LT | 630.87 ms | 3246.30 |
| BF16 のみ LT（再測） | 633.97 ms | 3230.42 |

prefill は誤差の範囲で改善しない。decode（`shelf-bench tg --context 128 --tokens 8`）は
**25.94 → 16.79 tok/s** と大きく悪化した。`MT128x128x32` を M=1 で起動するため、
1 行のために 128 行分の tile を計算する。

## 判定

**REJECT。** FP8 PTPC は手書きカーネルを維持する。

理由:

1. `OUTER_VEC_32F` を満たす solution が未チューニングの参照カーネルしかなく、
   prefill で手書きカーネルを上回らない。
2. decode では tile 形状が合わず 35 % 悪化する。
3. scale を落として F32 出力にする案もあるが、その場合は scale 適用の後段カーネルが
   必要になり、prefill の利得も期待できない。

将来 hipBLASLt 側に gfx1201 向けの scaled FP8 solution が追加された場合は、
`docs/perf/` の pp2048 と decode を再測したうえで再検討する。
