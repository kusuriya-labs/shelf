# EXL3 dense prefill 付随処理の融合

2048 token / prefill chunk 2048 / TP=2 / BF16 KV の prefill で、EXL3 付随処理の
どこまでを融合すべきかを測った記録。数値はすべて `shelf-bench pp --mode forward`
の `gpu_ms_median`（3 ペア交互実行）。

## EXP-EXL3RP-1　rowprepare を rotate_quantize に融合

### 仮説

現行は `rotate_quantize`（1 CTA = 1 行、1088〜4 割の K 群ごとに H128 と量子化）の後、
`rowprepare`（行 peak / 二乗和 → 行 scale、flagged 行の退避、非 flagged 行の再丸め）を
別 launch で走らせる。後者は `codes` を読み直すので、量子化の結果をレジスタに
保持したまま 1 カーネルで完結させれば `codes` の往復（行列あたり 35 MB × 400）が
消えるはず。

### 実装

`exl3_mul24_rotate_quantize_prepare_kernel<kRegs>` を追加した。group ループの
量子化バイトを lane あたり `kRegs` 個の uint32（k/128/8 の切り上げ）に保持し、
block 内 reduction で行 peak を決めた後、flagged 行は `codes_f` へ、それ以外は
`ratio = scale_g / peak` で再丸めして `codes` へ **1 回だけ**書く。reduction と
再丸めの式は `exl3_mul24_rowprepare_kernel` と同一にした。slot の atomic カウンタは
同じ launch 内で 0 にできないため `hipMemsetAsync` を前置した。

focused test（`test_exl3_mul24_linear`）に 2 経路の出力比較を追加し、
`codes` / `scales` / `rowscale` / flag / `codes_f` / `scales_f` が全形状で
byte 一致（slot 表が飽和する形状のみ、どの行が落ちるかが atomic 順で変わることを許容）。
VGPR は kRegs=17 で 61（spill なし、private 0）。

### 結果

| ペア | 現行（2 launch） | 融合（memset + 1 launch） | 差 |
| --- | ---: | ---: | ---: |
| 1 | 606.45 ms | 608.30 ms | +0.31% |
| 2 | 608.19 ms | 608.76 ms | +0.09% |
| 3 | 610.08 ms | 610.59 ms | +0.08% |

### 結論: REJECT

3 ペアとも改悪。`rowprepare` は 8.0 ms/rank（約 20 µs/呼び出し）しかなく、
`codes` の読み直しは L3 に載っていてほぼ無料だった。融合側は
(a) 前段の量子化が全 group を書き終えるまで barrier で待つため、行ごとの
量子化と再丸めのパイプラインが切れる、(b) slot カウンタの reset に
`hipMemsetAsync` が 1 launch 増える、の 2 つで相殺される。
書き込み 1 回化で減るのは高々 8 ms で、その大半が元から他の行の処理と
重なっていた。

## EXP-EXL3RP-2　flagged fixup の列タイル化

### 背景

dense prefill の出力は `prefill_super64` の epilogue が rotate-out まで済ませるが、
flagged 行（行内の 1 グループが scale エネルギー `SHELF_EXL3_INT8_FLAG`（既定 0.5）を
超える行、chunk あたり最大 `rows/8`）だけは per-group scale の `mul24_gemm` を通すため
`yprime2`（F32）に落ち、`launch_exl3_mul24_rotate_out_i32_flagged` が H128 と `svh` を
適用して BF16 へ書く。プロファイルでは 9.1 ms/rank。

旧カーネルは 1 CTA 8 行・1 wave が 1 行の全列（n/128 ブロック、5120 なら 40）を
逐次処理していた。flagged 行は 2048 行中 200 行程度なので、同時に走る wave は
200 本しかなく、1 wave あたり未完了ロード 1 本の連鎖になる。

### 実装

`exl3_mul24_rotate_out_i32_f4_kernel` に `kTiled` を足し、CTA を
`(row, column tile)` の 2 次元にした。wave `w` は `blockIdx.y * 8 + w` から
`8 * gridDim.y` 刻みで 128 列ブロックを担当する。grid.y は
`min(ceil(col_blocks / 8), max(1, 16384 / rows))` で上限を設け、`n` が大きい
形状（lm_head など）でも grid が `rows × 列タイル` に発散しないようにした。
flagged でない行の CTA は `slots[m]` を見て即座に抜ける。

### 結果

上限なしの列タイル（grid.y = ceil(col_blocks / 8)）:

| ペア | 旧（1 wave = 1 行全体） | 列タイル | 差 |
| --- | ---: | ---: | ---: |
| 1 | 610.94 ms | 608.21 ms | −0.45% |
| 2 | 612.22 ms | 607.49 ms | −0.77% |
| 3 | 611.83 ms | 608.78 ms | −0.50% |

grid 上限つき（`16384 / rows`、n=5120 では 5 タイルのまま）:

| ペア | 旧（1 wave = 1 行全体） | 列タイル（上限つき） | 差 |
| --- | ---: | ---: | ---: |
| 1 | 610.87 ms | 606.44 ms | −0.73% |
| 2 | 611.36 ms | 607.82 ms | −0.58% |
| 3 | 611.35 ms | 609.04 ms | −0.38% |

focused test は byte 一致（`fused_bad=0`、`rotate_out_i32` の f4 経路と比較）。

### 結論: ACCEPT

3 ペアとも改善。

## EXP-EXL3RP-3　flagged fixup を flag GEMM の epilogue に融合しない理由

`mul24_gemm`（flagged 行専用、30.8 ms/rank）の epilogue で rotate-out まで
済ませれば `yprime2` の書き＋読みが消える。しかし現行の `exl3_mul24_gemm_kernel` は
1 wave が 16 列（`col = lane & 15`）を持ち、CTA 全体でも 64 列しかない。
出力側の H128 は 128 列連続を 1 wave に集める必要があるため、そのままでは適用できない。
適用するには CTA を 128 列（8 wave）へ組み替え、`kBlockN` の LDS 段取りと
per-group scale の float 累算を epilogue に足す必要があり、30.8 ms の
230 TOPS を出している本体の register / occupancy を触ることになる。
列タイル化後の fixup は 6 MB/呼び出し（flagged 200 行 × 5120 列）の読み書きで、
DRAM 上限 600 GB/s に対して残る余裕は数 µs/呼び出し（0.2% 未満）しかない。
費用対効果が合わないため、fixup は独立 launch のまま残す。

## 付随処理の内訳（2048 行・chunk 2048・per rank）

| 段階 | ms | share | 融合の可否 |
| --- | ---: | ---: | --- |
| 入力回転・INT8 量子化 | 22.9 | 3.6% | producer フック側で伸ばす（GEMM 内は不可） |
| rowprepare | 8.0 | 1.3% | EXP-1 で REJECT |
| recon | 20.5 | 3.3% | 済（4 K-tile store）。bytes しか残らない |
| flagged 行 GEMM | 29.2 | 4.7% | これ自体が consumer。epilogue 融合は EXP-3 で却下 |
| flagged fixup | 6〜7（推定） | 1.0% | EXP-2 で改善済み |
| 出力回転・scale | 0（統合済） | — | 済 |
