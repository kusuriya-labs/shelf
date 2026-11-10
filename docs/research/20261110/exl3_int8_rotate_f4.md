# EXL3-INT8 回転の float4 化

## ID

20261110-exl3-int8-rotate-f4

## 背景

`src/shelf/kernels/optimized/linear/exl3_int8.hip` の次の 3 kernel は、128 thread / 128-block の
scalar/smem H128（`fwht128(v, smem, t)`）を使っていた。1 CTA が 1 個の 128 要素ブロックを担当し、
Hadamard の後半 2 段で shared memory と barrier を使う。

- `exl3_int8_rotate_quantize_kernel`（入力回転 + INT8 量子化）
- `exl3_int8_rotate_out_kernel`（出力回転、float yprime）
- `exl3_int8_rotate_out_i32_kernel`（出力回転、int32 acc）

`exl3_int8_prefill.hip` の `rotate_out_i32_prefill` は前半 5 段だけ shuffle の hybrid だった。
`exl3_mul24_grouped.hip` は既に `exl3_detail::fwht128(v0..v3, lane)`（float4）を使っており、変更しない。

## 実装

各 kernel の隣に float4 版を追加し、`SHELF_EXL3_INT8_ROTATE_F4`（既定 0）で切り替える。

- `exl3_int8_rotate_quantize_f4_kernel`: 1 CTA = 1 row、256 thread（8 wave）。各 wave が
  128-K group を `g = wave, wave+8, ...` で担当する。入力は bf16 4 個を 64-bit load、code は
  4 個を 32-bit store、amax は warp shuffle。
- `exl3_int8_rotate_out_f4_kernel`: 1 CTA = 8 row、256 thread（8 wave）、1 wave = 1 row。
  split-K の加算、H128、`svh` を wave 内で完結し、float4 load / 64-bit store。
- `exl3_int8_rotate_out_i32_f4_kernel`: 同上。通常 row は int4 load、flag row は float4 load。
- `launch_exl3_int8_prefill_rotate_out_i32` は `SHELF_EXL3_INT8_ROTATE_F4=1` のとき
  `launch_exl3_int8_rotate_out_i32` に委譲し、hybrid を full float4 に置き換える。

アラインメント: f4 経路は row stride が 4 要素の倍数のときだけ使い、そうでなければ scalar に
fallback する。量子化の scale と code は scalar と同一で、出力の積算順
（`(v * kInvSqrt128) * svh`、`(acc * rowscale) * kExl3Int8GridStep`）も scalar に合わせた。
この順序を合わせないと BF16 1 ulp 差が出る。

## 計測条件

`docs/perf/methodology.md` に従う。R9700 (gfx1201)、GPU1 を `HIP_VISIBLE_DEVICES=1` で隔離、
`flock /tmp/r9700.lock`。commit `6f22cc0`。kernel 単体は rows=2048、32 反復 × 7 サンプルの
p50 を、base/candidate を交互に 3 プロセスずつ。E2E は実モデル
`models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`、2048 token、chunk 2048、BF16 KV、
warmup 1 + 3 run を交互に 3 プロセスずつ。run 間の p50 のばらつきは全条件 30% 未満。

## 結果（kernel 単体、rows=2048、p50 µs）

| kernel | 形状 | scalar | float4 | 短縮 |
|---|---|---:|---:|---:|
| rotate_quantize | k=5120 | 167.58 | 50.71 | 69.7% |
| rotate_out_i32 | n=5120 | 144.45 | 49.63 | 65.6% |
| rotate_out_i32 | n=17408 | 726.95 | 380.57 | 47.6% |
| rotate_out | n=5120, splits=1 | 140.49 | 49.13 | 65.0% |
| rotate_out | n=5120, splits=4 | 620.92 | 344.01 | 44.6% |

n=17408 の短縮率が小さいのは、データ量 213 MB が L3 64 MB を超えて DRAM 律速になり、
CTA オーバーヘッドの占比が下がるためである。

## 結果（E2E prefill、2048 token、p50 ms）

| 設定 | scalar | float4 | 短縮 |
|---|---:|---:|---:|
| `PREFILL_ROTATE=0`（既定） | 1207.13 | 1049.30 | 13.07% |
| `PREFILL_ROTATE=1`（hybrid 出力回転） | 1218.91 | 1054.93 | 13.45% |

## N 分割（出力回転）

`rotate_out_i32` の f4 版は 1 CTA = 8 row で、N 方向は 1 wave が全 128 列ブロックをループする。
`blockIdx.y` で N を `SHELF_EXL3_INT8_ROTATE_TILE` 個の 128 列ブロックに分割できるようにし、
N=17408（136 ブロック、256 CTA）と N=5120（40 ブロック）で tile 幅を比較した
（rows=2048、f4、p50 µs）。

| tile | CTA 数（N=17408） | N=17408 | N=5120 |
|---|---:|---:|---:|
| 0（全列、現行） | 256 | 380.59 | 50.02 |
| 32（4096 列） | 1,280 | 372.89 | 44.84 |
| 16（2048 列） | 2,304 | 372.12 | 47.57 |
| 8（1024 列） | 4,352 | 371.81 | 48.90 |

N=17408 は 2.0〜2.3% の短縮で、tile=8〜32 の差は 0.3% 未満。N=5120 は tile=32 で約 10% 短縮するが、
tile=8 では悪化する（CTA 過多）。E2E 2048 token では tile=0 1055.03 ms に対し tile=32 1053.09 ms
（0.18%、ほぼ測定ばらつき）。

理由: N=17408 は 213.9 MB で L3 64 MB を超え、実効帯域は 562 GB/s（tile=0）→ 575 GB/s（tile=8）。
同じ読み書き比（int32 read : bf16 write = 2:1）の純コピーは 1:1 copy 505 GB/s total、
read-only 628 GB/s であり、575 GB/s はほぼ上限。したがって N=17408 の低い改善率は CTA 構成ではなく
DRAM 帯域が主因で、N 分割では 2% 程度しか回収できない。N=5120 は 62.9 MB で L3 に収まり、
実効帯域 1273 GB/s まで出るため、並列度を上げる N 分割が効く。

## デコード行数の下限

f4 版は `grid = ceil(rows/8)` なので rows が小さいと CTA が足りない。rows=1 の `rotate_out` は
scalar 4.1 µs に対し f4 36.1 µs、rows=16 では 5.3 µs 対 34.8 µs と大幅に遅い。そこで
`rotate_quantize` は rows>=32、`rotate_out` / `rotate_out_i32` は rows>=512 のときだけ f4 を使う。
rows=1/16 で decode の stage 時間が scalar と一致することを確認した。

## 正しさ

- `test_exl3_int8_linear`: `ROTATE_F4=0/1` とも ALL PASS。f4 では `out_rel=0`（scalar と bit 一致）。
- `test_exl3_int8_prefill_rotate`: `ROTATE_F4=0/1` とも bitwise PASS。
- `test_exl3_int8_decode` / `test_exl3_int8_tp` / `test_exl3_int8_grouped`: 両方 PASS。
- `shelf-bench quality`（context 2048、64 位置、語彙 248320）: `KL_MEAN=0`、
  `LOGIT_MAX_ABS_DELTA=0`、`TOP1_AGREEMENT=1`、`TOP5_OVERLAP=1`、`PPL_DELTA=0`。

積算順を scalar に合わせたため、f4 は速度だけでなく出力も bit 一致する。

## 判断

ACCEPT（opt-in）。既定は `SHELF_EXL3_INT8_ROTATE_F4=0` のままにし、実モデル 2048 token の
E2E 短縮 13% と bit 一致を確認済みの採用候補として残す。既定化する場合は環境変数の既定値を
反転するだけでよい。row 数の下限は decode 回帰を防ぐために実装側で固定した。

N 分割（`SHELF_EXL3_INT8_ROTATE_TILE`）は既定 0（全列）のまま、tuning knob として残す。
N=17408 の短縮は約 2% で、E2E ではほぼ測定ばらつきの範囲。N=5120 では tile=32 が約 10% 速いので、
L3 に収まる形状が多い workload では tile=32 を試す価値がある。

## 補足

同一環境の単体計測では、hybrid `rotate_out_i32_prefill` は scalar より約 8% 遅かった
（n=5120: 156.85 vs 144.45 µs、n=17408: 746.18 vs 726.95 µs）。hybrid は barrier が
scalar より 1 回多いためと考えられる。この観察は f4 の採否には影響しない。
