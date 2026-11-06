# EXL3 5 bpw decode の高速化（2026-11-07）

実験 ID: `EXL3-5BPW-1`

結論:

- **ACCEPT**: ka=5 half=0 専用の `decode_tile_ka5` と、decode / verify 用の split-K 選択
- **REJECT**: rows=1 での `n_tiles=2`
- **REJECT**: split-K reduce の warp-per-block 化
- **DEFER**: trellis の `[n_tile][k_tile][words]` repack

## 1. 背景

`models/Qwen3.8-27B-exl3` は 5 bpw の EXL3 checkpoint である。TP=1 の decode は
14.4〜14.6 tok/s で、`gemm_exl3_w16a16<1u,true,false>` が 1 token の 87.3 %
（56.47 ms）を占めていた。

decode 1 token あたりの重み traffic は language linear + lm_head で 16.22 GB である。
DRAM 640 GB/s の理論下限は 25.3 ms/token だが、実測は 69.0 ms/token で、
実効 235 GB/s（roofline の約 37 %）しか出ていない。

## 2. 試した候補

### 2-1. ka=5 専用 dequant path（ACCEPT）

generic の `decode_tile` は `args.ka` と `args.half` を実行時に見て 16 bit 窓を
切り出す。ka=5 half=0 では 1 tile が 80 word で、8 状態の窓は 4 word に収まる。
ka=4 の fast path と同じ形で、`off0 = 40 * lane + 1269` から 8 状態を固定手順で
抽出する `decode_tile_ka5` を追加した。

重要な点として、最初は `decode_tile` 内の実行時分岐として実装したが、これでは
**ka=5 を使わない経路まで遅くなった**。

| 条件 | runtime branch | 専用 template 実体 |
| --- | ---: | ---: |
| PP128、`SHELF_EXL3_KA5=0` | 389.4 tok/s | **449.1 tok/s** |
| PP128、`SHELF_EXL3_KA5=1` | 435.6 tok/s | **505.5 tok/s** |
| TG、`SHELF_EXL3_KA5=0` | 14.53 tok/s | **14.77 tok/s** |

そのため `gemm_exl3_w16a16_kernel` に `bool kKa5` を追加し、ka=5 half=0 のときだけ
`<..., true>` を launch する形にした。`kKa5=false` の実体は変更前と同じコードになる。

### 2-2. decode / verify の split-K 選択（ACCEPT）

`make_launch_plan` の既定 split を rows で分けた。

- rows == 1: `grid_x <= 160` なら 8、それ以外は 1
- rows 2〜8: 8
- rows >= 9: 従来（`grid_x <= 16` なら 8、`<= 160` なら 4、それ以外は 1）

rows=1 の大きな linear（lm_head）は split=1 の方が速く、verify の rows=8 は
lm_head も含めて split=8 の方が速い。

### 2-3. rows=1 の `n_tiles=2`（REJECT）

`SHELF_EXL3_NTILES=2` で A/B した。

| 条件 | tok/s |
| --- | ---: |
| n_tiles=1、split=8 | **14.80** |
| n_tiles=2、split=8 | 14.20 |
| n_tiles=2、split=16 | 13.95 |

CTA 数が半減し、LDS 使用量が増えるため decode では遅い。採用しない。

### 2-4. split-K reduce の warp-per-block 化（REJECT）

rows=1 の reduce は 8 warp のうち 1 warp しか使っていない。1 warp が 1 output block
を担当し、grid_x を 1/8 にする変更を試した。

| 条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| TG（rows=1） | 16.12 | 16.23 |
| DFlash2 TP2（verify rows=8） | 60.97 | **55.05**（rows<=8 版） |

rows=8 では並列度が落ちて明確に遅い。rows=1 だけに限定すると DFlash2 は 61.03 で
同等、TG の差は 0.7 % 程度。複雑さに見合わないため採用しない。

### 2-5. trellis repack（DEFER）

`[k_tile][n_tile][words]` から `[n_tile][k_tile][words]` へ repack すれば、1 CTA が
読む K 列は連続になる。ただし loader、dense recon、grouped expert、TP partition、
DFlash2 の全 consumer に layout 変更が波及する。

既存の isolation benchmark では、EXL3 の decode を除いた WMMA 構造だけで
hipBLASLt の 6.82 倍遅く、decode chain の追加は 1.40 倍だった。つまり支配的なのは
tile 構造と data supply であり、repack 単独の見込みは低い。今回は実装せず、
rows=1 専用 GEMV など構造側の変更を次の候補にする。

## 3. 結果

TP=1、device 1、bf16 KV、`--arena-gib 28`。

| 断面 | 変更前 | 変更後 | 比 |
| --- | ---: | ---: | ---: |
| TG ctx128 / 16 token | 14.49〜14.56 | **16.79〜16.83** | +15.7 % |
| PP 128 | 447.96 | **504.60** | +12.6 % |
| PP 512 | 1042.93 | 1053.62 | +1.0 % |
| PP 2048 | 1356.65 | 1364.05 | +0.5 % |

TP=2。

| 断面 | 変更前 | 変更後 | 比 |
| --- | ---: | ---: | ---: |
| TG ctx128 / 16 token | 22.99 | **26.52** | +15.4 % |
| PP 128 | 812.89 | **902.39** | +11.0 % |
| DFlash2 drafts=7 | 58.44 | **63.26** | +8.3 % |

DFlash2 の生成列は target-only と一致する。TG の `LOGITS` は ka5 の on / off で
bitwise 一致した（同じ split 条件）。

## 4. 受入

- `test_exl3_codec`、`test_exl3_linear`、`test_exl3_dense`、`test_exl3_grouped` はすべて PASS。
- `test_exl3_linear` に ka=5 の rows=1 / rows=16 / rows=64 split 自動の case を追加。
- `test_exl3_dense` に ka=5 の dense recon 比較を追加。
- `SHELF_EXL3_KA5=0` で従来経路に戻せる。
