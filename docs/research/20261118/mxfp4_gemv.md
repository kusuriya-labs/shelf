# MXFP4 W4A8 decode GEMV と WMMA の使い分け

実験 ID: `MXFP4-GEMV-1`

MXFP4 dense linear に GEMV 経路を追加し、decode / verify の小さい rows で
GEMV と WMMA のどちらを使うかを、重みのロード速度とカーネル時間の実測で決めた記録である。
現在形の contract は `docs/developer/kernels.md` と
`include/shelf/kernels/optimized/linear/config.h` に移した。

## 1. 背景

`linear/mxfp4.hip` の既存 variant はすべて WMMA
（`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`）で、1 wave が
16 output column x 16 activation row の tile を評価する。rows が 1〜4 の decode
では 16 row のうち 1〜4 row しか使わず、残りは捨てになる。加えて rows == 1 では
重み 16 byte を lane あたり 8 byte ずつ読むため、命令数に対して有効な出力が少ない。

`linear/lm_head_int8.hip` は同じ問題に対して、小さい rows を 1 warp = 1 output row の
GEMV（係数を scalar で畳む）で処理し、rows が増えると WMMA へ切り替える。
MXFP4 でも同じ構成にするのが本実験である。MXFP4 の GEMV は E2M1 の 4 bit 係数を
E4M3 へ展開してから FP8 dot4（`__builtin_amdgcn_dot4_f32_fp8_fp8`）で積和する。

## 2. 実装

`src/shelf/kernels/optimized/linear/mxfp4.hip` の `gemv_mxfp4_w4a8_kernel`。

- 1 lane = 1 output row。warp = 32 output row。block = 8 warp = 256 thread。
- native layout の 32 K block は 16 byte なので、lane あたり `uint4` 1 回で
  32 係数（nibble 32 個）を読む。weight の warp 幅は 32 row x 16 byte = 512 byte
  の連続 2 block になり、そのまま coalesce する。
- activation は同じ 32 K block の fragment 2 つを `uint4` 2 回で読む。
  fp8 fragment は 16 row tile を 32 lane slot x 32 byte で持つため、
  activation row t の 32 K 分は slot `t & 15` の 16 byte と slot `(t & 15) + 16` の
  16 byte にある（後者は `+512` byte）。全 lane が同じ 32 byte を読むので broadcast
  になる。
- 係数の展開は既存の `mxfp4_expand`（`v_perm_b32` 2 本で 4 係数、sign は bit3 を
  OR）を 4 byte ごとに呼ぶ。
- 32 K block の中は 8 本の dot4 を 2 本の依存鎖（p0/p1）に分けて畳む。
  unfold では `acc = fma(p0 + p1, 2^e8m0, acc)`、fold では `acc += p0 + p1` で
  weight E8M0 は operand 側に畳んである。
- K は block 内の 8 warp で連続に分担し、F32 accumulator を LDS に書いて
  warp 0 が warp 昇順で足す（既存 SplitK と同じ縮約順で決定的）。
  狭い out_features でも grid が 1〜2 block にならないようにするためである。
- row は `kRows` 個の accumulator を持ち、weight 16 byte を全 row で共有する。
  rows 1/2/3-4/5-8 を `kRows` 1/2/4/8 で受ける（lm_head_int8 と同じ段）。

## 3. 正しさ

- `shelf-bench gemm --dtype mxfp4 --impl gemv --mxfp4-fold 0 --check` で、
  rows 1〜8、out 17/24/100/128、`5120:5120`、`2560:640` の全条件が
  correctness path と bf16 bit 一致（`10240:5120` の一部 shape だけ
  bf16 tie の丸めで 40960 要素中 2〜4 要素が 1 ulp 差、max_rel 0.0072）。
- fold を有効にすると、既存 RowBlock1 と*同じ* 25/128 の差が出る。これは
  bench の合成 E8M0 が fold table の厳密域（d <= 8）を超えるためで、
  GEMV 固有の差ではない。
- `tests/unit/test_mxfp4_linear` に rows 1/2/3/4、out 24/100/128/17、
  fold あり・なしを追加した。correctness path と bf16 完全一致、
  double 参照に対して bf16-exact 100%、rel_l2 1.4e-3〜1.9e-3。
- `tests/unit/test_qwen35_linear_selector` に GEMV/WMMA の選択を固定した。

## 4. 速度（`shelf-bench gemm --dtype mxfp4`）

R9700 1 枚（`HIP_VISIBLE_DEVICES=1`、device 0）、Release、2026-11-18。
`--warmup 30 --samples 15 --launches 15`。`auto`（旧経路 = RowBlock1 / SplitK）と
`--impl gemv` を同じ binary で交互に 2 回ずつ測り、warm-up 後の run を使った
（最初の process は clock が低いため捨てた）。p50 µs。GB/s は重み byte / p50。

| shape | rows | WMMA us | WMMA GB/s | GEMV us | GEMV GB/s | 比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 24:5120 | 1 | 47.8 | 2 | 5.1 | 16 | 9.37x |
| 48:5120 | 1 | 10.5 | 12 | 5.1 | 24 | 2.07x |
| 512:5120 | 1 | 10.9 | 120 | 5.2 | 254 | 2.12x |
| 1024:5120 | 1 | 11.6 | 227 | 5.6 | 468 | 2.07x |
| 2560:640 | 1 | 5.8 | 141 | 4.3 | 190 | 1.35x |
| 3072:5120 | 1 | 12.1 | 650 | 7.8 | 1008 | 1.55x |
| 5120:3072 | 1 | 10.9 | 722 | 7.8 | 1007 | 1.39x |
| 5120:5120 | 1 | 15.5 | 845 | 10.6 | 1241 | 1.47x |
| 5120:6144 | 1 | 18.3 | 859 | 11.8 | 1336 | 1.56x |
| 6144:5120 | 1 | 18.2 | 864 | 11.9 | 1327 | 1.54x |
| 10240:5120 | 1 | 24.5 | 1069 | 17.6 | 1493 | 1.40x |
| 12288:5120 | 1 | 27.0 | 1165 | 20.3 | 1553 | 1.33x |
| 17408:5120 | 1 | 35.3 | 1261 | 26.1 | 1710 | 1.36x |
| 34816:5120 | 1 | 150.5 | 592 | 152.8 | 583 | 0.98x |
| 24:5120 | 2 | 47.9 | 2 | 6.2 | 13 | 7.72x |
| 48:5120 | 2 | 10.4 | 12 | 6.1 | 20 | 1.70x |
| 512:5120 | 2 | 10.6 | 123 | 6.1 | 216 | 1.75x |
| 1024:5120 | 2 | 11.7 | 224 | 6.4 | 409 | 1.83x |
| 2560:640 | 2 | 5.9 | 138 | 4.7 | 173 | 1.26x |
| 3072:5120 | 2 | 12.3 | 637 | 8.5 | 923 | 1.45x |
| 5120:3072 | 2 | 10.9 | 722 | 8.3 | 942 | 1.31x |
| 5120:5120 | 2 | 15.7 | 837 | 11.0 | 1189 | 1.42x |
| 5120:6144 | 2 | 18.8 | 837 | 13.1 | 1199 | 1.43x |
| 6144:5120 | 2 | 18.5 | 850 | 12.7 | 1241 | 1.46x |
| 10240:5120 | 2 | 24.9 | 1054 | 20.9 | 1257 | 1.19x |
| 12288:5120 | 2 | 27.5 | 1143 | 23.0 | 1368 | 1.20x |
| 17408:5120 | 2 | 35.3 | 1262 | 30.4 | 1467 | 1.16x |
| 34816:5120 | 2 | 149.8 | 595 | 153.4 | 581 | 0.98x |
| 24:5120 | 4 | 44.0 | 2 | 7.8 | 10 | 5.62x |
| 48:5120 | 4 | 10.6 | 12 | 7.7 | 16 | 1.38x |
| 512:5120 | 4 | 10.9 | 120 | 7.1 | 184 | 1.53x |
| 1024:5120 | 4 | 12.1 | 218 | 7.9 | 330 | 1.52x |
| 2560:640 | 4 | 6.2 | 132 | 5.2 | 158 | 1.20x |
| 3072:5120 | 4 | 12.6 | 622 | 10.7 | 734 | 1.18x |
| 5120:3072 | 4 | 11.3 | 694 | 10.5 | 746 | 1.07x |
| 5120:5120 | 4 | 16.1 | 812 | 14.2 | 924 | 1.14x |
| 5120:6144 | 4 | 19.0 | 826 | 16.1 | 975 | 1.18x |
| 6144:5120 | 4 | 18.8 | 835 | 15.8 | 995 | 1.19x |
| 10240:5120 | 4 | 25.2 | 1040 | 22.9 | 1143 | 1.10x |
| 12288:5120 | 4 | 28.3 | 1110 | 25.7 | 1226 | 1.10x |
| 17408:5120 | 4 | 34.6 | 1287 | 33.5 | 1329 | 1.03x |
| 34816:5120 | 4 | 152.8 | 583 | 155.9 | 572 | 0.98x |
| 24:5120 | 8 | 44.6 | 2 | 10.2 | 8 | 4.37x |
| 48:5120 | 8 | 11.6 | 11 | 9.3 | 13 | 1.25x |
| 512:5120 | 8 | 11.5 | 114 | 9.3 | 142 | 1.25x |
| 1024:5120 | 8 | 12.9 | 204 | 10.3 | 256 | 1.25x |
| 2560:640 | 8 | 6.5 | 125 | 6.1 | 135 | 1.08x |
| 3072:5120 | 8 | 13.1 | 598 | 14.2 | 554 | 0.93x |
| 5120:3072 | 8 | 11.8 | 665 | 15.9 | 496 | 0.75x |
| 5120:5120 | 8 | 16.7 | 785 | 22.5 | 582 | 0.74x |
| 5120:6144 | 8 | 19.7 | 797 | 26.0 | 606 | 0.76x |
| 6144:5120 | 8 | 19.6 | 804 | 23.3 | 675 | 0.84x |
| 10240:5120 | 8 | 25.9 | 1012 | 32.4 | 810 | 0.80x |
| 12288:5120 | 8 | 29.1 | 1080 | 36.4 | 863 | 0.80x |
| 17408:5120 | 8 | 35.8 | 1246 | 50.8 | 877 | 0.70x |
| 34816:5120 | 8 | 157.0 | 568 | 166.3 | 536 | 0.94x |

読み取れること:

- rows 1〜4 では、重みが L3 に収まる全 shape で GEMV が勝つ（1.03x〜9.37x）。
  narrow out（24 / 48 / 512 / 1024）は WMMA が block 数不足で律速しており、
  GEMV の K 分割がそのまま効く（2.1x）。
- rows 8 では GEMV の命令数が 8 倍になり、out_features 3072 以上で負ける。
  out_features <= 2560 では依然として勝つ（48/512/1024 で 1.25x、2560:640 で 1.08x）。
  ただし rows == 8 の 2048:5120 / 2560:5120 を追加で測ると GEMV が 1.39x / 1.41x 速く、
  3072:5120 で 0.92x と逆転する。境界は 2560 と 3072 の間にある。

### 4.1 重みのロード速度が判定基準になる

同じ rows でも shape によって勝敗が変わるため、判定は geometry ではなく
「WMMA 経路が重みをどの速度で流せているか」で読む。上の表の WMMA GB/s を見ると
境界がそのまま現れている。

- 重みが L3 に収まる shape では WMMA が 650〜1400 GB/s を出す。DRAM 天井
  （636 GB/s）を超えているので、この帯域では命令側が律速であり、命令数が少ない
  GEMV が勝つ。GEMV は 1000〜1800 GB/s まで上がる。
- 重みが L3 を超える shape では WMMA が 568〜595 GB/s に落ち、DRAM 天井に張り付く。
  ここでは GEMV も 536〜583 GB/s で、勝敗は 0.94x〜0.98x の測定ゆらぎになる。

境界を確かめるため、L3（64 MiB）をまたぐ shape を rows 1/2/4 で測った。

| shape | 重み | rows | WMMA us | WMMA GB/s | GEMV us | GEMV GB/s | 比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20480:5120 | 50.0 MiB | 1 | 37.8 | 1386 | 29.2 | 1796 | 1.30x |
| 20480:5120 | 50.0 MiB | 2 | 38.9 | 1348 | 33.3 | 1574 | 1.17x |
| 20480:5120 | 50.0 MiB | 4 | 39.0 | 1343 | 36.8 | 1424 | 1.06x |
| 24576:5120 | 60.0 MiB | 1 | 45.1 | 1395 | 34.9 | 1802 | 1.29x |
| 24576:5120 | 60.0 MiB | 2 | 45.7 | 1377 | 39.9 | 1576 | 1.14x |
| 24576:5120 | 60.0 MiB | 4 | 46.6 | 1350 | 43.9 | 1433 | 1.06x |
| 28672:5120 | 70.0 MiB | 1 | 100.9 | 727 | 108.1 | 679 | 0.93x |
| 28672:5120 | 70.0 MiB | 2 | 98.8 | 743 | 103.6 | 709 | 0.95x |
| 28672:5120 | 70.0 MiB | 4 | 99.6 | 737 | 104.5 | 702 | 0.95x |
| 32768:5120 | 80.0 MiB | 1 | 136.0 | 617 | 141.0 | 595 | 0.97x |
| 32768:5120 | 80.0 MiB | 2 | 136.9 | 613 | 142.1 | 590 | 0.96x |
| 32768:5120 | 80.0 MiB | 4 | 138.7 | 605 | 144.9 | 579 | 0.96x |

L3 に収まる 60 MiB までは WMMA が 1350〜1395 GB/s、GEMV が 1.06x〜1.29x 勝つ。
L3 を超える 70 MiB から WMMA が 600〜740 GB/s（DRAM 天井）に落ち、GEMV は
0.93x〜0.97x で負ける。したがって判定は次の 2 段で足りる。

1. 重み byte が L3（64 MiB）以下か。超えるなら WMMA。
2. rows が 4 以下か。5〜8 は out_features <= 2560 のときだけ GEMV。

## 5. E2E（`shelf-bench tg`、ctx 128、tokens 32、prefill-chunk 128、R9700 1 枚）

`models/Qwen3.8-27B-Quark-AWQ-MXFP4`、同一 build で selector の GEMV 選択だけを
無効化した A/B。`--warmup 1`。

| 条件 | gpu_tokens_per_sec_p50 | wall tok/s |
| --- | ---: | ---: |
| 旧経路（SplitK / RowBlock1） | 28.91 | 28.51 |
| 新経路（GEMV） | 31.11 | 30.71 |

decode が +7.6%。`SHELF_QWEN35_PERF_STATS=1` で `LINEAR_MXFP4_W4A8` は
496 dispatch / 496 optimized / fallback 0 である。

## 6. 採否

- ACCEPT: MXFP4 W4A8 の GEMV（1 lane = 1 output row、32 K block を `uint4` 1 回、
  8 本の dot4、8 warp の K 分割と LDS 縮約）。
  rows 1〜4 で 1.03x〜9.37x、rows 5〜8 の narrow out で 1.08x〜1.25x。
- ACCEPT: 判定を重み byte（L3 64 MiB）と rows（4、narrow は 8）で行う。
  WMMA が DRAM 天井に張り付く帯では GEMV を選ばない。
- REJECT: rows 8 を超える GEMV。kRows の accumulator 数に比例して命令数が増え、
  out_features 3072 以上で WMMA に負ける（0.70x〜0.84x）。
- REJECT: GEMV への fold 常用。fold table の lookup が per-block rescale の
  削減を上回る傾向は RowBlock / SplitK と同じで、production の
  `mxfp4_gemm_folds_row_ref` は Prefill2D のみのまま据え置く。

## 7. 再現

```sh
HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype mxfp4 --impl auto \
  --shapes 24:5120,48:5120,512:5120,1024:5120,2560:640,3072:5120,5120:3072,5120:5120,5120:6144,6144:5120,10240:5120,12288:5120,17408:5120 \
  --rows 1,2,4,8 --warmup 30 --samples 15 --launches 15 --device 0

HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype mxfp4 --impl gemv \
  --shapes 20480:5120,24576:5120,28672:5120,32768:5120 \
  --rows 1,2,4 --warmup 30 --samples 15 --launches 15 --device 0

HIP_VISIBLE_DEVICES=1 ./build/tests/test_mxfp4_linear
HIP_VISIBLE_DEVICES=1 ./build/tests/test_qwen35_linear_selector
```
