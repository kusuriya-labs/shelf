# MoE decode の launch 下限と op 別 roofline（`LINEAR_BF16` 43 % の実体）

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` decode。動機は「MoE decode の最大項目
`LINEAR_BF16` を削る」だったが、先に **項目別の roofline を checkpoint から直接計算**して、
どこに余地があるのかを測った。実行経緯の記録で、現在の性能値は `docs/perf/current.md` を正本とする。

## 計測口と注意点（per-op 計測自体が decode を 1.8 倍遅くする）

- per-op GPU time: `SHELF_QWEN35_PERF_TIMING=1` + `--perf-stats-output`（同一 dir へ
  `semantic_timing.csv`）。集計は `seq,kernel_id,path,implementation,logical_consumed,duration_us`
  を kernel_id / path で summarise する
- **この計測は dispatch の前後に event ペアを置く**。MoE decode の context 64 は
  timer off で 59 tok/s、timer on で **33 tok/s**（1 token 1355 dispatch × 約 8 µs が下限として乗る）。
  よって csv の `mean_us` は絶対値として使わない。`share` は相対比較にだけ使い、
  絶対値は uninstrumented の `gpu_ms` と突き合わせる
- dispatch 数: `SHELF_QWEN35_PERF_STATS=1` の census。context 4096 / 8 token で
  `logical_dispatches 10840` / `physical_kernel_launches 10928` → **1 token あたり logical 1355 /
  physical 1366**

## launch 下限の実測

`shelf-bench elementwise --variant optimized --profile swiglu --features 128 --rows 1`
（device 0、samples 40、warmup 20）。1 launch 384 B 読むだけの kernel を連打した:

| `--launches` | 1 | 2 | 4 | 8 | 16 | 32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p50 / launch | 9.72 µs | 7.60 | 5.89 | 4.15 | 3.84 | **3.59 µs** |

**この platform の launch + 最小 kernel の下限は約 3.6 µs**。1355 dispatch/token なら
**4.9 ms/token** が launch だけで決まる。単発（`--launches 1`）は 9.7 µs で、per-op csv の
`mean_us` 下限（約 8 µs）と一致する。

## weight traffic の floor（checkpoint から計算）

`model.safetensors` のヘッダから text 経路（`model.language_model.layers.0..39` + `lm_head`）の
tensor 形状を直接数えた。layer_types は `LLL A` × 10（linear_attention 30 / full_attention 10）、
experts 256 / top_k 8。**decode は 1 token でこれらの weight を 1 回ずつ読む**（L3 64 MB ≪ 4.4 GB、
かつ各 weight は 1 token に 1 回参照）。

| family | bytes / token | 636 GB/s at | floor ms / token |
| --- | ---: | ---: | ---: |
| lm_head（bf16 248320×2048） | 1017.1 MB | 100 % | 1.60 |
| linear_attention（GDN 30 層） | 2023.1 MB | | 3.18 |
| full_attention（10 層） | 545.3 MB | | 0.86 |
| routed experts（MXFP4、256 のうち 8） | 534.8 MB | | 0.84 |
| shared expert | 251.8 MB | | 0.40 |
| router | 41.9 MB | | 0.07 |
| norms / other | 0.3 MB | | 0.00 |
| **合計** | **4.414 GB** | | **6.94 ms → 144 tok/s** |

`docs/research/20261103/qwen36_35b_a3b_moe_structure.md` §9 の推算（4.41 GB / 145 tok/s）と
一致したが、こちらは checkpoint の形状から出した数で、DRAM 実測 636 GB/s を使っている。

## 実測 decode の内訳（context 64、59.01 tok/s = 16.95 ms/token）

| 区分 | ms / token | 割合 |
| --- | ---: | ---: |
| weight traffic の roofline floor | 6.94 | 41 % |
| launch 下限（1355 × 約 3.6 µs） | 4.9 | 29 % |
| kernel が自分の roofline まで届いていない分 | 5.1 | 30 % |

`LINEAR_BF16` は 43〜50 % の share だが、family の floor（lm_head + GDN + attention + shared +
router = 6.10 ms/token）に対し、instrumented mean から launch 下限を引いた実測は約 8 ms/token。
**bandwidth の余地は 2 ms/token（E2E の 11 %）で、43 % の share の割に小さい**。残り 5 ms の
本体は launch 数と、小さな op 1 本あたり 3.6 µs の下限である。

op 別の self-roofline（context 64、timer on。`mean_us − 8 µs`（launch 下限）× count/token）:

| kernel | mean µs | count / token | 実 ms/token | その op の floor | 余地 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LINEAR_BF16 | 28.31 | 391 | 7.9 | 6.10 | 1.8 |
| LINEAR_MXFP4_W4A8_GROUPED | 22.82 | 80 | 1.19 | 0.84（expert weight） | 0.35 |
| **MOE_COMBINE** | **40.39** | 40 | **1.30** | 約 0.03（36 KB/token） | **1.27** |
| **MOE_PLAN_SCAN** | **27.40** | 40 | **0.78** | ほぼ 0（256 語の scan） | **0.78** |
| MOE_ROUTER | 14.87 | 40 | 0.27 | ほぼ 0 | 0.27 |
| GDN_RECURRENCE | 20.86 | 30 | 0.39 | state traffic | 0.2 |
| RMS_NORM | 9.36 | 131 | 0.18 | 0.02 | 0.16 |

上位の余地は **traffic がほぼゼロの op（combine / plan scan / router）の latency** だった。

## MOE_COMBINE が 1 workgroup / 行だった

`launch_moe_combine` は `<<<rows, kCombineThreads=128>>>` で、**decode（rows=1）は 1 workgroup
・128 thread が 2048 feature を 16 回ループ**していた。expert 8 行 × 2048 × 2 B = 32 KB を 1 CU で
読む形なので、latency 律速で 32 µs 級。feature を grid.y に分割した
（`grid = dim3(rows, ceil(features / 128))`、1 thread = 1 feature）。per-element の accumulate
順序（expert index 昇順）はそのままなので出力は bit 一致。

## MOE_PLAN_SCAN が thread 0 のみ

`moe_slot_scan_kernel` は `<<<1, 256>>>` で、**thread 0 だけが experts 256 ぶんの
`tile_counts` を依存ループで読み、行 layout を全部書いていた**。decode（tiles=1）でも 256 回の
global load が直列につながり 27 µs。expert 合計を全 thread で LDS へ集約する前段を足し、prefix の計算と書き込みは thread 0 のまま
にした。group table の意味（expert 昇順・`group_capacity` 超過の打ち切り）は変えていない。

## 実測（device 1。`GREEDY_TOKEN_SUM` anchor は全 run で不変）

| 条件 | 変更前 | combine だけ | combine + plan scan |
| --- | --- | --- | --- |
| `tg --context 64 --tokens 32` | 58.74 / 58.86 / 59.17 | 62.11 / 63.09 / 63.14 | **64.49 / 64.53 / 64.32** |
| `tg --context 4096 --prefill-chunk 2048` | 56.19 / 56.25 | 60.40 / 60.43 | 58.79 / 61.64 |
| anchor | `2199` / `2192` | 同一 | 同一 |

context 64 で **+9.2 %**、context 4096 で +7 %（4096 は run 揺れが 2.9 ms/token ある）。
per-op（timer on、context 64）は `MOE_COMBINE` 40.39 → 11.83 µs、`MOE_PLAN_SCAN` 27.40 →
20.21 µs。plan scan の残り（prefix 決着と書き込みは thread 0 直列）は次の手。

## 次の lever（実測に基づく優先順）

1. **dispatch 数を減らす**。1 token 1355 dispatch × 3.6 µs = 4.9 ms（29 %）。100 dispatch/token を
   消すたびに 0.36 ms/token = 2 %。小さい elementwise 系（RMS_NORM 131、RESIDUAL_ADD 80、
   L2_NORMALIZE 60、SILU 60、SIGMOID 50、MUL 40、SCALE 30、ROPE 20、SPLIT 10）は
   481 dispatch/token で、そこだけの fusion 余地は 1.7 ms/token
2. **weight の bytes を減らす**。bf16 のまま残っている backbone（GDN 2023 MB + attention 545 MB）と
   lm_head（1017 MB）を fp8 E4M3 にすると 1.79 GB/token 削れて **1.9 ms/token（11 %）**。ただし
   HF / f64 oracle との絶対照合 gate が無いので、数値の受け入れ根拠が先に必要
3. **1 token の traffic を按分する**。MTP / drafter（`docs/developer/dflash2.md`）は accepted token
   ごとに 4.41 GB を分けられるので、原理的には 44 % 以上の余地がある。MoE 側の drafter 配線は未実装
4. `MOE_PLAN_SCAN` の残り（prefix 自体は thread 0 直列のまま）と `MOE_ROUTER`（14.9 µs）

## 受け入れ

- `tests/unit/test_moe_combine.hip`: feature 幅を production の 2048 へ上げ（128 × 16 block）、
  `run_partial_feature_block_case` を追加（features=200・stride=256・rows=2）。
  全 feature column が参照と一致すること、および **feature domain の外が sentinel（0x0001）の
  まま**であることを確認する。grid.y 分割を消せば f ≥ 128 の column が参照と合わなくなり、
  kernel の `f >= features` guard を消すれば domain の外が書かれる（両方が FAIL として出る）
- combine / plan scan の出力 layout は既存の `test_moe_combine`（router → scan → scatter → combine
  の full chain）、`test_moe_program_execute`、`test_moe_layer_e2e`（f64 参照 fixture）、
  `test_moe_router`、`test_moe_grouped_linear` が覆う。いずれも PASS
- **fail-first は tamper で実演**: scan の `expert_row_begin[e] = slots` を `slots + 1u` にずらすと
  `test_moe_combine` が exit 1（plan の d2h 段階で失敗）、`test_moe_program_execute` は停止せず
  ctest の timeout で捕まる（clean なエラーではなくハングで出る。レイアウト誤りはこの経路で
  しか検出されないことに注意）。restore 後は 5 本すべて PASS
- E2E: `GREEDY_TOKEN_SUM=2199`（context 64 / 32 token）と `2192`（context 4096 / 8 token）が
  変更前後で不変。combine は per-element の accumulate 順序を変えないので bit 一致のはずで、
  greedy 列が動かないことがその確認になっている

## fusion の economics を rows=1 の単発 launch で実測した（2026-11-05 追加）

「FusedMoE にすると何 µs 儲かるのか」を推定ではなく実測で決めるため、decode の geometry
（`rows=1`、`k=2048`）で profile 単体の launch 時間を測った。device 1、`--warmup 10 --samples 20`、
`--launches 1` と `--launches 10` で**差が無い**（＝back-to-back launch 下限に張り付いている）。
`optimized` variant の単発 launch で、`correctness` variant は常に 2.5 µs 台（reference kernel）。

| profile（単発 launch µs） | f=1 | f=8 | f=64 | f=256 | f=512 | f=1024 | f=2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LINEAR_BF16` | 1.77 | 2.44 | 2.44 | 2.50 | 2.56 | 2.59 | 2.56 |
| `ADD_BIAS_BF16` | 1.77 | 1.77 | 1.77 | 1.98 | 3.44 | 3.43 | 3.44 |
| `MUL_ADD_BF16` | 0.78 | 0.78 | 0.78 | 0.78 | 1.79 | 1.92 | 2.02 |
| `RESIDUAL_WARP` | 1.79 | 1.79 | 1.80 | 1.80 | 1.80 | 1.78 | 1.80 |
| `RMS_NORM_WARP` | 1.83 | 1.80 | 1.98 | 2.40 | 2.52 | 2.52 | 2.40 |
| `RMS_RESIDUAL_WARP` | 2.41 | 2.40 | 2.52 | 3.41 | 3.48 | 3.39 | 3.41 |
| `RMS_SCALE_WARP`（k=1024） | 1.78 | 1.78 | 2.52 | 3.37 | 2.43 | 3.44 | — |
| `RMS_RESIDUAL_SCALE`（production、block reduce） | — | 3.45 | — | — | — | — | 4.20 |
| `ACTIVATION_QUANTIZE_MXFP4_ROWREF`（k=1024） | 1.78 | 1.78 | 1.78 | 1.78 | 2.52 | 2.56 | — |

帰結（この表から直接言えること）:

1. **1 dispatch の下限が支配的で、fold した work は安い**。`RESIDUAL` 単体 1.80 µs を
   `RMS_NORM` へ fold すると 2.40 → 3.41 µs（+1.01 µs）で、**0.8 µs + launch 1 本ぶん（3.6 µs）
   の浮き**がある。`MUL_ADD`（scale 平面 16 KB 書き）も単体 1.92 µs → fold で +0.97 µs。
   つまり **2 dispatch を 1 に統合する收益は 2.5〜3.5 µs**（work の増加を引いた実値）
2. `LINEAR_BF16` は f=512 でも f=2048 でも 2.56 µs で、**launch 下限に貼っている**。だから
   shared expert の gate と up（各 f=512）を 1 本へ統合（f=1024 で 2.59 µs）は
   **40 層で 2.5 µs × 40 = 0.10 ms/token = 約 2 %**（decode 4.97 ms 基準）
3. 逆に **block reduce の RMS（4.20 µs）は warp reduce（2.40 µs）より 1.8 µs 遅い**。production の
   `RMS_RESIDUAL_SCALE` / `RMS_F32_SCALE`（`norm.hip`）は block reduce + LDS scratch で、
   feature 方向を grid で分割していない。**ここは fusion ではなく kernel 側の修正**で、
   40 層 × 2 本 × 1.8 µs = 0.14 ms/token = 2.9 % の余地がある（未試作）
4. `ACTIVATION_QUANTIZE_MXFP4_ROWREF`（1.78 µs）と grouped GEMM は launch 下限に貼っているので、
   activation 量子化を GEMM へ fold しても收益は quantum 1 本ぶん（≈1.8 µs）/ 層にとどまる

### MoE FFN 1 層の dispatch 一覧（`lower_to_primitives.cpp` から確定）

`LINEAR_BF16`(router) → `MOE_ROUTER`（+ `MOE_PLAN_SCAN` / `MOE_PLAN_SCATTER`）→
`ACTIVATION_QUANTIZE_MXFP4_ROWREF` + `LINEAR_MXFP4_W4A8_GROUPED`（gate/up、**w13 融合平面を
2 view で読む = ここは既に fused**）→ `SWIGLU`（routed）→ 量子化 + grouped GEMM（down）→
`LINEAR_BF16` × 4（shared gate / up / down / scale）→ `SWIGLU`（shared）→ `SIGMOID` →
`MOE_COMBINE`（routed 合計 + shared×gate、**fused**）。1 層 16 dispatch、40 層で **640 dispatch/token**
（decode 全体 1355 dispatch/token の 47 %）。

fused になっていない所は 3 つ: ① activation 量子化が GEMM と別 kernel ② `SWIGLU` が GEMM の
epilogue ではない ③ shared expert の gate / up が 2 本の `LINEAR_BF16`（dense 側は w13 平面で
統合済みなので、**shared expert だけが未統合**）。vLLM の `fused_moe_kernel` は ①②を持たず
（`moe_kernel_quantize_input` と `apply_moe_activation` が別 kernel）、gather と
`MUL_ROUTED_WEIGHT` を epilogue でやっている。**SHELF の①②は vLLM と同じ形**で、削れるのは
③と router bias（`MOE_ROUTER` へ吸収可、約 1 µs/層）と plan の scan/scatter 統合になる。
