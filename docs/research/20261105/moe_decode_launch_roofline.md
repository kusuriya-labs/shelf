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
