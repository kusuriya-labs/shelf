# qsa_select の WMMA 化: score / top-k 分離

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、`--kv-cache-dtype fp8`。

## 結論

`qsa_select` の dot を WMMA に置き換えた。dot と top-k を同じ kernel に置くと
WMMA の staging と top-k の occupancy が干渉して遅くなったため、次の 2 kernel に分けた。

- score kernel: 16 query row × block tile を WMMA で計算し、bounded global score buffer に書く。
- top-k kernel: 1 row/CTA で score buffer を読み、candidate sort + bitonic merge で top-k を作る。

32K prefill は scalar の 13.93 s / 2,352 tok/s から **13.19 s / 2,484 tok/s** になった。
qsa_select は 2,434 ms（両 rank）から **1,207 ms** に減った。

## 不採用にした案

WMMA と top-k を同じ kernel に入れる 2 案は scalar より遅かった。

| 方式 | 32K prefill | qsa_select（1 rank、trace） |
|---|---:|---:|
| scalar | 13.93 s / 2,352 tok/s | 1.21 s |
| 4 row/CTA WMMA | 14.87 s / 2,203 tok/s | 2.15 s |
| 1 row/CTA WMMA | 15.14 s / 2,164 tok/s | 2.19 s |

`qsa_select` の dot は GEMV で、scalar 版は 4 warp で 128 dimension を分担して
global から直接読む。WMMA 版は K tile の shared staging、fragment load、block barrier が
増える。4 row/CTA では M16 を埋められるが、4 row 分の top-k state で occupancy が落ちた。

## 採用した構成

- score buffer: `kQsaScoreRowTile=2048` × `kQsaScoreBlockTile=8192` × f32 = 64 MiB。
- top-k state: 2048 × 2 × `kQsaSelectMaxTopK=1024` × (f32 + i32) = 32 MiB。
- 合計 96 MiB を executor workspace に追加し、`program_needs_qsa_workspace` がある
  program だけ確保する。Qwen4Exp の auto reserve にも同じ量を加える。
- score kernel は request ごとに launch する。同一 request の 16 row は同じ slot を共有する。
  grid は (row group, block tile) の 2 次元。
- top-k kernel は block tile ごとに 1 回 launch し、state を global に持ち越す。
  32K は 8192 block で 1 tile、262K は 8 tile になる。

## 計測

条件: `shelf-bench pp --prompt-tokens 32768 --prefill-chunk 2048 --kv-cache-dtype fp8
--device 2,3 --tensor-parallel 2 --warmup 0 --runs 1`、`SHELF_EXL3_MUL24_PREFILL_AB=B`。

| 状態 | 32K prefill | qsa_select（両 rank） |
|---|---:|---:|
| scalar | 13.93 s / 2,352 tok/s | 2,434 ms |
| score / top-k 分離 | **13.19 s / 2,484 tok/s** | **1,207 ms** |

内訳（両 rank、trace）: score 734 ms、top-k 481 ms。
chunk 末尾 32K の 12 layer 合計（1 rank）は score 43.1 ms + top-k 27.9 ms。

8K prompt は 3.46 s / 2,368 tok/s。

## 検証

- `test_qsa_select`: PASS。scalar と同じ参照に対して、split 経路でも top-k、tail bit、
  同点時の小さい index 優先を確認する。
- `test_paged_attention`: 56 PASS。
- `test_qsa_attention_mask`: PASS。
- QSA / paged attention 系 ctest: 7/7 PASS。

## 残る課題

- score kernel は 137 GFLOP を 367 ms/rank で処理しており、tensor core の利用率はまだ低い。
  K tile staging と block barrier が次の候補である。
- top-k kernel は 240 ms/rank。32K では block tile あたりの score 読み出しと merge が残る。
- 現在の 32K prefill の最大要素は MoE の grouped EXL3 GEMM である。
