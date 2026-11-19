# Prefill chunk 長の速度と品質

実験 ID: `PREFILL-CHUNK-1`

`models/Qwen3.8-27B-Quark-AWQ-MXFP4`、1 カード（device 1）、bf16 KV。
`--prefill-chunk 4096` は 2048 より速いが、multi-chunk prefill の品質が
single-chunk と一致しない。既定値の変更は保留する。

## 1. 速度

`--prompt-tokens 8192`、chunk 2048 / 4096 を交互に 5 回（各 1 run、warmup 1）。

| 回 | chunk 2048 | chunk 4096 |
| ---: | ---: | ---: |
| 1 | 3,887.1 ms | 3,740.8 ms |
| 2 | 3,904.3 ms | 3,754.8 ms |
| 3 | 3,915.1 ms | 3,764.9 ms |
| 4 | 3,922.6 ms | 3,772.7 ms |
| 5 | 3,927.6 ms | 4,511.7 ms（外れ値） |

外れ値を除いた p50 は 3,915.1 → 3,764.9 ms で **3.8% 短縮**。
全 5 回の p50 でも 3,915.1 → 3,772.7 ms で 3.6% 短縮。
外れ値は同一 GPU の連続実行による clock 低下と見られ、再測定が必要。

## 2. 品質ゲート

`shelf-bench quality`、corpus は prose 16,384 token、context 8,192、
scored positions 512、bf16 KV。

| 比較 | KL mean | top-1 一致 | logit max Δ |
| --- | ---: | ---: | ---: |
| chunk 2048 → 4096 | **0.6429** | **89.45%** | 24.08 |
| 同上、`SHELF_GDN_RECURRENCE_EXACT=1` | 0.6107 | 88.09% | 22.23 |

閾値（KL 0.01 以下、top-1 99% 以上を目安）を大きく外れる。**4096 は採用できない。**

## 3. 切り分け

| 条件 | chunk 比較 | KL | top-1 |
| --- | --- | ---: | ---: |
| context 2,048 | 2,048 vs 4,096（どちらも 1 chunk、2,048 rows） | 0 | 100% |
| context 2,048 | 2×1,024 vs 1×2,048 | **2.398** | **67.97%** |
| context 4,096 | 2×2,048 vs 1×4,096 | 0.515 | 90.63% |
| context 8,192 | 4×2,048 vs 2×4,096 | 0.643 | 89.45% |

- context 2,048 で chunk 4,096 を使っても、実際の row 数は 2,048 なので
  1 chunk のままで完全一致する。**chunk 設定そのものは数値を変えない。**
- 同じ context 2,048 を 1,024×2 に分割しただけで KL 2.4、top-1 68% になる。
  **最初の chunk 境界で既に壊れている。**
- `SHELF_GDN_RECURRENCE_EXACT=1` でも改善しない。lossy 分解は原因ではない。

## 4. 原因の仮説

single-chunk では GDN recurrence の状態が f32 accumulator のまま次の token に
渡るが、multi-chunk では chunk 境界で **fp16 の recurrent state** に丸めて
保存・再ロードする。state は `GdnStatePool` の `fp16_t` で、
`gdn_recurrence` は境界で fp16 に落とす。

- conv state は bf16 だが、入力 activation 自体が bf16 なので丸め誤差はない。
- recurrent state は f32 → fp16 → f32 の round-trip になり、境界ごとに
  仮数 10 bit へ落ちる。36 層の GDN でこれが増幅される。
- 512 token のような短い chunk を多数回通す serving では、この誤差が
  さらに蓄積する。Flash-Next の長文で観測された差とも整合する。

## 5. 次の実験

recurrent state を f32 にしたとき、2×2,048 が 1×4,096 とどこまで一致するかを
品質ゲートで測る。

- state サイズ: 36 layer × 48 v_head × 128 × 128 × 4 B = 113 MB/request
  （fp16 の 56.6 MB から倍）。4 request でも 452 MB で、arena 31 GB に収まる。
- 併せて conv state の dtype と、decode / verify / DFlash2 の state copy
  （`recurrent_state_bytes`）の整合を確認する。
- f32 化で品質が通れば chunk 4,096 を既定候補にできる。通らなければ
  chunk 境界の別要因（KV page / RoPE position / 層別 state）を event で追う。

## 6. 決定

- `--prefill-chunk` の既定 2,048 は変更しない。
- ② GDN chunk-parallel は、f32 state の保持と chunk 境界の再現性を
  前提条件として進める。
