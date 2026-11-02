# Kernel accuracy audit

- 実施日: 2026-11-03
- 対象 build: `bb2e56e` + 本監査の harness 変更、Release / gfx1201
- GPU: Radeon AI PRO R9700、device 1
- 目的: optimized kernel と production correctness 実装の差分を kernel 単位で把握する

## 方法

`shelf-bench` の `--check` を使い、optimized path の出力を production correctness path と比較した。
GEMM は Qwen3.8-27B の実 shape（`--actual-qwen-shapes`）、それ以外は実際の Qwen profile または
既定 shape で確認した。判定は既存の tolerance に従う。

## 結果

| kernel | 最悪 max relative error | mismatch 率 | 判定 |
| --- | ---: | ---: | --- |
| gemm bf16 | 0.7812% (`mlp_down`, rows=128) | 約 0.11% | OK |
| gemm fp8 | 0.7812% (`mlp_gate_up` / `attn_q`, rows=128) | 約 0.11% | OK |
| gdn-recurrence（lossy） | 0.801% (rows=2048, nr=4) | mismatches 0 | OK |
| activation-quantize | 0% | 0% | 完全一致 |
| rmsnorm | tolerance 内 | - | PASS |
| l2-normalize | 4.5e-08 | - | PASS |
| kv-append bf16 / fp8 | tolerance 内 | - | PASS |
| paged-attention decode | tolerance 内 | - | PASS |
| rope | 0% | 0% | 完全一致 |
| gdn-conv1d | 0% | 0% | 完全一致 |

## GEMM の誤差

BF16 / FP8 とも、rows=1 では 1 ULP 以下、rows>=16 では最大 2 ULP に相当する。
どちらも入力を FP32 で accumulate し、出力時に BF16 へ 1 回丸める実装である。
したがって差分の主因は accumulation order の違いであり、出力 BF16 の丸めで増幅される。

BF16 で `--impl wmma|wide|splitk|exact` を比較したが、rows=64 の max relative error は
wmma / wide が 0.6623%、splitk / exact が 0.7576% で、exact 系の方が小さくならない。
accumulation order を correctness と一致させない限り 2 ULP を下回らない。

FP8 も同じ傾向で、activation / weight の量子化 code は optimized と correctness で共通である。
したがって kernel 側の精度改善余地は小さく、品質を上げるなら activation / weight の
量子化方式（group size、scale 方式）を変える領域になる。

## GDN recurrence の誤差

production 既定は lossy WMMA decomposition（1 回の WMMA）である。
`SHELF_GDN_RECURRENCE_EXACT=1` で high+low の 4 項 decomposition に切り替えた場合を比較した。

| 条件 | lossy | exact |
| --- | ---: | ---: |
| max relative error（全体最悪） | 0.801% | 0.663% |
| mean absolute error（rows=2048, nr=16） | 1.56e-05 | 9.59e-06 |
| rows=1 p50 | 21.47 us | 21.04 us |
| rows=64, nr=1 p50 | 35.63 us | 36.30 us |
| rows=2048, nr=1 p50 | 847.76 us | 900.82 us |
| rows=2048, nr=16 p50 | 654.17 us | 736.73 us |

E2E（TP2 / FP8 KV / prompt 1024 / 32 token）では decode は
24.47 tok/s（lossy）と 24.46 tok/s（exact）で差がなかった。
prefill も 297.9 ms と 299.1 ms で差は 0.4% であった。

このため、GDN recurrence は exact を既定にしても decode の性能影響は観測されない。
一方で、exact にすると `WmmaVerify` の lossy 専用経路が無効になるため、
spec verify の rows=2/4/8 と DFlash2 の acceptance rate は別途確認が必要である。

## 判断

- GEMM bf16 / fp8: 現在の差は BF16 出力の 2 ULP で、kernel 変更による改善は見込めない。
  check の tolerance は現行の 2e-2 を維持し、ULP を併記するのが妥当。
- GDN recurrence: exact を既定候補とする。採用前に spec verify / DFlash2 の
  acceptance と E2E を測る。lossy へ戻す逃げ道は
  `SHELF_GDN_RECURRENCE_LOSSY=1` として残す。
- FP8 paged prefill の `--check` は、correctness 参照が P 量子化を模擬しないため
  per-element 比較では FAIL する。prefill の合否は row-L2 で判定する必要がある。

## harness 変更

- `shelf-bench gemm --dtype fp8 --check` を実装した。activation quantize と
  linear の correctness dispatch を同一 workspace で順に実行して比較する。
- FP8 weight code の乱数生成で NaN code（`0x7F` / `0xFF`）を除外した。
  除外前は全出力が NaN になり check が成立しなかった。
