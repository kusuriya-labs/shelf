# Gate 6 long-context measurement

- 実施日: 2026-11-03
- 対象: Qwen3.8-27B-FP8、TP2、FP8 KV、`--rope-factor 2`
- GPU: Radeon AI PRO R9700 × 2（device 0,1）
- build: `bb2e56e` + 計測用 harness 変更

## 262,144 トークン（production selector）

`shelf-bench pp`、`--prompt-tokens 262128`（要求 seq len 262144）、
`--prefill-chunk 2048`、warmup 1 + 5 run。

| 指標 | 値 |
| --- | ---: |
| gpu_ms p50 | 272468.22 ms |
| gpu_ms min / max | 272112.75 / 272787.81 ms |
| gpu_tokens_per_sec p50 | 962.05 |
| run 間の p90/p10 | 272688.71 / 272200.69 ms |

production selector の paged-attention 上限は 262144 であった（現行の上限は下記の
採用で 524288）。この計測は当時の上限内で完結する。

## 524,288 トークン

### production 経路

`shelf-bench pp --prompt-tokens 524272`（要求 seq len 524288）を実行した。
`max_visible` が 262144 を超えた時点で paged-attention が correctness へフォールバックし、
42 分経過しても完了しなかったため中断した。

```
paged-attention: outside measured envelope (rows=2048 max_visible=264192)
falling back to correctness path
```

Gate 6 の production 計測は、この selector envelope が blocker である。
（後述のとおり envelope を 524,288 へ拡大して解消した。）

### selector を 524,288 へ広げた計測

`kMeasuredMaxVisible24` を 524288 にして同じ条件を実行した。

| run | 指標 | 値 |
| --- | --- | ---: |
| 1 run | gpu_ms | 952161.75 ms |
| 1 run | gpu_tokens_per_sec | 550.61 |
| 1 run | kv_reserved_bytes | 14616043520 |

#### 採否判断

`kMeasuredMaxVisible24 = 524288` を production selector の envelope として採用した。
根拠は上記 1 run と、後述の `visible=524288` で optimized decode 経路が correctness と
一致した確認である。

アンカー付きの production 値（warmup 1 + 5 run の p50）は未取得のままなので、
`docs/perf/current.md` には 524,288 の速度を載せない。上記の 1 run 値のみが実測値である。
production 値の取得は未完了として残す。

採用に伴い `tests/unit/test_qwen35_paged_attention_selector` の expected を
524288（measured max）/ 524289（above measured）基準へ更新し、`24 x 4` の
rule も境界検証に加えた。

## 追加で確認したこと

- optimized paged-attention の decode 経路は `visible=524288` で correctness と PASS。
- FP8 paged prefill の `--check` は、correctness 参照が P 量子化を模擬しないため
  per-element 比較では FAIL する。prefill の長 visible 検証には row-L2 の判定が必要。
- `shelf-bench pp` の TP working-set reserve は 1 GiB 固定で、executor 実測 1.84 GiB に
  足りなかった。auto 計算へ変更して計測を成立させた。
- `shelf-compute` の single-shot は prefill を 1 step しか実行せず、残りを decode 側で
  処理していた。全 prefill chunk を完了してから decode 計測を開始するよう修正した。
