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

## 4. 原因の確定と修正

原因は **GDN recurrent state の fp16 丸め**だった。single-chunk では状態が
f32 accumulator のまま次 token に渡るが、multi-chunk では境界で fp16 の
state pool に丸めて保存・再ロードする。36 層分の丸めが context とともに
増幅し、最初の境界で既に KL 2.4 になっていた。

`GdnStatePool` と recurrence kernel の state を **f32** に変更した。
`GdnStateScalar = float` を導入し、pool / kernel / spec history / prefix cache /
footprint の state サイズと load/store を一貫させた。

- state サイズ: 36 layer × 48 v_head × 128 × 128 × 4 B = **113 MB/request**
  （fp16 の 56.6 MB から倍）。4 request でも 452 MB。
- conv state は bf16 のまま（入力 activation が bf16 なので丸め誤差なし）。
- history capture は異なる launch 形状の reference と比べるため、テストは
  bit 一致から相対 1e-3 の比較に変更した（B1）。

## 5. f32 state の結果

| 条件 | chunk 比較 | KL | top-1 |
| --- | --- | ---: | ---: |
| context 2,048 | 2×1,024 vs 1×2,048 | **0** | **100%** |
| context 4,096 | 2×2,048 vs 1×4,096 | **0** | **100%** |
| context 8,192 | 4×2,048 vs 2×4,096 | **0** | **100%** |

multi-chunk と single-chunk が bit 一致した。品質ゲートを通過する。

## 6. ① の採否

- 速度: chunk 2,048 3,946 ms → 4,096 3,792 ms（**3.9% 短縮**、f32 state）。
- 品質: KL 0 / top-1 100%。**① は通過**。
- decode への影響: 32.41 tok/s（f32）vs 32.82 tok/s（fp16）で **約 1.25% 低下**
  （state traffic が 56.6 → 113 MB/token）。prefill の改善と引き換えに許容する。
- `--prefill-chunk 4096` を推奨設定とする。既定値の変更は serving の
  メモリ / TTFT 判断として別途行う。

## 7. 次の段

② GDN chunk-parallel は、f32 state を前提に recurrence の row 方向の
並列性を上げる。現在は head 方向（48）だけが並列で、row は serial。
conv / recurrence の event 内訳を取り、chunk 4,096 での上限を確認してから
scan の並列化方式を決める。
