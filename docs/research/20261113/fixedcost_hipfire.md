# PLaMo c=1 の producer 側融合

対象は R9700 / gfx1201、ROCm 7.14、PLaMo 3 Translate 2B EXL3-INT8 4.05 bpw、BF16 KV、非投機 decode。比較元は `ca29aa1`。`/tmp/r9700.lock` と GPU UUID `GPU-a8e9ddefa2d60f55` を使った。

## 実 step の内訳

`plamo_linear_timing.patch` で 8 回目の decode-only step を一回捕捉した。表の回数は外側の semantic dispatch 数であり、linear の launch 数ではない。HIP event の範囲には host による launch 間隔と event の影響があり、分類の合計は step wall time にならない。

| 種類 | 回数 | event 合計 ms | 備考 |
| --- | ---: | ---: | --- |
| EXL3 linear | 97 | 4.480 | 169 本を group 化 |
| RMSNorm | 145 | 1.907 | Q/K の head norm を含む |
| RoPE | 48 | 0.452 | Q/K |
| 残差 | 48 | 0.424 | |
| paged attention | 24 | 0.364 | |
| KV append | 24 | 0.229 | |
| SwiGLU | 24 | 0.217 | |
| sampling | 1 | 0.015 | |
| embedding / scale / gather | 各 1 | 0.030 | 3 種の合計 |

linear 内部の `group_rotate` / `group_out` の単独 event は今回の実 step では分離していない。既存の [shape 別計測](../20261112/plamo3_exl3_linear.md) は各約 10 µs だが、event と launch の影響を含む。また [第2弾](../20261112/plamo3_exl3_linear2.md) の c=1 経路は入力または出力を既に GEMM に融合する形状を含み、169 本分をそのまま削減可能な時間とすることはできない。同資料の c=1 物理 linear launch は 230 回。

## 実装と見積もり

- `f924502`: SwiGLU が BF16 の product と、down 用の H128 回転・INT8 code と scale を一度に出力する。24 箇所でそれぞれ 1 launch を削減する。既存の中間 BF16 丸めを保つ。
- `521caa2`: 入力 RMSNorm と後続の projection ごとの SUH、H128、INT8 量子化を融合する。QKV 24、gate/up 24、最終 norm→head 1 の計 49 箇所で選択された。各 128 要素 block が norm の sum を再計算して並列性を確保し、既存 BF16 出力も生成する。4 bit QKV 等の従来経路は回転を GEMM 内に融合済みのため、49 箇所すべてで launch が減るわけではない。
- 対象は PLaMo の 1 行 decode に限定。prefill、投機、値トレース、layer override、操作単位の診断時は従来経路。`SHELF_PLAMO_EXL3_PRODUCER_ROTATE=0`、`SHELF_PLAMO_EXL3_NORM_ROTATE=0` で個別に戻せる。

候補側の同じ診断 capture では RMSNorm と後続 projection を合わせた event 合計が 4.169 ms、SwiGLU と down の合計が 1.126 ms、その他の linear が 0.663 ms。対応する baseline の RMSNorm + SwiGLU + linear は 6.604 ms、候補は 5.959 ms。差の約 0.65 ms は同じ区間の診断上の概算であり、clean throughput や個別案の寄与ではない。2 案の相互作用があるため単独寄与は未測定。

## 暫定速度・品質と検証

継続投入の初回は baseline 155.28、候補 159.08 tok/s。残りの ABAB と品質全例は共有 GPU ロックの待機中であり、これは採用判定ではない。品質は CPU BF16 比較の各方向 18 文・875 位置。KL はどちらも 0.02 未満、既存 main 比の top-1 低下は 1 ポイント未満。

| 方向 | KL | top-1 一致率 | 既存 main top-1 | 最大 logits 差 |
| --- | ---: | ---: | ---: | ---: |
| ja-en | 0.0183034 | 92.9143% | 92.9143% | 5.500000 |
| en-ja | 0.0151106 | 93.6000% | 93.7143% | 6.066406 |

`test_plamo_exl3_projection` と、新規 `test_fixedcost_hipfire_producer` が通過した。後者は通常入力とゼロ入力で、融合 producer の BF16 出力と従来量子化経路の code / scale の整合、RMSNorm 値の誤差を検査する。`--all` は実行していない。反復速度の確定までは速度に基づく採否は未確定である。

## 他の候補

- attention reduce→O 回転: attention 出力は 24 箇所。別担当の attention 融合との競合を避けて未実装。
- 残差加算→次の RMSNorm: 48 回の残差と全幅 norm の境界が候補。残差値は次の layer でも必要であり、出力の保存と norm 用入力の別名関係を確認する必要がある。既存の一般的な残差＋RMSNorm 融合は探索範囲では見つからなかった。
- final norm→head 入力回転は上の 49 箇所に含む。head→argmax は logits の外部出力と sampling の契約も考慮する必要があるため未実装。既存 sampling は分割 argmax を備える。
- hipfire で却下された FWHT 分割、gate/up と SwiGLU の GEMM 融合、QK/RoPE/KV の融合は再実装していない。head の wave64 は hipfire の別 checkpoint で微小な改善にとどまった。これらは shelf での評価結果ではない。

## 出典

設計の参考: `warpfront/hipfire` commit `89e32856d25d6f6e6645319f2ec2c641c46be525`、Apache-2.0。`kernels/src/fused_rmsnorm_mq_rotate.hip`、`fused_silu_mul_mq_rotate.hip`、`ffn_grouped_activation_rotate_i8.gfx1201.hip`、`attention_flash_q8_0_reduce_gated_mq_rotate.gfx1100.hip`、`fused_rmsnorm_proj4_mq4g256v2_fold.gfx1201.hip`、`argmax_pair_merge.hip` と `docs/perf-checkpoints/` を参照。shelf の H128、重み固有 SUH、INT8 量子化と実行計画に合わせて独自に実装した。

計測集計と条件は [JSON](fixedcost_hipfire.json)、再実行用 runner は [速度](fixedcost_hipfire_runs.py) と [品質](fixedcost_hipfire_quality.sh)。
