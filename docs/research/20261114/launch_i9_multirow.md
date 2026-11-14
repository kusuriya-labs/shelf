# I9 の複数行 verify 調査

## 結果

I9 を保持したまま目標を満たす変更は得られなかった。production code は変更していない。実モデルの stream 計測では、直前の GEMM が split-K 12 の I9 が長く、特に 3 projection の場合に未融合の出力回転と I8 の合計を上回った。I9 の後続 GEMM の中央値には目立つ変化がなかった。これは重複した出力回転と読み込みの影響を支持するが、L2 ヒット率や stall を測定しておらず、原因を確定したものではない。

対象は 4bc0e412 相当の `ln-integ/build-gfx1201`、R9700 gfx1201、Qwen3.8-27B EXL3 mul24、BF16 drafter / KV、code prompt、drafts 7、chain、temperature 0。指定 GPU、ライブラリ、`flock /tmp/r9700.lock`、`timeout -k 5 280` を使用し、GPU process ごとに lock 内で load average < 8 を確認した。rocprofv3 は使用していない。build と kernel の変更は行っていない。

## 実モデルの kernel 切り分け

`SHELF_GRAPH_VERIFY=0` で 32 token を生成し、最初の 8 行 forward の各 `hipLaunchKernel` 前後に HIP event を挿入した。I9 有効と `SHELF_DISABLE_BOUNDARY_I9=1` は別 process 各 1 回。表は kernel の分類ごとの中央値、単位は µs。無効側は出力回転と I8 の連続 2 kernel の event 間隔の和の中央値。直前 GEMM の grid.z は split-K 数、次の GEMM の grid.x を分類に用いた。

| 直前 split-K | I9 grid.z | 件数 | I9 有効 | 出力回転 + I8 | 次の GEMM 有効 / 無効 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 63 | 19.48 | 23.88 | 183.04 / 182.88 |
| 12 | 1 | 46 | 29.88 | 24.90 | 87.26 / 87.24 |
| 12 | 2 | 16 | 41.82 | 27.76 | 83.98 / 84.12 |

I9 の grid は `(8, 40, ceil(projection数/2))`、block は 256 thread。各 CTA は 5,120 要素について source の各 split を読み、出力回転・residual・二乗和を再計算する。split-K 12 では yprime の論理読み出し量は 320 CTA で約 78.6 MB、640 CTA で約 157.3 MB。これは CTA による要求量であり、実際の DRAM 転送量ではない。cache hit は未測定。3 projection では 2 projection 対を別 CTA とするため二重の全行走査がある。

同 build の code object metadata では I9 の静的 LDS 11,328 bytes、VGPR 35、SGPR 40、private segment 0、wave32、spill なし。実 occupancy は未測定。比較対象の I8 は同じ LDS、VGPR 18、SGPR 26。イベントの作成・挿入や host の投入待ちが区間を長くする可能性があり、初回の外れ値も観測された。中央値は分布を示す診断値であり、graph replay からこの差を直接差し引くことはできない。L2 を冷却した合成実験や実 weight 読み込みを挟む実験は未実施。

## graph replay と既存案

各条件 1 process、256 token。graph launch 前後の HIP event による 8 行 graph の初回を除く 43 replay の平均。tok/s は同じ process の生成結果。ABAB の反復ではなく、採用判定用の平均 ± SD ではない。全条件の code prompt 生成 ID は同一。

| 条件 | kernel node | replay ms | tok/s |
|---|---:|---:|---:|
| 現行 | 1,001 | 34.5025 | 124.66 |
| I9 無効 | 1,126 | 33.8254 | 126.50 |
| I8 無効（I9 も無効） | 1,251 | 34.2965 | 124.24 |
| I1 無効（I8 / I9 も無効） | 1,376 | 34.2572 | 124.37 |
| 既存 `SHELF_BOUNDARY_I9_SERIAL=1` | 1,001 | 41.6179 | 108.05 |

serial は行ごとに 1 CTA で重複を避ける既存 kernel の試験であり、新実装ではない。現行より約 7.12 ms 遅く、採用しない。graph の追加 descriptor kernel 1 個を除く forward launch は現行 1,000。各無効化は階層的に他の融合も無効にするので、個々の効果として差を加算できない。前回の 5 ペア計測値と計測手法は `shelf-launch-reduction/dflash2_why.md` にある。

## 受入と未解決

改善候補が得られなかったため、最終品質・速度受入の ABAB 5 ペア、baseline / 修正前 / 修正後の 3 条件、logits・3 prompt・DFlash2 99/99・tree・関連 ctest は今回実行していない。上表の 1 prompt 一致は全品質基準の達成を意味しない。非投機速度、8 / 10 行の全 launch 数、および目標とする baseline 比 +1% の再受入も未実施。I9 有効の replay は同一調査内の無効条件より 0.677 ms 遅く、目標 (1) を満たさない。他の目標は今回のデータでは判定できない。

追加 launch と grid-wide sync を使わず、異なる CTA 間で bit 一致を保って norm と出力回転を共有する方法は未実装。split-K 12 の重複読み込み量、L2 hit、占有率、graph 内の依存と間隔の寄与を分ける追加計測が必要。行数やモデルによる新たな分岐、I9 無効化への切替は行っていない。
