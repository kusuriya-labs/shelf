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

## 追加調査: 冷却・常駐条件・CTA の担当数

I9 を保持したまま無効時以下の replay 時間を得る目標は、今回も達成できなかった。案 C は既存 I9 を改善したが、実モデルの単発 graph replay では I9 無効に届かないため、production には採用しなかった。以下の試作と生ログは `/tmp/ln_i9/` に置いた。前回の `.i9-diagnostic/` も同所に移した。

### 冷却を挟む単体計測

8 行、hidden 5,120、split-K 2 / 12、後続 2 / 3 projection。既存 I9 と、出力回転に続く既存 I8 相当を HIP event で測り、各条件 40 回の中央値を採った。冷却は各測定直前に 128 MiB の別バッファを GPU で読み出した。実 weight を読んだものではない。入力も合成値である。単体の出力回転は既存 kernel と同じ split 加算と回転を持つ試作用 kernel である。実モデルの 1 projection は直前 GEMM、下表の projection は後続 group の数であり、両者は別の意味である。

| split-K | 後続 projection | 冷却 | I9 µs | 出力回転 + I8 µs |
|---:|---:|:---:|---:|---:|
| 2 | 2 | あり | 20.90 | 20.68 |
| 2 | 3 | あり | 25.28 | 23.32 |
| 12 | 2 | あり | 34.90 | 23.30 |
| 12 | 3 | あり | 44.16 | 25.90 |

split-K 12 / 3 projection の 44.16 / 25.90 µs は前回の実モデル 41.82 / 27.76 µs と方向・大きさが近い。しかし split-K 2 は実モデルで I9 が速いという関係を再現していない。冷却なしの split-K 12 / 3 projection は 39.24 / 23.24 µs。したがって重複読み込み量の寄与と整合するが、帯域や cache miss を測った証明ではなく、全形状の実モデル時間を再現したとも言えない。

### 案 A の常駐上限

指定の 256 thread block、全 grid 同時常駐という安全条件を、`hipOccupancyMaxActiveBlocksPerMultiprocessor` で調べた。R9700 は `multiProcessorCount=32`、LDS を使わない小 kernel の上限が 8 block / multiprocessor、合計 256 CTA だった。8 行の I9 は 320 または 640 CTA なので、資源使用量が最小の kernel ですら指定条件を満たせない。10 / 16 行も同様である。実際の案 A kernel の occupancy は未測定だが、上限を超える対象では実装しても安全条件により fallback するため、barrier kernel の実行実験は行わなかった。カウンタの初期化・再利用や memory ordering も未検証である。

### 案 C の試作

各 CTA の二乗和計算を維持し、正規化・回転の担当 group を 1 個から 2 個へ増やして CTA 数を半減する試作を、新規 kernel として build した。同一 binary で `SHELF_DISABLE_BOUNDARY_I9_SHARED=1` により既存 I9 に戻す切替を一時的に設けた。これは共有 barrier 案ではない。合成入力の 1 / 8 / 10 / 16 行、split-K 2 / 12、2 / 3 projection、residual 加算の両順序の 32 条件で、旧 I9 に対して residual・norm・codes・scales は bit 一致した。乱数入力を使用したが網羅的な品質検証ではない。

前回と同じ HIP event 挿入による実モデル 8 行 stream の各 1 process の中央値（µs）。分類は直前 GEMM の split-K と I9 の projection 対数。

| split-K | projection 対 | 件数 | 旧 I9 | 案 C | 出力回転 + I8 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 63 | 19.00 | 17.12 | 23.96 |
| 12 | 1 | 46 | 29.98 | 24.10 | 24.68 |
| 12 | 2 | 16 | 41.16 | 31.42 | 27.68 |

同一 binary の 256 token graph replay、最初を除く 43 回の平均、条件ごとに 1 process: 旧 I9 34.4618 ms、案 C 34.2272 ms、I9 無効 33.7933 ms。生成速度は順に 124.94 / 125.40 / 126.66 tok/s。この単発測定は ABAB 受入ではないが、案 C は I9 無効より 0.434 ms 長かった。group を 4 個担当する別試作の冷却単体値は split-K 12 / 2 projection 32.00 µs、3 projection 32.04 µs（既存 I9 34.96 / 44.52 µs、未融合 23.08 / 25.80 µs）。2 group 案の同条件 29.80 / 36.80 µs に対し、一方は悪化し他方も未融合を超えた。4 group 案の bit 一致と実モデルは測っていない。

案 B は実装・計測していない。直前 GEMM 内の split reduction を launch を増やさずに完了する設計や、その丸め順と workspace の成立は未検証。案 C の production 変更は取り消したため、新しい env 切替も残していない。最終受入の ABAB、logits、3 prompt、DFlash2 / tree、ctest、launch 数、非投機速度は実施しておらず、残りの目標の達否は未確認。前回の品質値を今回の新しい受入結果として扱わない。
