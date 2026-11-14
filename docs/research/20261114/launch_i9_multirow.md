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

## 追加調査: split-K に連動した選択適用

既存の launch 計画の直前 GEMM の split-K が 12 の場合に限って I9 を使わず、出力回転と I8 へ戻す。split-K 2 と、今回現れなかった他の split-K は従来どおり I9 を使う。行数やモデル名による新しい分岐は設けず、`SHELF_DISABLE_BOUNDARY_I9` は引き続き全無効にできる。案 C は後述の replay 比較で採用しなかった。

対象 GPU、model、BF16 drafter / KV、code prompt、draft 7、chain、256 token、最大長 1024、temperature 0 は前節と同じ。GPU process は指定 GPU、ライブラリ、`flock /tmp/r9700.lock` と `timeout -k 5 280` を用い、lock 内で load average < 8 を確認した。各 run は別 process、交互に 5 ペア。± は run 間の標本 SD。測定中の区分切替、案 C、形状出力は一時実装で、commit には含めない。

実行時に I9 の対象として解決された形状は、1 / 8 / 10 / 16 行の各 forward で同じだった。split-K 2・後続 2 projection は 63 か所、split-K 12・後続 2 projection は 46 か所、split-K 12・後続 3 projection は 16 か所。後続 projection の出力幅は順に `(17408, 17408)`、`(10240, 6144)`、`(12288, 1024, 1024)`。ただし split-K は launch 計画の結果であり、一般に行数で変わり得る。

### 8 行の区分別 replay

`graph_events.cpp` の event 間隔を使い、初回を除く 43 replay を各 run で平均した。以下は各区分だけを無効化した値で、差は全有効 minus 区分無効。負なら無効化で遅くなる。対応する all-on run は直前に交互実行した別 process であり、表の全有効値は行ごとに異なる。

| 無効化した区分 | 全有効 ms | 区分無効 ms | 差 µs | kernel node 有効→無効 |
|---|---:|---:|---:|---:|
| split-K 2 | 34.8787 ± 0.2592 | 34.7882 ± 0.2386 | +90.6 ± 50.2 | 1001→1064 |
| split-K 12・後続 2 projection | 34.9004 ± 0.1653 | 34.3249 ± 0.0233 | +575.6 ± 149.3 | 1001→1047 |
| split-K 12・後続 3 projection | 34.9426 ± 0.1467 | 34.4702 ± 0.0695 | +472.4 ± 91.8 | 1001→1017 |
| split-K 12 全体 | 34.9566 ± 0.0974 | 34.2211 ± 0.0789 | +735.5 ± 21.5 | 1001→1063 |
| 全 I9 | 34.9478 ± 0.1542 | 34.1102 ± 0.0984 | +837.6 ± 63.1 | 1001→1126 |

split-K 12 の個別区分差の和は約 1048 µs で、同時無効の 736 µs と一致しない。run 間変動と区分間の相互作用を含むため、これらの観測値の加算性は確認できない。先行する stream kernel の中央値からの単純見積もりと graph の差には大きな開きがあるが、event 挿入、cache、依存・間隔を含む異なる測定を直接差し引いて kernel 外の寄与と断定することはできない。約 0.5 ms の内訳は未解決。split-K 12 全体の差を増加した 62 node で割ると約 11.9 µs/node だが、kernel の計算変更を含む正味差であり launch 単価ではない。

### 他の行数と案 C

1 行 decode も同じ一時区分切替を 5 ペアで比較した。graph event を使用したため下記の生成速度はフック付きであり、フックなしの最終受入とは分ける。差は replay が全有効 minus 無効、tok/s が無効 minus 全有効。

| 無効化した区分 | replay 差 µs | 生成速度差 tok/s |
|---|---:|---:|
| split-K 2 | −76.1 ± 71.0 | −0.106 ± 0.078 |
| split-K 12・後続 2 projection | +251.2 ± 28.8 | +0.232 ± 0.029 |
| split-K 12・後続 3 projection | +75.6 ± 26.0 | +0.070 ± 0.028 |
| split-K 12 全体 | +325.8 ± 28.1 | +0.294 ± 0.034 |
| 全 I9 | +325.1 ± 41.3 | +0.328 ± 0.040 |

split-K 2 の 1 行 replay 差は SD と同程度であり、その値だけで効果を断定しない。split-K 12 全体の無効化は 1 行と 8 行の両方で改善した。tree budget 10 の候補条件は全有効 41.4924 ± 0.0111 ms、候補 40.5109 ± 0.0218 ms、ペア差 981.4 ± 26.3 µs/replay の改善。budget 16 は現行 executor の graph 対象外なので、verify GPU event / round の全有効 49.301 ± 0.016 ms、候補 47.580 ± 0.052 ms、ペア差 1.721 ± 0.053 ms を記録した。この区間には staging 等も含まれる。各 5 ペア。tree の各区分の単発比較は傾向の確認に限る。

案 C は split-K 12・後続 3 projection のみに使用し、同・後続 2 projection は両条件で未融合に揃えた。8 行 graph replay 5 ペアの案 C minus 未融合は +85.9 ± 28.1 µs で、未融合より遅いため採用しない。先行の 32 合成条件での bit 一致は最終品質受入とは別である。

### フックなし速度受入

同一時期に baseline `44da8fe`、修正前 `4bc0e412`、修正後を各比較 5 ペアで実行した。比率は各ペア群の平均の比、差の SD はペア差の標本 SD。code prompt の生成 ID は全 run で一致した。

| 条件 | 比較元 tok/s | 修正後 tok/s | 差 tok/s | 比較元比 |
|---|---:|---:|---:|---:|
| DFlash2 baseline | 123.078 ± 0.063 | 124.892 ± 0.100 | +1.814 ± 0.127 | +1.47% |
| DFlash2 修正前 | 123.216 ± 0.118 | 124.848 ± 0.078 | +1.632 ± 0.143 | +1.32% |
| 非投機 baseline | 30.880 ± 0.130 | 31.920 ± 0.096 | +1.040 ± 0.040 | +3.37% |
| 非投機 修正前 | 31.510 ± 0.019 | 31.810 ± 0.012 | +0.300 ± 0.012 | +0.95% |

DFlash2 の baseline 比 +1% と、非投機が修正前を下回らない目標を満たした。非投機の比較元同士に変動があるため、baseline 比と修正前比の値はそれぞれ同時期の独立した 5 ペアとして扱う。

### 品質・launch 数の受入

- BetterBench corpus、context 128、256 位置、BF16 KV の baseline logits と bit 一致。mean KL 0、top-1 一致率 100%、最大絶対差 0。
- code / prose / ja の非投機 greedy は、各 256 token の生成 ID が共有 baseline と完全一致。
- `test_compute_dflash2.py` の同じ `run_suite` を BF16、FP8、Octave の各 dtype ごとに実行し、各 33/33、合計 99/99 成功。1 lock の保持を 280 秒以内にするため dtype ごとに分割した。
- `test_compute_dflash2_tree.py` は各 compute process を個別に lock / timeout して全 prompt 成功。chain 対 target の既存差は 12 / 132 / 213 token のまま。tree の各比較も既存テストの許容条件を満たした。
- `test_exl3_mul24_linear`、`test_exl3_mul24_grouped`、`test_exl3_launch_plan`、`test_plamo3_lowering`、`test_plamo_exl3_projection` の ctest 5/5 成功。`run_server_regression.py --all` は実行していない。
- 非投機 `shelf-bench tg --mode forward --context 128` の perf stats は 1,061 physical launch/token。trace の decode step も 1,061。非投機 compute、chain 8、tree budget 8 / 10 の capture は各 1,063 kernel node。verify graph は descriptor 準備 kernel 1 個を含み、forward 内は 1,062 node。bench と compute graph は前後処理を含む範囲が異なるため同じ数として扱わない。全 I9 無効フラグでは chain 8 の kernel node が 1,126 となり、全無効化が維持されることを確認した。launch 数は記録のみで、1,000 以下は要求していない。

約 0.5 ms の残差の原因は、kernel 外と特定できていない。16 行には verify graph がなく、同一の replay 指標では比較できない。これらの制限と、今回実測しなかった split-K 値の性能は未解決である。
