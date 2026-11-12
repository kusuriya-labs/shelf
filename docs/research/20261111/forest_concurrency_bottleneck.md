# Forest の同時生成時のボトルネック


## 結論

- c=2・同種では N=10 の時間/token 悪化 +0.802 ms のうち、算術分解で verify の差が約 57%、発行数減が約 28%、残りが約 15% だった。通常速度の 2 回平均は chain 比 -9.02%。
- c=4・同種では悪化 +0.506 ms/token の約 69% が発行数減、約 21% が verify の差、約 9% が残差だった。N=10 は batch 発行数が 17.24→16.19 token/round、通常速度は -10.47%。追加行が常に受理増をもたらすわけではない。
- 8→10 行の c=1 verify 増分は約 0.86 ms、32→40 行の c=4 は約 4.74 ms。前者も無料ではなく、後者は request 数分の行が増えるため絶対額が大きい。EXL3 が時間の大半を占めるが、律速がメモリ帯域かどうかは未確定。
- c=4・同種の同期診断では選択・metadata・replay・gather の増分合計が約 2.3 ms/round、追加同期なし診断の損益分岐に必要な削減は約 8.2 ms/round。別実行・異なる同期条件なので厳密な上限ではないが、固定費削減だけでの逆転は見込みにくい。c=3・混在は通常速度で +1.77% と条件差がある。

寄与は異なる生成軌跡の観測値を算術的に分けたもので、項ごとの介入実験ではない。通常速度の反復間には大きい変動もある。

## 測定方法

R9700（gfx1201）、Qwen3.8-27B EXL3 mul24、DFlash2 K=7、BF16 KV、greedy、
thinking off、prefix cache 有効、最大 sequence 長 1024。明示的な budget 0 を chain、
10 と 12 を tree とした。各 request は最大 512 token、各構成 2 回、2 回目は
budget の順序を逆にした。各試行で server を起動し直し、GPU 使用中は
`flock /tmp/r9700.lock` を保持した。`--all` 回帰と rocprofv3 は使っていない。

A は code 1・13・12・21、B は code 1・shell 0・日本語 3・reasoning 2 の先頭から
同時数分を取り、[forest_serve.md](forest_serve.md) の c=2〜4 と揃えた。c=1 は
A/B とも同じ code 1 になるので 1 条件として扱う。既存の c=1 の +9% は 8 prompt の
比の平均であり、この 1 prompt だけで再現することを要求したものではない。

通常速度の実行と診断実行は別にした。server が compute から token を受け取った
単調時計時刻を記録し、c=2 以上は最後の初回 token から最初の最終 token までの
全 request の token 数を区間秒数で割る。これは先行調査と同じ区間であり、prefill と
共通区間外を除く。c=1 は先行調査に従い compute の request log の生成数 / decode_ms
を使う。生成完了と 512 token をすべて確認する。

診断では `SHELF_DFLASH2_BATCH_PHASE_TRACE`、`SHELF_DFLASH2_COST_TRACE` を指定し、
各 request の受理 draft 数・発行数、batch の行数・wall と hipEvent の drafter / verify、
host 区間を記録した。通常実行に追加同期はない。host 区間の計時は GPU 投入と完了を
混ぜるため、非同期 GPU 作業が後の区間の待機に現れる。区間名のみから kernel 費用を
足し合わせない。別の `SHELF_DFLASH2_BATCH_PHASE_SYNC` 診断は各区間に同期を挿入し、
投入後の GPU 実行を各区間に含めて測る。同期により並行性と wall が変わるので通常速度の
比較には使わない。木選択には probability kernel・D2H 待ち・木構築、metadata には
snapshot と metadata 構築・H2D、replay には conv 復元と compact log、gather には
KV・tap gather と drafter context 追加を含む。

request あたりの round 数にはページ境界で投機できない単一 token round を含む。
最初の 1 token は prefill で生成されるので、round の発行数の合計は 511 token/request
になる。全員の共通区間を round に対応させる際は同じ単調時計上の round 終了時刻で
選別した。server が token を受け取るまでに非同期配送があるため、区間両端に数 round の
ずれがあり得る。round wall と token 数の比は検算であり、HTTP・scheduler・区間端の
差を完全に除いた厳密な同一区間の因果推定ではない。

測定・集計のコードは [scripts/measure.py](forest_concurrency_bottleneck_scripts/measure.py)、
[scripts/summarize.py](forest_concurrency_bottleneck_scripts/summarize.py)、各試行の token
時刻・速度と集計は [結果 JSON](forest_concurrency_bottleneck_results.json) に記録した。

## 寄与の計算

各条件の共通区間 round をまとめ、batch あたりの発行数を `E`、round wall を `T`、
verify の hipEvent 時間を `V`（いずれも round 平均）とする。単位は `E` が token、
`T` と `V` が ms。chain を添字 c、tree を t とすると、時間 / token の差は次の
恒等式で分けられる。

```text
Tt/Et - Tc/Ec = (Vt - Vc)/Et
              + ((Tt - Vt) - (Tc - Vc))/Et
              + Tc × (1/Et - 1/Ec)
```

順に verify event 差、verify 以外の wall 差、発行数差の項である。round 数が変わり、
生成列・ページ制約・稼働 request 数も変わり得るので、これは実測値の算術分解であり、
同じ軌跡を保った介入実験ではない。event と host の重なりもある。verify 以外の差には
drafter、選択、metadata、replay、gather、単一 token round、host 待機等が混在し、
すべてを木の固定費とみなさない。負の項は差を縮める方向を表す。

木の仮想的な損益分岐は、発行数を `Et` に保ったまま round wall を
`Tc × Et/Ec` まで短縮できるかで見る。必要削減量は `Tt - Tc × Et/Ec`。
これは発行数や verify が変わらない理想化であり、改善実装の実測値ではない。

## 通常速度

各セルは 2 回の平均、括弧内は個別値と chain 比。c=1 は request log の値を使う。
全試行で各 request は 512 token を生成し `length` 終了した。

| c | prompt | chain tok/s（各回） | N=10 tok/s（chain 比、各回） | N=12 tok/s（chain 比、各回） |
| ---: | --- | ---: | ---: | ---: |
| 1 | A | 84.05 (83.70, 84.40) | 85.05 (+1.19%; 85.20, 84.90) | 84.30 (+0.30%; 84.40, 84.20) |
| 2 | A | 136.05 (141.65, 130.45) | 123.78 (-9.02%; 123.89, 123.66) | — |
| 2 | B | 127.90 (133.35, 122.46) | 116.69 (-8.76%; 115.48, 117.91) | — |
| 3 | A | 147.13 (147.03, 147.23) | 144.76 (-1.61%; 144.69, 144.84) | — |
| 3 | B | 121.12 (120.90, 121.33) | 123.27 (+1.77%; 123.21, 123.32) | — |
| 4 | A | 183.06 (182.68, 183.44) | 163.90 (-10.47%; 163.88, 163.92) | — |
| 4 | B | 154.17 (148.11, 160.23) | 145.62 (-5.55%; 146.95, 144.28) | — |

c=2 の chain と c=4・B の chain には大きい反復差がある。前者の 2 回目だけとの比較でも
N=10 は A -5.11%、B -4.71%。c=4・B の先行測定は N=10 +0.83% であり、今回の
2 回だけで一般的な低下を断定できない。request の完了順や batch 構成の変化も影響し得る。

## request ごとの受理数

round あたりの受理数は draft の受理数で、発行数には correction / bonus も含む。
次は c=2・A の追加同期なし診断で、2 回とも同じ値だった。request ID は当該 server
実行内の識別子。全 request が prefill の 1 token を除く 511 token を発行した。

| request | 構成 | round 数 | 受理 draft / round | 発行 token / round | batch verify 行 / round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | chain | 112 | 3.562 | 4.562 | 6.63 |
| 1 | N=10 | 120 | 3.258 | 4.258 | 7.15 |
| 2 | chain | 122 | 3.189 | 4.189 | 6.77 |
| 2 | N=10 | 121 | 3.223 | 4.223 | 7.08 |

batch verify 行には単一 token fallback の行を含めず、その round は 0 行として平均した。
この例では tree の追加行が request 1 の発行数増にはつながっていない。request 2 は
発行数がわずかに増えた。これはこの生成軌跡の観測であり、draft の confidence と実際の
利益の相関を検証した結果ではない。各 c・prompt・構成・反復・request の全値は結果 JSON の
`diagnostic_summary.requests` に記録した。

## 共通区間の round と wall の整合

追加同期なしの診断 2 回を、共通区間の round 数で重み付けして集計した。`E` は
batch あたりの発行 token、`T` は `DFCOST` wall、`V` は verify の hipEvent 時間。
検算速度は `1000 × E/T`、右端は同じ診断実行の token 到着時刻による速度の 2 回平均。

| c | prompt | budget | round 数 | E | T ms | V ms | E/T tok/s | 到着時刻 tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 0 | 224 | 4.562 | 54.31 | 38.70 | 84.02 | 84.11 |
| 1 | A | 10 | 240 | 4.258 | 50.62 | 37.80 | 84.13 | 84.22 |
| 1 | A | 12 | 240 | 4.258 | 51.03 | 38.21 | 83.45 | 83.54 |
| 2 | A | 0 | 224 | 8.696 | 66.39 | 44.98 | 130.98 | 129.88 |
| 2 | A | 10 | 240 | 8.450 | 71.28 | 48.82 | 118.54 | 117.84 |
| 2 | B | 0 | 224 | 8.000 | 62.58 | 45.03 | 127.84 | 127.49 |
| 2 | B | 10 | 246 | 7.764 | 67.01 | 47.76 | 115.87 | 115.45 |
| 3 | A | 0 | 230 | 12.148 | 80.88 | 50.16 | 150.19 | 150.08 |
| 3 | A | 10 | 222 | 12.270 | 84.42 | 51.49 | 145.35 | 144.69 |
| 3 | B | 0 | 240 | 10.133 | 83.43 | 49.38 | 121.46 | 121.04 |
| 3 | B | 10 | 222 | 10.198 | 82.58 | 51.18 | 123.50 | 123.19 |
| 4 | A | 0 | 216 | 17.241 | 93.72 | 54.35 | 183.96 | 183.83 |
| 4 | A | 10 | 228 | 16.193 | 96.22 | 56.10 | 168.29 | 167.20 |
| 4 | B | 0 | 220 | 14.109 | 94.06 | 53.88 | 150.00 | 148.46 |
| 4 | B | 10 | 236 | 14.136 | 95.30 | 56.11 | 148.33 | 148.28 |

両速度の差は最大約 1.1%。c=1・chain では全 round の `DFCOST` 合計 6.114 秒に対して
request log の decode は 6.119 秒だった（最初の診断実行）。c=2 以上は共通区間端の
非同期配送と scheduler を含むため、全 request の decode_ms を単純合算して batch wall と
比較しない。診断と通常速度は別実行なので各表の値も一致する必要はない。

## 速度差の分解

以下は前記の同一診断の `T/E` を式で分けた値（ms/token）。正は tree の悪化、負は改善。
`V` は batch verify に限り、page 境界などの単一 token fallback は「その他」に含む。

| c | prompt | verify 差 | その他の wall 差 | 発行数差 | 合計 | 損益分岐に必要な wall 削減 ms/round |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | A | -0.212 | -0.655 | +0.850 | -0.016 | -0.07 |
| 2 | A | +0.455 | +0.124 | +0.223 | +0.802 | 6.77 |
| 2 | B | +0.353 | +0.218 | +0.238 | +0.808 | 6.27 |
| 3 | A | +0.108 | +0.180 | -0.066 | +0.222 | 2.72 |
| 3 | B | +0.177 | -0.260 | -0.052 | -0.136 | -1.39 |
| 4 | A | +0.108 | +0.047 | +0.352 | +0.506 | 8.20 |
| 4 | B | +0.158 | -0.070 | -0.013 | +0.075 | 1.06 |

c=2・A の悪化 +0.802 ms/token の約 57% は verify、15% はその他の wall、28% は
発行数減の項。c=4・A の +0.506 の約 21%、9%、69% はそれぞれ同じ項で、こちらでは
発行数減が最大である。これらは丸めた割合であり、前述のとおり木固有の因果寄与そのもの
ではない。c=3・B では verify が増えてもその他と発行数の項で相殺された。負の損益分岐
削減量は、この診断ではすでに chain を上回ることを示す。

## verify 行数の傾き

追加同期なし診断の 2 回分をまとめ、該当する batch 行数の hipEvent 中央値を取った。
行数は合計であり、page 境界により同じ合計でも request ごとの内訳が違う場合がある。

| c | 構成 | 合計行 | verify ms 中央値 | 件数 |
| ---: | --- | ---: | ---: | ---: |
| 1 | chain | 8 | 40.949 | 150 |
| 1 | N=10 | 8 | 41.292 | 28 |
| 1 | N=10 | 10 | 41.807 | 110 |
| 1 | N=12 | 12 | 42.909 | 82 |
| 4 | chain | 32 | 58.788 | 38 |
| 4 | N=10 | 32 | 58.739 | 10 |
| 4 | N=10 | 40 | 63.527 | 12 |

chain 8 → tree 10 は +0.858 ms、約 0.429 ms/行、+2.1%。chain 32 → tree 40 は
+4.739 ms、約 0.592 ms/行、+8.1%。同じ N=10 内の 8→10 と 32→40 では、それぞれ
約 0.257、0.599 ms/行。追加 2 行は c=1 でも無料ではないが割合は小さく、c=4 では
4 request 分の追加行による絶対額と割合が大きい。これらは異なる prefix・行配置の
自然発生 round であり、同一入力・同一 kernel の行数だけを変えた掃引ではない。

先行の c=4 semantic timing は chain 32 行と N=10 40 行で EXL3 が 48.438→51.117 ms、
記録 kernel 合計が 62.946→66.462 ms だった（[forest_serve.md](forest_serve.md)）。
EXL3 は合計の約 77% で、その診断での増分 3.516 ms の約 76% を占める。ただし event
記録負荷のある別実行であり、この観測だけから重み読み出し律速の有無や帯域使用率を
確定できない。8〜64 行の制御された掃引と帯域計測は未実施。

## 追加同期による段階別診断

各行は別実行 2 回の共通区間を round 数で重み付けした ms/round。列は drafter、
選択、snapshot・metadata、verify、結果待ち / copy、accept、conv 復元・compact replay、
KV・tap gather と context 追加、単一 token fallback。各区間の完了を待つよう同期を
追加している。同期待ちは最後の列に別掲するが、各段階の値に含まれるため加算しない。
自然に行われる同期や pageable 転送が選択・metadata・verify・結果待ちに含まれるため、
追加同期の待ちだけを host 待ちの全量とはみなせない。H2D と CPU 構築、KV gather と tap
append、自然な同期待ちの内訳はこの計測ではさらに分離できない。

| c | prompt | budget | round | draft | 選択 | metadata | verify | 結果待ち | accept | replay | gather・追加 | 単一 token | 追加同期待ち | wall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 0 | 224 | 9.54 | 0.00 | 0.28 | 45.95 | 0.11 | 0.02 | 0.34 | 1.47 | 2.09 | 11.32 | 59.97 |
| 1 | A | 10 | 240 | 7.82 | 0.36 | 0.29 | 37.99 | 0.05 | 0.02 | 0.58 | 1.25 | 2.55 | 9.40 | 50.98 |
| 1 | A | 12 | 240 | 7.82 | 0.36 | 0.28 | 38.48 | 0.05 | 0.02 | 0.58 | 1.27 | 2.55 | 9.41 | 51.49 |
| 2 | A | 0 | 224 | 9.11 | 0.00 | 0.22 | 45.40 | 0.11 | 0.02 | 0.66 | 2.50 | 3.41 | 12.18 | 61.58 |
| 2 | A | 10 | 240 | 9.04 | 0.72 | 0.37 | 46.35 | 0.06 | 0.02 | 1.10 | 2.45 | 7.02 | 12.35 | 67.26 |
| 2 | B | 0 | 224 | 9.06 | 0.00 | 0.17 | 45.10 | 0.09 | 0.01 | 0.66 | 2.53 | 2.38 | 12.14 | 60.15 |
| 2 | B | 10 | 246 | 9.01 | 0.73 | 0.39 | 46.30 | 0.05 | 0.02 | 1.11 | 2.57 | 4.34 | 12.34 | 64.64 |
| 3 | A | 0 | 230 | 15.03 | 0.00 | 0.32 | 50.14 | 0.10 | 0.02 | 0.97 | 3.73 | 6.29 | 19.75 | 76.79 |
| 3 | A | 10 | 222 | 14.47 | 1.02 | 0.49 | 51.17 | 0.06 | 0.03 | 1.68 | 3.67 | 9.25 | 19.47 | 82.05 |
| 3 | B | 0 | 240 | 14.72 | 0.00 | 0.16 | 50.45 | 0.10 | 0.01 | 0.93 | 3.65 | 8.33 | 19.37 | 78.51 |
| 3 | B | 10 | 222 | 14.66 | 1.12 | 0.51 | 51.30 | 0.06 | 0.02 | 1.57 | 3.79 | 7.54 | 19.59 | 80.72 |
| 4 | A | 0 | 216 | 18.00 | 0.00 | 0.31 | 54.54 | 0.10 | 0.02 | 1.31 | 4.93 | 7.72 | 24.33 | 87.19 |
| 4 | A | 10 | 222 | 17.56 | 1.35 | 0.45 | 57.03 | 0.06 | 0.02 | 2.01 | 5.00 | 11.72 | 24.40 | 95.45 |
| 4 | B | 0 | 220 | 17.39 | 0.00 | 0.41 | 54.00 | 0.09 | 0.02 | 1.25 | 4.84 | 9.26 | 23.60 | 87.49 |
| 4 | B | 10 | 236 | 17.50 | 1.36 | 0.49 | 56.05 | 0.06 | 0.02 | 1.93 | 4.96 | 9.94 | 24.06 | 92.53 |

c=1・chain の同期診断は 1 回に外れ値があり、verify の 8 行中央値 41.51 ms に対して
2 回を合わせた上表の平均は 45.95 ms になった。追加同期なしの表では平均 38.70 ms
で、両表の単純な差を同期のコストとは解釈できない。同期診断では実行経路と待機位置が
変わり、単一 token fallback の回数も変化する。追加同期なしの host 区間も JSON に保存
しているが、例えば chain の metadata 区間には先に投入した drafter の待ちが入り得る。

## 改善余地

同期診断で、選択 + metadata + replay + gather・追加の各列の tree − chain を足すと、
c=2・A は約 1.26、c=2・B は 1.44、c=3・A は 1.84、c=4・A は 2.26、c=4・B は
2.24 ms/round だった。選択は tree だけ、他の列は chain にも必要な仕事を含む。
対応する追加同期なし診断の損益分岐削減量は、それぞれ 6.77、6.27、2.72、8.20、
1.06 ms/round。c=2 と c=4・A では、上記の差だけを完全に消すという楽観的な比較でも
必要量より小さい。一方 c=4・B は数値上は届く可能性があり、c=3・A も差が比較的小さい。
同期条件と生成軌跡が異なる値を並べた目安で、削減可能量の厳密な上限でも、実装時の
速度予測でもない。特に単一 token fallback は tree の構築費とは別に変動する。

行数を減らす方向は検討に値する。先行の c=4・A は N=9 が -6.40%、N=10 が -11.82%
で N=9 の方が損失は小さかったが、c=3・A は N=9 -6.13%、N=10 -1.69% と逆だった。
固定行数を下げるだけで常に改善するとは言えない。request ごとに行数を変える場合は
verify kernel の行配置と metadata の費用も再測定する必要がある。

c=4・A の request 別 round 数は chain が 110 / 110 / 160 / 108、N=10 が
114 / 125 / 152 / 129 だった（両反復で一致）。1 件では round が減り、3 件では増えた。
有利な request だけを tree にする余地はあるが、この結果に oracle を当てても batch の
行数と他 request の進行が変わるため、個々の round 数を足して速度向上を予測できない。
現在の batch は tree mode を共通に選ぶため、request ごとに chain と tree を混ぜるには
別途設計が必要である。confidence で有利な request を識別できるか、追加の記録と実験が
必要であり、今回の測定では示していない。

## 再実行

指定 worktree で configure / build 済み、`/tmp/shelfvenv` と記載のモデル配置を前提にする。
`measure.py` は各試行の server を `flock` で起動し、完了ごとに JSON を更新する。
既存の結果を再開する場合は同じキーを省略する。ログは `build-gfx1201/forest-c2-runs/`
に置く。下記の集計・マージは各測定プロセスが終了してから行う。

```sh
nohup setsid env FOREST_RESULTS=build-gfx1201/forest-c2-fresh.json /tmp/shelfvenv/bin/python docs/research/20261111/forest_concurrency_bottleneck_scripts/measure.py > build-gfx1201/forest-c2-measure.log 2>&1 < /dev/null &
nohup setsid env FOREST_RESULTS=build-gfx1201/forest-c2-sync-results.json /tmp/shelfvenv/bin/python docs/research/20261111/forest_concurrency_bottleneck_scripts/measure.py --sync-only > build-gfx1201/forest-c2-sync.log 2>&1 < /dev/null &
# 両プロセスの完了・エラーを確認した後
/tmp/shelfvenv/bin/python docs/research/20261111/forest_concurrency_bottleneck_scripts/merge.py build-gfx1201/forest-c2-fresh.json build-gfx1201/forest-c2-sync-results.json
FOREST_RESULTS=build-gfx1201/forest-c2-fresh.json /tmp/shelfvenv/bin/python docs/research/20261111/forest_concurrency_bottleneck_scripts/summarize.py
FOREST_RESULTS=build-gfx1201/forest-c2-fresh.json /tmp/shelfvenv/bin/python docs/research/20261111/forest_concurrency_bottleneck_scripts/tables.py
```
