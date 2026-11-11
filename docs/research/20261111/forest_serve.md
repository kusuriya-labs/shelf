# Forest serve 統合調査

## 統合時点の状況

`feature/dflash2-policy` と `feature/forest-chunk` を統合し、両者の単一 request API と chunk verify の既定値の不整合を修正した。serve の tree verify は未接続であり、以下の性能・品質評価は未実施である。

| 項目 | 状態 |
| --- | --- |
| c=1 の chain / tree N=10 | 未測定 |
| c=2〜4 の N=8、9、10、12 | 未測定 |
| first-mismatch の logit 差、KLD、生成品質 | 未測定 |
| 自動 budget の規則 | 未決定 |

## 実装上の確認事項

- serve の session pool は `step_buffers=false` で生成されるため、単一 request 向け tree の metadata、history、selector scratch を持たない。共有 batch context に必要な領域を確保し、request ごとの候補選択・木構築・受理経路 replay・KV と tap の gather を接続する必要がある。
- batch verify の attention region では、行単位の committed prefix も region の先頭行に合わせる必要がある。この不整合を修正した。
- 現在の tree chunk の選択条件は batch 全体の行数が 32 以下であり、例えば c=4、各 9 行では選択されない。request ごとの行数に基づく選択と kernel の境界条件を検証してから c=4 を評価する必要がある。
- tree 無効時の serve DFlash2 テスト、および統合された tree / adaptive の CPU テストを実行した。結果は作業報告を参照。

## request ごとの tree chunk 選択

GDN verify の dispatch と kernel launch の両方で、tree chunk の上限を batch 全体の行数から request ごとの最大行数に変更した。最大行数を指定しない既存の直接呼び出しは、従来どおり batch 全体の行数で判定する。

| 検証 | 結果 |
| --- | --- |
| 4 request × 9 行、合計 36 行の tree chunk と request ごとの個別実行 | 出力が一致 |
| tree chunk と CPU 参照・serial、および枝なし tree と chain chunk | 既存 GPU テストが通過 |
| tree / adaptive / GDN recurrence selector の関連 CPU テスト | 通過 |
| tree 無効の server DFlash2 テスト | 21 項目が通過 |

この段階では serve の tree 接続自体は未実装だった。共有 batch context の tree metadata、request ごとの候補選択、attention と RoPE の指定、受理経路の replay と KV・tap gather、CLI/server の budget 指定、VRAM 見積りの対応が残る。そのため、serve の数値ゲート、M1 と M2 の速度表、および budget 規則は未測定・未決定である。単体 kernel の一致を serve の数値ゲートの代わりにはできない。

## serve の木接続

`--dflash2-tree-budget` を serve へ配線し、共有 batch context に request ごとの木の
metadata と compact log を用意した。各 request の drafter logits から候補を選び、
祖先 mask と depth に対応する RoPE 位置で target verify を実行する。受理後は経路の
compact log を replay し、KV と target tap を受理経路順に集める。tree budget が
`K+1` の場合は chain 経路を使う。prefix capture 中は tree の物理行を KV page 境界でも制限する。

| 段階 | 確認 | 結果 |
| --- | --- | --- |
| 設定配線 | build、tree 無効の `test_server_dflash2.py` | 成功、20 項目通過 |
| 共有領域 | `test_dflash2_footprint` | 0 の従来容量と非ゼロ時の metadata 増分が一致 |
| 候補選択 | GPU の全語彙正規化と CPU 参照からの木構築 | 選択された token・parent・depth が一致 |
| forward | 同一 prefix・同一 N=10 の単発 decoder と batch context、10 行 × 248320 logits | 最大絶対差 0、全行 bit 一致 |
| forward | 同一 prefix の batch chain と budget 8、8 行 × 248320 logits | 全行 bit 一致 |
| 確定 | shelf-server と単発 decoder の N=10、code prompt、128 token | 最初の不一致は 118、非投機 top-1/top-2 差 0.020216 |
| 確定 | chain と budget 8、128 token | 生成 token が完全一致 |

forward の比較は 1 request、prefix token ID `248041,77091`、tree chunk lossy、
BF16 KV で実行した。N=10 の木の token は
`60,198,271,248046,13784,13784,510,220,16,220`、parent は
`-1,0,0,0,1,2,1,0,1,1` で、両経路の入力が一致した。serve は request の
sampling が greedy のときに木を使用する。c=2 以上の serve 全体の性能・品質は
この確認に含まれない。

## 数値ゲートで見つかった state 復元誤差

日本語 prompt 4 の N=12 では、当初の lossy replay で非投機との最初の不一致が
117 token 目となり、非投機の top-1/top-2 差は 0.147789 で基準 0.1 を超えた。
同じ入力で 2 回再現した。位置 143 の単一行 target decode の logits を比較すると、
非投機では token 12512 が 15.473950、token 196531 が 15.326161 だったのに対し、
tree の state を引き継いだ側ではそれぞれ 15.430992、15.460729 だった。全語彙の
最大絶対差は 0.232613、RMS 差は 0.055046 だった。prefix capture を無効にした
160 token の比較では不一致がなく、page 境界による round の分割も結果に影響する。
検証 kernel だけを exact にした別実験でも最初の不一致は 115 token 目・差 0.045752
となり、誤差が replay だけで発生すると断定はできない。

検証の chunk kernel は lossy のまま、受理経路の compact log replay の積だけを
exact decomposition に変更した。N=12 の同じ 512 token の再実行では最初の不一致は
115 token 目、非投機 top-1/top-2 差は 0.045752 となった。CPU 参照との tree replay、
既存 chain の replay、および serve の N=8 / 10 の関連テストも通過した。
このゲートは最初の不一致だけを判定し、その後の全位置の数値精度を保証しない。

## c=1 の数値ゲートと速度

R9700、Qwen3.8-27B EXL3 mul24、DFlash2 K=7、BF16 KV、greedy、thinking off。
`shelf-server --max-concurrent-requests 1 --max-seq-len 1024` を使用し、policy の
既定である lossy chunk verify と tree の exact replay で測定した。対象は
`gdn_recurrence_scripts/measure.py` の 24 prompt のうち表の 8 件で、各 512 token。
各 prompt について chain・N=10・N=12 を交互に切り替え、それぞれ 2 回実行した。
GPU は各実行を `flock /tmp/r9700.lock` で保護した。

速度は server の request log が出す `generated tokens / decode_ms` で、prefill 時間は
分母に含まない。各速度は 2 回の算術平均、chain 比の平均は prompt ごとの比の算術平均。
右端は非投機 greedy と最初に異なった位置における非投機 top-1/top-2 logit 差で、
2 回の大きい方を chain / N=10 / N=12 の順に示す。非投機基準は M11 と同じ 512 token の
記録を使用し、入力 ID と基準ファイルの SHA-256 を確認した。

| prompt | chain tok/s | N=10 tok/s (chain 比) | N=12 tok/s (chain 比) | 最初の不一致の最大差（0 / 10 / 12） |
| --- | ---: | ---: | ---: | ---: |
| code 1 (1) | 84.14 | 77.40 (-8.01%) | 75.58 (-10.17%) | 0.01429 / 0.02022 / 0.02022 |
| code 2 (13) | 79.82 | 82.55 (+3.43%) | 82.33 (+3.15%) | 0.01471 / 0.01471 / 0.01471 |
| shell 1 (0) | 69.79 | 69.33 (-0.67%) | 70.62 (+1.18%) | 0.05124 / 0.05124 / 0.05124 |
| shell 2 (16) | 77.22 | 75.25 (-2.55%) | 74.85 (-3.06%) | 0.01984 / 0.00925 / 0.00925 |
| 日本語 1 (3) | 36.92 | 40.32 (+9.20%) | 40.08 (+8.56%) | 0.00363 / 0.00363 / 0.00363 |
| 日本語 2 (4) | 43.50 | 44.52 (+2.34%) | 46.97 (+7.97%) | 0.03569 / 0.03569 / 0.04575 |
| reasoning 1 (2) | 74.60 | 73.54 (-1.43%) | 79.31 (+6.31%) | 0.03577 / 0.03306 / 0.03306 |
| reasoning 2 (18) | 108.82 | 104.70 (-3.79%) | 105.96 (-2.63%) | 0.00121 / 0.00121 / 0.00121 |
| 平均 | 71.85 | 70.95 (-0.18%) | 71.96 (+1.41%) | — |

48 回すべて 512 token を生成し、数値ゲートを通過した。差の最大は 0.051245。
各回の速度、最初の不一致位置、差、生成 token のハッシュ、および基準のハッシュは
[forest_serve_c1_results.json](forest_serve_c1_results.json) に記録した。tree 無効の
生成列は replay 修正前後の 16 回すべてで一致した。

N=10 の平均 chain 比は -0.18%、N=12 は +1.41% で、この測定では単発 CLI の M11 で
報告された N=10 の約 +10% と同程度の向上は得られなかった。両測定は実行経路と
prompt 数が異なる。tree logits の host 転送と prefix capture 時の page 境界も
serve の処理に含まれるが、それぞれの寄与は分離測定していない。今回の確認は c=1 の
生成列の最初の不一致に限り、c=2 以上の性能・数値ゲート、KLD、生成品質、自動 budget は
未評価である。

## 関連回帰

final build で `test_server_dflash2.py` は 20 項目、Octave の DFlash2 は 26 項目、
unaligned prefill は 7 項目、長い prefix は 8 項目が通過した。関連 ctest の tree、
footprint、candidate selector、tree chunk、compact log commit も通過した。tree 無効の
`test_compute_dflash2.py` は replay 精度修正前に 99 項目が通過し、修正前後の
budget 0 の 512 token 生成列は 16 回すべて一致した。

既存 `test_compute_dflash2_tree.py` は、lossy な chain と非投機の完全一致、および
`tree-extra 0` と chain の完全一致を前提にして失敗した。shell prompt では修正前後とも
chain と非投機は位置 37 で異なり、非投機の top-1/top-2 差は 0.051245 だった。
`tree-extra 0` は非投機と一致し、budget 8 は chain と完全一致した。chain、tree-extra 0、
budget 8 はそれぞれ replay 精度修正前後で同じ生成列だった。テストは chain と
`tree-extra 0` を既定の lossy 数値ゲートで判定し、budget 8 と chain の完全一致は
引き続き要求する。
更新後の `test_compute_dflash2_tree.py` は shell・code・reasoning の全 3 prompt、
各 256 token で通過した。

## c=1 の round 内訳と同期削減

前節と同じ R9700・モデル・greedy・BF16 KV、512 token、N=10、c=1 で、
prompt 1、0、13、4 を調べた。変更前の serve は計測専用ビルドで各木 round の
host 時間を採り、verify GPU は hipEvent、replay と gather・append は追加同期を
含む wall 時間で測った。CLI は M11 の同じ prompt の 2 回平均で、round 数で割った。
同期付き診断は通常の速度測定と異なる。serve の表は木 round のみで、ページ境界の
単一 token round を含まない。列の時間は重複するものがあり、単純に足せない。

| prompt | CLI 発行 token/round | serve 発行 token/木 round | CLI round ms/token | serve 診断 ms/token | CLI verify GPU ms/round | serve verify GPU ms/木 round | serve 未使用 logits D2H ms/木 round |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.44 | 4.49 | 9.82 | 12.43 | 42.14 | 39.93 | 1.11 |
| 0 | 4.60 | 4.03 | 11.58 | 12.94 | 42.09 | 40.19 | 1.12 |
| 13 | 4.87 | 4.83 | 10.90 | 11.63 | 41.92 | 40.18 | 1.14 |
| 4 | 2.69 | 2.55 | 19.73 | 21.68 | 41.73 | 40.71 | 1.11 |

CLI の round 時間は M11 の `DFLASH2_ROUND_MS / DFLASH2_ROUNDS` を発行数で割った。
serve の診断時間は各木 round の drafter 投入・待機、選択、metadata、verify、
転送、walk、replay、gather・append の wall 時間合計を発行数で割った。これには
計時のための同期が含まれ、両者は厳密な同一条件の差分ではない。

| 項目 | CLI ms/round | serve ms/木 round | 読み方 |
| --- | ---: | ---: | --- |
| drafter GPU | 8.43〜8.47 | 未分離 | CLI は hipEvent。serve の draft 待機は 8.17〜11.95 ms で、GPU 実行を含む |
| 候補選択 | 0.38 | 0.37〜0.38 | host 時間。候補確率の GPU 計算・転送・待機を含む |
| metadata | 未分離 | 0.25〜0.31 | snapshot と metadata の host 投入を含む |
| accept walk | 0.003〜0.014 | 約 0.004 | host 時間 |
| replay | 0.09〜0.12 | 0.47〜0.63 | CLI は投入時間、serve は追加同期付き wall 時間 |
| KV・tap gather / context 追加 | 0.014〜0.025 / 0.046〜0.048 | 合計 1.28〜1.31 | CLI は投入時間、serve は追加同期付き wall 時間 |
| logits D2H | なし | 1.11〜1.14 | serve は全語彙を使用せず転送していた |
| batcher | 対象外 | 0.01〜0.02 | `DFHOST` と `DFCOST` の wall 平均差。stdio と token 送出は未分離 |

CLI の `DFLASH2_GDN_MODE=tree-history` は tree 設定に基づく固定ラベルである。
chunk verify の実装では CLI も受理経路の compact replay に exact を指定するため、
このラベルだけで serve と復元方式が異なるとは判断できない。

prefix cache の影響は、prompt 1、N=10、512 token の stdio 診断で cache のみを
切り替えて調べた。有効時は 120 round・4.258 token/round・verify 7.150 行/round、
無効時は 95 round・5.379 token/round・verify 9.968 行/round だった。生成列も変わる
ため同一軌跡の比較ではないが、ページ境界の制約が受理数に影響することを示す。
cache 無効を本測定の代替にせず、既存の capture 動作は維持した。

変更前の木 round には draft の D2H、候補確率の D2H と明示同期、全 logits の D2H、
sampled token の D2H があり、metadata に複数の同期コピーを使用していた。変更後は
通常の木経路から全 logits と draft の D2H を除き、候補確率の取得時の同期にまとめた。
これは呼び出し元で確認した回数であり、kernel や runtime 内部を含む全同期の総数ではない。
trace が draft token を読む場合は従来どおりコピーする。

各変更後の 4 prompt・N=10 の 512 token 速度を示す。各セルは 1 回のため変動を含む。
最初の変更は未使用 logits の転送削除、次は draft の host 同期削除。2 つ目の変更前後の
生成列は 4 件とも完全一致した。

| prompt | 変更前 tok/s | logits 削除後 tok/s（前比） | draft 同期削除後 tok/s（前比） |
| ---: | ---: | ---: | ---: |
| 1 | 76.55 | 77.26 (+0.93%) | 82.20 (+6.40%) |
| 0 | 69.07 | 70.24 (+1.68%) | 76.56 (+9.00%) |
| 13 | 82.53 | 83.84 (+1.58%) | 91.93 (+9.65%) |
| 4 | 45.95 | 44.20 (-3.81%) | 47.36 (+7.16%) |

### verify の GPU 診断

N=10 の verify が 10 行となった最初の round を、CLI と serve でそれぞれ
semantic timing の hipEvent により別実行で測った。64 token の診断で、event 記録の
負荷を含む。表の合計は記録された kernel 時間の和であり、通常実行の forward wall
時間ではない。線形は `LINEAR_EXL3_INT8` と `LINEAR_BF16` の合計。

| prompt | CLI 合計 / GDN / attention / 線形 ms | serve 合計 / GDN / attention / 線形 ms |
| ---: | ---: | ---: |
| 1 | 47.855 / 1.275 / 0.255 / 37.972 | 47.471 / 1.263 / 0.274 / 37.603 |
| 0 | 47.685 / 1.274 / 0.256 / 37.806 | 47.812 / 1.278 / 0.277 / 37.880 |
| 13 | 47.693 / 1.283 / 0.243 / 37.821 | 47.710 / 1.272 / 0.263 / 37.806 |
| 4 | 47.772 / 1.271 / 0.236 / 37.882 | 47.693 / 1.268 / 0.250 / 37.821 |

この診断では verify の主要 kernel に、CLI と serve の速度差を説明する大きな差は
現れなかった。異なる入力位置と実行時状態での単発診断であり、全 round の差を
保証するものではない。

### 受理経路の GPU 診断

128 token の別実行で、compact commit、KV・tap gather、drafter の context 追加を
hipEvent で測った。比較のため診断ビルドのみで replay の積を lossy に切り替えた。
計時は各木 round で追加同期を行い、通常測定の速度には使用しない。exact と lossy は
生成経路が異なり得るため、差は同一入力 state での kernel 単体比較ではない。

| prompt | exact commit ms/round | lossy commit ms/round | exact gather ms/round | exact context 追加 ms/round |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.394 | 0.295 | 0.056 | 1.264 |
| 0 | 0.379 | 0.294 | 0.057 | 1.259 |
| 13 | 0.387 | 0.296 | 0.065 | 1.251 |
| 4 | 0.322 | 0.272 | 0.040 | 1.272 |

exact の増分はこの診断で約 0.05〜0.10 ms/木 round であり、前節の日本語 N=12
で lossy が数値ゲートを超えた事実を踏まえ、精度は変更しなかった。commit の event は
replay kernel の投入前から完了までで、conv 復元や replay 行の H2D 待ちは含まない。
CLI の M10 における shell N=10 の別診断は復元 0.508、gather 0.076 ms/round だった。

変更前の同期付き診断では、compute が報告する decode 時間から `DFCOST` の全 round
wall 時間を引いた残りは、4 prompt で 2.99〜5.19 ms/512 token だった。この残りは
compute 内の round 外処理をまとめた値で、stdio、送出、scheduler を個別には分離できない。
HTTP 側の通信や client への配送時間もこの差には含まれない。

診断値を発行 token あたりに換算した差を、round 時間と発行数の寄与に算術的に
分解すると次のようになる。前者は `(serve round - CLI round) / CLI 発行数`、
後者は `serve round × (1 / serve 発行数 - 1 / CLI 発行数)`。前記のとおり同期方法・
対象 round が異なるので、これは厳密な因果分解ではない。

| prompt | serve − CLI ms/token | round 時間の項 | 発行数の項 |
| ---: | ---: | ---: | ---: |
| 1 | +2.61 | +0.45 | +2.16 |
| 0 | +1.36 | -0.25 | +1.61 |
| 13 | +0.73 | +0.64 | +0.09 |
| 4 | +1.96 | +0.85 | +1.10 |

prompt 1 では変更前の全 logits D2H 1.11 ms/木 round が round 時間の項に含まれ、
発行数の項は prefix cache のページ制約とも整合する。前者は今回削除した。後者を
除くために cache capture を無効化すると意味が変わるため、今回の修正には含めていない。

### 候補選択と host 待機の診断

同期削減後の 64 token の別実行で、候補確率 kernel の GPU event と、その結果の
D2H 呼び出しおよび明示同期を測った。木構築は host 時間。各行は木 round の平均。
`hipMemcpyAsync` の host buffer は pageable であり、serve の転送呼び出し時間には
同じ stream に先行投入された drafter の完了待ちが含まれる。したがって転送の純粋な
コピー時間として扱わない。CLI は候補確率の前に draft 同期を行う。

| prompt | CLI 確率 GPU / D2H 呼出 / 同期 / 構築 ms | serve 確率 GPU / D2H 呼出 / 同期 / 構築 ms |
| ---: | ---: | ---: |
| 1 | 0.307 / 0.344 / 0.037 / 0.009 | 0.306 / 9.667 / 0.034 / 0.022 |
| 0 | 0.308 / 0.343 / 0.032 / 0.008 | 0.306 / 9.607 / 0.038 / 0.021 |
| 13 | 0.308 / 0.341 / 0.030 / 0.007 | 0.307 / 9.658 / 0.034 / 0.020 |
| 4 | 0.308 / 0.341 / 0.030 / 0.008 | 0.306 / 9.634 / 0.036 / 0.020 |

この GPU 時間は top-k 後の全語彙正規化と候補確率の計算である。候補の top-k 自体は
drafter 内で実行され、上記とは別の処理である。

別の 64 token・追加同期付き診断で、drafter 内の top-k に限定して hipEvent を
記録した。serve の平均は prompt 1 / 0 / 13 / 4 の順に 0.148 / 0.153 / 0.157 /
0.151 ms/木 round、平均 draft 行数は 5.18 / 5.76 / 6.14 / 5.62 だった。CLI の
c=1 も同じ単一 request の drafter 実装を使うが、top-k の CLI 単独 event 値は
この診断では採っていない。

## c=1 の最終測定

前節と同じ 8 prompt、各 512 token、chain / N=10 / N=12 を各 2 回、構成を交互に
切り替えて測定した。速度は `generated tokens / decode_ms`、各セルは 2 回平均、
chain 比の平均は prompt ごとの比の算術平均。基準ファイル 8 件の SHA-256 は前節と
一致する。計測バイナリと最終ビルドの SHA-256 はいずれも
`906f697e04d4b6e0eb3afdae4d8a7c00a7a97e6a385864237efccef9b10965f4`。

| prompt | chain tok/s | N=10 tok/s（chain 比） | N=12 tok/s（chain 比） | 最初の不一致の最大差（0 / 10 / 12） |
| --- | ---: | ---: | ---: | ---: |
| code 1 (1) | 84.08 | 84.23 (+0.17%) | 82.78 (-1.54%) | 0.01429 / 0.02022 / 0.02022 |
| code 2 (13) | 78.51 | 91.85 (+17.00%) | 91.63 (+16.72%) | 0.01471 / 0.01471 / 0.01471 |
| shell 1 (0) | 73.01 | 75.35 (+3.20%) | 76.60 (+4.91%) | 0.05124 / 0.05124 / 0.05124 |
| shell 2 (16) | 76.04 | 82.86 (+8.98%) | 82.65 (+8.70%) | 0.01984 / 0.00925 / 0.00925 |
| 日本語 1 (3) | 37.90 | 44.10 (+16.36%) | 43.77 (+15.49%) | 0.00363 / 0.00363 / 0.00363 |
| 日本語 2 (4) | 42.17 | 47.43 (+12.48%) | 50.82 (+20.51%) | 0.03569 / 0.03569 / 0.04575 |
| reasoning 1 (2) | 74.75 | 79.58 (+6.46%) | 84.07 (+12.47%) | 0.03577 / 0.03306 / 0.03306 |
| reasoning 2 (18) | 109.00 | 115.99 (+6.42%) | 116.81 (+7.17%) | 0.00121 / 0.00121 / 0.00121 |
| 平均 | 71.93 | 77.67 (+8.88%) | 78.64 (+10.55%) | — |

N=10 / N=12 の 8 prompt 平均は変更前の -0.18% / +1.41% から +8.88% /
+10.55% となり、目標の +5% を両方で超えた。参考として同一 8 prompt の CLI M11 は
+6.3% / +6.1% だった。測定時刻が異なるため、直接の対照試験ではない。

48 回とも 512 token を生成し、first-mismatch の非投機 top-1/top-2 logit 差は
最大 0.051245 で基準 0.1 以下。budget 0 の 16 回は変更前の対応 prompt と生成列の
SHA-256 がすべて一致した。各回・診断値・基準 hash は
[forest_serve_optimization_results.json](forest_serve_optimization_results.json) に記録。

最終コードでは関連 ctest 7 件と `test_server_dflash2_tree.py` が通過。最初の変更後に
`test_server_dflash2.py` は 20 項目、`test_compute_dflash2_tree.py` は 3 prompt すべて
通過した。2 つ目の変更は budget 0 の経路を変更せず、前述の 16 回の生成列でも一致を確認した。
サーバー全回帰は実行していない。c=2 以上は本測定に含まれない。

## 同時生成時の速度と数値：c=2

R9700、同じ target / drafter、K=7、BF16 KV、greedy、thinking off、最大長 1024。
`shelf-server` の同時 request 上限を 2 にし、policy と prefix cache は既定のままにした。
各 request は最大 512 token、構成を chain / N=9 / 10 / 12 と逆順に切り替えて各 2 回実行した。
A は code prompt 1・13、B は code 1・shell 0。prompt 番号は上記 24 件の基準と同じ。

server が compute から受け取った各 token の単調時計時刻を計測用の一時 Python copy で記録し、
全員の最初の token のうち最も遅い時刻から、最後の token のうち最も早い時刻までの token 数を
区間秒数で割った。prefill と、1 件でも生成を終了した後の時間を含まない。時刻記録の
オーバーヘッドを含む。セルは 2 回平均、括弧は同じ prompt 条件の chain 平均との比。

| prompt | chain tok/s | N=9 tok/s | N=10 tok/s | N=12 tok/s |
| --- | ---: | ---: | ---: | ---: |
| A：同種 | 130.76 | 124.77 (-4.58%) | 124.32 (-4.93%) | 126.12 (-3.55%) |
| B：混在 | 122.51 | 118.36 (-3.39%) | 117.82 (-3.83%) | 131.19 (+7.09%) |

32 request とも 512 token を生成した。c=1 の非投機基準で比較した first-mismatch の
非投機 top-1/top-2 logit 差の最大は 0.05738 で、0.1 以下だった。結果の詳細は
[forest_serve_concurrency_results.json](forest_serve_concurrency_results.json) に記録した。
N=12 は B で向上するが A では低下したため、この 2 条件だけでは一律の改善とは言えない。

## 同時生成時の速度と数値：c=3

同じ方法で最大同時 request 数を 3 にし、A は code 1・13・12、B は code 1・shell 0・
日本語 3 を使った。各セルは 2 回の平均。

| prompt | chain tok/s | N=9 tok/s | N=10 tok/s | N=12 tok/s |
| --- | ---: | ---: | ---: | ---: |
| A：同種 | 148.56 | 139.45 (-6.13%) | 146.06 (-1.69%) | 138.11 (-7.04%) |
| B：混在 | 121.69 | 126.57 (+4.02%) | 126.47 (+3.94%) | 122.25 (+0.47%) |

48 request とも 512 token を生成した。A・N=12 の code 1 は、非投機基準との最初の
不一致が 196 token 目で、その位置の基準 top-1/top-2 logit 差は 0.15756 と 0.1 を
超え、2 回とも再現した。他の 3 構成と B の最大差は 0.09210 以下だった。
この N=12 は数値ゲート不合格であり、原因を追加調査する。

c=3・A・N=12 の別診断では `SHELF_GDN_VERIFY_CHUNKED_EXACT=1` を指定すると、
code 1 の最初の不一致は 118 token 目・差 0.02022 に変わり、3 request の差は
0.08289 以下だった。これは lossy verify の数値差が影響することを示すが、最初の
不一致だけの評価であり、その後の全位置が正しいことを意味しない。速度表は既定の
lossy verify であり、この診断値を含まない。

## 同時生成時の速度と数値：c=4

同じ方法で最大同時 request 数を 4 にし、A は code 1・13・12・21、B は code 1・
shell 0・日本語 3・reasoning 2 を使った。各セルは 2 回の平均。

| prompt | chain tok/s | N=9 tok/s | N=10 tok/s | N=12 tok/s |
| --- | ---: | ---: | ---: | ---: |
| A：同種 | 185.24 | 173.38 (-6.40%) | 163.35 (-11.82%) | 165.96 (-10.41%) |
| B：混在 | 147.34 | 149.49 (+1.46%) | 148.56 (+0.83%) | 150.99 (+2.48%) |

64 request とも 512 token を生成した。A の code 13 は chain・N=9 ともに
c=1 非投機との最初の不一致で基準差 0.12203 とゲートを超えた。ただし両者の生成列は
512 token すべて一致した。N=10/12 および B の差は最大 0.09974 以下だった。
これらは first-mismatch のみの結果で、chain が c=1 非投機と異なることも考慮する必要がある。

### c=4 の verify GPU 診断

A の別実行で、4 request の batch round を `DFCOST` の hipEvent と semantic timing で
計測した。表の verify は指定行数が現れた round の GPU 時間中央値で、括弧は件数。
semantic timing はその行数の最初の 1 round で記録した kernel 時間の和であり、
event の記録負荷を含むため verify 中央値と直接加算・比較しない。EXL3 は
`LINEAR_EXL3_INT8`、attention は `PAGED_ATTENTION`、GDN は `GDN_RECURRENCE`。
N=12 は page 境界の制約で 48 行が現れず、46 行を示す。

| 構成 | 実行行数 | verify GPU ms/round | EXL3 ms | attention ms | GDN ms | 記録 kernel 合計 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain | 32 | 59.158 (19) | 48.438 | 0.617 | 3.228 | 62.946 |
| N=9 | 36 | 61.393 (9) | 49.968 | 0.704 | 3.429 | 65.026 |
| N=10 | 40 | 63.313 (6) | 51.117 | 0.806 | 3.465 | 66.462 |
| N=12 | 46 | 65.761 (1) | 52.662 | 2.425 | 3.582 | 69.974 |

この条件では行数増加に伴って verify と EXL3 の時間が増えた。semantic の attention
は単発値であり、46 行の値を行数だけによる増分とは断定できない。詳細は
[forest_serve_concurrency_profile.json](forest_serve_concurrency_profile.json) に記録した。
