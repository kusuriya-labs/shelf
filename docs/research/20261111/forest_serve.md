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
