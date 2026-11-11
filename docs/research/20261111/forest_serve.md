# Forest serve 統合調査

## 現状

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

serve の tree 接続自体は未実装である。共有 batch context の tree metadata、request ごとの候補選択、attention と RoPE の指定、受理経路の replay と KV・tap gather、CLI/server の budget 指定、VRAM 見積りの対応が残る。そのため、serve の数値ゲート、M1 と M2 の速度表、および budget 規則は未測定・未決定である。単体 kernel の一致を serve の数値ゲートの代わりにはできない。
