# docs/perf

このdirectoryは **現在の性能値・再現条件・測定方法** のみを置く。

- [current.md](current.md) — 現在再現可能な performance baseline（未計測の場合はその状態を書く）
- [methodology.md](methodology.md) — 今後も使う測定原則

## ルール

- ここに置くのは「現行の再現可能な値」だけ。日付・revision・commit・計測コマンドを必ず併記する。
- 新しい benchmark を実行していない場合、数値を更新しない。
- 過去 baseline / Gate / 採否判断 / 実験ログはここへ置かない（Git history を参照）。
- 「latest」と書く場合は revision または日付を必ず添える。
- 個別の最適化経緯や試行錯誤をここへ持ち込まない。
