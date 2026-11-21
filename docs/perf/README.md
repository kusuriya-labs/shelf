# docs/perf

このdirectoryは **現在の性能値・再現条件・測定方法** のみを置く。

- [current.md](current.md) — 現在再現可能な performance baseline（未計測の場合はその状態を書く）
- [methodology.md](methodology.md) — 今後も使う測定原則・GPU 排他（`/tmp/*.lock`）・品質比較手順
- [exl3-int8-prefill-quality.md](exl3-int8-prefill-quality.md) — EXL3-INT8 mul24 prefill A/B の品質と速度
- [quality-bf16/README.md](quality-bf16/README.md) — Qwen3.8-27B の BF16 基準品質測定と [結果一覧](quality-bf16/results.md)
- [kv-octave.md](kv-octave.md) — Octave KV cache の容量・decode / prefill 性能
- [qwen-embed-host.md](qwen-embed-host.md) — 16 GiB VRAM での Qwen3.8-27B 3.05bpw / 128Ki 実測
- [prefill-chunk-reference.md](prefill-chunk-reference.md) — prefill chunk の参考測定

実験の経緯・採否判断・棄却根拠の正本は [../research/README.md](../research/README.md) を参照する。
`shelf-bench` の使い方は [../user/bench.md](../user/bench.md) を参照する。

## ルール

- ここに置くのは「現行の再現可能な値」だけ。日付・revision・commit・計測コマンドを必ず併記する。
- 新しい benchmark を実行していない場合、数値を更新しない。
- 過去 baseline / Gate / 採否判断 / 実験ログはここへ置かない（Git history を参照）。
- 「latest」と書く場合は revision または日付を必ず添える。
- 個別の最適化経緯や試行錯誤をここへ持ち込まない。
