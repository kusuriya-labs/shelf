# Roadmap

Shelf の現在の作業対象と、今後予定している作業の一覧である。
完了済み項目は載せない。過去の経緯は Git history を参照。

現行のゴールと性能目標は `docs/developer/objective.md`、現在の性能は
`docs/perf/current.md` を参照。

## 1. Kernel / runtime 最適化

- 不要な dispatch / workspace / copy の削除
- KernelId 粒度の再確認
- state transition / dependency の簡略化
- kernel family / KernelConfig の整理

## 2. Host backend に TP=2 を追加

## 3. DFlash2 mixed batch の production 配線

scheduler / executor は decode region と verify region を 1 つの `ScheduledBatch` に
載せられる（契約は `test_dflash2_mixed_batch` が固定する）。`ContinuousBatcher` は
現状同一 step 内で verify round と decode batch を別々に発行している。

1 batch に統合するには `dflash2_spec_step_batch` を prepare（propose + GDN snapshot +
verify token staging）と commit（accept + GDN restore + ring append）へ分割し、
batcher が decode region を前置した `ScheduledBatch` を組んで 1 回だけ `execute_batch`
する。実装上の制約（mixed batch に載せられる verify request は 1 つまで、
`capture_rows` と row index の扱い）がある。

drafter の batched proposal は `v_dot2c_f32_bf16`（`ExactRowsDot2`）を選ぶが、
draft GPU 時間は request 数に比例して増える。

## 4. リファレンス実装（vllm-mxfp4）との差分から採用するもの

優先順は次のとおり。

1. **small-M split-K decode GEMM（M≤64）** と **kernel trace 内訳に基づく launch 削減**。
   内訳は rocprofv3 kernel-trace から取り直す（[../perf/methodology.md](../perf/methodology.md) §1）。
2. **dynamic verify width**（round ごとの draft 数可変）。
3. 自前計測のうえで採否: fp8 residual stream、KV 容量の決め方、
   GDN conv+recurrent 統合。

## 5. serving の改善対象（BetterBench で測定）

`vendor/BetterBench` によるカテゴリ別比較の手順は
[../perf/methodology.md](../perf/methodology.md) §9。
BF16 / FP8 構成でのカテゴリ別計測は未実施であり、`docs/perf/current.md` に
現在のカテゴリ別数値は無い。

改善対象は長文 prefill のスループットである。

1. **長文 prefill のスループット**: 16k 以降で参照実装との差が開く。
   chunk サイズの拡張は効果が小さく、64k のコストは attention 側
   （O(n²) と leg）が支配する。prefill attention は fp8 QK/PV leg と
   query tile 64 行を持ち、row bucket 契約は R8192 まで拡張済みである。
   参照は長文側の差を paged attention backend / prefill attention の
   fp8 QK/PV leg / epilogue store width に分けて計測している。
   leg と tiling は実装済みなので、次の最大項目は gemm 側である。


## Server

`shelf-server` は Feature Complete として維持する。API 機能追加は現在の
roadmap の主対象ではない。Server の現在の surface と詳細 contract は
[server_status.md](server_status.md) を入口に各 doc を参照。
