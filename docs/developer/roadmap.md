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

1. **BF16 small-M（M≤64）** と **kernel trace 内訳に基づく launch 削減**。
   FP8 / MXFP4 の in-block split-K は実装済みである。BF16 の
   block-per-output-row `SmallRows` と M=9..16 の Wmma 切替は実装したが、
   非投機 decode が横ばい（EXL3 28.39 → 28.40、MXFP4 27.27 → 27.27）で、
   DFlash2 の accept もプロンプト依存だったため production selector には
   接続していない（[../research/20261116/bf16_small_rows.md](../research/20261116/bf16_small_rows.md)）。
   次は、非投機で明確な利得が出る形状に限定した再評価と、
   内訳の rocprofv3 kernel-trace 取り直しである
   （[../perf/methodology.md](../perf/methodology.md) §1）。
2. **dynamic verify width**（round ごとの draft 数可変）。
3. 自前計測のうえで採否: fp8 residual stream、KV 容量の決め方、
   GDN conv+recurrent 統合。
4. MXFP4 linear の A 側量子化は、採用判断（logit 変動ゲートへ切り替える可否）と
   本実装が未完了（[../research/20261103/mxfp4_gate_a_prime.md](../research/20261103/mxfp4_gate_a_prime.md)
   §11〜§13）。

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

## 6. メモリ footprint の見積

- DFlash2 の drafter footprint を詰める。session slot の見積 0.34 GiB に対し実測
  0.094 GiB で、TP2 の KV auto-fit が約 1 GiB 過剰に確保している
  （`dflash2_footprint.cpp` の spec workspace 項）。
- working set reserve を正確にする。既定は auto（`executor_footprint.h` の
  `52 × Npad × hidden × 2`）だが、`create_model_executor` を plan（graph lowering +
  `build_program_set`）と create（arena 確保）へ分けられれば program workspace の
  実値を使える。TP2 では rank の shard 分だけ過大（見積 2.06 GiB / 実測 1.16 GiB）。

## 7. CLI / E2E

- 起動時の引数の名前を整理し、compute で使用可能なものはすべて E2E で通す
  （命名を OpenAI API 準拠にしたい）。

## 8. Multi-request + 投機推論

2026-11-06 時点の実測にもとづく残課題。測定はすべて 1 rank（TP4）per-rank shape
`2560:2560`、`shelf-bench gemm` p50。

- 既定の `--max-concurrent-requests` は 4（`shelf-compute` / `shelf-server` とも）。
  DFlash2 の verify 行数は `4 × block_size = 32`。`kDFlash2SpecMaxVerifyRows = 512`。
- GEMM は rows=4→16 でほぼ時間が変わらない（FP8 9.6→9.8 µs、BF16 36.4→36.9 µs）。
  並列 4 は GEMM 側では完全に余っており、並列数を上げる余地が大きい。
- 負荷生成器は用意済み（`tools/bench/shelf_bench_suite.py track-d` の
  `--concurrency` / `--requests`）。`shelf-bench batch` は合成トークンで即 EOS する
  ため decode スループットの計測には使わない。
- やること:
  - `--max-concurrent-requests 4 → 16` の効果（スループットと KV 容量）を測る。
    `--max-num-batched-tokens` の既定 2048 が row bucket を決めている点に注意。
  - 投機推論は `requests × block_size` 行が flat 領域に収まることを確認し、accept 率と
    verify の内訳（draft / verify GPU ms）を同時に記録する。
  - rows 8〜64 の BF16 / FP8 GEMM 効率（2〜4 倍の余地）を詰める。
  - router 側は decode で 1.73 ms/token（4.5%）。`moe_router` は帯域限界の 3.6 倍、
    `moe_slot_scan` は prefill で 15 倍。融合（router + top-k + slot_scan + scatter）が候補。

## 9. Pipeline parallel

PP の layer split・host hand-off・E2E は実装済み。以下は未対応または未検証。

- DFlash2 speculative decode、prefix cache、hidden taps / `--dump-logits`、MTP block
  との併用（現行は PP と併用できない）。
- decode と prefill が同じ batch に混在する実行（§3 の mixed batch と同じ。
  PP coordinator は現在 `unsupported` を返す）。
- stage 間 overlap（現行は pinned host 経由の直列 hand-off。性能最適化は未着手）。
- 3 stage 以上の実機検証（この環境は 2 GPU のため PP=2 のみ実測）。
- stage 0 の KV page 数に全 stage を合わせる制約の解消
  （非対称 layer 分割または stage ごとの page 数指定）。
- dense checkpoint の PP=2 prefill / decode を warmup + 複数 run で測る。
  grouped expert（Qwen3.6-35B-A3B）の PP=2 は E2E で prompt 512 / 1024 / 1943 token を
  2 run ずつ測り `docs/perf/current.md` へ記録済み。


## Server

`shelf-server` は Feature Complete として維持する。API 機能追加は現在の
roadmap の主対象ではない。Server の現在の surface と詳細 contract は
[server_status.md](server_status.md) を入口に各 doc を参照。
