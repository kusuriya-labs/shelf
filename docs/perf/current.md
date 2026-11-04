# Current performance baseline

この文書は Shelf の qualified performance baseline を記録する。
ここに載る値は各節に記した measurement revision での実測結果であり、それ以後に追加された
変更に対する性能を保証しない。

測定原則は [methodology.md](methodology.md) を正本とする。

## 現在の状態

**現在の build に対する qualified baseline は、2026-11-03 の long-context prefill 1 断面のみである。**

これまでに記録していた値は、計測の前提だった構成がソースから削除されたため、
再現できなくなっている。該当する計測を削除した。

| 削除した計測 | 失われた前提 |
| --- | --- |
| HF FP8 E2E prefill / decode（TP=1・TP=2） | resident model host（`shelf-model-host` / `--model-host`）を削除 |
| Target prefill pp2048・non-spec decode tg128 | 計測対象が `models/Qwen3.8-27B-Quark-AWQ-MXFP4`。MXFP4 / AWQ 量子化は非対応 |
| TP=2 decode（`test_qwen35_tp_e2e`） | 計測条件の `models/Qwen3.8-27B-PSQ` を削除 |
| DFlash2 decode・batched verify scaling | 融合カーネルが有効な構成と MXFP4 model での計測 |
| BetterBench によるカテゴリ別比較 | 参照実装が MXFP4 構成。現在の build では同じ条件を作れない |
| fp8 KV prefill の leg / query tile 比較、attention kernel 実効レート | kernel 構成が変更済み |
| GPU-MCU decode backend | 実装も `exp/gpu-mcu` branch もリポジトリに無い |

削除した数値そのものは Git history に残る。実験の経緯・採否判断は
[../research/](../research/) が正本である。

## 再計測が必要な断面

現在の performance target（`models/Qwen3.8-27B-FP8`、Qwen3.8-27B BF16、drafter DFlash2）で
次を測る。条件・コマンドは [methodology.md](methodology.md) に従い、measurement revision・
date・build・GPU 割り当て・生成結果の一致アンカーとともに同じ節へ書く。

| 断面 | 計測口 | 主指標 |
| --- | --- | --- |
| prefill（pp2048） | `shelf-bench pp --mode forward` | `gpu_tokens_per_sec` |
| non-spec decode（tg128 / ctx2048） | `shelf-bench tg --mode greedy` | `gpu_tokens_per_sec` |
| E2E decode（BF16 / HF FP8、TP=1・TP=2） | `shelf-compute` | `DECODE_TOKENS_PER_SEC` |
| DFlash2 decode | `shelf-compute --dflash2-model-dir <dir> --dflash2-drafts 7` | `DECODE_TOKENS_PER_SEC`、`DFLASH2_ROUNDS`、`DFLASH2_MEAN_ACCEPTED` |
| DFlash2 batched verify scaling | `test_dflash2_batch_perf` | round 内訳（`draft_gpu_ms` / `verify_gpu_ms`） |
| TP=2 scaling | `test_qwen35_tp_e2e` | decode tok/s p50 |
| カテゴリ別 serving | `vendor/BetterBench`（methodology §9） | decode t/s、TTFT p50 |
| fp8 KV の数値品質 | `tools/quality_gate.py` / `shelf-bench quality` | ppl 差、mean KL、top-1 一致率 |

## 2026-11-03 long-context prefill

Qwen3.8-27B-FP8 / TP2 / FP8 KV / `--rope-factor 2`、selector の範囲内
（`max_visible <= 262144`）で計測した。warmup 1 + 5 run の p50。

| 条件 | 値 |
| --- | ---: |
| prompt tokens | 262,128（要求 seq len 262,144） |
| prefill chunk | 2048 |
| gpu_ms p50 | 272468.22 |
| gpu_tokens_per_sec p50 | 962.05 |
| run 間 p10 / p90 | 272200.69 / 272688.71 ms |

- measurement revision: `bb2e56e` + 計測用 harness 変更
- GPU: Radeon AI PRO R9700 × 2（device 0,1）
- command: `shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --tp 2 --device 0,1 --kv-cache-dtype fp8 --prompt-tokens 262128 --prefill-chunk 2048 --runs 5 --warmup 1`
- 524,288 は production の値が未取得（`kMeasuredMaxVisible24 = 524288` を採用済みだが、
  warmup + 5 run の production 計測は未実施。1 run の参考値のみ research 記録にあり）。詳細は
  [../research/20261103/gate6_long_context.md](../research/20261103/gate6_long_context.md)。

## 2026-11-04 Qwen3.6-35B-A3B MoE decode（初回）

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）を TP=1 / bf16 KV で
計測した。**これは fallback 混在の初期 baseline であり、最適化経路の decode 速度ではない**。
decode 1 step あたり bf16 線形 171 本が selector 却下で correctness path へ落ちている
（内訳は [../research/20261104/moe_gate_d_slot_layout.md](../research/20261104/moe_gate_d_slot_layout.md)
§11。MoE 系の kernel は 1 本も fallback していない）。

| 条件 | 値 |
| --- | ---: |
| context tokens | 64 |
| decode tokens | 32 |
| warmup | 1 |
| gpu_ms p50（clean 4 run） | 5512.97 |
| gpu_tokens_per_sec p50 | 5.81 |
| run 間 | 5490.07 / 5511.22 / 5514.72 / 5521.73 ms |
| 生成一致 anchor | `GREEDY_TOKEN_SUM=15232`（5 run すべて一致） |

- measurement revision: `d92c9e3` + `LoadedModel` 配線と plan scratch / router 契約の修正
- GPU: Radeon AI PRO R9700（`--device 0`、測定時は device 0 / 1 idle）
- command: `shelf-bench tg --mode greedy --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --context 64 --tokens 32 --warmup 1`
- 5 run 目は `SHELF_QWEN35_PERF_STATS=1 --perf-stats-output` を付けた（gpu_ms 5502.91 /
  5.82 tok/s）。上の p50 は stats なしの 4 run から取った
- 同じ checkpoint の `shelf-compute` E2E（prompt 6 token、`--max-num-batched-tokens 64
  --max-concurrent-requests 1`）は DECODE_TOKENS_PER_SEC 8.05。条件が違うので直接比較しない
- **この数値は正しさの anchor を含まない**。E2E の生成列は correctness backbone と optimized
  MoE の混合経路であり、HF との数値照合は未実施（§8 の 40 層 E2E gate）

## 記録規則

- 新しい計測をしていない数値は更新しない。
- 速度の数値は、生成結果が一致している rep からのみ採用する。一致アンカー
  （`GENERATED_IDS` の sha1、`GREEDY_TOKEN_SUM` など）を同じ節に併記する。
  `GENERATED_IDS` の sha1 は次で取る。

  ```sh
  grep '^GENERATED_IDS=' <output> | sed 's/^GENERATED_IDS=//' | sha1sum
  ```

- 速度に関する主張には計測条件（revision / commit / GPU / コマンド）を添える。
- 絶対値は GPU の温度・クロック状態でも振れる。同一条件の交互測定と run 間中央値で
  判定し、日付の違う値を直接比較しない。
- ここに無い性能値が必要な場合は推測せず、methodology の手順で新規計測し、
  日付・revision・commit とともに追記する。
