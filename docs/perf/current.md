# Current performance baseline

この文書は Shelf の qualified performance baseline を記録する。
ここに載る値は各節に記した measurement revision での実測結果であり、それ以後に追加された
変更に対する性能を保証しない。

測定原則は [methodology.md](methodology.md) を正本とする。

## 現在の状態

**現在の build に対する qualified baseline は、2026-11-03 の long-context prefill、
2026-11-04 の MTP decode（Qwen3.8-27B MXFP4）、2026-11-04 の 9B BF16 linear allowlist
である。**

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
- command: `shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --tensor-parallel 2 --device 0,1 --kv-cache-dtype fp8 --prompt-tokens 262128 --prefill-chunk 2048 --runs 5 --warmup 1`
- 524,288 は production の値が未取得（`kMeasuredMaxVisible24 = 524288` を採用済みだが、
  warmup + 5 run の production 計測は未実施。1 run の参考値のみ research 記録にあり）。詳細は
  [../research/20261103/gate6_long_context.md](../research/20261103/gate6_long_context.md)。

## 2026-11-04 Qwen3.6-35B-A3B MoE decode

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）を TP=1 / bf16 KV で
計測した。decode の correctness path は `EMBEDDING_LOOKUP`（1 本 / token）だけである。

| 条件 | 値 |
| --- | ---: |
| context tokens | 64 |
| decode tokens | 32 |
| warmup | 1 |
| gpu_ms p50（clean 6 run） | 497.27 |
| gpu_tokens_per_sec p50 | 64.35 |
| run 間 | 496.28 / 496.70 / 497.21 / 497.34 / 497.42 / 498.87 ms |
| 生成一致 anchor | `GREEDY_TOKEN_SUM=2199`（11 run すべて一致） |

- measurement revision: `4b8d413` + `RMS_NORM` @2048（per-feature）と `PAGED_ATTENTION`
  `(q_heads 16, kv_heads 2, head_dim 256)` の registered 化、split-KV splits 決定の一本化、
  `EMBEDDING_LOOKUP` @2048 の registered 化、`MOE_COMBINE` の feature 方向分割と
  `MOE_PLAN_SCAN` の expert 合計 LDS 前段（2026-11-05、
  [../research/20261105/moe_decode_launch_roofline.md](../research/20261105/moe_decode_launch_roofline.md)）
- GPU: Radeon AI PRO R9700（`--device 0`、測定時は device 0 / 1 idle）
- command: `shelf-bench tg --mode greedy --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --context 64 --tokens 32 --warmup 1`
- kernel 単位 counter の内訳（`SHELF_QWEN35_PERF_STATS=1`、32 token 区間）:
  logical_dispatches 43360 = optimized_operations 43360、**correctness_fallbacks 0**
  （`physical_kernel_launches` 43392 は grouped 系の multi-launch ぶん。`EMBEDDING_LOOKUP`
  登録前は fallback 32、`RMS_NORM` / `PAGED_ATTENTION` 登録前は 2944）
- split の on / off は `physical_kernel_launches - logical_dispatches` で見てよい
  （context 4096 / 8 token で 88 = grouped の 8 + split attention の 80 = 10 層 × 8 decode）。
  context 64 では 32（grouped のみ）で split は入らない
- 前の値（同一条件、combine / plan scan 変更前）は `gpu_ms p50` 542.29 / **59.01 tok/s**。
  その前は `gpu_ms p50` 3128.75 / **10.22 tok/s**（anchor `GREEDY_TOKEN_SUM=2987`）。その前は
  5512.97 / 5.81 tok/s（anchor 15232）。登録と計測の経緯は
  [../research/20261104/moe_backbone_linear_allowlist.md](../research/20261104/moe_backbone_linear_allowlist.md) と
  [../research/20261104/moe_rmsnorm_paged_attention_allowlist.md](../research/20261104/moe_rmsnorm_paged_attention_allowlist.md)
- `shelf-compute` E2E（prompt 14 token、`--max-num-batched-tokens 64 --max-concurrent-requests 1`、
  8 decode）は DECODE_TOKENS_PER_SEC 56.52（1 run）。tg とは条件が違うので直接比較しない
- 長い context（device 1、各 2 run、`--prefill-chunk 2048`）。split-KV 有効後:
  `--context 4096` は 56.19 / 56.25 tok/s、`--context 2048` は 56.60 / 55.01 tok/s。
  capability を外した同一 tree の A/B は 51.67 / 51.73 と 55.02 / 55.11 なので、
  **context 4096 で +8.7 %、context 2048 は差が無い**（visible 2048 は attention が短く、
  split の効く余地が無い）。`q_per_kv = 8` の split-KV は bf16 で有効（`max_visible_tokens >= 2048`）、
  fp8 は非 split のまま
- 同じ条件を MoE combine / plan scan の変更後に測ると `--context 4096` は **58.79 / 61.64**
  （anchor `GREEDY_TOKEN_SUM=2192` 不変。combine だけの中間測定で 60.40 / 60.43）。
  context 4096 の run 揺れは 2.9 ms/token（5.3 %）あるので、2 run の差を性能主張に使わない
- context 2048 は split の on / off で greedy の合計が 745578 → 827368 に変わる
  （reduction 順序の差。split は softmax を可視域ごとに online で走らせ最後に合成する）。
  数値品質は context 2048 / 64 positions の A/B で mean KL 0.00623・top-1 一致 0.984・
  ppl 差 −2.70 %（TP1 と TP2 の差と同じ帯。根拠は research 記録）
- context 64 は split の閾値（2048）未満なので無変更。3 run で 58.74 / 58.86 / 59.17 tok/s、
  anchor `GREEDY_TOKEN_SUM=2199`
- MoE prefill chunk（decode、tg）: `--prefill-chunk 1024 / 2048 / 4096` は 51.36 / 51.28 /
  49.31 tok/s。context 8192 では chunk 8192 が 52.61 / 54.93、chunk 4096 が 55.09 / 54.86 tok/s
  （device 1、各 2 run。chunk 8192 は通るようになったが decode は速くない）
- MoE prefill throughput（`shelf-bench pp --prompt-tokens 8192 --runs 3 --warmup 1`、device 1）。
  `pp` が grouped expert checkpoint を load できるようになった（`LoadedModel` 配線）。
  `--prefill-chunk 1024 / 2048 / 4096 / 8192` で **3648 / 3759 / 3684 / 3569 tok/s**。
  bucket R8192（group 69632 行）は elementwise kernel の grid.z 分割で表現されるので拒否されないが、
  **最良は chunk 2048 で 8192 は 5.3 % 低い**。elementwise 単体の roofline は 69632 行で
  607 GB/s（DRAM 実測 636 GB/s の 95 %）なので、chunk 8192 が遅い理由は elementwise ではない
  （経緯は `docs/research/20261104/moe_group_row_elementwise_envelope.md`）
- 数値の連続性は teacher-forced A/B（base の生成列をプロンプトへ埋めた 14 token 入力、
  `--dump-logits` 9 行）で top-1 一致 7/9、入力列が同一だった row 0〜6 はすべて一致。
  HF との絶対照合は未実施（§8 の 40 層 E2E gate）

## 2026-11-04 MTP decode（Qwen3.8-27B MXFP4）

`models/Qwen3.8-27B-Quark-AWQ-MXFP4` を TP=1 / BF16 KV で計測した。target-only と
MTP drafter（`--mtp --mtp-drafts 7`）を交互に実行し、warmup 1 + 5 run の p50 を取った。
全 run で `GENERATED_IDS` が target-only と一致した。

| prompt | target t/s | MTP K=7 t/s | speedup | accepted/7 | full accept | prefill target/MTP ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 26.03 | 56.55 | 2.17x | 6.88 | 1.00 | 405.3 / 508.0 |
| 2048 | 26.25 | 37.91 | 1.44x | 4.25 | 0.58 | 818.5 / 977.7 |
| 4096 | 25.93 | 49.46 | 1.91x | 6.00 | 0.89 | 1682.6 / 1968.6 |

- measurement revision: `2c3cdb1`
- GPU: Radeon AI PRO R9700 × 1（`--device 0`、他の GPU は idle）
- command（MTP）: `shelf-compute --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 --input-ids-file <prose N tokens> --max-new-tokens 64 --max-seq-len 4096（4096 は 8192） --device 0 --prefix-cache 0 --mtp --mtp-drafts 7`
- prompt: `tests/data/mtp_perf/prose.txt` を tokenizer で 1024 / 2048 / 4096 token に伸ばしたもの
- 生成一致 anchor（`GENERATED_IDS` の sha1）: 1024 `6c07c6fe36929d9ad2f750304422b1a6f6035edc`、
  2048 `08e9e2c949de068a2d18621e676067db5170e070`、4096 `4aba2b4cee835fa60e6696f5528fb4ece21d6e42`
- run 間の p50 は上表。1024 の MTP は 9-run 再測でも p50 56.71（54.73–80.86、80.86 の 1 run を除くと
  54.73–56.75）。2048 / 4096 の MTP は 5-run の範囲がそれぞれ 36.42–38.08 / 49.18–49.59。
- 参考: K=4 の 5-run p50 は 1024 41.00、2048 34.95、4096 37.73。計測した範囲では K=7 が上回る。
- target の prefill chunk は target-only と同じ `max_batch_rows` を使う。MTP 側の first pass だけ
  512 行以下に分割するため、MTP の prefill は target より長い（上表の prefill 差）。

## 2026-11-04 MoE teacher-forced quality（初回）

`shelf-bench quality` を `LoadedModel` 経由にしたことで、grouped expert checkpoint も
teacher forcing できる。

| 条件 | 値 |
| --- | ---: |
| model | Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 / TP=1 / bf16 KV |
| corpus | `vendor/BetterBench/corpus/v1` を `tools/quality_gate.py corpus`（2191 token、sha256 `9136d38b…`） |
| context / scored positions | 128 / 64 |
| `QUALITY_PPL` | 75.146 |
| 再現性 | 同一条件の self-compare: mean KL 0、top-1 一致 1.000、ppl 差 0（決定論的） |

- **これは HF との絶対照合ではない**。生成列の一致 anchor は `docs/perf/current.md` の MoE
  decode 節と research 記録にあり、数値の正しさの gate は未実施のまま（§8）
- split-KV が走る条件（context 2048 / 64 positions / score start 2049）での on / off A/B:
  ppl 31.754（split）/ 30.896（非 split、−2.70 %）、mean KL 0.00623、top-1 一致 0.984、
  top-5 overlap 0.966、max abs logit delta 1.216。reduction 順序の差として
  TP1 と TP2（mean KL 0.00494・top-1 0.9375）と同じ帯
- 同じ `quality` 経路の dense 側確認（Qwen3.8-27B-FP8、corpus 2203 token、context 128 /
  positions 32）: TP1 `QUALITY_PPL` 4.3905、TP2 4.5421（+3.45 %）、mean KL 0.00494、
  top-1 一致 0.9375、top-5 overlap 0.9563。TP1 と TP2 の差は reduction 順序によるもので、
  `quality` の `LoadedModel` 化で生じた変更ではない（`--tensor-parallel 2` は dense のみ受理）

## 2026-11-04 Qwen3.5-9B BF16 linear allowlist

`models/Qwen3.5-9B`（BF16）の hidden 4096 系 linear shape を
`kLinearShapeRules` に追加した。201 token prompt / 8 decode token / BF16 KV で、
変更前と変更後をそれぞれ 5 run 計測した。

| 項目 | 変更前 p50 | 変更後 p50 |
| --- | ---: | ---: |
| prefill ms | 12392.6 | 294.3 |
| decode tok/s | 3.77 | 13.69 |

- measurement revision: 変更後 `500367c`、変更前 `5b4885f`（target-only 経路は同一。
  変更前は MTP 無効の target-only build で計測）
- GPU: Radeon AI PRO R9700 × 1（`--device 0`）
- command: `shelf-compute --model-dir models/Qwen3.5-9B --input-ids-file <prose 201 tokens> --max-new-tokens 8 --max-seq-len 512 --device 0 --prefix-cache 0`
- prompt: `tests/data/mtp_perf/prose.txt` を Qwen3.5-9B の tokenizer で 201 token に切ったもの
- 生成一致 anchor（`GENERATED_IDS` の sha1）: `db823356300f09e2258f9f9615ffd0df7abebb74`
- 変更前の decode は 5 run 中 1 run が 2.48 tok/s、他は 3.77–3.78。変更後は
  13.62–13.70 で安定。
- 変更前は `SHELF_LINEAR_DEBUG=1` で 7 shape の `LINEAR_SELECT_MISS` が出ていた。
  変更後は通常 decode で miss が出ない。生成列は correctness mode と一致する。

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
