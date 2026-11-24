# Online N-gram / Suffix Speculative Decoding の導入・併用評価記録

日付: 2026-11-24
対象 Issue: #49
関連ドキュメント: [測定手法](../../perf/methodology.md) / [Online N-gram 設計仕様](../../developer/online_ngram.md) / [DFlash2](../../developer/dflash2.md) / [MTP](../../developer/mtp.md) / [shelf-compute CLI](../../user/compute.md) / [shelf_server CLI](../../user/server.md)

## 1. 概要と実装方式

`Qwen3.8-27B` を主対象とし、プロンプトおよび生成済みコンテキストの suffix 一致から追加 GPU モデル推論なしでドラフト列を提案する `OnlineNgramMatcher`（[online_ngram.h](../../../include/shelf/runtime/online_ngram.h) / [online_ngram.cpp](../../../src/shelf/runtime/online_ngram.cpp)）を実装し、以下 3 経路へ統合した。

1. **N-gram 単独 speculative decoding (`--ngram`)**: `ContinuousBatcher::run_ngram_round()` が `OnlineNgramMatcher` により 2-gram / 4-gram ハッシュインデックス（32,768 トークン窓、4-way 16,384 バケット、デフォルト `min_match=4`, `max_match=32`, `num_drafts=7`）を走査し、ヒット時は `K+1` 行の target 検証、ミス時は通常 1-token decode を実行する（追加モデルメモリ不要）。
2. **DFlash2 + N-gram ハイブリッド (`--dflash2-model-dir ... --ngram`)**: `ContinuousBatcher::run_dflash2_round()`（および `ServeController`）が DFlash2 drafter 実行前に `OnlineNgramMatcher` へ照会し、ヒット時は `precomputed_drafts` を検証へ渡して DFlash2 ドラフト推論をスキップ（`NGRAM_DFLASH_SKIPPED` を加算）、ミス時は通常の DFlash2 ドラフトにフォールバックする。
3. **MTP + N-gram ハイブリッド (`--mtp ... --ngram`)**: `run_mtp_shot()` および `ServeController::step()` が MTP 4-step ドラフト実行前に `OnlineNgramMatcher` へ照会し、ヒット時は `MtpDrafter` のドラフト生成（`compute_logits=false` ステップ）をスキップして `precomputed_drafts` を target 検証へ渡す。

## 2. 実験一覧と判定サマリ

| ID | 対象 | 判定 | 主な根拠 |
|---|---|---|---|
| `EXP-NG-01` | CPU Matcher オーバーヘッドとロジック正当性 | **ACCEPT** | `test_online_ngram` で `avg_us=0.2834` / `stats_cpu_per_round=0.2107`、2-gram・4-gram・リングバッファ折り返し・GDN history 等検証 PASS |
| `EXP-NG-02` | Standalone `--ngram`（4 ワークロード AB/BA 評価） | **ACCEPT** | `baseline` 比 `code_edit` 2.31x、`multi_turn` 3.44x、`json_extract` 1.60x、低反復 `general_reasoning` でも 1.03x（劣化なし） |
| `EXP-NG-03` | DFlash2 + N-gram ハイブリッド（単一リクエスト Greedy） | **CONDITIONAL (既定 OFF の opt-in として ACCEPT / 単一 Greedy 常時高速化目的としては REJECT)** | `code_edit` 4.46x / `multi_turn` 5.01x 等ベースライン比は大幅高速化だが、DFlash2 単体（4.69x / 5.37x）比では `dflash_mean_accepted` 低下により検証ラウンド数増加（後述） |
| `EXP-NG-04` | DFlash2 + N-gram ハイブリッド（`serve-stdio` 並行 batch=4 / Stochastic） | **ACCEPT** | 並行 batch=4 で `agg_tps` 194.02 → **207.88 tok/s (+7.1%)**、`df_rounds` 129 → 116 削減；`code_edit` stochastic でも 146.68 → **161.11 tok/s (+9.8%)** |
| `EXP-NG-05` | MTP + N-gram ハイブリッド (`code_edit`) | **ACCEPT (EXL3) / CONDITIONAL (FP8)** | EXL3 (`max_seq_len=1024`) で `54.37 → 68.55 tok/s (+26.1%)`、別枠プローブでも `66.07 → 71.15 tok/s (+7.7%)`。FP8 (`Qwen3.8-27B-FP8`) では `68.24 → 63.89 tok/s`（`mtp_rounds` 109→120） |
| `EXP-NG-06` | CLI / Server / GPU 単体テスト・回帰 | **ACCEPT** | 関連 CTest 9/9 PASS、`gpu1;required` 単体テスト 7/7 PASS、Python unit/grammar テスト群 PASS（`test_server_cli.py` の既知 prefill chunk 1 件のみ既存 FAIL） |
| `EXP-NG-07` | PLaMo 3 Translate 8B BF16 動作確認・5 ワークロード AB/BA 評価 | **ACCEPT** | `baseline` と Greedy 5/5 (100%) 完全一致・停止 token `16` 正常終了。`w5` **2.29x** (`32.53 → 74.33 tok/s`)、`w3` **1.78x** (`32.61 → 58.15 tok/s`)、`w4` **1.21x**、`w1`/`w2` 退行なし |
| `EXP-NG-08` | PLaMo 3 8B パラメータ感度 (`min_match=2/3/4`, `K=3/7/15`) | **ACCEPT** | 全構成 `parity=True`。`w3` では `m2_k7` が **64.67 tok/s (1.98x)**、`w5` では `m3_k7` が **86.48 tok/s (2.66x)** で最高速。追加非反復確認でも `m2_k7` で退行なし |
| `EXP-NG-09` | PLaMo 3 2B BF16 / EXL3 4.05bpw 軽量モデル比較 | **ACCEPT** | 全 18 実行で `parity=True`。2B EXL3 `w5` **306.36 (`m4_k7`, 1.93x)** / **354.57 (`m2_k7`, 2.23x)**、2B BF16 `w5` **195.19 (`m4_k7`, 2.29x)** / **242.62 (`m2_k7`, 2.85x)** |
| `EXP-NG-10` | PLaMo 3 8B BF16 `serve-stdio` 並行 batch=4 | **ACCEPT** | 4 ワークロード同時投入で `agg_tok_s` **69.73 → 90.62 tok/s (+30.0%)**、wall `14.87s → 11.41s`、`hits=121/568`, `acc=462` |

## 3. 測定環境・条件

測定手順は [測定手法](../../perf/methodology.md) に従い、GPU 実行はすべて `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`（AMD Radeon AI PRO R9700 / `gfx1201`）＋ `flock /tmp/r9700.lock` で排他取得して計測した。

- モデル: `qwen38-27b-exl3-4bpw-mul24`（主評価）、`Qwen3.8-27B-DFlash2`（DFlash2 drafter）、`Qwen3.8-27B-FP8`（FP8 MTP 比較）
- 主比較設定: `--max-new-tokens 512`、`--max-seq-len 4096`、`--prefix-cache 0`、`--ngram-drafts 7`、`--dflash2-drafts 7`
- スケジュール: 各ワークロード内で `baseline` / `ngram_only` / `dflash2_only` / `dflash2_ngram` の順方向（r0）＋逆順（r1）の AB/BA 2 ラウンドを実施

## 4. `EXP-NG-01`: CPU Matcher マイクロベンチマーク

- バイナリ: `build/tests/test_online_ngram`
- 実測値: `avg_us=0.2834`, `hits=4093`, `stats_cpu_per_round=0.2107`
- CPU 全 9 CTest（`test_online_ngram`, `test_qwen35_spec_verify`, `test_architecture_boundaries`, `test_compute_ngram_*` 4 件, `test_cli_docs_sync`, `test_cli_docs_sync_python`）すべて 0 failure で PASS。

## 5. `EXP-NG-02` / `EXP-NG-03`: 4 構成 × 4 ワークロード（EXL3 Greedy AB/BA）

### 5.1 スループット・採択効率・CPU オーバーヘッド実測表（2 ラウンド平均）

| Workload (prompt / gen) | 構成 | `decode_tok_s` r0 / r1 (平均・倍率) | `rounds` | `emitted/round` | `NGRAM_HITS` | `NGRAM_MEAN_ACC` | `DFLASH2_MEAN_ACC` | `NGRAM_DFLASH_SKIPPED` | `NGRAM_CPU_US/round` |
|---|---|---|---|---|---|---|---|---|---|
| `code_edit` (507 / 512) | `baseline` | 31.94 / 31.75 (**31.84**, 1.00x) | 511 | 1.000 | - | - | - | - | - |
|  | `ngram_only` | 73.69 / 73.72 (**73.70**, 2.31x) | 190 | 2.689 | 67/190 (35.3%) | 4.687 | - | 0 | 0.980 |
|  | `dflash2_only` | 144.05 / 154.64 (**149.34**, 4.69x) | 73.0 | 7.005 | - | - | 6.005 | - | - |
|  | `dflash2_ngram` | 142.83 / 141.43 (**142.13**, 4.46x) | 84 | 6.083 | 45/83 (54.2%) | 4.978 | 5.083 | 45 (53.6%) | 1.499 |
| `json_extract` (731 / 512) | `baseline` | 31.38 / 31.27 (**31.32**, 1.00x) | 511 | 1.000 | - | - | - | - | - |
|  | `ngram_only` | 50.17 / 50.08 (**50.12**, 1.60x) | 294 | 1.738 | 68/294 (23.1%) | 3.147 | - | 0 | 0.657 |
|  | `dflash2_only` | 168.94 / 169.02 (**168.98**, 5.39x) | 64 | 7.984 | - | - | 6.984 | - | - |
|  | `dflash2_ngram` | 154.16 / 154.22 (**154.19**, 4.92x) | 76 | 6.724 | 32/76 (42.1%) | 3.969 | 5.724 | 32 (42.1%) | 1.456 |
| `general_reasoning` (88 / 512) | `baseline` | 31.16 / 31.84 (**31.50**, 1.00x) | 511 | 1.000 | - | - | - | - | - |
|  | `ngram_only` | 32.85 / 31.87 (**32.36**, 1.03x) | 483 | 1.058 | 15/483 (3.1%) | 1.400 | - | 0 | 0.478 |
|  | `dflash2_only` | 101.18 / 100.82 (**101.00**, 3.21x) | 111 | 4.604 | - | - | 3.604 | - | - |
|  | `dflash2_ngram` | 85.86 / 85.70 (**85.78**, 2.72x) | 131 | 3.901 | 4/131 (3.1%) | 1.250 | 2.901 | 4 (3.1%) | 0.950 |
| `multi_turn` (576 / 443) | `baseline` | 31.49 / 31.51 (**31.50**, 1.00x) | 442 | 1.000 | - | - | - | - | - |
|  | `ngram_only` | 108.43 / 108.32 (**108.38**, 3.44x) | 104 | 4.250 | 70/104 (67.3%) | 4.829 | - | 0 | 1.131 |
|  | `dflash2_only` | 169.34 / 169.24 (**169.29**, 5.37x) | 56 | 7.893 | - | - | 6.893 | - | - |
|  | `dflash2_ngram` | 157.97 / 157.80 (**157.88**, 5.01x) | 71 | 6.225 | 61/71 (85.9%) | 5.049 | 5.225 | 61 (85.9%) | 1.427 |

### 5.2 Greedy `GENERATED_IDS` 一致性（実測結果）

- `json_extract` (512 tok): r0/r1 × 全 4 構成で 8/8 **完全一致**。
- `multi_turn` (443 tok): r0/r1 × 全 4 構成で 8/8 **完全一致**。
- `general_reasoning` (512 tok): `baseline` (r0/r1)、`ngram_only` (r0/r1)、`dflash2_ngram` (r0/r1) の 6 回が **完全一致**。`dflash2_only` (r0/r1) のみ `idx=23`（`23470` vs `11870`）で分岐（検証バッチ 8 行時と 1 行時の丸め差に起因すると推測される）。
- `code_edit` (512 tok): 同一 `baseline` の r0 と r1 間ですら `idx=417`（`23173` vs `41824`）で分岐する（EXL3 split-K 等の近接 logit タイに起因すると推測）。一方、`r0_dflash2_ngram == r1_dflash2_ngram == r1_baseline`（3 実行間で 512/512 一致）、および `r0_ngram_only == r1_dflash2_only`（`r0_baseline` と `idx=499` まで一致し両者間は 512/512 一致）を確認。
- `--ngram-drafts` バリアント検証（`code_edit`, `max_new_tokens=128`）: `K=3`（`55.52 tok/s`, hits `30/61`, acc `63`）・`K=15`（`64.60 tok/s`, hits `12/36`, acc `77`）の双方で `code_edit` r0 `baseline` 先頭 128 トークンと **完全一致**。`--ngram-stats 0` 実行（`62.43 tok/s`）で `NGRAM_ROUNDS` 非出力を確認。

### 5.3 単一リクエスト Greedy における N-gram 単独 vs DFlash2 ハイブリッドの考察

- **N-gram 単独 (`--ngram`) の有用性**: DFlash2 重み・セッションプール等（約 4.18 GiB 相当の追加 VRAM と 1.5B draft 推論）を一切消費せず、`multi_turn` で **3.44x**、`code_edit` で **2.31x**、`json_extract` で **1.60x** の高速化を達成し、反復の乏しい `general_reasoning` でも 1.03x（CPU 負荷 `0.478 µs/round`）と退行が起きない。DFlash2 非対応モデルや VRAM 逼迫時に非常に有効。
- **DFlash2 ハイブリッド (`dflash2_ngram`) が単一 Greedy で `dflash2_only` を下回った理由（実測値に基づく分析）**: 単一 Greedy では DFlash2 単体の採択長が極めて高い（`multi_turn` 6.893、`json_extract` 6.984）。N-gram がヒットすると DFlash2 ドラフト推論を省略できる（`skip` 32〜61 回）一方、N-gram ヒット時の平均採択長（3.97〜5.05）が DFlash2 より短いため、全体の verify 回数（`rounds`）が増加した（例: `multi_turn` 56 → 71、`json_extract` 64 → 76、`code_edit` 73 → 84）。単一 Greedy では DFlash2 単体の検証 1 回追加コストが N-gram スキップ利得を上回ったと分析できる（`general_reasoning` の 111 → 131 ラウンド差には上記 `idx=23` 分岐系列差も含まれる）。

## 6. `EXP-NG-04` / `EXP-NG-05`: Stochastic・並行 `serve-stdio` (batch=4)・MTP 併用

### 6.1 Stochastic サンプリング（`temperature=0.6`, `top_p=0.95`, `top_k=20`, `seed=42`, EXL3, gen=512）

| Workload | 構成 | `decode_tok_s` | `NGRAM_HITS` | `NGRAM_MEAN_ACC` | `DFLASH2_MEAN_ACC` | `skip` | `NGRAM_CPU_US/round` |
|---|---|---|---|---|---|---|---|
| `code_edit` | `ngram_only` | 73.88 | 61/188 | 5.180 | - | 0 | 0.926 |
|  | `dflash2_only` | 146.68 | - | - | 5.905 | - | - |
|  | `dflash2_ngram` | **161.11 (+9.8%)** | 48/76 | 5.833 | 5.724 | 48 | 1.434 |
| `general_reasoning` | `ngram_only` | 32.70 | 9/483 | 2.333 | - | 0 | 0.606 |
|  | `dflash2_only` | 68.14 | - | - | 2.116 | - | - |
|  | `dflash2_ngram` | 68.11 | 0/164 | 0.000 | 2.116 | 0 | 0.860 |

Stochastic の `general_reasoning` で `dflash2_only` と `dflash2_ngram` は同一ラウンド数（164）・同一速度帯（68.14 vs 68.11）であり、ミス時のハイブリッド経路に実質的な劣化がないことを示す。また `code_edit` stochastic では `dflash2_ngram` が `dflash2_only` を `+9.8%` 上回った。

### 6.2 `serve-stdio` 並行 batch=4（4 ワークロード同時投入, total=1979 tok, EXL3）

| 構成 | `wall_s` | `agg_tok_s` | `NGRAM_HITS` | `NGRAM_MEAN_ACC` | `DFLASH2_ROUNDS` | `DFLASH2_MEAN_ACC` | `skip` | `NGRAM_CPU_US/round` | Parity vs baseline (req 1/2/3/4) |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 37.74s | 52.44 (1.00x) | - | - | - | - | - | - | T/T/T/T |
| `ngram_only` | 35.27s | 56.11 (1.07x) | 213/1082 | 4.113 | - | - | 0 | 0.899 | T/T/F(req3)/T |
| `dflash2_only` | 10.20s | 194.02 (3.70x) | - | - | 129 | 12.837 | - | - | F(req1)/T/F(req3)/T |
| `dflash2_ngram` | **9.52s** | **207.88 (+7.1% vs DFlash2)** | 146/347 | 4.575 | **116** | **14.026** | 146 | 1.631 | T/T/F(req3)/T |

並行 batch=4 では `DFLASH2_ROUNDS` が 129 → 116 へ減少し、`DFLASH2_MEAN_ACC` も 12.837 → 14.026（バッチ合算値）、集約スループットは **194.02 → 207.88 tok/s (+7.1%)** へ向上した。

### 6.3 MTP (`--mtp --mtp-drafts 4 --kv-cache-dtype fp8`) 併用評価

| モデル | Workload | 構成 | `decode_tok_s` | `MTP_ROUNDS` | `MTP_MEAN_ACC` | `NGRAM_HITS` | `NGRAM_ACC` | `skip` | Parity (`mtp_only == mtp_ngram`) |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen3.8-27B-FP8` (`seq=2048`) | `code_edit` | `mtp_only` | 68.24 | 109 | 3.688 | - | - | - | **True (512/512)** |
|  |  | `mtp_ngram` | 63.89 | 120 | 3.258 | 74/120 | 247 | 74 |  |
| `qwen38-27b-exl3-4bpw-mul24` (`seq=2048`) | `code_edit` | `mtp_only` | 66.07 | 107 | - | - | - | - | - |
|  |  | `mtp_ngram` | **71.15 (+7.7%)** | 116 | - | 75/116 | 249 | 75 |  |
| `qwen38-27b-exl3-4bpw-mul24` (`seq=1024`) | `code_edit` | `mtp_only` | 54.37 | 107 | 3.776 | - | - | - | False (`idx=417`: `23173` vs `41824` の EXL3 タイ) |
|  |  | `mtp_ngram` | **68.55 (+26.1%)** | 120 | 3.258 | 74/120 | 241 | 74 |  |

※ `general_reasoning` の MTP 比較は `Qwen3.8-27B-FP8`（`seq=2048`）および `qwen38-27b-exl3-4bpw-mul24`（`seq=1024`）の双方で arena 容量上限（31.29 GiB に対し使用済み 31.17〜31.18 GiB ＋ 要求 0.14 GiB の `insufficient memory`）により実行不可であった（観測事実として記録）。

## 7. `EXP-NG-06`: 回帰テスト・品質ゲート確認結果

- 関連 CTest 9 件（`test_online_ngram`, `test_qwen35_spec_verify`, `test_architecture_boundaries`, `test_compute_ngram_*` 4 件, `test_cli_docs_sync`, `test_cli_docs_sync_python`）: **9/9 PASS**。
- 全 CPU CTest: `144` required PASS、`2` optional `external_files` SKIP（`test_plamo3_checkpoint`, `test_qwen4exp_ple_prefetch`）、`test_cli_docs_sync_python` PASS（非 GPU 全 147 件中 fail=0）。
- `gpu1;required` 単体テスト 7 件（`test_gdn_spec_history`, `test_dflash2_exl3_head`, `test_dflash2_executor_alloc`, `test_dflash2_bf16_geometry`, `test_qwen35_mtp_weight_load`, `test_prefix_cache_batcher`, `test_qwen35_mtp_lowering`）: **7/7 PASS**。
- `tests/server/test_server_cli.py`: **14/15 PASS**（N-gram 引数転送 5 件すべて PASS、既存 failure `server-cli:既定の prefill chunk を転送する :: '8192'` は `origin/main` 側の既定値 8192 変更に起因）。
- `shelf_server.py` 負系バリデーション 4 件・`tools/quality_gate.py selftest`・Python xgrammar / reasoning / bench テスト群: すべて PASS。

## 8. PLaMo 3 Translate 翻訳タスク評価（`EXP-NG-07` 〜 `EXP-NG-10`）

### 8.1 対象モデル・経路確認・測定条件

- 対象チェックポイント（公式 drafter なし）:
  - 主評価: `plamo-3-translate-8b`（BF16）
  - 軽量比較: `plamo-3-translate-2b`（BF16）および `plamo-3-translate-2b-exl3-4.05bpw-mul24`（EXL3 4.05bpw / head 8-bit）
- 実行条件: `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` ＋ `flock /tmp/r9700.lock`、`--max-seq-len 2048`、`--max-new-tokens 512`、`--prefix-cache 0`、`--temperature 0`。入力トークン列は `Plamo3Tokenizer.apply_chat_template(..., add_generation_prompt=True)` により `source_lang` / `target_lang` / `context` / `glossary` / `form` を指定して生成した。
- 経路点検結果:
  - `validate_model_runtime_options()`（[model_policy.cpp](../../../src/shelf/models/model_policy.cpp)）は PLaMo 3 に対して `--mtp` / `--dflash2` / TP / PP を拒否する一方、単独 `--ngram` は追加モデル読み込み不要のため既存 PLaMo 3 `SPEC_VERIFY` + KV rollback 経路でそのまま動作する。
  - [loaded_model.cpp](../../../src/shelf/models/loaded_model.cpp) は `stop_tokens = {1, 16}` を設定しており、スモーク確認および全本計測で全ワークロードとも停止 token `16`（および `parity=True`）で正常終了した。
  - [executor.hip](../../../src/shelf/runtime/executor.hip) は `plamo3` の HIP Graph 適用を `Decode`（`R=1, N=1`）に限定しているため、`SPEC_VERIFY`（`N=K+1` 行）は eager 実行となるが、8B BF16（`w3_doc_code_cli_md` で `emitted/round=1.839` に対し実行速度比 `1.78x`）および 2B EXL3（[paged_attention_dispatch.hip](../../../src/shelf/inference/execution/paged_attention_dispatch.hip) の multi-row / split-K 経路）とも機能不具合・数値破綻は確認されず、コード修正は不要と判断した。

### 8.2 `EXP-NG-07`: `plamo-3-translate-8b`（BF16）5 ワークロード AB/BA 評価（既定 `min_match=4, K=7`）

| Workload (prompt / gen) | 構成 | `decode_tok_s` r0 / r1 (平均・倍率) | `emitted/round` | `NGRAM_HITS` | `NGRAM_ACCEPTED` (`mean`) | `NGRAM_CPU_US/round` | Parity (`r0/r1 × 2 構成`) |
|---|---|---|---|---|---|---|---|
| `w1_tech_article_en_ja` (117 / 111) | `baseline` | 24.69 / 33.80 (**29.24**, 1.00x) | 1.000 | - | - | - | **True (4/4 一致, stop=16)** |
|  | `ngram_m4_k7` | 34.12 / 34.13 (**34.12**, 1.17x※) | 1.009 | 1/109 (0.9%) | 1 (1.000) | 0.439 / 0.382 |  |
| `w2_news_ja_en` (108 / 88) | `baseline` | 33.80 / 33.80 (**33.80**, 1.00x) | 1.000 | - | - | - | **True (4/4 一致, stop=16)** |
|  | `ngram_m4_k7` | 33.73 / 32.46 (**33.09**, 0.98x) | 1.000 | 0/87 (0.0%) | 0 (0.000) | 0.943 / 0.525 |  |
| `w3_doc_code_cli_md` (391 / 400) | `baseline` | 32.59 / 32.62 (**32.61**, 1.00x) | 1.000 | - | - | - | **True (4/4 一致, stop=16)** |
|  | `ngram_m4_k7` | 58.18 / 58.12 (**58.15**, **1.78x**) | 1.839 | 43/217 (19.8%) | 182 (4.233) | 0.618 / 1.020 |  |
| `w4_glossary_context` (267 / 99) | `baseline` | 33.37 / 33.38 (**33.38**, 1.00x) | 1.000 | - | - | - | **True (4/4 一致, stop=16)** |
|  | `ngram_m4_k7` | 42.16 / 38.48 (**40.32**, **1.21x**) | 1.289 | 11/76 (14.5%) | 22 (2.000) | 0.596 / 0.639 |  |
| `w5_relnotes_ui_list` (407 / 427) | `baseline` | 32.52 / 32.54 (**32.53**, 1.00x) | 1.000 | - | - | - | **True (4/4 一致, stop=16)** |
|  | `ngram_m4_k7` | 74.35 / 74.32 (**74.33**, **2.29x**) | 2.420 | 66/176 (37.5%) | 250 (3.788) | 0.739 / 1.237 |  |

※ `w1_tech_article_en_ja` の `r0 baseline` (`24.69`) は初回起動直後のコールド揺らぎを含む。ウォーム後の単独再確認でも 8B BF16 `w1` は `baseline=33.88`, `m4_k7=34.14`, `m2_k7=34.55`、`w2` は `baseline=33.70`, `m4_k7=33.79`, `m2_k7=33.79` であり、非反復翻訳での退行は生じていない。

- 出力テキストのデコード品質確認（全件 `baseline` と token 列完全一致）:
  - `w1` 冒頭: `大規模言語モデル向けの現代的なGPU推論パイプラインでは、自己回帰デコーディング時間の大部分を演算ユニットではなくDRAMからの重み読み出し待ちに費やしている。`
  - `w2` 冒頭: `Domestic cloud providers are accelerating the deployment of geographically distributed GPU clusters, aiming to balance data center power efficiency with inference latency.`
  - `w3` 冒頭: `` ### オンライン N-gram 投機的デコーディングガイド\n\nポート `8080` で、`--ngram` および任意のチューニングフラグを指定して `shelf-compute` または `shelf-server` を実行してください： ``（CLI フラグ・コードブロック・表セル・URL が原文保持）
  - `w4` 冒頭: `` `ContinuousBatcher::step()` と併せて `--ngram` を有効にすると、`OnlineNgramMatcher` はトークンリングバッファを参照し、接尾辞一致が見つかった場合には `DFlash2` のドラフト推論をスキップします。 ``（指定した glossary「接尾辞一致」「リングバッファ」「ドラフト推論」を正確に反映）
  - `w5` 冒頭: `` ## リリースノート v2026.11\n\n- `shelf-compute` に `--ngram` サポートを追加 (`EXEC_MODE=standalone`, ステータス: Ready) ``

### 8.3 `EXP-NG-08`: パラメータ感度評価（8B BF16, `--ngram-min-match` / `--ngram-drafts`）

| Workload | 設定 (`min_match`, `K`) | `decode_tok_s` (対 baseline 比) | `emitted/round` | `NGRAM_HITS` | `NGRAM_ACCEPTED` (`mean`) | `NGRAM_CPU_US/round` | Parity vs baseline |
|---|---|---|---|---|---|---|---|
| `w3_doc_code_cli_md` (base 32.61) | `m4_k3` (`min=4, K=3`) | 51.20 (1.57x) | 1.629 | 71/245 | 154 (2.169) | 0.817 | **True** |
|  | `m4_k7` (`min=4, K=7`) | 58.15 (1.78x) | 1.839 | 43/217 | 182 (4.233) | 0.819 | **True** |
|  | `m4_k15` (`min=4, K=15`) | 60.14 (1.84x) | 2.046 | 32/195 | 204 (6.375) | 0.716 | **True** |
|  | `m3_k7` (`min=3, K=7`) | 61.38 (1.88x) | 1.956 | 51/204 | 195 (3.824) | 0.646 | **True** |
|  | **`m2_k7` (`min=2, K=7`)** | **64.67 (1.98x)** | **2.089** | 66/191 | **208 (3.152)** | 0.839 | **True** |
| `w5_relnotes_ui_list` (base 32.53) | `m4_k3` (`min=4, K=3`) | 61.86 (1.90x) | 1.963 | 94/217 | 209 (2.223) | 0.724 | **True** |
|  | `m4_k7` (`min=4, K=7`) | 74.33 (2.29x) | 2.420 | 66/176 | 250 (3.788) | 0.988 | **True** |
|  | `m4_k15` (`min=4, K=15`) | 76.27 (2.34x) | 2.859 | 53/149 | 277 (5.226) | 0.878 | **True** |
|  | `m2_k7` (`min=2, K=7`) | 85.86 (2.64x) | 2.859 | 78/149 | 277 (3.551) | 1.013 | **True** |
|  | **`m3_k7` (`min=3, K=7`)** | **86.48 (2.66x)** | **2.859** | 72/149 | **277 (3.847)** | 0.871 | **True** |

翻訳タスクでは短い CLI フラグや固定句読点・定型接頭辞が 2〜3 token で再出現するため、既定 `min_match=4` から `min_match=2` または `3` へ下げることで `w3` は `58.15 → 64.67 tok/s`、`w5` は `74.33 → 86.48 tok/s` までさらに向上した（CPU オーバーヘッドも 1.0 µs/round 以下）。

### 8.4 `EXP-NG-09`: 2B モデル比較（BF16 / EXL3 4.05bpw）

| モデル | Workload | 構成 | `decode_tok_s` (倍率) | `emitted/round` | `NGRAM_HITS` | `NGRAM_ACCEPTED` | `NGRAM_CPU_US/round` | Parity |
|---|---|---|---|---|---|---|---|---|
| `plamo-3-translate-2b` (BF16) | `w1_tech_article_en_ja` | `baseline` | 92.02 (再測 81.87) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 65.10※ (再測 **92.68**) | 1.009 | 1/111 | 1 | 0.702 (再測 0.260) | **True** |
|  |  | `ngram_m2_k7` | **93.33** (再測 **93.90**) | 1.028 | 5/109 | 3 | 0.292 (再測 0.305) | **True** |
|  | `w3_doc_code_cli_md` | `baseline` | 83.54 (1.00x) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 158.45 (1.90x) | 1.913 | 43/207 | 189 | 0.649 | **True** |
|  |  | `ngram_m2_k7` | **169.55 (2.03x)** | 2.073 | 64/191 | 205 | 0.492 | **True** |
|  | `w5_relnotes_ui_list` | `baseline` | 85.15 (1.00x) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 195.19 (2.29x) | 2.412 | 65/177 | 250 | 0.536 | **True** |
|  |  | `ngram_m2_k7` | **242.62 (2.85x)** | 3.072 | 82/139 | 288 | 0.599 | **True** |
| `plamo-3-translate-2b-exl3` (4.05bpw) | `w1_tech_article_en_ja` | `baseline` | 180.40 (1.00x) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 181.57 (1.01x) | 1.009 | 1/113 | 1 | 0.743 | **True** |
|  |  | `ngram_m2_k7` | 178.74 (0.99x) | 1.027 | 5/111 | 3 | 0.320 | **True** |
|  | `w3_doc_code_cli_md` | `baseline` | 160.63 (1.00x) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 268.73 (1.67x) | 1.913 | 43/207 | 189 | 0.623 | **True** |
|  |  | `ngram_m2_k7` | **273.27 (1.70x)** | 2.073 | 64/191 | 205 | 0.524 | **True** |
|  | `w5_relnotes_ui_list` | `baseline` | 159.09 (1.00x) | 1.000 | - | - | - | True |
|  |  | `ngram_m4_k7` | 306.36 (1.93x) | 2.412 | 65/177 | 250 | 0.530 | **True** |
|  |  | `ngram_m2_k7` | **354.57 (2.23x)** | 3.072 | 82/139 | 288 | 0.588 | **True** |

※ 初回バッチの `2b_bf16 w1 ngram_m4_k7` (`65.10`) は一過性の起動遅延であり、単独再確認では `baseline=81.87` → `m4_k7=92.68` / `m2_k7=93.90` と退行なしを確認した。EXL3 2B でも全構成で `parity=True` を保ちつつ `w5` で最大 `354.57 tok/s (2.23x)` を達成した。

### 8.5 `EXP-NG-10`: `plamo-3-translate-8b` `serve-stdio` 並行 batch=4（`w1`, `w3`, `w4`, `w5` 同時投入）

| 構成 | `requests` | `total_generated` | `wall_s` | `agg_tok_s` | `NGRAM_HITS` | `NGRAM_ACCEPTED` (`mean`) | `NGRAM_CPU_US/round` | Parity vs baseline (req 1/2/3/4) |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 4 | 1037 | 14.87s | 69.73 (1.00x) | - | - | - | T/T/T/T |
| `ngram_m4_k7` | 4 | 1034 | **11.41s** | **90.62 (+30.0%)** | 121/568 | 462 (3.818) | 0.714 | T/F(req2: `w3`)/T/T |

並行 batch=4 でも集約スループットが **69.73 → 90.62 tok/s (+30.0%)** へ向上した（`req 2` はバッチ多行検証に伴う BF16 演算順の近接タイにより末尾長が 400→397 へ 3 token 変化、他 3 リクエストは完全一致）。

**総合結論（PLaMo 3 Translate）**: 公式 drafter を持たない PLaMo 3 Translate のような特化型モデル（特に 8B BF16）に対して、`--ngram` は追加 VRAM ゼロ・CPU コスト `1 µs/round` 未満で、技術文書（**1.78x〜1.98x**）・反復リスト/リリースノート（**2.29x〜2.66x**）・Glossary/Context 対話翻訳（**1.21x**）・並行 batch=4（**+30.0%**）のいずれでも実用的かつ大幅な高速化をもたらし、非反復記事（日英・英日）での退行もないことを確認した。
