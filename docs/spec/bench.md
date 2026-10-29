# shelf-bench 仕様

対応実装: `apps/bench/`（ターゲット `shelf-bench`。共有ヘルパは `apps/common/`）。
運用手順は `docs/guide/bench.md`。計画上の位置づけ（フェーズ 0 の計測手段）は
`draft_docs/directory_design.md` §5。

---

## 1. 概要

**モデルの速度だけを測る** CLI。入力は合成（トークン列・バッファ）で、**答えは生成・比較しない**。

| 項目 | 状態 |
| --- | --- |
| `pp`（prefill） / `tg`（decode）の壁時計計測・統計・CSV | 対応 |
| `kernel <name>` の微計測（イベント、5 種） | 対応 |
| `--arena-mib 0`（自動算出）と実効値の出力 | 対応 |
| GPU event による `gpu_ms_*` | **未対応**（`wall_*` のみ） |
| `--device cpu` | **未対応**（HIP 専用） |
| JSON 出力 / 複数 GPU・TP / roctx / `--gap-keepalive` | **未対応** |

* 正しさ（parity・決定性）は `shelf-validate`、このツールは計測のみ
* 参考: `phaseshift-bench`（CLI 語法・終了コード・集計規約・CSV 追記の様式の出所）

---

## 2. ビルドと実行形態

```bash
cmake --build --preset hip-rel  # → build/hip_rel/apps/shelf-bench（**計測はこれ**）
cmake --build --preset hip      # → build/hip/apps/shelf-bench（Debug・動作確認用）
cmake --build --preset cpu      # → build/cpu/apps/shelf-bench（--help のみ）
```

* 初回は `cmake --preset <name>` で設定する。**性能値はすべて `hip-rel`（Release）で測定する**
  （2026-10-28 決定。Debug は桁違いに遅く性能値にならない）

* ソース: `apps/bench/{main,bench_common,pp,tg,kernel}.cpp`（+ `bench_common.hpp`）。
  モデル読取・重みアップロードは `apps/common/app_support.*` を `shelf-run` と共有
* `SHELF_ENABLE_HIP` で `SHELF_HAVE_HIP` が定義され、本体の経路が有効になる。
  **HIP 非対応ビルドでは `--help` / 起動時検査のみ動作し、本体は exit 1**
* `kernel` の計測は bench 自前の HIP ストリームを使う（実行器を経由しない）

```bash
./build/hip/apps/shelf-bench pp --model-dir <dir> [--prompt-tokens 128] ...
```

---

## 3. CLI

値の形式は `--key value` と `--key=value` の両対応。`--help` / `-h` は**値を取らないフラグ**。
未知のオプション・値の欠落・余分な位置引数は起動エラー。

```text
usage: shelf-bench <command> [options]

commands:
  pp                 prompt processing (prefill) を測る
  tg                 token generation (decode) を測る
  kernel <name>      1 つのカーネルを測る（名称一覧は kernel --help）
```

```text
usage: shelf-bench pp --model-dir DIR [options]

options:
  --model-dir DIR     モデルディレクトリ（--model の別名・必須）
  --prompt-tokens N   供給するトークン数（既定 128）
  --prefill-chunk N   1 ステップの行数（既定 0 = min(prompt, 64)）
  --runs N            計測回数（既定 3。中央値を主指標にする）
  --warmup N          計測前に破棄する回数（既定 0）
  --gpu N             GPU 番号（既定 0）
  --seed N            合成トークン列のシード（既定 1234）
  --arena-mib N       アリーナサイズ MiB（既定 0 = グラフから自動算出、正の値で固定）
  --output PATH       CSV を追記（ファイルが無ければヘッダを書く）
  --help, -h          このヘルプ（GPU を触らず exit 0）
```

```text
usage: shelf-bench tg --model-dir DIR [options]

  --model-dir DIR     モデルディレクトリ（--model の別名・必須）
  --context N         prefill するコンテキスト長（既定 128）
  --tokens N          測定するデコード歩数（既定 8）
  --prefill-chunk N   prefill の 1 ステップの行数（既定 0 = min(context, 64)）
  --warmup N          測定前に破棄するデコード歩数（既定 0）
  --mode M            forward（既定・logits を取らない）| greedy（毎歩 argmax）
  --gpu N             GPU 番号（既定 0）
  --seed N            合成トークン列のシード（既定 1234）
  --arena-mib N       アリーナサイズ MiB（既定 0 = 自動算出、正の値で固定）
  --output PATH       CSV を追記
  --help, -h          このヘルプ（GPU を触らず exit 0）
```

```text
usage: shelf-bench kernel <name> [options]

names: gemm / rmsnorm / elementwise / kv-append / paged-attention（kernel --help に出る）

common options:
  --warmup N       破棄する発射回数（gemm/rmsnorm/elementwise 既定 100、KV 系 既定 30）
  --samples N      サンプル数（既定 50）
  --launches N     1 サンプル内の発射回数（既定 20）
  --gpu N          GPU 番号（既定 0）
  --output PATH    CSV を追記
  --help, -h       このヘルプ（GPU を触らず exit 0）
```

### 3.1 起動時検査（exit 2 で拒否）

| 条件 | 制約 |
| --- | --- |
| コマンド | `pp` / `tg` / `kernel` のいずれか（未知は exit 2） |
| `--model-dir` | `pp` / `tg` で必須（`kernel` では不要） |
| `pp`: `--prompt-tokens` / `tg`: `--context`, `--tokens` | `> 0` |
| `--prefill-chunk` | `1 ≤ N ≤ kMaxStepRows（64）` |
| `--runs`（pp） | `≥ 1` |
| `--warmup` / `--arena-mib` | `≥ 0`（`--arena-mib 0` = 自動算出） |
| `--gpu` / `--seed` | `≥ 0` |
| `tg`: `--mode` | `forward` / `greedy` |
| `kernel <name>` | §6 の 5 種のみ |
| 位置引数 | 各コマンドとも余分な位置引数は拒否 |
| 未知の `--key` | そのコマンドの既知キー集合（§3 の usage と `kernel` の名前別一覧）に無いものは exit 2 |

実行時の検査（デバイス・ロード・カーネル発射・アリーナ枯渇）は **exit 1**。

---

## 4. 計測方法

### 4.1 pp / tg（壁時計）

* **計測は壁時計時間のみ**。実行器が毎ステップ末で stream を同期するため GPU 時間とほぼ等価。
  GPU event 計測は未実装（`stream_` が private）
* 区間の外: モデル読取・重みアップロード・`declare_states`・グラフ構築・
  （tg）コンテキスト prefill・（各コマンドの）warmup
* `pp`: `reset_states()` → チャンクループを 1 区間として測る。**各 run を独立させる**
  （行数と KV 使用行を詰め直す）
* `tg`: コンテキスト prefill（計測外）→ `--warmup` ステップを破棄 → `--tokens` ステップを 1 区間として測い、
  1 ステップごとの `last_step_ms` も取得する
* `--mode greedy` のみ、区間に logits の D2H と argmax が入る

### 4.2 kernel（イベント）

規約は `phaseshift-bench` と同じ:

1. 自前の `hipStreamNonBlocking` とイベント 2 点を作る
2. `--warmup` 回発射して破棄し、`hipStreamSynchronize`
3. `--samples` 回、各サンプルで **イベント0 → `--launches` 回発射 → イベント1 →
   `hipEventSynchronize` → `hipEventElapsedTime / launches`** を記録
4. µs/launch の統計（§8）を返す

### 4.3 統計の定義

`summarize(xs)` はソート後:

* `min` / `max` / `mean`（算術平均）/ `stddev`（**母標準偏差**）
* percentile は近似ランク: `xs[round(q * (n - 1))]`（`p10` / `p50` / `p90` / `p95`）
* 入力が空なら全 0

---

## 5. 入力（合成）

* `pp` / `tg` のトークン列: LCG（`seed` 既定 1234、`x = x*6364136223846793005 + 1442695040888963407`、
  上位ビット）で `vocab_size` 未満にクランプ。**内容は不問**（計測対象はカーネル経路）
* `kernel` のバッファ: `hipMemset` による **0 埋め**。値はタイミングに影響せず NaN も生まない
  （rmsnorm の全ゼロ入力、attention の全ゼロ q は有限値で終わる）
* KV fixture（`kv-append` / `paged-attention`）: ブロック表は恒等（0,1,2,…）、
  `row_sequence_slots` は全 0（シーケンス 0）、`row_positions` は `seq_len - 1`

---

## 6. カーネル一覧と契約

| name | 測るもの | 固有オプション（既定） | 制約 | throughput |
| --- | --- | --- | --- | --- |
| `gemm` | `launch_gemm_bf16`（**生産と同じ設定選択**: rows 1 = `ExactRows`、rows ≥ 2 = `ExactRowsDot2`） | `--rows`（`1,16`）、`--n`（2048）、`--k`（2048） | `1 ≤ rows ≤ 64`、`k % 8 == 0`、`n,k > 0` | `2*rows*n*k / p50` → GFLOP/s |
| `rmsnorm` | `launch_rmsnorm`（生産と同じ: bf16 入出力・`PER_FEATURE`・`DIRECT`、group = features、eps = 1e-5） | `--rows`（1024）、`--features`（2560） | ともに `≥ 1` | `2*入力 + 重み` → GB/s |
| `elementwise` | `launch_residual_add_bf16` | `--rows`（1024）、`--features`（2560） | ともに `≥ 1` | `入力2 + 出力1` → GB/s |
| `kv-append` | `launch_kv_append_bf16` | `--rows`(1), `--kv-heads`(8), `--head-dim`(256), `--page-tokens`(16), `--num-pages`(4), `--seq-len`(16) | すべて `≥ 1`、`seq_len ≤ page_tokens*num_pages` | `K/V の読取+書出` → GB/s |
| `paged-attention` | `launch_attention_paged_bf16` | 上記 + `--q-heads`(32) | 同上 + `head_dim ≤ 256`（kPagedAttentionHeadLimit）、`q_heads % kv_heads == 0` | 空（読み出し量は可視長に依存するため指標化しない） |

* いずれも**モデル不要**。ブロック表の幅は `ceil(seq_len / page_tokens)`、行数は `--rows`
* 主指標は `us_p50`

---

## 7. 実行処理

* **pp**: config 読み → チャンク決定（既定 `min(prompt, 64)`）→ グラフ（`rows = chunk`）→
  shard 開放 → 実行器構築 → 重みアップロード + `declare_states`（**アリーナ自動算出はここで確定**）→
  合成トークン生成 → warmup → `runs` 回（各回: `reset_states` → チャンクループ）→ 統計
* **tg**: config 読み → decode グラフ（`rows = 1`）と prefill グラフ（`rows = chunk`、`chunk = 1` なら不要）→
  **decode プランの weights でアップロード**（weight id は config 由来で両プラン共通）→
  コンテキスト prefill（計測外）→ warmup → 測定区間 → 統計
* **kernel**: デバイス選択 → 合成バッファ確保 → 自前 stream で §4.2 の計測 → 1 シェイプずつ出力
* エラー時: `run_step` が未消化のカーネルをドレインしてから返るため、呼び出し側の解放は安全
  （`hint` 行は失敗時点の**実効アリーナ**を表示）

---

## 8. 指標

| 指標 | 定義 |
| --- | --- |
| `wall_ms_median` / `_min` / `_p10` / `_p90` / `_stddev` | `pp` の run 別壁時計時間の統計（主指標は median） |
| `run_wall_ms[i]` | `pp` の個票 |
| `wall_us_per_token` | `pp`: `median_ms * 1000 / prompt_tokens`、`tg`: `wall_ms_total * 1000 / tokens` |
| `wall_tokens_per_sec` | `1e6 / wall_us_per_token` |
| `wall_ms_total` | `tg` の測定区間（`--tokens` ステップ）合計 |
| `first_token_ms` | `tg` の最初の 1 ステップの `last_step_ms`（`step_once` の結果） |
| `step_ms_{mean,p50,p90,max}` | `tg` の 1 ステップごとの `last_step_ms` 統計 |
| `context_prefill_ms` | `tg` のコンテキスト prefill（**計測区間外・参考値**） |
| `us_min` / `us_p10` / `us_p50` / `us_p90` / `us_p95` / `us_mean` / `us_stddev` | `kernel` の µs/launch 統計（主指標は p50） |
| `throughput` / `throughput_unit` | §6 の式による p50 ベース換算（`GFLOP/s` / `GB/s` / 空） |
| `arena_mib` | **実効**アリーナ（MiB）。`pp`/`tg` のみ（自動拡張後の値） |

---

## 9. 終了コード

| コード | 条件 |
| --- | --- |
| `0` | 成功、または `--help` / `-h` |
| `1` | 実行時エラー（デバイス・設定・ロード・カーネル発射・アリーナ枯渇・HIP 非対応ビルドの本体） |
| `2` | 用法の誤り（§3.1 のいずれか、および未知コマンド / 未知カーネル名） |

`--help` は**装置初期化とモデルロードより前**に処理される（`phaseshift-bench` と同一契約）。

---

## 10. 出力

* **stdout**: `key=value` の逐次。先頭は `shelf-bench <command>`、次が `schema_version`
* **CSV**: `--output PATH` のみ。追記、ヘッダはファイルが無い・空のときだけ。

| コマンド | schema_version | 列 |
| --- | --- | --- |
| `pp` | 1 | `schema_version,device,gpu,model,prompt_tokens,prefill_chunk,runs,warmup,arena_mib,wall_ms_median,wall_ms_min,wall_ms_p10,wall_ms_p90,wall_ms_stddev,wall_us_per_token,wall_tokens_per_sec` |
| `tg` | 2 | `schema_version,device,gpu,model,context,tokens,prefill_chunk,warmup,arena_mib,mode,context_prefill_ms,wall_ms_total,wall_us_per_token,wall_tokens_per_sec,first_token_ms,step_ms_mean,step_ms_p50,step_ms_p90,step_ms_max` |
| `kernel` | 3 | `schema_version,device,gpu,kernel,shape,warmup,samples,launches,us_min,us_p10,us_p50,us_p90,us_p95,us_mean,us_stddev,throughput,throughput_unit` |

* **schema_version が違うため、3 コマンドの行を同じファイルに混ぜない**（1 コマンド = 1 行、
  `gemm` は 1 シェイプ = 1 行）
* `model` はディレクトリ文字列、`shape` はカンマを持たない `AxBxC` 形（CSV を壊さないため）

---

## 11. リソース仕様

### 11.1 アリーナ（HIP）

* `HipStepExecutorOptions::arena_bytes` の既定 **`0` = 自動算出**。
  `declare_states(g)` で **`64 MiB + 2 × Σ（全 primitive 出力の要素数 × sizeof(float)）`** を確保する
  （バンプ割当でステップ内再利用が無いため、必要量は行数に比例する）。
  式は `shelf-validate` と同一（`apps/shelf_validate.cpp` の `arena_bytes_for`）
* **複数グラフ対応**: 必要量が declare 時を超えていれば `run_step` の**開始前**に自動で拡張する
  （縮めない）。`pp` は prefill グラフで declare するため通常は拡張されない。
  `tg` は decode グラフ（rows=1）で declare し、コンテキスト prefill で拡張される
* 指定値（`arena_bytes > 0` / `--arena-mib N>0`）を使うとそれが**上限**になり、枯渇時は
  `insufficient_memory`（`arena exhausted`）を返す（null 出力でのカーネル発射は行わない）。
  実効値は `HipStepExecutor::arena_bytes()` / 出力の `arena_mib`

### 11.2 実測した必要量（開発モデル `Qwen3.5-9B`、32 層、vocab 248,320・**Release**）

> 2026-10-28、R9700 / gfx1201、`hip-rel`（Release）で計測。
> 計測: `shelf-bench pp --model-dir …/Qwen3.5-9B --prompt-tokens 64 --prefill-chunk <rows>
> --runs 1 --warmup 0 --gpu <n> [--arena-mib N]`（最小必要は 64 MiB 刻みで二分探索）。

| prefill 行数 | 最小必要（二分探索） | 自動算出の実効値 | 従来の既定 64 MiB |
| --- | --- | --- | --- |
| 1（decode） | ≤ 64 MiB | 98 MiB | ○ |
| 16 | 193〜256 MiB | 622 MiB | × |
| 32 | 385〜448 MiB | 1,181 MiB | × |
| 64 | **777〜784 MiB** | **2,298 MiB** | × |

* 自動算出は最小必要に対して、行数とともに **約 1.5 → 2.4 → 2.6 → 2.9 倍**と膨らむ（過大確保）。
  予算計画（`draft_docs/memory_budget.md` / `directory_design.md` D2）と整合させることが残課題。
  根本の削減は D4（declare/run 二相でバッファ寿命の共有）
* 従来の既定 64 MiB では **`shelf-run --device hip` の 16 トークン prefill が枯渇した**
  （自動算出と `--arena-mib` 追加で解消）
* 渡すプランで自動算出は 1 MiB ずれる: `shelf-bench pp` の rows=64 グラフは **2,298 MiB**、
  `shelf-validate` の prefill グラフは **2,299 MiB**（kv_capacity の差。出力例は
  `docs/guide/bench.md` §3 に別掲）
* **参考（旧 `Qwen3.5-4B`・履歴・Debug ビルド）**: 最小必要 1: ≤64 / 16: 129〜192 /
  32: 257〜384 / 64: 641〜768 MiB、自動算出 93 / 538 / 1012 / 1961 MiB（約 2.5 倍）

### 11.3 容量の目安（開発モデル `Qwen3.5-9B` bf16、1 GPU）

| 項目 | 目安 |
| --- | --- |
| 重み payload（mtp を含む・ディスク） | 19.31 GB（safetensors ヘッダ集計の実測） |
| うちアップロード対象（`language_model.*` ＋ `lm_head`） | 17.91 GB（`visual` 0.91 GB と `mtp` 0.49 GB は読まない） |
| アリーナ（rows=64、自動算出） | **2,298 MiB**（pp グラフ・実測／validate グラフは 2,299 MiB）／最小必要 **777〜784 MiB**（§11.2） |
| VRAM 常駐合計（重み + KV 状態 + arena、`free` 差分） | **20.56 GB（実測 2026-10-28）** |
| KV（既定 context 128、single sequence） | 数十 MB 程度（1 tok = 32 KiB） |
| approx `Qwen3.5-9B-FP8`（FP8 W8A8）の重み | ディスク 14.01 GB（単一 13.52 GB ＋ `mtp` 0.49 GB）／うち `language_model` ＋ `lm_head` = 12.61 GB（bf16 5.27 GB ＋ `F8_E4M3` 5.30 GB）。**VRAM 常駐は未計測** |

* GPU が占有されていると重みアップロードが `out of memory` で失敗するため
  `--gpu` で空きを選ぶ（`rocm-smi`）
* 実走の実測（prefill / decode / stop）は `draft_docs/shelf_validate.md` §7「V0 実績（9B）」

---

## 12. 現行の対応外（計画のみ）

| 項目 | 内容 |
| --- | --- |
| GPU event 計測 | `gpu_ms_*` 列（`HipStepExecutor` にイベント計測を追加する必要がある） |
| `--device cpu` | CPU 経路は `shelf-run` / `shelf-validate` 側に維持 |
| JSON 出力 | 既存は stdout + CSV のみ |
| 複数 GPU / TP / `--model-host` / roctx / `--gap-keepalive` | `phaseshift-bench` にあるもの。要件が出た時点で追加 |
| 生成結果の検証 | 本ツールの範囲外（`docs/guide/validate.md`） |

---

## 13. 調査記録（既知の観測）

実装時に調査した 2 件の記録（いずれも対応済み・残課題を明記）。

### 13.1 アリーナ枯渇（対応済み: 自動算出 + 拡張 + `--arena-mib`）

* 症状: 従来既定 64 MiB で `prefill chunk 0: ... arena exhausted`（`shelf-run` 16 トークンで再現）
* 原因: ステップ内で再利用しないバンプ割当 × 固定 64 MiB。899 primitives の出力と
  一時バッファ（`rope` の f32 展開、`gdn` の prep、norm 変換）を全部保持する
* 対応: §11.1。残課題は過大確保（×2 余白）の是正 = D4/D2 と統合

### 13.2 OOM 経路の GPU メモリフォルト（防御は実装、根拠は未確定）

* 観測: アリーナ枯渇の直後に `HSA_STATUS_ERROR_MEMORY_FAULT`
  （`gemm_bf16_exact_rows_dot2_kernel`、`grid=[32768,1,1]` → N=262,144）とハング
* **コード監査**: アリーナ割当は 14 箇所すべて発射前に検査済み（null 出力への発射は構造的になし）
* **再現**: 早い失敗（64 MiB、warmup/runs あり）と遅い失敗（640 MiB）の 2 条件で**再現せず**
* **矛盾**: 観測グリッドの N はこのモデルのどの GEMM 幅とも不一致
  （vocab 248,320 → grid 31,040。262,144 は `max_position_embeddings`）。同時刻に他ジョブが
  同一 GPU で動作していたため**他プロセス起因の可能性が高い**が断定できず
* **防御として実装済み**: `run_step` のエラー経路でキューをドレイン、`~HipStepExecutor` の先頭で
  stream 同期、失敗時に `logits_dev_` を無効化。`--arena-mib 1` でエラー + ヒントが出ることと
  破棄が安全なことを確認済み

### 13.3 既存の nodiscard 警告

* `step_executor.cpp` の既存行（`hipSetDevice` / `hipMemset` / デストラクタの `hipFree` 等）に
  **28 箇所**の `ignoring return value ... nodiscard` がある（フル再コンパイル時に出現）。
  今回の追加行は `(void)` か代入で回避済み。既存行は未修正（別作業）
