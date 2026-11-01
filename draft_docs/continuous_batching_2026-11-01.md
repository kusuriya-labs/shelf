# Continuous Batching（T15）の実装記録 — スケジューラと実行器のバッチ化

日付: 2026-11-01
対象: 最大 4 スロットの連続バッチング。スロットごとの状態・ブロック表・
step のバッチ化・終了と再利用。

---

## 1. 実装済み

### 1.1 スケジューラ（純粋・GPU 非依存）

`src/runtime/generate/batch_scheduler.{hpp,cpp}`。

* `BatchScheduler(max_slots=4, prefill_chunk)`。受付は空きまで即時、溢れは FIFO の
  queued。終了で空いたスロットへ投入順に詰める。
* Prefill は `prefill_chunk` 行ずつに分割（`PrefillAction{slot, begin, rows, base_pos,
  tokens}`）。チャンクの間でも cancel できる。
* Decode は「1 スロット 1 行」を 1 ステップにまとめる
  （`DecodeAction{rows:[{slot, position, token}]}`、スロット番号昇順）。
* 終了は `SlotEnd{stop, length, cancelled, error}`。EOS は生成列に載せない。
  `take_finished` で取り出した時点でスロットを free に戻し queued を詰める。
* cancel は実行中・queued の両方。queued の cancel も終了イベントとして出る。
* 検証: `unit.runtime_batch_scheduler`（受付・チャンク・2/4 行バッチ・EOS/length・
  再利用の FIFO・cancel・4 並列）。

### 1.2 実行器のスロット状態と行メタデータ

`HipStepExecutorOptions::max_sequences`（既定 1）と
`HipStepExecutor::run_batch(g, tokens, n, BatchRows{slots, positions})`。

* **KV**: `pages × max_sequences` の global page プール。ブロック表は
  `slot * pages + 論理 page`（identity）で `max_sequences × pages` エントリ。
  `block_table_stride = pages`（1 シーケンス分）、`num_pages = pages × max_sequences`
  （カーネルはブロック表の値をこの値で bound する）。
* **conv / recurrent（GDN）**: 1 スロット分の要素数を `max_sequences` 倍して
  スロット先頭オフセットで並べる。GDN のリクエスト記述子
  （`GdnRequestDescriptor{slot, row_begin, row_count}`）を、行の slot が
  連続する区間ごとに作ってデバイスへ送る。
* `row_sequence_slots` / `row_positions` を行ごとにアップロード。
  KV 容量チェックはバッチ内の最大位置 + 1 で行う。
* **attention の経路選択**: prefill 用タイルカーネルは 1 タイル（32 行）が
  1 スロットを共有する（`row_sequence_slots[row0]` をタイル全体で使う）。
  複数スロットを混ぜるステップでは **decode 用カーネル**（行ごとに slot / 位置 /
  ブロック表を引く）を強制する。単一スロットの prefill は従来のタイル経路のまま。

### 1.3 検証（合成グラフ・2 スロット）

`unit.hip_step_executor` に追加。同じモデル（tiny）で

1. 系列 A = `[7, 111]`、系列 B = `[512, 900]` をそれぞれ単独で decode
2. リセット後、1 ステップで rows=4・slots=`[0,0,1,1]`・positions=`[0,1,0,1]`
3. バッチの行 1（A の最終行）と行 3（B の最終行）を単独実行の logits と比較

結果（全層 full_attention / 全層 linear_attention の両方）:

| 層構成 | slot0 の差 | slot1 の差 |
| --- | --- | --- |
| full attention | 1.19e-07 | 2.38e-07 |
| linear attention（GDN） | 2.38e-07 | 2.38e-07 |

**スロットごとの KV・GDN 状態が分離され、バッチ結果が単独実行と一致**することを確認。
既存の「batch == seq」検証（同一系列 4 行）も通る。

### 1.4 パイプライン・エンジン・Server への結線（2026-11-01 続き）

* `runtime::pipeline::StageGraphs` に `decode_batch`（rows=2..N）を追加し、
  `step_batch_prefill` / `step_batch_decode` を実装。PP では各ステージへ同じ
  行メタデータを渡す。
* `app::make_pipeline`: `PipelineOptions::max_sequences` に応じて rows=2..N の
  decode グラフを作り（logits は全行）、`StageGraphs` へ配線。handoff バッファは
  `max(chunk_rows, max_sequences)` 行に広げた。
* `HipStepExecutor::reset_slot(slot)`: 新しい要求をスロットへ入れるとき、その
  スロットの KV ページ・conv/recurrent・アリーナ scratch を 0 に戻す。
  `reset_states` もアリーナを 0 に戻すようにした（従来はフラグだけで未使用だった）。
* `shelf-compute`: 直列 `handle` を **スケジューラループ**に置き換え。
  受付 → `next_prefill`（チャンク）→ `next_decode`（バッチ）→ `take_finished`
  の順で回し、cancel は `request_id` でスケジューラへ。`--max-batch N`（既定 4）。
* `apps/server`: `EngineClient(max_inflight=4)` と `--max-inflight`（既定 4）で
  HTTP の同時要求をエンジンへ流す。

### 1.5 検証（実モデル・同時要求）

| 検証 | 結果 |
| --- | --- |
| JSONL: 4 リクエスト同時 vs 直列（9B-FP8、長いプロンプト） | 生成列一致 |
| JSONL: 同一プロンプト 2 本同時 vs 直列（9B bf16） | 一致 |
| JSONL: 5 本同時（4 スロット + queued 1） | 5 本とも length で完了（queued が再利用で入る） |
| JSONL: 同時 2 本のうち 1 本を cancel | cancel 側 `cancelled`、他方 `length` で完走 |
| HTTP: 4 並列 vs 直列（9B bf16 / 9B-FP8） | content / finish / usage が一致 |
| HTTP 単独 E2E（9B-FP8 / 27B-FP8） | 全項目 PASS（cancel・切断を含む） |

### 1.6 分かったこと: FP8 のバッチサイズ依存

* **bf16 9B ではバッチと単独が一致**（合成グラフ・実モデルとも）。
* **FP8 9B では rows>=2 の decode が rows=1 と異なる GEMM 経路**（decode1 の
  GEMV ではなく RowBlock/2D 系）になり、logits が ~0.1〜0.2 ずれる。
  そのため **top1/top2 の margin が小さいプロンプト**では生成列が変わる
  （例: `[88,11,264]` vs `[88,11,31336]`。単独時の margin 1.7e-2）。
  長いプロンプト（margin が大きい）では一致した。
* これは T08 の parity 方針でいう T2（丸め・経路差）に該当するが、
  **同じ要求の出力が同時実行の有無で変わりうる**ため、サービスとしては
  避けたい。次の対応案:
  1. rows<=4 の decode で FP8/MXFP4 の linear を **行ごとに decode1 GEMV** で
     発射する（`linear` の multi-slot 分岐。バッチの旨味は attention/GDN 側に残る）
  2. rows>=2 でも decode1 と同一の累積順を持つ GEMV を用意する
  まず 1 を実装して、bf16 と同じ「バッチ不変」を確認する。

---

## 2. 残り（次ラウンド）

| 項目 | 状態 |
| --- | --- |
| スケジューラ | 実装済み（純粋・単体テストあり） |
| 実行器の複数スロット状態 + run_batch | 実装済み（合成グラフで一致） |
| パイプライン（PP）への batch 伝搬 | 未着手。`StageGraphs` に 2..4 行の decode グラフを追加し、`pipeline_step_batch` を用意する |
| グラフ生成（行数 2..4） | 未着手。`app::make_pipeline` が `max_sequences` 分の decode グラフを作る |
| エンジン（`shelf-compute`）のスケジューラループ | **実装済み**（受付→prefill→decode バッチ→終了/再利用、cancel は `request_id`） |
| Server の同時要求 | **実装済み**（`--max-inflight` 既定 4） |
| mixed prefill + decode の同時実行 | 未着手（今は **prefill 優先**。新しい要求の prefill チャンクが decode を一時的に止める） |
| FP8 のバッチ不変性 | 未着手（1.6 のとおり。行ごと decode1 GEMV を次の候補に） |
| ブロック表の非 identity 配置 | T16（prefix cache と共通） |

### 設計メモ

* 1 ステップの decode は「アクティブなスロット数 = 行数」のグラフを使う。
  行数 1..4 の decode グラフを起動時に作り、n で選ぶ（graph の leading rows と
  n は一致させる）。
* prefill は 1 リクエストずつチャンクで流す（混合バッチは次段）。decode が
  詰まっている間に新規要求の prefill を 1 チャンクだけ挟む方針にする。
* `pipeline_step_batch` は PP の各ステージへ同じ行メタデータを渡す（hidden の
  行順は不変）。
