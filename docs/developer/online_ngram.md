# Online N-gram / Suffix 投機的デコード

`include/shelf/runtime/online_ngram.h` と `src/shelf/runtime/online_ngram.cpp` は、現在のリクエストの履歴（入力プロンプト + 生成済みトークン）から最長 suffix 一致を検索して draft トークンを提案する `OnlineNgramMatcher` を定義する。照合は CPU で行い、単独実行と DFlash2 / MTP との併用に対応する。単独実行の GDN 履歴と logits の領域は、実行可能な verify 行数に合わせて確保する。

## インデックスと一致規則

- **ローリング履歴**: リクエストごとに直近最大 `32,768` トークンをリングバッファ（`kWindowCapacity = 32768`）に保持する。バッファは初期化・`reset()` 時に固定長で確保され、`ingest()` / `propose()` のホットパスでは動的メモリ確保を行わない。
- **固定サイズハッシュ表**: 2-gram と 4-gram の 2 テーブル（各 `16,384` バケット × 最新 `4` 位置）を持ち、`ingest()` ごとに最新位置を先頭へシフト挿入する。
- **実トークン照合**: `propose()` は 4-gram バケットおよび 2-gram バケットの候補位置を走査し、最大 `max_match`（既定 `32`）トークンまで後方照合を行う。選択基準は **最長一致優先**、同長なら **最新位置優先** である。
- **最小一致長と固定長パディング**: `match_len >= min_match`（未指定時は greedy `4` / stochastic `6`）を満たす候補を採用し、一致位置の後続トークンを最大 `K` トークンコピーする。残余トークン数が `K` 未満の場合は末尾トークンでパディングし、固定長を使う単一要求とハイブリッドの verify 用に `K` 個の候補を返す。複数要求の単独モードでは実際にコピーできた候補だけを使う。
- **適応型クールダウン**: `record_outcome(accepted_drafts)` により `accepted == 0` が連続 `2` 回発生すると `16` ラウンドのクールダウンに入る。クールダウン中は `propose()` が提案をスキップして余計な verify を防ぐが、`ingest()` による履歴登録は継続する。

## 実行モードとの統合

- **DFlash2 ハイブリッド（`--dflash2-model-dir` + `--ngram`）**: `ContinuousBatcher::run_dflash2_round` および `SpeculationRunner::run_dflash2` は各ラウンドで `OnlineNgramMatcher::propose()` を事前ゲートとして呼び出す。ヒット時は `DFlash2SpecBatchSlot::precomputed_drafts`（単発経路では `dflash2_spec_step` の `precomputed_drafts`）へ候補を渡し、`dflash2_propose_*` をスキップする。`dflash2_spec_step_batch` は `precomputed_drafts != nullptr` 時も `device_token_bridge` を維持して `[anchor, drafts...]` を非同期転送し、verify 用 HIP Graph（`dflash_verify_graph_cache`）のヒットと `dflash2_append_target_taps` による次ラウンドへの整合性を保つ。構造化出力（`structured_dflash`）では `DraftMaskState::Idle` で N-gram を優先判定し、`PendingMask` を経て同じ verify 経路へ合流する。
- **MTP ハイブリッド（`--mtp` + `--ngram`）**: `mtp_draft` / `mtp_draft_batch` が `precomputed_drafts != nullptr` を受け取ると、`first_pass` のみ `compute_logits=false` で実行してドラフタの KV キャッシュ・GDN 状態を追従させ、`lm_head` と自己回帰ステップ `1..K-1` をスキップする。
- **N-gram 単独モード（`--ngram`）**: スケジューラが選んだ decode 要求を同じ `ScheduledBatch` で処理する。候補のない要求は先頭の `DECODE` 領域で 1 行を使い、候補のある要求は後続の `SPEC_VERIFY` 領域で直前の確定トークンと候補を検証する。各要求の入力行と出力行は連続し、受理後に必要な GDN 履歴もその要求の先頭行を基準に復元する。補正、追加トークン、EOS の確定は通常の decode と同じコミット処理を使う。
- **行数の制御**: 単一要求は設定した `K`（3、7、15 のいずれか、既定 7）で固定長検証する。対応する実行経路では 4、8、16 行の verify 専用 HIP Graph キャッシュを使う。複数要求では要求ごとに 1 行を確保し、残りを入力順に候補へ割り当てる。合計は 16 行、スケジューラのトークン予算、出力行容量のうち最小の値を上限とする。3〜7 件では、生成トークン数が最も少ない要求より `K` トークンを超えて先行した要求を通常の decode で進める。最大同時要求数が 8 件以上なら、通常の各要求 1 行と単独 verify 1 件を収容できる出力領域を使う。KV ページの追加割り当てにも通常の安全判定を適用する。実行中の要求が 8 件以上の間、prefill がある場合、または全要求を同じ decode ステップに載せられない場合は通常のスケジューリングを使う。要求が減れば照合を再開する。履歴への登録は継続する。GDN の畳み込みまたは再帰カーネルが途中状態を保存できない形状では、単独モードの要求は通常の decode で処理する。

## 統計出力契約

`--ngram-stats 1`（既定）またはサーブ統計有効時は以下を出力する。

- `NGRAM_ROUNDS`: `propose()` が評価された総ラウンド数（クールダウンによるスキップ含む）
- `NGRAM_HITS`: N-gram 提案が採用されたラウンド数
- `NGRAM_ACCEPTED_DRAFTS`: N-gram 提案で target に受理された draft トークン総数
- `NGRAM_MEAN_ACCEPTED`: ヒット 1 回あたりの平均受理 draft 数（`accepted_drafts / hits`）
- `NGRAM_DFLASH_SKIPPED`: N-gram ヒットにより DFlash2 / MTP の draft forward をスキップした回数
- `NGRAM_CPU_US_PER_ROUND`: ラウンドあたりの平均 matcher CPU 消費時間（µs）
