# Hy4-Flash DFly2 Adaptive `K`（`K ∈ {2, 3, 4, 5}` 動的切替）

## 目的

Hy4-Flash の DFly2 投機デコード候補数 `K` を直近の平均受理長および現在のアクティブバッチサイズ `B` に応じて `K ∈ {2, 3, 4, 5}` の範囲で動的に切り替え、高受理レジーム（構造化出力・反復列など）では `K=5` まで拡大し、低受理・高バッチ時は `K=2..3` へ縮小して 1 ラウンドあたりの検証コストを抑える。

## 設計と実装概要

### 1. Adaptive `K` ポリシー（`Hy4DflyAdaptiveController`）

- `include/shelf/models/hy4_flash/generate.h` / `src/shelf/models/hy4_flash/generate.hip` に `Hy4DflyAdaptiveController` を実装した。
- 初期値 `current_k` は解決済みの `--dfly-drafts` / `SHELF_HY4_DFLY_DRAFTS`（既定 `4`）を用いる。
- 投機ラウンドごとに平均受理数 `accepted_per_lane`（`sum(keep - 1) / B`）を EMA（`ema_alpha = 0.25`、初回標本は直接代入で初期化）で更新する。
- HIP Graph の過剰な再キャプチャを防ぐため、最小切替間隔 `min_switch_interval = 8` ラウンド（`rounds_since_switch >= 8` のときのみ切替許可）およびヒステリシス幅を設けた。

| バッチサイズ | レジーム | 目標 `K` | 下降閾値（`acc_ema <`） | 上昇閾値（`acc_ema >=`） |
| :--- | :--- | ---: | ---: | ---: |
| `B <= 16` | 極低受理 | `2` | `< 1.05`（`K >= 3` から） | — |
| `B <= 16` | 低受理 | `3` | `< 1.65`（`K >= 4` から） | `>= 1.25`（`K <= 2` から） |
| `B <= 16` | 標準 | `4` | `< 3.05`（`K = 5` から） | `>= 1.95`（`K <= 3` から） |
| `B <= 16` | 高受理 | `5` | — | `>= 3.35`（`K <= 4` から） |
| `B > 16` | 低受理 | `2` | `< 1.65`（`K >= 3` から） | — |
| `B > 16` | 標準 | `3` | `< 3.05`（`K >= 4` から） | `>= 1.95`（`K <= 2` から） |
| `B > 16` | 高受理（上限抑制） | `4` | — | `>= 3.35`（`K <= 3` から） |

- `Residency` / `GraphInput` / `BatchGraphInput` / `sparse_top_slot` / `Target::state` の固定容量は最大 `K=5`（`max_verify_rows = 6`）のまま保持し、`Target::draft_tokens` / `Hy4BatchSession::set_draft_tokens(K)` / `GraphKey::draft_tokens` によりメモリ再確保なしで `K` を切り替える。
- 単一スロット（`generate`、TP>1 では `broadcast(actual, keep)` 後の共通 `keep` を各 rank で評価して同期）および連続バッチ（`Hy4BatchSession::step` / `generate_batch`）の双方で動作する。

### 2. CLI・環境変数・メタデータ・統計契約

- `Residency::resolve_adaptive_draft`（`include/shelf/models/hy4_flash/residency.h` / `src/shelf/models/hy4_flash/residency.cpp`）で `SHELF_HY4_DFLY_ADAPTIVE`（`0` / `1`、範囲外は起動エラー）を解決する。
- `shelf-compute hy4 generate` / `hy4 serve`（`src/apps/compute/hy4_compute.cpp`）に `--dfly-adaptive` フラグ（既定オフ）を追加した。
- `shelf-server`（`src/apps/server/shelf_server.py`）に `--dfly-adaptive`（環境変数 `SHELF_HY4_DFLY_ADAPTIVE` 対応、CLI 優先）を追加し、`--dfly-dir` 指定時に `shelf-compute` へ転送する。`/v1/meta` の `speculative` は `{"method": "dfly2", "drafts": opts.dfly_drafts, "adaptive": bool(opts.dfly_adaptive)}` を返す。
- JSONL 応答の `stats` イベントに `draft_tokens`・`k_switches`・`adaptive`、`done` イベントおよび CLI stderr ログに `draft_tokens`・`k_switches` を出力する。

## 検証結果

### CPU・文書同期・CLI 単体テスト

- 必須 CPU CTest（`ctest --test-dir build -L "cpu" -LE "optional"`）: 133/133 PASS（`test_hy4_session`、`test_hy4_routes`、`test_cli_docs_sync` 含む）
- オプション CPU テスト: `test_cli_docs_sync_python` PASS、`tests/unit/test_hy4_flash.py`（9/9 PASS）、`tests/server/test_server_cli.py`（45/45 PASS）
- `test_hy4_session`: `Hy4DflyAdaptiveController` の合成受理率遷移（低受理 → 標準 → 高受理 → 低受理、`B=1` / `B=16` / `B=32`）、最小切替間隔 8 ラウンド、ヒステリシス境界帯域でのチャタリング抑止、`SHELF_HY4_DFLY_ADAPTIVE=0/1/不正値` 検証 PASS。
- CLI 異常値拒否: `shelf-compute`（`SHELF_HY4_DFLY_ADAPTIVE=abc/true/2`、`--dfly-drafts 0/1/6/-1/abc`）および `shelf_server.py`（`SHELF_HY4_DFLY_ADAPTIVE=2` 等）が起動エラーで拒否。

### GPU 回帰テスト（単一 R9700, `flock /tmp/r9700.lock`, `SHELF_HY4_CHECK_EVENT_REUSE=1`）

- `test_hy4_reduce` / `test_hy4_peer_transfer`: PASS
- `test_hy4_graph BODY DRAFT kparity`: PASS
  - `k_parity_single`（`K=2,3,4,5`、`graph=0/1`、prompt 4/510/2046）: PASS
  - `k_adaptive_single`（48 token 生成、`graph=0/1` とも `rounds=21`, `k_final=3`, `switches=2`、非投機 greedy 完全一致）: PASS
  - `k_parity`（`K=2,3,4,5`、`B=1,2,4,8,16`、512/2048 境界混在、`graph=0/1`）: PASS
  - `k_dynamic_session`: PASS
  - `k_adaptive_batch`（`Hy4BatchSession` `B=2`、48 token 生成、`graph=0/1` とも `rounds=42`, `k_final=3`, `switches=2`、非投機 greedy 完全一致）: PASS
- `test_hy4_graph BODY DRAFT batch`（論理 `EP=1,2,3,4`、`draft=0/1`、`graph=0/1`、capture フォールバック、2047/2053 疎境界混在、`Hy4BatchSession` FIFO/キャンセル/再利用）: PASS
- `test_hy4_graph BODY DRAFT transport`（論理 `EP=3,4`、`push` / `peer` / `host`、`draft=0/1`、`B=1,2,4,8,16`）: PASS
