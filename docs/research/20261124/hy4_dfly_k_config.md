# Hy4-Flash DFly2 候補数 `K` の設定対応（`K ∈ {2, 3, 4, 5}`, 既定 `4`）

## 目的

Hy4-Flash の DFly2 投機デコード候補数 `K` を従来の `5` 固定（検証行数 `6`）から `K ∈ {2, 3, 4, 5}`（検証行数 `block_rows = K + 1 ∈ {3, 4, 5, 6}`）へ一般化し、既定値を `K = 4` に変更する。あわせて `K=4` / `K=3` / `K=2` 実行時に `rows == 6` 専用経路から低速フォールバックへ落ちないようカーネルとバッファ設計を整える。

## 設計と実装概要

### 1. CLI・環境変数・`/v1/meta` 契約

- `Residency::parse_draft_tokens` / `Residency::resolve_draft_tokens` を追加し、`{2, 3, 4, 5}` のみを受理して範囲外を起動時エラーで拒否するようにした。環境変数 `SHELF_HY4_DFLY_DRAFTS` が指定された場合はそれを解決し、未指定時は `Residency::default_draft_tokens = 4` を用いる。
- `shelf-compute hy4 generate` / `shelf-compute hy4 serve`（`src/apps/compute/hy4_compute.cpp`）に `--dfly-drafts` を追加し、GPU 初期化前に値域検証を行う。
- `shelf-server`（`src/apps/server/shelf_server.py`）に `--dfly-drafts`（`choices=(2, 3, 4, 5)`, default `4`, env `SHELF_HY4_DFLY_DRAFTS`）を追加し、`--dfly-dir` 指定時に `shelf-compute` へ転送する。`/v1/meta` の `speculative` フィールドは `{"method": "dfly2", "drafts": opts.dfly_drafts}` を返す。

### 2. エンジン・HIP Graph・固定バッファ設計

- 将来の動的 `K` 切替時にメモリ再確保を不要とするため、`Residency`、`Hy4StepParams`、`GraphInput`、`BatchGraphParams`、`BatchGraphInput`、`sparse_top_slot`（stride 6）、`transfer_packet_bytes` の固定確保容量は `max_draft_tokens = 5`（`max_verify_rows = 6`）のまま維持した。
- 実行時の `K` は `Residency::draft_tokens`、`Target::draft_tokens`、`Hy4BatchOptions::draft_tokens`、`Hy4BatchSession::set_draft_tokens(int)` / `draft_tokens()` および `GraphKey::draft_tokens` に保持し、同一セッション内で `K` ごとの HIP Graph を安全に分離した。
- `generate.hip` の `forward`、`draft_steps`、`propose`、`verify_speculation`、`generate`（`accepted_histogram` 長 `K + 1`）、`gather_block_rows(src_stride, slot_stride)`、`fuse_block_anchors(block_rows)`、`draft_batch_steps`、`propose_batch`、`decode_batch`、`generate_batch` を `draft_tokens = K` / `block_rows = K + 1` で一般化した。DFly2 の 5 層内部ループ（`i < 5`）は変更していない。
- `wmma.hip` の `shelf_hy4_shared_gate_up` に `CompactRows = 3, 4, 5` の特殊化を追加し、`rows ∈ {1, 3, 4, 5, 6, 16}` がすべてコンパクト経路を通るようにした。

## 実機計測結果（R9700 / `gfx1201`, 単一 GPU `EP=1`）

### `microbench`（単位: us）

| カーネル | rows=1 | rows=3 (`K=2`) | rows=4 (`K=3`) | rows=5 (`K=4`) | rows=6 (`K=5`) | rows=16 | rows=80 (`B=16,K=4`) | rows=96 (`B=16,K=5`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear` | 15.299 | 15.443 | 15.726 | 16.361 | 15.886 | 21.151 | 36.286 | 41.803 |
| `fixed_ffn` | 30.744 | 33.345 | 33.242 | 34.132 | 36.265 | 49.560 | 110.717 | 118.143 |
| `mla_qb` | 21.328 | 24.381 | 24.517 | 25.489 | 26.637 | 42.917 | 81.434 | 85.467 |
| `mla_o` | 56.481 | 94.171 | 94.523 | 95.251 | 96.038 | 120.823 | 238.884 | 254.866 |
| `lm_head` | 474.393 | 486.694 | 491.527 | 495.340 | 502.347 | 527.895 | 1321.501 | 1389.912 |
| `lm_head_argmax` | 480.092 | 487.883 | 491.009 | 493.672 | 496.834 | 517.827 | 1285.770 | 1345.920 |

### `stepbench`（1 ラウンド時間・5 回試行中央値, `EP=1`, 単位: ms/step）

| 条件 | `K=2` (`rows=3/48`) | `K=3` (`rows=4/64`) | `K=4` (`rows=5/80`) | `K=5` (`rows=6/96`) | `K=4` vs `K=5` 短縮 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `B=1, draft=1, graph=1` | 39.375 | 46.554 | 51.115 | 56.581 | -5.466 ms (-9.7%) |
| `B=1, draft=1, graph=0` | 40.094 | 46.083 | 50.212 | 55.245 | -5.033 ms (-9.1%) |
| `B=16, draft=1, graph=1` | 59.932 | 71.222 | 84.083 | 95.191 | -11.109 ms (-11.7%) |
| `B=16, draft=1, graph=0` | 61.094 | 72.069 | 84.917 | 95.514 | -10.597 ms (-11.1%) |

`B=1` および `B=16` のいずれにおいても、`K` の減少に伴い 1 ラウンド時間が単調に短縮し、既定値 `K=4` は `K=5` 比で `B=1` 時に約 `5.47 ms`、`B=16` 時に約 `11.11 ms` 短縮することを確認した。

## 検証結果

- 必須 CPU CTest（`ctest --test-dir build -L "cpu" -LE "optional"`）: 133/133 PASS（`test_cli_docs_sync`、`test_hy4_session`、`test_hy4_routes` 含む）
- オプション文書同期・Python 単体テスト: `test_cli_docs_sync_python` PASS、`tests/server/test_server_cli.py`（39/39 PASS）、`tests/unit/test_hy4_flash.py`（9/9 PASS）
- CLI 異常値拒否確認: `shelf-compute`（`--dfly-drafts 0/1/6/-1/abc`、`SHELF_HY4_DFLY_DRAFTS=6`）および `shelf_server.py`（同範囲外・env）が起動エラーで拒否
- GPU 回帰テスト（R9700, `flock /tmp/r9700.lock`）:
  - `test_hy4_reduce` / `test_hy4_peer_transfer`: PASS
  - `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph batch`（EP=1, 2, 3, 4、`draft=0/1`、`graph=0/1`、capture フォールバック、2047/2053 疎境界）: PASS
  - `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph`（既定 EP=1/3/4 capture フォールバック・単一/バッチ `B=1,2,4,8,16` × `K=2,3,4,5` × `graph=0/1`・512/2048 境界混在 greedy 完全一致・`B=1` バッチセッションおよび動的 `K` 切替 `Hy4BatchSession` パリティ）: PASS
  - `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph transport`（EP=3, 4、`push` / `peer` / `host`、`draft=0/1`、`B=1,2,4,8,16`）: PASS
