# Hy4-Flash `ignore_eos`・簡易 Prefix Cache・AgentX (`swe_smith`) ベンチマーク基盤

## 概要

本作業では、`shelf-server` / `shelf-compute` / `Hy4-Flash` に対し次の 3 点を実装・検証した。

1. `ignore_eos` パラメータ（`shelf-server` → `stdio_server` / `ContinuousBatcher` / `MtpDrafter` / `DFlash2` および `hy4_compute` / `Hy4BatchSession` / `generate`）と `GET /readiness` ヘルスチェックの追加
2. `Hy4-Flash` マルチターン会話向けの `slot` 戦略ベース簡易 Prefix Cache（差分 prefill、SWA 窓スナップショット復元、DSA group 境界整合、DFly2 ON/OFF 両対応）
3. TokenSpeed / EvalScope `swe_smith` 互換 Agentic ベンチマーク枠組み（[evalscope_agentic.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/tools/bench/evalscope_agentic.py)）の整備（`conc=32` 対応を含む）

## 設計と実装要点

### 1. `ignore_eos` と `GET /readiness`

- [openai_protocol.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/server/serverlib/openai_protocol.py): `ChatRequest.ignore_eos`（既定 `false`）を追加し、allowlist 検証と bool 型チェックを実装した。
- [chat_service.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/server/serverlib/chat_service.py) / [compute_client.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/server/serverlib/compute_client.py): `generate` / `generate_stream` へ `ignore_eos` を伝播し、`usage.prompt_tokens_details.cached_tokens` を `restored_tokens` から反映する。
- [shelf_server.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/server/shelf_server.py): `GET /readiness`（compute 生存時 `200 {"ready": true}`、停止時 `503 {"ready": false}`）を追加し、`/v1/meta` 契約を維持した。`--prefix-cache-strategy` に `off` を追加し、Hy4 でも `--prefix-cache 0|1` / `off` を切り替え可能にした。
- [stdio_server.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/compute/stdio_server.hip) / [continuous_batcher.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/shelf/runtime/continuous_batcher.cpp) / [runtime_request.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/shelf/runtime/runtime_request.cpp) / [mtp_drafter.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/shelf/models/qwen35/runtime/mtp_drafter.cpp): `ignore_eos: true` 時は stop token 停止をバイパスして厳密に `max_new_tokens` まで生成し、`finish_reason = "length"` を返す。

### 2. Hy4-Flash 簡易 Prefix Cache

- [generate.h](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/include/shelf/models/hy4_flash/generate.h) / [generate.hip](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/shelf/models/hy4_flash/generate.hip):
  - 完了したリクエストのスロット状態（`history` トークン列、`max_written`、および prefill 完了時点での SWA 30 層リングスナップショット）を `Hy4PrefixSlot` に保持し、次ターン入力との最長共通プレフィックスを評価する。
  - **厳密一致保証**:
    - DSA（12 層、`ratio=4`）が `pending` を参照せず既に確定済みの圧縮グループ `0 .. M/4-1` のみを参照するよう、復元長 `M` を `4` の倍数へ切り下げる。
    - 最終トークンの隠れ状態 `target->state`・`sparse_top_slot`・logits・DFly2 状態を確実に再構築するため、`M <= input_ids.size() - 1` を必須とし、最低 1 トークンの prefill を実行する。
    - SWA（30 層、`swa_slots = min(tokens, 518)`）では、`max_written <= max(0, M - 511) + swa_slots` が成立する場合は GPU 上のリングをゼロコピーでそのまま再利用し、DFly2 の投機書き込みや途中分岐により初期リングが上書きされた場合は、prefill 終了時に退避した host スナップショット（`snap_pos` まで）を H2D 復元して差分 prefill を行う。
    - キャンセル・エラー・途中で失敗したスロットは実行開始前に即時無効化（`valid = false`）し、汚染スロットの再利用を完全に禁止する。
    - `Hy4BatchSession::Impl::fill()` では全空きスロットから最長一致を選択し、ミス時は未登録スロットまたは LRU スロットへ割り当てる。
- [hy4_compute.cpp](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/src/apps/compute/hy4_compute.cpp): `--prefix-cache 0|1`（既定 `1`）を追加し、`pong` の `capabilities.prefix_cache`、`stats` の `prefix` 統計、`done` の `prompt_tokens` / `restored_tokens` / `prefill_tokens` / `cache_checkpoint_tokens` を返す。

### 3. AgentX / EvalScope `swe_smith` 互換ベンチマーク

- [evalscope_agentic.py](file:///home/homelab1/coding-local/shelf/.worktrees/hy4-agentic-bench-pc/tools/bench/evalscope_agentic.py):
  - `prepare` / `run` / `collect` の 3 サブコマンドを提供。
  - プロファイル `hy4-light`（4096/800/4t, conc 1,4,8,16）、`hy4-cached`（8192/800/5t, conc 1,4,8,16）、`single-r9700`（4096/800/4t, conc 1,4,8）、`hy4-c32`（4096/800/4t, conc 1,4,8,16,32）を備え、`--concurrencies` で任意（`conc=32` を含む）指定が可能。
  - `tokenizers` + `jinja2` による実モデル tokenizer 読み込みと、`transformers` 不在環境・CI 向けのフォールバックに対応。
  - `--runner builtin`（標準 asyncio + aiohttp OpenAI SSE マルチターン実行、`ignore_eos: true`、`stream_options.include_usage: true`、`/v1/meta` 採取、EvalScope 互換 `benchmark_summary.json` / `benchmark_data.db` 出力）および `--runner evalscope` をサポート。

## 検証結果

### CPU CTest・単体テスト

- `ctest --test-dir build -L "cpu" -LE "optional"`: **108/108 PASS**（`test_hy4_session`, `test_hy4_routes` を含む）
- `python3 tests/unit/test_hy4_flash.py`: **9/9 PASS**
- `python3 -I tests/server/test_server_cli.py`: **26/26 PASS**（Hy4 `--prefix-cache 0` / `off` 転送および非 Hy4 の `off` 時 `--prefix-cache 0` / `--prefix-cache-strategy radix` 転送を確認）
- `python3 -I tests/server/test_server_ignore_eos_readiness.py`: **3/3 PASS**（`ignore_eos` パース、SSE streaming 時の `usage.prompt_tokens_details.cached_tokens` 反映、`GET /readiness` 200/503、`/v1/meta` 維持）
- `python3 tests/server/run_server_regression.py --group bench`: **2/2 PASS**（`test_shelf_bench_suite.py`, `test_evalscope_agentic.py` の合成/JSONL `prepare`、mock HTTP OpenAI SSE での `run --runner builtin`・`c=32` 並列・`ignore_eos: true`・`--runner evalscope` コマンド構築・`collect` CSV/JSON 集計）
- 実モデル tokenizer（`Hy4-flash`）での `evalscope_agentic.py prepare --profile single-r9700`: `first_prompt_tokens_min=4096, max=4096` で生成成功

### R9700 GPU 検証（単一 GPU + `share_expert_weights`、`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`）

- `SHELF_HY4_CHECK_EVENT_REUSE=1 ./build/tests/test_hy4_graph`（`Hy4-flash-exl3-4bpw-mul24` + `engram-e4m3fn` + `Hy4-flash-dfly-4bpw`）:
  - `ep=1 target=7/7 draft=7/7 capture_fallback=pass`
  - `ep=3 target=7/7 draft=7/7 capture_fallback=pass`
  - `ep=4 target=7/7 draft=7/7 capture_fallback=pass`
- `./build/tests/test_hy4_prefix_cache`（Prefix Cache OFF との全生成トークン ID 100% 一致を検証）:
  - **DFly2 ON（`Hy4BatchSession`）**:
    - 525 tokens・3 ターン（512 SWA 窓境界 + 4 非倍数長）: `r2=544/564`, `r3=580/605`, ターン 2 全体時間 `696.7 ms -> 294.8 ms`（約 2.36x）
    - 2051 tokens・疎境界越え: `restored=2060/2094`, prefill 時間 `1913.8 ms -> 95.5 ms`（約 20.0x 高速化）
    - SWA ラップ後の prompt 境界復元（`ignore_eos=true` で 96 tokens 生成後）: スナップショット復元 `restored=640/665`, SWA ラップ前途中分岐の安全ミス `restored=0`
    - 並列 `B=2` 混在 + 不一致ミス + 途中キャンセル後の再利用禁止: `r201=540`, `r202=1128`, 全ケース生成 ID 一致
  - **単一スロット `generate()`（DFly2 ON、`ignore_eos=true`）**:
    - `restored=624/638`, `prefill_s=0.526656 s -> 0.122527 s`（約 4.30x 高速化）, `draft_rounds=3, accepted=15`, 生成 ID 一致
  - **DFly2 OFF（`Hy4BatchSession`）**:
    - 525 tokens・3 ターン: `r2=544/564`, `r3=580/605`, `1250.9 ms -> 848.0 ms`
    - 2051 tokens・疎境界越え: `restored=2060/2094`, prefill 時間 `1944.4 ms -> 95.5 ms`（約 20.4x 高速化）
    - SWA スナップショット復元 `640/665`・安全ミス `0`・`B=2` 混在 `540 / 1128`: 全ケース生成 ID 一致

