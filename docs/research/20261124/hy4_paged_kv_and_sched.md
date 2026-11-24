# Hy4-Flash Paged KV プール化と Decode 優先割り込みスケジューリング

基準は `1a061f54`。ROCm 10.2、gfx1201、Release、`SHELF_HY4_ONLY=OFF`、単一 R9700（`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、`flock /tmp/r9700.lock`、42 層の routed expert は第 1 MoE 層の重みを共有）で実装・検証した。

## 概要

1. **DSA 12 層の Paged KV プール化**: SWA 30 層は固定長リングバッファ（最大 518 tok/slot、294 B/tok/layer）を各スロットに維持しつつ、DSA 12 層（`layer == 0 || layer % 4 == 1`）の Octave 4-bit KV（294 B/tok/layer）と圧縮 Indexer `index_keys`（4-token グループ単位・256 B/group = 64 B/tok/layer）を `kv_page_tokens = 256`（`kv_page_groups = 64`）単位の共有ページプールへ移行した。
2. **LRU Prefix Cache Eviction と Admission Control**: `Hy4BatchSession` の Prefix Cache を固定スロット配列（`slot_caches`）から可変エントリプール（`prefix_entries`）へ変更し、完全ページの参照カウント共有と末尾部分ページの CoW（`copy_dsa_page_prefix`）を実装した。ページ不足時は未使用 Prefix Cache を LRU 順に解放し、それでも不足する新規リクエストは `waiting` キューへ留める（banker-safe な Admission Control）。
3. **Prefill 間の Decode 優先割り込み（HoL ブロッキング解消）**: `Hy4BatchSession::step()` で、アクティブな Decode スロットが存在する間に完了前の長文 Prefill チャンクが連続する場合、`--prefill-decode-interval N`（既定 `1`、`SHELF_HY4_PREFILL_DECODE_INTERVAL`、`0` で無効）ごとに Decode ステップを優先割り込み実行する。

## 設計と実装要点

### 1. DSA 共有 KV ページプールと Kernel 参照

- [residency.h](../../../include/shelf/models/hy4_flash/residency.h) / [residency.cpp](../../../src/shelf/models/hy4_flash/residency.cpp):
  - `kv_page_tokens = 256`、`kv_page_groups = 64` を定義し、`Residency` に `kv_pool_tokens`（`--kv-pool-tokens` または `SHELF_HY4_KV_POOL_TOKENS`、未指定時は `max_slots * kv_page_stride` を上限に残余 VRAM から自動算出、TP=1 では `graph_headroom` 256 MiB を優先確保）、`kv_pages_total`、`kv_page_stride`、`page_table[4]`（デバイス上の `[max_slots][kv_page_stride]` int32 配列）、`index_pending[4]`（`max_slots * 12 * 128 * sizeof(uint16_t)`）、`kv_free_pages`、`kv_page_refs`、`slot_page_tables` を追加した。
  - 論理 `max_seq_len = 1,048,576` × `max_slots = 16` の `reserve()` でも、SWA 30 層（約 4.36 MiB/slot）と `page_table` / `index_pending` 以外の残余 VRAM 内に共有 DSA ページプールが収まり、`reserve()` が成立する。
- [attention.h](../../../include/shelf/models/hy4_flash/attention.h) / [attention.hip](../../../src/shelf/models/hy4_flash/attention.hip):
  - `kv_record_const` / `kv_record`（`pos >> 8` ページ、`(pos & 255)` オフセット）、`index_group_const` / `index_group`（`group >> 6` ページ、`(group & 63)` オフセット）を追加し、`unpack`、`unpack_quant`、`unpack_batch`、`save_records`、`save_records_batch`、`pool`、`pool_batch`、`save_pending_batch`、`index_scores`、`index_scores_wmma`、`index_key_peak`、`index_select`、`attention_inner`、`attention_batch` を `page_table` 経由参照へ更新した（`page_table == nullptr` の単体テスト構成との互換も維持）。
  - Prefix 復元時の末尾部分ページ CoW 用に `copy_dsa_page_prefix` カーネルを追加し、DSA 12 層の `cache` と完了済み 4-token グループの `index_keys` を GPU 上でコピーする。
  - [test_hy4_graph.cpp](../../../tests/kernels/test_hy4_graph.cpp) の `reset_history` / `check_prefill` も DSA ページ配置と `index_pending` へ追従させた。

### 2. Prefix Cache 共有・LRU Eviction・Admission Control・HoL 解消

- [generate.h](../../../include/shelf/models/hy4_flash/generate.h) / [generate.hip](../../../src/shelf/models/hy4_flash/generate.hip):
  - `Target::forward` / `generate` / `generate_batch` の入出力に DSA ページ割当・解放（例外安全解放を含む）を組み込んだ。
  - `Hy4PrefixSlot` に `pages`、`live_swa_slot`、`live_swa_gen` を追加。`Hy4BatchSession::Impl::fill()` では全 `prefix_entries` から最長共通プレフィックスを探索し、ページ境界で割れる完全ページは `retain_pages()` によるゼロコピー参照カウント共有、端数ページは他参照が無ければ引き継ぎ・他参照ありなら `alloc_single_page()` + `copy_dsa_page_prefix()` で分離する。SWA は該当スロットが未上書き（`slot_swa_gen` 一致）ならゼロコピー再利用、それ以外は `swa_snap`（host スナップショット）から H2D 復元する。
  - `unmapped_reserved_pages()` と `avail_for_new()` で実行中スロットの予約済み最大必要ページを控除した実質空きページを計算し、不足時は保護対象以外の `prefix_entries` を `last_used` の古い順（LRU）に解放する。それでも確保できないリクエストは `waiting` に留めて Admission Control する。
  - キャンセルされたリクエストは `finish(slot, "cancelled")` で `prefix_entries` へ保存せず、`slot_swa_gen[slot]` 更新と `release_slot_pages(slot)` のみを行うため、汚染スロットの再利用を防ぎつつ既に確定済みの別 Prefix Cache エントリも毀損しない。
  - `Hy4BatchSession::step()` では `prefill_decode_interval`（既定 `1`）に従って未完了 Prefill チャンク間に `decode_batch()` を割り込ませ、完了済みチャンクの初回 1-step バッチ投入時のみ全投入チャンクを完了させてバッチ効率を維持する。
- [hy4_compute.cpp](../../../src/apps/compute/hy4_compute.cpp) / [shelf_server.py](../../../src/apps/server/shelf_server.py):
  - `--kv-pool-tokens` と `--prefill-decode-interval` を追加し、`pong.capabilities` に `paged_kv` / `prefill_decode_interval`、`stats.kv` に `page_tokens`（256）・`total_pages`・`used_pages`・`free_pages`・`capacity_tokens`・`free_tokens` を出力する。

## 検証結果

### CPU CTest・CLI ドキュメント・Server モック試験

- `ctest --test-dir build -L "cpu" -LE "optional" --output-on-failure`: **133/133 PASS**（`test_hy4_session` での `prefill_decode_interval` / `pages_for_tokens` 検証、`test_cli_docs_sync`、`test_fp8_kernel_spill_contract` を含む）
- `ctest --test-dir build -R "^(test_cli_docs_sync_python|test_hy4_flash_cpu)$" --output-on-failure`: **2/2 PASS**
- Server 単体・モック試験: `test_server_cli.py`、`test_server_ignore_eos_readiness.py`（3/3）、`test_compute_client_reader.py`、`test_inference_budget.py`、`test_reasoning_parser.py`、`test_evalscope_agentic.py`（4/4）、`test_shelf_bench_suite.py`（13/13）、`test_strict_tool_stream_unit.py`（1/1）、`test_dflash2_grammar_protocol.py`（3/3）すべて **PASS**（`test_tool_structural_tag.py` は `xgrammar` 未導入、`test_plamo3_template.py` は `SHELF_PLAMO_MODEL_DIR` 未設定のためスキップ／対象外）

### GPU 回帰試験（`SHELF_HY4_CHECK_EVENT_REUSE=1`）

- `./build/tests/test_hy4_reduce`: **PASS**
- `./build/tests/test_hy4_peer_transfer`: **PASS**（push / peer、corrupt=0 / 1 すべて通過）
- `SHELF_HY4_CHECK_EVENT_REUSE=1 ./build/tests/test_hy4_graph "$BODY" "$DRAFT"`:
  - `ep=1 target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass`
  - `ep=3 target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass`
  - `ep=4 target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass`
- `SHELF_HY4_CHECK_EVENT_REUSE=1 ./build/tests/test_hy4_graph "$BODY" "$DRAFT" batch`:
  - `ep=1`: `draft=0` / `draft=1` とも `graph=0` / `graph=1` で `batch=2..16` 全件 `greedy=N/N`、`session=pass`
  - `ep=2 / 3 / 4`: `draft=0` / `draft=1` とも `graph=0` / `graph=1` で `batch=2, 4, 8, 16` 全件 `greedy=N/N`、`session=pass`、位置 2047 / 2053 混在・capture 失敗フォールバック通過

### Paged KV・Prefix Cache・LRU Eviction・Admission Control・HoL 解消試験（`test_hy4_prefix_cache`）

単一 R9700 上で `./build/tests/test_hy4_prefix_cache "$BODY" "$DRAFT"` を実行し、全ケース `PASS`（`ALL_HY4_PREFIX_CACHE_TESTS_PASSED`）を確認した。

- `draft=1`（42 層、DFlash2 併用）:
  - `swa_525_3turn`: `r2=544/564`、`r3=580/605`、`hits=2`、`saves=3`、`timing_525_t2: cold=698.6ms cached=296.5ms`
  - `sparse_2051`: `restored=2060/2094`、`cold_pp=1918.7ms` -> `cached_pp=95.3ms`（Prefix Cache 利用により prefill 約 20 倍高速化、出力完全一致）
  - `swa_snap_diverge`: `restored=640/665`、`early_miss=0`、`PASS`
  - `concurrent_b2_mismatch_cancel`: `r201=540`、`r202=1128`、`r304=400`、`PASS`（キャンセル要求の非保存と既存キャッシュ分離を確認）
  - `single_slot_generate`: `restored=624/638`、`hits=1`、`saves=2`、`PASS`
  - `reserve_1Mi_c16`: 論理 `max_seq_len=1,048,576`、`max_slots=16` で `kv_pages=21589`（`capacity_tokens=5526784`、`graph_room=1`）を事前確保 `PASS`
  - ページ境界・SWA 境界・疎境界（`len=255, 256, 257, 512, 2048, 4097`）: 全 6 長で `graph==eager=8/8` 完全一致・ページリーク 0 `PASS`
  - `lru_eviction`: `kv_pool_tokens=1024`（4 ページ）予算下で未使用 Prefix Cache を LRU 順に解放（`evictions=2`、`p2_restored=300`）`PASS`
  - `admission_control`: 4 ページ予算へ 3 要求（各 2 ページ必要）同時投入時に `active=2/queued=1->all_resumed_completed`（出力完全一致）`PASS`
  - `hol_interleave`: 短文 Decode（64 tok + 48 tok 生成）実行中に 6144 tok 長文 Prefill を投入したところ、`interval=0` では `during_pp=0` / `max_short_itl=6411.4ms`（`p95=69.3ms`）、`interval=1` では `during_pp=6` / `max_short_itl=1206.3ms`（`p95=1155.0ms`）へ最大 ITL を約 5.3 分の 1 に低減しつつ全生成 ID が基準と完全一致 `PASS`
- `draft=0`（DFlash2 なし）:
  - `swa_525_3turn`: `r2=544/564`、`r3=580/605`、`timing_525_t2: cold=1251.1ms cached=850.8ms`
  - `sparse_2051`: `restored=2060/2094`、`cold_pp=1946.7ms` -> `cached_pp=95.8ms`
  - `swa_snap_diverge`: `restored=640/665`、`early_miss=0`、`PASS`
  - `concurrent_b2_mismatch_cancel`: `r201=540`、`r202=1128`、`r304=400`、`PASS`

### 基準（`1a061f54`）との A/B 性能比較（単一 R9700、`EP=1`、`5 trial` 中央値）

| ベンチマーク条件 | 基準 `1a061f54` | Paged KV 実装 | 差分 |
| :--- | :--- | :--- | :--- |
| `prefillbench`（`8192` tok、`chunk=1024`、`draft=0`） | `8.528664 s`（`960.5 tok/s`） | `8.537730 s`（`959.5 tok/s`） | `+0.11%` |
| `stepbench`（`B=1`、`draft=1`、`graph=0`） | `55.242 ms` | `55.474 ms` | `+0.42%` |
| `stepbench`（`B=1`、`draft=1`、`graph=1`） | `56.616 ms` | `56.899 ms` | `+0.50%` |
| `stepbench`（`B=16`、`draft=1`、`graph=0`） | `95.635 ms` | `95.752 ms` | `+0.12%` |
| `stepbench`（`B=16`、`draft=1`、`graph=1`） | `95.371 ms` | `95.773 ms` | `+0.42%` |

## 判定・結論

**採用（ACCEPT）**。
- `page_table` 間接参照による性能劣化は `prefillbench` / `stepbench`（`B=1, 16`、`Graph OFF/ON`）とも `+0.11%〜+0.50%` と誤差範囲内に抑えられた。
- 静的 per-slot 確保で不可能だった論理 `1Mi` × `c=16` 予約が単一 R9700 上（21,589 ページ / 5,526,784 token 共有プール）で成立し、境界一致・LRU Prefix Cache Eviction・Admission Control・Prefill 間の Decode 優先割り込み（最大 ITL `6411.4ms` -> `1206.3ms`）まで実機確認を完了した。
