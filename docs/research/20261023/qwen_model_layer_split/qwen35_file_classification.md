# qwen35 ファイル分類（model 非依存化のための A/B/C）

目的: `models/qwen35/` を肥大化させずに `models/qwen4exp/` を追加できるように、
既存 qwen35 の実装を 3 分類する。本ドキュメントは分類と移動予定の正本である。

- A: 完全 model 非依存。`include/shelf/kernels/` および
  `src/shelf/kernels/` へ移す。
- B: Qwen35/Qwen4Exp の family 共通。geometry / small POD builder として
  `models/common/qwen_hybrid/` へ移す。
- C: Qwen35 固有。`models/qwen35/` に残す。

制約: runtime polymorphism（IModel / IBackend / Registry / Plugin）を導入しない。
ホットパスに virtual dispatch を入れない。汎用 Transformer framework を作らない。

## Phase 1: kernels（A）

移動元 `models/qwen35/kernels/optimized/` → 移動先 `shelf/kernels/optimized/`。
namespace は既に `shelf::kernel` であり `qwen35` 依存がない。そのまま移す。

| ファイル | 分類 | 備考 |
| --- | --- | --- |
| `linear/bf16.h` `.hip` | A | BF16 GEMM |
| `linear/fp8.h` `.hip` | A | FP8 GEMM |
| `linear/config.h` | A | |
| `linear/gemm_policy.h` | A | |
| `attention/paged_attention.h` | A | main paged attention |
| `attention/paged_decode.hip` | A | |
| `attention/paged_prefill.hip` | A | |
| `attention/rope.h` `.hip` | A | partial rotary |
| `activation_quantize.h` `.hip` | A | |
| `embedding.h` `.hip` | A | |
| `elementwise.h` `.hip` | A | SiLU/Sigmoid/Mul/ResidualAdd/Split |
| `rmsnorm.h` `.hip` | A | mode/geometry は引数化済み |
| `l2_normalize.h` `.hip` | A | |
| `output_gather.h` `.hip` | A | |
| `kv_append.h` `.hip` | A | paged KV append |
| `kv_relocate.h` `.hip` | A | |
| `gdn/conv1d.h` `.hip` | A | depthwise causal conv |
| `gdn/recurrence.hip` | A | gated delta rule |
| `gdn/fused_gdn_*.hip` | A | |
| `sampling.h` `.hip` | A | |
| `stochastic_sampling.h` `.hip` | A | |
| `swiglu_quantize.h` `.hip` | A | Qwen35 dense SwiGLU 用だが kernel 自体は model 非依存 |
| `rmsnorm_quantize.h` `.hip` | A | |
| `resid_rmsnorm_quantize.h` `.hip` | A | |
| `attn_post_quantize.h` `.hip` | A | |
| `detail/*` `gdn/detail/*` | A | device ヘルパ |

`kernels/correctness/` は model 参照実行系のため Phase 1 の対象外。
`kernels/dflash2/` `kernels/eagle4/` は spec decode 固有 kernel であり当面 qwen35 配下に残す。

### qwen35 状態型への依存（Phase 1 で除去）

`paged_attention.h` / `kv_append.h` / `kv_relocate.h` および対応 `.hip` は
`models/qwen35/state/kv_cache_types.h` を include し、`shelf::qwen35::fp8e4m3_storage_t` と
`shelf::qwen35::KVCacheDType` を参照する。kernel 層から qwen35 への依存であり、
`shelf/kernels/kv_types.h` へ型を移して `shelf::kernel` namespace に置く。

## Phase 2: state（A/B）

| ファイル | 分類 | 移動先 |
| --- | --- | --- |
| `state/paged_types.h` | A | `shelf/inference/state/paged_types.h` |
| `state/kv_cache_types.h` | A | `shelf/kernels/kv_types.h`（Phase 1） |
| `state/paged_kv_pool.h` `.cpp` | A | `shelf/inference/state/` |
| `state/sequence_slot_pool.h` `.cpp` | A | `shelf/inference/state/` |
| `state/paged_sequence_state.h` `.cpp` | A | `shelf/inference/state/` |
| `state/gdn_state.h` | A | `shelf/inference/state/` |
| `state/gdn_state_pool.h` `.cpp` | A | `shelf/inference/state/` |

`GdnStatePoolLayout::from_text_config(Qwen35TextConfig)` は state 層から Qwen35 への
依存であり除去する。state 層は `GdnStatePoolLayout` のみを受け取る。
Qwen35 側に `make_qwen35_gdn_state_layout(Qwen35TextConfig)` を置く。
将来 Qwen4Exp は `make_qwen4exp_gdn_state_layout(...)` を置く。

`PagedSequenceState` は `GdnStatePool` の直接所有をやめ、
generic sequence state（slot / generation / position / max_seq_len /
block_table / `PagedKVPool*`）のみを担当する。Qwen35 の GDN state slot は
Qwen35 runtime 側で同じ sequence slot をキーに管理する。
StateBundle の interface は作らない。

### 実施状況と再検討の結果

config 依存の除去（`GdnStatePoolLayout::from_text_config` の削除と
`make_qwen35_gdn_state_layout` / `make_qwen35_gdn_state_layout_tp` への移設、
model 依存デフォルトの 2 引数 `GdnStatePool::create` 削除）と
namespace の `shelf::state` への分離は完了した。

`PagedSequenceState` の `GdnStatePool` 保持については、
保持する方針に変更した。根拠は次のとおりである。

- `PagedSequenceState` / `PagedKVPool` / `GdnStatePool` は
  いずれも `shelf::state`（generic 層）であり model 境界に抵触しない。
- GDN は Qwen35 / Qwen4Exp が同一 FP16 実装を共有するため、
  Qwen4Exp も同じ `GdnStatePool` を使う。
- 真に model 固有なのは QSA indexer key state / PLE state であり、
  これらは `PagedSequenceState` にフィールドを追加せず、
  同じ sequence slot をキーに model runtime 側で管理する
  （Qwen4Exp 実装時にこの設計を維持する）。
- `create_paged_sequence_state` が GDN initialize を抱えるのは
  「GDN state は必須」という両 model 共通の前提によるもので、
  model 固有の制約ではない。

StateBundle の interface は作らず、この境界を Qwen4Exp 実装時も維持する。

### GDN state dtype の方針

Qwen35 の GDN recurrent state は FP16。Qwen4Exp も FP16 で実装し lossy を許容する。
FP16/FP16 切り替えのための storage variant 抽象と runtime branch は導入しない。
Qwen35 と Qwen4Exp は同一の最適化済み GDN kernel を共有し、
head 数 / head_dim / conv kernel の geometry 差のみで吸収する。
既存 Qwen35 の GDN 精度・計算経路は変更しない。

## Phase 3: runtime infrastructure（A/C）

| ファイル | 分類 | 移動先 |
| --- | --- | --- |
| `runtime/sampling_params.h` `.cpp` | A | `shelf/inference/scheduling/` |
| `runtime/scheduled_batch.h` `.cpp` | A | `shelf/inference/scheduling/` |
| `runtime/schedule_plan.h` | A | `shelf/inference/scheduling/` |
| `runtime/kv_capacity_manager.h` `.cpp` | A | `shelf/inference/scheduling/` |
| `runtime/kv_banker.h` `.cpp` | A | `shelf/inference/scheduling/` |
| `runtime/token_budget_scheduler.h` `.cpp` | A | `shelf/inference/scheduling/` |
| `runtime/program_executor.h` `.hip` | A（common 部分） | `shelf/inference/execution/` |
| `runtime/optimized_dispatch.hip` | A（common 部分） | `shelf/inference/execution/` |
| `runtime/physical_launch.h` `.hip` | A（common 部分） | `shelf/inference/execution/` |
| `runtime/*_selector.cpp` | A | `shelf/inference/execution/selectors/` |
| `runtime/*_dispatch.hip` | A | `shelf/inference/execution/` |
| `runtime/*_fusion.hip` | A | `shelf/inference/execution/` |
| `runtime/executor.h` `.hip` | C | qwen35 に残す |
| `runtime/dflash2_*` | C | spec decode |
| `runtime/eagle4_*` | C | spec decode |
| `runtime/gdn_spec_history.*` | C | spec decode |
| `runtime/spec_decode.cpp` | C | spec decode |
| `runtime/prefix_cache.*` | B（metadata 部分のみ抽出） | Phase 5 |
| `runtime/continuous_batcher.*` | C | qwen35 に残す |
| `runtime/tensor_parallel.*` | C | qwen35 に残す |
| `runtime/decode_perf_stats.*` | A | `shelf/inference/execution/`（model 非依存のため移動済み） |

`HostExecutionContext` は common 化せず巨大な全 model union にしない。
common context と Qwen35 extension（`GdnSpecHistory` / `tree_anc_*` / DFlash2）を分離する。

### 実施状況と調査結果

`runtime/*_selector.h/.cpp` 11 種は `HostExecutionContext` に依存せず、
唯一の共通定数 `kQwen35OptimizedMaxRows` だけを参照していた。
これらは `shelf/inference/execution/selectors/` へ移動済み
（namespace `shelf::execution`、定数は `selector_limits.h` の `kOptimizedMaxRows`）。
qwen35 側の残存利用は `models/qwen35/runtime/selector_types.h` の using 宣言で集約する。

`HostExecutionContext`（`program_executor.h`）は 63 フィールドを持ち、
40 ファイルが参照する。フィールドの使用分布調査の結果、
Qwen35 extension として分離すべき候補は以下である。

- spec tree: `tree_anc_offsets` / `tree_anc_positions` / `tree_committed_end` /
  `tree_committed_end_rows` / `tree_row_parent`
- spec history: `gdn_spec_history`（`GdnSpecHistoryDeviceView`） /
  `gdn_compact`（`GdnCompactLogDeviceView`）
- spec visible range: `min_visible_tokens` / `max_visible_tokens`

これらを使うのは `paged_attention_dispatch.hip` /
`gdn_conv_dispatch.hip` / `gdn_recurrence_dispatch.hip` の spec verify 経路のみである。
`stochastic_topk_*`（9 フィールド）は sampling 共通のため common 側に残す。

common 化の手順は次の順で行った（すべて完了）。

1. spec 専用 9 フィールドを `SpecDecodeContext`
   （`shelf/inference/execution/spec_decode_context.h`）へ分離した。
   参照は `ctx.<field>` から `ctx.spec.<field>` へ統一した。
2. `program_executor` / `optimized_dispatch` / `physical_launch` /
   9 種の dispatch / 7 種の fusion / `kv_calib_dump` / `decode_perf_stats` を
   `shelf/inference/execution/` へ移動し、namespace を
   `shelf::execution` へ統一した（42 ファイル）。
   qwen35 側の利用は `models/qwen35/runtime/execution_types.h` の
   using 宣言（113 識別子）で集約した。
3. correctness 参照実装（`model_dispatch_correctness` + `detail/*.inc` +
   `standalone/`）は `shelf/kernels/correctness` へ移動した。
   依存していた `ModelDispatchStateView` / `DispatchStagingPool` /
   `ProgramStagingMeta` も generic 層へ移設済み。
4. `Qwen35KernelMode` / `read_qwen35_kernel_mode` は汎用機能のため
   `KernelMode` / `read_kernel_mode` へ改名した。
   環境変数 `SHELF_QWEN35_KERNEL_MODE` は公開インターフェース
   （`docs/perf/current.md` の測定条件を含む）のため不変とした。
   `execute_program*` 本体は common として移動し、
   Qwen35 に残るのは `executor.hip`（Executor 本体と spec decode 統合）のみ。

移動後、`shelf/inference` / `shelf/kernels` から
`models/qwen35` / `qwen35` 文字列を参照しないことを
境界テストが静的に検証している。

## Phase 4: lowering（B）

`model/lower_to_primitives.cpp` の `lower_qwen35_to_primitives()` 全体は generic 化しない。
Qwen35 と Qwen4Exp は forward topology が異なる。

- Qwen35: RMSNorm → GDN/full attention → residual → RMSNorm → dense SwiGLU → residual
- Qwen4Exp: HyperConnection → GDN/QSA → HyperConnection → MoE → HyperConnection

共有する単位は次の small builder / validation のみ。
配置は `models/common/qwen_hybrid/`。

| 記述 | 分類 |
| --- | --- |
| `GdnGeometry` | B |
| `GatedAttentionGeometry` | B |
| `validate_gdn_geometry` | B |
| `validate_gated_attention_geometry` | B |
| `emit_gdn` | B |
| `emit_gated_attention` | B |
| `lower_qwen35_to_primitives` 全体 | C |
| decoder layer topology | C |

これらの API は `Qwen35TextConfig` / `Qwen35LayerWeights` を直接受け取らず、
model-neutral な small POD geometry と weight view を受け取る。

### 実施状況

Phase 4 は完了済みである。

- `geometry_check.h`（`checked_u32` / `checked_product` / `checked_double` /
  `validate_matrix`）を共通化し、Qwen35 側のローカル定義を using 宣言へ置換
- `gdn_geometry.h` / `attention_geometry.h` に geometry POD と
  `validate_gdn_geometry` / `validate_gated_attention_geometry` を抽出
- `emitter.h`（`Emitter` = `PrimitiveGraph` 操作ラッパー）を移設
- `gdn_lowering.h` に `emit_gdn`、`attention_lowering.h` に
  `emit_gated_attention` を抽出
- `lower_to_primitives.cpp` は 1299 行 → 1030 行。
  `lower_qwen35_to_primitives` 本体と decoder topology、
  MLP / MTP / residual 構築は Qwen35 側に残す（分類どおり C）

Qwen4Exp の lowering はこの共通 builder を呼び、
HyperConnection / MoE の topology を自前のファイルで構築する。

## Phase 5: PrefixCache（B/C）

`prefix_cache.cpp`（約 1400 行）は model 間で snapshot 対象が異なる。

- Qwen35: KV / GDN state / spec ring
- Qwen4Exp: KV / GDN FP16 state / QSA indexer key state / PLE conv state / PLE ngram context

（GDN state は Qwen4Exp も FP16 である。`docs/developer/qwen4exp.md` 5.0 の
精度方針を正本とする。）

`Prefix radix tree` / `LRU` / `page sharing` の metadata algorithm は generic にできるが、
model-specific state snapshot は model 側に残す。virtual callback は入れない。
まず `PrefixIndex`（metadata）の抽出のみを行う。

### 調査結果（2026-10-23）

`prefix_cache.cpp` 1592 行 / `prefix_cache.h` 433 行の構成を分析した。

| 領域 | 行（概算） | 型依存 | 分類 |
| --- | ---: | --- | --- |
| radix tree（`find_child` / `walk` / `split_node` / `insert_path`） | 170 | なし | A |
| entry / LRU（`find_entry` / `find_lru_entry` / `remove_entry`） | 45 | なし | A |
| page 管理（`alloc_page` / `release_page` / cold page） | 50 | `PagedKVPool` | A |
| GDN state slot（`acquire_entry_state_slot` / `demote_entry_state_to_cold`） | 45 | `GdnStatePool` | A |
| spec ring（`acquire_entry_ring_slot` / `save_entry_ring` / `load_entry_ring`） | 50 | `SpecRingRef` / `SpecRingOut` | C |
| cold / hot migration（`copy_page_*` / `copy_state_*`） | 360 | `PagedKVPool` / `GdnStatePool` / ring | A+C |
| save / restore 統合（`prepare_save` / `fill_pages` / `restore`） | 580 | 上記の合成 | B |

重要な発見は、`PrefixCache` が直接依存する
`PagedKVPool` / `GdnStatePool` / `PagedSequenceState` は
Phase 2 で `shelf::state`（generic 層）へ移動済みだということである。
つまり KV page と GDN state の snapshot は Qwen35 / Qwen4Exp で共有でき、
現状の `PrefixCache` を Qwen4Exp がほぼそのまま使える。
真の model 固有は spec ring（`SpecRingRef` / `SpecRingOut` /
`PrefixEntry.ring_slot` の 3 フィールド / ring buffer 管理）のみである。

Qwen4Exp が追加する `QSA indexer key state` / `PLE conv state` /
`PLE ngram context` は、spec ring と同じ「model 固有 state スロット」の
パターンに収まる。懸念される失敗パターンは、
`PrefixEntry` に QSA / PLE フィールドが平積みされて全 model union になることである。

判断と方針:

- 現時点では `PrefixIndex` 抽出を行わない。実施順 6（Qwen4Exp 実装後に
  本当に重複した部分だけ 2 回目の共通化を行う。PrefixCache /
  ContinuousBatcher はこのタイミングで判断する）に従う。
- Qwen4Exp 実装時に snapshot 対象が実際のコードとして重複した時点で、
  `PrefixIndex`（radix tree / LRU / page 追跡）の抽出と、
  model 固有 state スロットの共通化（slot index + kind 形式）を判断する。
- その際も virtual callback は入れない。state の save / restore 本体は
  model 側に置く。

## 共通化しない対象

- MTP（Qwen4Exp の forward contract が未確定）
- HyperConnection / Gated Residual
- QSA indexer
- MoE（512 experts / top10 / shared expert）
- PLE / NGramEmbedding
- final hyper_connection_mixer
