# Executor / TP ランタイムの共通化: 依存調査と保留判断

2026-11-17。

## 1. 動機

hy_v4 で TP=2 を行う際、TP ランタイムが
`src/shelf/models/qwen35/runtime/tensor_parallel.cpp` にあるため、qwen35 配下の
ファイルを編集する必要が生じる。新しい family の実装作業としては直感に反する。

現行の契約は `docs/developer/tensor_parallel_execution.md` の
「multi-rank runtime の family 境界」と `docs/developer/architecture.md` にあり、
`TpRankRuntime` / `TpCoordinator` は `shelf::qwen35` に置くが family をまたぐ判断は
持たない、としている。本調査はこの記述が配置として正しいかを検証するもの。

## 2. 調査結果

### 2.1 Executor の依存

`include/shelf/models/qwen35/runtime/executor.h` の `Executor` 本体は、
ほぼすべて共通層の型で構成されている。`ExecutorConfig`、`RuntimeModelView`、
`LayerOverrideSource`、`ExecutorModelInput`、`GraphCacheSlot`、`BatchExecutionOutput`、
`PendingBatch`、`ExecuteBatchOptions` はモデル固有の定数を持たない。

state pool（`SequenceSlotPool` / `GdnStatePool` / `PagedKVPool`）はすでに
`include/shelf/inference/state/` の `shelf::state` にある。
`execution_types.h` / `scheduling_types.h` / `state_types.h` / `gdn_spec_history.h`
は `shelf::execution` / `shelf::scheduling` / `shelf::state` の using であり、
型定義は持たない。

移行時に qwen35 側に残す必要がない主な型:

- `runtime::DecodeBackend`（`decode_backend.h:9-11`）は中身が `enum class { Hip }` のみ
- `PipelineStage` は実体が `shelf::qwen_hybrid::PipelineStage`
- `Qwen35TensorParallelContext` は
  `include/shelf/models/common/qwen_hybrid/tensor_parallel_context.h:14-32` の alias
- `EffectiveRopeProfile` は RoPE / YaRN の一般パラメータのみ

### 2.2 移行を阻む結合点

Executor が family を名乗らない設計になっているにもかかわらず、次の 4 点が残る。

1. `create_model_executor`（`executor.h:264-270`、`executor.hip:2846-2882`）が
   `Qwen35Model&` を直接受け、`lower_qwen35_to_primitives` を呼ぶ。
2. 文字列による family 判定。`executor.hip:1009` の
   `declared_model_type == "qwen3_5"` と `executor.hip:1030` の `"plamo3"`。
3. GDN layout の検証（`executor.hip:2003-2056`）が
   `make_qwen35_gdn_state_layout_local` / `qwen35_stage_gdn_layers` /
   `qwen35_stage_attention_layers` を使う。
4. RoPE テーブル初期化（`executor.hip:2414-2451`）の
   `EffectiveRopeProfile::Type::Yarn` 分岐。

workspace 見積もりの `exl3_workspace_bytes` と `qwen35_padded_row_count` は
namespace と名前のみが qwen35 で、ロジックは model 非依存。

### 2.3 影響範囲

- `executor.h` を直接 include するコードファイル: 31 個
  （`include/shelf/` 12、`src/` 7、`tests/` 12）
- `qwen35::Executor` 系の明示修飾を持つファイル: 24 個
- 和集合: 40 個
- CMake / 境界テスト: `cmake/targets.cmake:433,449,455`、
  `tests/unit/test_architecture_boundaries.cpp:171-172,222-228`

### 2.4 `shelf_runtime` への直接配置が不可な理由

`shelf_runtime` は `cmake/targets.cmake:188-202` で 8 ソースと
`shelf_core` / `shelf_gpu` / `shelf_kernels_fused` のみに依存する。
これを守る仕組みが 3 重にある。

1. `tests/unit/test_architecture_boundaries.cpp:171-172` が
   `{"shelf_runtime", "shelf_qwen35"}` と `{"shelf_runtime", "shelf_qwen35moe"}`
   の禁止エッジを検査する。
2. 循環依存。既存の `shelf_qwen35 → shelf_runtime`
   （`cmake/targets.cmake:339`）に加えて `shelf_runtime → shelf_qwen35` を
   作ることはできない。
3. 文字列検査。`test_architecture_boundaries.cpp:222-228` が
   `include/shelf/runtime/**` と `src/shelf/runtime/**` に
   `qwen35` / `Qwen35` / `shelf/models/` が現れないことを検査する。

Executor が必要とする `shelf_inference_*` / `shelf_kernels_correctness` /
`shelf_qwen35_kernels_optimized` / `shelf_qwen35` は `shelf_runtime` に
追加できない。

## 3. 選択肢と採否

| 選択肢 | 内容 | 判断 |
| --- | --- | --- |
| `shelf_runtime` へ直接配置 | Executor を `shelf_runtime` に移す | REJECT。境界テストと循環依存で不可 |
| `shelf_executor` を新設 | 兄弟ターゲットを作り、qwen35 固有部分を注入コールバックへ追い出す | 保留。境界テストの契約を緩和する設計判断が必要 |
| 現状維持 | 依存の実態に合わせて qwen35 配置を維持 | 保留。hy_v4 の具体化を待つ |
| 調査結果のみ記録 | 本ファイルのように依存マップと選択肢を残す | ACCEPT |

`docs/research/20261105/moe_pipeline_parallel.md:28-42` の採否判断と同じく、
family をまたぐ層は `shelf::models` の interface 経由で注入する方式が
既に採用されている。Executor 本体を動かすかどうかは、hy_v4 が
qwen35 runtime を再利用するかどうかで決まる。

AGENTS.md は「仮定の将来機能を軸にアーキテクチャを再設計しない」と定めており、
hy_v4 は現時点でリポジトリに存在しない。そのため今回は設計判断を保留する。

## 4. 今回の対応

TP prefix cache の VRAM reserve を capture slot に合わせる起動修正を
分離してコミットした。これは Qwen4Exp の radix + 262144 起動を可能にする
独立したバグ修正であり、Executor 共通化とは依存しない。

Executor 共通化は hy_v4 の実装が具体化した時点で、本ファイルの
選択肢から改めて選ぶ。
