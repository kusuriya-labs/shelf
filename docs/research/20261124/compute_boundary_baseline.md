# compute 責務分離リファクタリング記録

日付: 2026-11-24
対象: `src/apps/compute` のモデル非依存化と `shelf::engine` の導入
基準 HEAD: `92273f9c92364ab76e53e2cb1abc299784f115a4`（未コミット変更なし。`run.sh` と
指示書は未追跡のまま保持）

## P0: ベースライン

### 環境

- GPU: Radeon AI PRO R9700、`--device 2` を使用
- build: 既存 `build/`（Release、`SHELF_BUILD_TESTS=ON`、gfx1201）
- モデル:
  - target: `models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`
  - drafter: `models/kusuriya-labs/Qwen3.8-27B-DFlash2-EXL3-INT8-4bpw`

### 単発推論（greedy、max_new_tokens=4、max_seq_len=512）

```text
GENERATED_IDS=60,198,262,220
DECODE_TOKENS=3
MODEL_LOAD_MS=10885.3
PREFILL_MS=62.2
DECODE_TOKENS_PER_SEC=19.25
```

### DFlash2（drafts=2）

```text
GENERATED_IDS=60,198,410,220
DECODE_TOKENS=3
DECODE_TOKENS_PER_SEC=21.93
DFLASH2_ROUNDS=2
DFLASH2_MEAN_ACCEPTED=0.500
```

### MTP

`--mtp --mtp-drafts 2` は model load 後の arena 確保で `hipMalloc failed: out of memory`。
この環境では MTP 単発のベースラインを取得できなかった。変更後も同じモデルでは
arena の KV 確保後に drafter 実行領域が不足し、単発 MTP は完走しない。
MTP の回帰判定は、モデルが小さい環境で `test_compute_mtp_batch.py` /
`test_compute_mtp_prefix.py` を実行する必要がある。

### 境界テスト

`ctest --test-dir build -R '^test_architecture_boundaries$'` は変更前から PASS。

## P1: alias 除去

- `src/apps/compute` から `qwen35::runtime::*` の alias 参照を削除し、
  `shelf::runtime` / `shelf::scheduling` の共通型へ置換した。
- `compute_cli` は `shelf::runtime::SamplingConfig` と
  `shelf::models::max_mtp_drafts()` を使う。
- `compute_common.h` は `shelf::runtime::FinishReason` のみを持つ。
- `compute_runtime.h`（alias だけのヘッダ）は削除した。

検証: `shelf-compute` build PASS、単発 GENERATED_IDS が P0 と一致。

## P2: モデル固有の識別・制約を models へ移動

新規:

- `include/shelf/models/model_info.h` / `src/shelf/models/model_info.cpp`
  - family 表示、geometry、expert 情報、weight encoding 内訳
- `include/shelf/models/model_policy.h` / `src/shelf/models/model_policy.cpp`
  - `ModelRuntimeRequest` → `ResolvedModelOptions`
  - Plamo3 の rope / max_seq / 並列制約
  - Qwen4Exp の DFlash2 drafts 上限、host token ids、private target policy
  - RoPE profile 解決、pipeline hidden width、MTP reserve 計算
- `include/shelf/models/private_target_weights.h` /
  `src/shelf/models/private_target_weights.cpp`
  - `model.language_model.*` tensor 名を app から撤去

`LoadedModel` に `borrow_rank_model` / `borrow_stage_model` を追加し、
TP rank / pipeline stage の family 分岐を models に集約した。

アプリの `main.hip` は family を判定せず、`models::resolve_model_options()` の結果を
`ComputeConfig` に渡すだけにした。

## P3: ComputeSession

新規:

- `include/shelf/engine/compute_session.h`
- `src/shelf/engine/compute_session.hip`
- `src/shelf/engine/compute_session_impl.h`（engine 内部だけが使う Pimpl 定義）
- CMake target `shelf_engine`

`Qwen35ComputeRuntime` を `shelf::engine::ComputeSession` に移した。
公開 API は token 入出力、step、request view、mask、cancel/retire、resource snapshot、
model info、debug capture である。`Executor` / TP rank / arena / prefix cache 実体は
公開ヘッダに出さない。

`request()` は内部 `RuntimeRequest*` ではなく、次回 step まで有効な読み取り専用
`RequestView` を返す。ログ・hidden dump は `copy_logits` / `copy_hidden` /
`copy_target_hidden` / `memory_debug` に分離し、通常経路では D2H copy を行わない。

`src/apps/compute` から `shelf/models/qwen35` / `qwen4exp` / `plamo3` の include と
`qwen35::` / `Qwen35` / `qwen4exp_model()` / `dense_model()` / `moe_model()` 参照が
なくなった。

## P4: single_request の driver 化

- `run_single_shot` は `ComputeSession` の submit/step/request/診断 API だけを使う。
- MTP / DFlash2 の drafter 生成、TP proposal、rank binding、snapshot、spec verify は
  `shelf::engine::SpeculationRunner`（`src/shelf/engine/speculation.hip`）へ移した。
- 表示用の metric は `SpeculationMetric` として返し、app が従来のキーで出力する。

検証: DFlash2 の `GENERATED_IDS` と統計キーが P0 と一致。単発 greedy も一致。

## P5: stdio_server の I/O と実行制御の分離

- `ServeController`（`include/shelf/engine/serve.h` / `src/shelf/engine/serve.hip`）が
  inflight、MTP batch、DFlash2 session、mask upload、cancel、stats を所有する。
- `stdio_server.hip` は JSONL の parse/emit、`poll(2)`、request_id とイベントの対応、
  `ping` / `cancel` / `draft_masks` / `mask` / `shutdown` の受付だけを持つ。
- `nlohmann/json` は engine に入れていない。base64 mask の decode も app 側。
- GPU owner 単一 thread の契約は維持している。

検証:

- `test_compute_service.py`: 26 PASS
- `test_compute_cancel.py`: 8 PASS
- `test_compute_token_mask.py`: PASS
- `test_compute_concurrency.py`: 30 PASS
- `test_compute_dflash2.py`: 99 PASS
- `test_compute_dflash2_tree.py`: PASS
- `test_compute_thinking_budget.py`: 6 PASS（DFlash2 節はモデル指定がなく SKIP）
- `test_compute_input_ids.py`: PASS（fake Plamo3 config の早期エラー順序を維持）
- `test_compute_mtp_batch.py`: 実行環境で SKIP。`Qwen3.8-Flash-Next` を単一 GPU の
  EP=1 で読むと weights が arena に収まらず、target baseline の起動で失敗する。
  コード起因ではなく、この作業環境の GPU 数不足である。

## P6: 依存規約テスト

`tests/unit/test_architecture_boundaries.cpp` を拡張:

- `src/apps/compute` の family include と concrete model / Executor / TP / HIP 参照を検出
- 文字列リテラルとコメントを除去してから型参照を検査（help 文字列の誤検知を回避）
- `include/shelf/engine/**` と `src/shelf/engine/**` の family / serving schema 参照を検出
- MTP は `shelf::runtime::mtp`、DFlash2 は `shelf::runtime` / `shelf::runtime::dflash2` の型を使うため、
  engine に family 例外を置かない
- `shelf-compute` が `shelf_engine` を link し、family / `shelf_models` を直接 link
  しないことを検査
- `shelf_engine` が family target を直接 link しないことを検査
- `shelf_mtp` が family target を直接 link しないことを検査

negative test: `main.hip` に `#include <shelf/models/qwen35/model/qwen35_model.h>` を
一時挿入すると `test_architecture_boundaries` が FAIL し、戻すと PASS することを確認。

## 品質・性能メモ

- 単発 greedy の token 列は P0 と完全一致（`60,198,262,220`）。
- DFlash2 の token 列は P0 と完全一致（`60,198,410,220`）。
- 単発 decode は 19.25 → 19.13 tok/s、DFlash2 は 21.93 → 21.52 tok/s。
  同一 GPU・同一 cache ではないため、計測ノイズの範囲として扱う。
- 通常経路に新しい GPU 同期・D2H copy・allocation は追加していない。
  debug capture は呼ばれたときだけ copy する。
- 演算、量子化、tensor shape、GDN recurrence、KV format は変更していない。

## Dead code 削除（レビュー反映）

2026-11-24 のレビューで、共通 runtime へ移した後も
`src/shelf/models/qwen35/` に旧実装が残っていることを確認した。
CMake の production target には共通側だけが登録されており、旧側は未コンパイルだった。

参照検査:

```bash
find src/shelf/models/qwen35 -type f \( -name '*.cpp' -o -name '*.hip' \) | while read f; do
  grep -Rq --exclude-dir=.git --exclude-dir=build -- "$f" cmake src include tests tools \
    || echo "$f"
done
```

削除:

- `src/shelf/models/qwen35/dflash2/**`
- `src/shelf/models/qwen35/kernels/**`
- `src/shelf/models/qwen35/runtime/dflash2_footprint.cpp`
- `src/shelf/models/qwen35/runtime/dflash2_session_pool.cpp`
- `src/shelf/models/qwen35/runtime/dflash2_spec_batch.cpp`
- `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp`
- `src/shelf/models/qwen35/runtime/gdn_spec_history.hip`
- `src/shelf/models/qwen35/runtime/pipeline_parallel.cpp`
- `src/shelf/models/qwen35/runtime/runtime_request.cpp`
- `src/shelf/models/qwen35/runtime/spec_decode.cpp`
- `src/shelf/models/qwen35/runtime/token_budget_scheduler.cpp`
- 未参照だった `include/shelf/models/qwen35/kernels/**` と
  `include/shelf/models/qwen35/state/**`

残したもの:

- `src/shelf/models/qwen35/runtime/{executor_model,tp_rank_model_dense,pipeline_stage_model_dense}.cpp`
  （`shelf_qwen35_runtime` の production source）
- `include/shelf/models/qwen35/dflash2/**`
  （`tests/support/dflash2_tp_harness.h` が使う alias header）
- `include/shelf/models/qwen35/runtime/**` の alias header
  （他 family / テストが使う）

再発防止として `test_architecture_boundaries` に
`check_no_orphan_model_sources` を追加した。
`src/shelf/models` 配下の `.cpp` / `.hip` が `cmake/targets.cmake` の
どの target にも現れない場合に FAIL する。

## 残る課題

- MTP 単発はこの GPU/モデル組み合わせで arena が不足し、変更前後とも完走しない。
  MTP の最終受入は小さいモデル環境で `test_compute_mtp_batch.py` /
  `test_compute_mtp_prefix.py` を実行する必要がある。
- TP=2 / PP / Qwen4Exp / Plamo3 の実機 E2E はこの作業では再実行していない。
  境界テストと build のみ確認した。
- engine から family 型を完全に除去した。MTP は `shelf::runtime::mtp`、DFlash2 は `shelf::runtime` / `shelf::runtime::dflash2` の型を使う。
  Qwen4Exp 固有の replicated head / per-rank MTP input 判定は
  `TpRankModel` の model-neutral な virtual に移した。
- 追加修正: MTP drafter を `models/qwen35/runtime` から
  `shelf/runtime/mtp`（namespace `shelf::runtime::mtp`）へ移し、DFlash2 の `shelf_dflash2` と対になる `shelf_mtp` target を作った。
  DFlash2 の session / spec decoder は元から `shelf_runtime` にあり、
  engine はそちらを直接参照する。
