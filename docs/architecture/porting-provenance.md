# 移植由来（Porting Provenance）

計画 Phase 0 の要件「取り込むコードについて、由来をファイル単位で管理する」に対応する記録。

## ライセンス方針

- **shelf:** MIT（`LICENSE`, Copyright (c) 2026 kusuriya-labs）。
- **PhaseShift**（`~/work/phaseshift-bf16-fp8-tp4`）: **自己制作**（CopenDeCamp = 本プロジェクト所有者）。
  第三者ライセンス上の義務はなく、**著作権表記は不要**。移植元は追踪目的で下表に記録するのみ。
- **XeStrata**（`~/work/references/XeStrata-exl3`）: **LGPL-3.0-or-later**（REUSE.toml 準拠、
  一部ファイルは MIT / Apache-2.0 等）。**Phase 1 ではコードを一切コピーしていない。**
  Kernel 配置規則などの「設計」のみ参照している（著作権で保護される表現の複製は行わない）。
  LGPL コードを移植する PR では、ファイル単位の SPDX ヘッダ保持・LGPL 本文の同梱・
  再リンク可能性（Static Library 運用時は特别注意）を必須とする。法務レビューは別途。

## Phase 1 移植台帳（移植元: PhaseShift、著作権表記不要）

| shelf ファイル | 移植元（PhaseShift） | 変更内容 |
|---|---|---|
| `include/shelf/status.hpp` | `include/phaseshift/core/status.h` | namespace を `shelf` に。`hip_error`/`rccl_error` を削除（backend 固有）。`not_found` を追加。`checked_align_offset_from_base` を削除（memory 実装時の再導入候補）。マクロ `SHELF_RETURN_IF_ERROR`/`SHELF_TRY` を追加 |
| `src/foundation/include/shelf/foundation/numeric.hpp` | `include/phaseshift/core/memory/types.h` | namespace を `shelf` に。`bf16_to_f32` を追加（新規実装） |
| `tests/architecture/test_architecture_boundaries.cpp` | `tests/unit/test_architecture_boundaries.cpp` | shelf の依存ルールに合わせて規則を全面書き換え（機構：include 走査＋モデル名 raw 走査は踏襲） |

## PR-004（Phase 3 前倒し: BF16 Weight Reader / Catalog）

- `src/artifact/safetensors/`・`src/artifact/common/`・`src/weights/catalog/` は
  **新規実装**（PhaseShift `io/safetensors_reader` の API 形状のみ参照。mmap でなく
  ifstream、pimpl でなく素の RAII。仕様の直接実装なので移植コードはない）。
- ベンダー依存: **nlohmann/json 3.12.0（MIT）** を `third_party/nlohmann/` に同梱
  （PhaseShift の vendor と同一ファイル、改変なし。詳細は third_party/README.md）。
  artifact の PRIVATE 依存に限定し、公開ヘッダに json 型を漏らさない。
- safetensors は PhaseShift で先に読み書き検証済みのコンテナ。GGUF/EXL3 の
  マルチコンテナ解決（Phase 3 完了条件）は GGUF Reader 導入時に拡張。

## PR-005（Phase 4: Primitive IR / CPU 参照 Executor）

`src/ir/`・`src/backend/cpu/` は **新規実装**（移植コードなし）。
PhaseShift の inference/program 系は Phase 6 の Lowering 検討時に設計参照予定。
Primitive 集合は Phase 4 候補から GDN・ExpertCompute を除外（Layer Parity 段階で追加）。

## PR-006（Qwen3.5-4B text path 動作・CPU 参照）

`src/model/qwen35/`・IR 拡張 Kind（slice/reshape/sigmoid_gate/rmsnorm_gated/
causal_conv_step/gdn_step/attention_gqa_step）・`apps/shelf_run.cpp` は **新規実装**。

数式契約の根拠（設計参照のみ・コード移植なし）:
- `~/work/references/qwen35-f64-oracle/ARCHITECTURE.md`（transformers op-for-op 仕様。
  oracle 側の `py/ref_forward.py` は transformers 5.20 の rope/GDN API 変更に伴い
  oracle リポジトリ側で適合修正済み）
- Ground truth: 同 oracle `--load full --pure`（f64）の logits.bin。
  shelf f32 CPU と top-1 一致・最大差 <5e-5。

PhaseShift（`~/work/phaseshift-bf16-fp8-tp4`、自己制作）の qwen35 runtime family は
GPU 高速化段階で kernel・paged KV を移植予定（未移植）。

## PR-008a（PhaseShift HIP kernel 移植開始）

`src/backend/hip/kernels/gemm/linear_bf16.hip` と公開ヘッダは PhaseShift
`kernels/optimized/linear/{bf16.h,config.h}` の **self-work vendoring**
（名前空間 `ps::kernel`→`shelf::hipk`・型は shelf `bf16_t` へ置換のみ、
数式変更なし）。PhaseShift は自己制作のため著作権表記不要。

## 設計のみ参照（コード未移植）

| shelf 側の設計 | XeStrata からの参照点 |
|---|---|
| Kernel を演算名で整理する配置規則（plan §6） | `src/kernels/`, `include/strata/` のレイアウト（表現は複製せず） |
| 後続 Phase の移植候補 | `src/artifact/`（EXL3 Reader）、`src/core/exl3_expert_runtime.*`、`src/prefill/` 等 — Phase 5+ で LGPL 手続きを踏んで個別審査の上移植 |

| 設計参照点（PhaseShift 側） | 内容 |
|---|---|
| `include/phaseshift/quantization/quant_contract.h` ほか | QuantSpec / Codec 契約（Phase 3 で移植候補） |
| `include/phaseshift/core/memory/tensor.h` | GPU Tensor ビュー（Phase 2 で Device 層と合わせて移植候補） |

## PR-003（Phase 2: Device/Buffer/Queue/Event）

`src/backend/common/` と `src/backend/hip/` は **新規実装**（コード移植なし）。
PhaseShift の `core/gpu/`・`core/memory/` は設計のみ参照（RAII・error マッピング方針）。
環境依存の事実記録: GPU は gfx1201 (R9700) x4、ROCm SDK は pip の `_rocm_sdk_devel`。
SYCL ランタイムは未導入のため `backend/sycl` は空で保持（ユーザー指示: sycl は将来必ず使う）。

## PR-008（PhaseShift HIP kernel 資産の完全移植 + GPU step executor）

`src/backend/hip/kernels/` の以下は PhaseShift `kernels/optimized/` の
**self-work vendoring**（`ps::kernel`→`shelf::hipk`、`ps::bf16_t`→
`shelf::bf16_t`、include 書き換えのみ。数式・カーネルシンボル
（`phaseshift_*`）は不改変。PhaseShift は自己制作のため著作権表記不要）:

| shelf TU | 移植元 |
|---|---|
| `attention/attention_paged_decode.hip` / `attention/attention_kv_append.hip` | `kernels/optimized/attention/{paged_bf16,kv_append}.hip`（2 プール separate-pool addressing、ABI は `paged-kv-abi.md` rev2 に一致） |
| `attention/attention_rope.hip` | `kernels/optimized/rope/`（f32 入力・device inv_freq、D=256/rotary=64 ペアカーネル） |
| `gdn/gdn_{qk_prepare,conv_silu,conv1d,recurrence}.hip` | `kernels/optimized/gdn/`（conv state は shelf では fp32 `[slot][layer][C][3]` oldest-first に拡張 — PhaseShift の bf16 `[3][C]` からの**意図的乖離**、recurrent state fp32 不変） |
| `elementwise/elementwise_{rmsnorm,ops,embedding,l2_normalize}.hip` | `kernels/optimized/{norm,elementwise,embedding}/`（zero-centered `ONE_PLUS` gamma ネイティブ、embedding error-word プロトコルの `ProgramStatus` は wire 値同一の自前宣言） |
| `elementwise/aux_ops.hip` | 新規（row slice / sigmoid-gate / argmax / bf16↔f32 変換） |

`src/backend/hip/step_executor.{hpp,cpp}` は **新規実装**: IR の全 Kind を
上記カーネルへディスパッチ（bf16 激活 / f32 GDN 状態・logits、KV は
bf16 separate pools、1 文脈 identity ブロックテーブル）。CPU reference
executor が数値 golden のままで、GPU 経路は greedy 一致で検証
（Qwen3.5-4B: 21 token 全一致、約 18 ms/token ≈ 55 tok/s、CPU 比約 190 倍）。

## PR-009d（backend/hip カーネル配置整理）

`src/backend/hip/kernels/` と `include/shelf/backend/hip/kernels/` を
ファミリー別サブディレクトリへ（`.hip` と公開ヘッダは同一ミラー配置、
共有 device ヘルパは `kernels/detail/` に残置）:

| ディレクトリ | 内容 |
|---|---|
| `gemm/` | linear_bf16（ExactRows / ExactRowsDot2 / Wmma 系） |
| `attention/` | paged decode・kv_append・rope・kv 型 |
| `gdn/` | qk_prepare・conv(+silu)・recurrence・型 |
| `elementwise/` | rmsnorm・ops(silu/mul/swiglu/residual)・embedding・l2_normalize・aux(glue) |

CMake は `GLOB_RECURSE`。レイアウトの動機は XeStrata の演算名整理規則
（設計参照のみ）と rebuild-plan §6。公共インクルードパスは
`shelf/backend/hip/kernels/<family>/<name>.h` に統一。
