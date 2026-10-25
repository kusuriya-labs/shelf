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

## 設計のみ参照（コード未移植）

| shelf 側の設計 | XeStrata からの参照点 |
|---|---|
| Kernel を演算名で整理する配置規則（plan §6） | `src/kernels/`, `include/strata/` のレイアウト（表現は複製せず） |
| 後続 Phase の移植候補 | `src/artifact/`（EXL3 Reader）、`src/core/exl3_expert_runtime.*`、`src/prefill/` 等 — Phase 5+ で LGPL 手続きを踏んで個別審査の上移植 |

| 設計参照点（PhaseShift 側） | 内容 |
|---|---|
| `include/phaseshift/quantization/quant_contract.h` ほか | QuantSpec / Codec 契約（Phase 3 で移植候補） |
| `include/phaseshift/core/memory/tensor.h` | GPU Tensor ビュー（Phase 2 で Device 層と合わせて移植候補） |
