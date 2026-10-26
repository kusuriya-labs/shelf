# shelf 依存ルール（Architecture Contract）

> 計画 §3.1・§7 に対応。`tests/architecture/test_architecture_boundaries.cpp` が
> この表を機械的に検査する。表とテストは必ず同期させること。

## 責務と禁止依存

| モジュール | 責務 | やってはいけないこと | include で禁止（一部） |
|---|---|---|---|
| foundation | 基本型、Tensor、エラー処理 | SYCL・モデル依存処理 | model/ir/artifact/quant/weights/compiler/backend/placement/runtime, sycl/, hip/ |
| model | モデル構造と演算意味論 | GPU確保、ファイル読込 | artifact/quant/backend/compiler/runtime/placement, sycl/ |
| ir | 演算とState依存の表現 | Kernel選択 | backend/compiler/runtime/placement, sycl/ |
| artifact | GGUF、EXL3等の解析 | モデルのforward | model/runtime/backend, sycl/ |
| quant | QuantSpec、Codec | モデル固有処理 | model/backend/runtime, sycl/ |
| weights | Tensor解決・Weight Binding | 推論実行 | runtime/api, sycl/ |
| compiler | IR最適化・実行計画構築 | Token生成ループ | runtime/api |
| backend | CPU/SYCL Kernel実行 | Qwen等のモデル判定 | model/compiler/runtime/placement |
| backend/common | Device/Buffer等の中立値型契約 | HIP/SYCL型の露出、万能仮想Backend化 | sycl/, hip/ |
| backend/hip | HIP (ROCm) 実装 | モデル判定、SYCL混在 | sycl/ |
| backend/sycl | SYCL 実装（backend/hip の移植・AdaptiveCpp） | HIP混在、モデル判定 | hip/ |
| placement | デバイス配置・常駐管理 | 演算の数値的定義 | model/runtime, sycl/ |
| runtime | Session・Batch・State・実行制御 | ファイル形式の解析 | artifact/api, sycl/ |
| api | 外部向けFacade | GPU Kernelの直接実装 | sycl/ |
| include/shelf | 最小の外部公開API | 内部モジュールの露出 | sycl/, backend/, foundation/ |

## 追加規則

- **モデル名の禁止:** `qwen / llama / deepseek / mistral / phi-` のリテラルは
  `src/model/` 以外ではソース本文中も禁止（`if (model == Qwen...)` 分岐を防ぐ）。
  ただし `apps/` は composition root（モデル結合を明示的に行う層）として走査対象外。
- **SYCL型の禁止（現状）:** `src/backend/sycl/` は SYCL ランタイム導入まで空。
  導入後は `<sycl/sycl.hpp>` の include を `src/backend/sycl/` のみに限定する。
- **HIP型の封じ込め:** `<hip/...>` を include できるのは `src/backend/hip/` のみ。
  公開 Facade・runtime・compiler から HIP/SYCL 型は漏らさない
  （中立語彙は `backend/common` の DeviceInfo/BufferView 等）。
- **循環依存:** CMake の `target_link_libraries` が依存の単方向を保証する。
  `shelf_X` が `shelf_Y` にリンクされる場合、`shelf_Y` のヘッダが `shelf_X` を
  include してはならない（上表がこれを内包する）。
- **公開 API:** `include/shelf/` は Facade（PR-008+ で `shelf_api` が実装）。
  モジュール間共有ヘッダは `src/<module>/include/shelf/<module>/` に置く。

## 見直しの運用

ルールを緩める変更は、契約文書の更新とセットでレビューする。
禁止を破るコードが必要な場合、責務境界の設計ミスの兆候
（計画 §10「設計見直しのシグナル」）として扱う。
