# EXL3 / mul24 P0 リファクタリング結果

## 目的と範囲

`docs/research/20261110/exl3_refactor_p0_baseline.md` の状態から、挙動・精度・性能を
変えずに公開ヘッダー・命名・共通処理・Launch/Workspace の責務を整理した。カーネル最適化
（FWHT 実装変更、WMMA タイル、split-K 改良）は含めない。

## 実施内容

| Phase | commit | 内容 |
|---|---|---|
| 0 | `de3aec2` | ベースライン固定（テスト 10 本、リソース、性能） |
| 1 | `6f1a6a9`, 以降 | device helper を `detail/` へ移動、公開ヘッダーを `exl3.h` に統合、`exl3_int8` → `exl3_mul24` |
| 2 | `0d5b262` | I/O 契約を `exl3.h` / `detail/types.h` に明記、境界テスト追加 |
| 3 | 以降 | mul24 workspace layout を `make_mul24_workspace_layout()` へ集約、golden テスト |
| 4 | 以降 | mul1 / mul24 の launch policy を host-only の `exl3/launch_plan.cpp` へ分離、golden テスト |
| 5 | `e38d102` | `Exl3ExecutionPolicy` を導入、verify の固定 tuning capacity を検証 |
| 6 | 以降 | `exl3_mul24.hip` を `exl3/decode.hip` / `exl3/prefill.hip` に分割、共通 helper を `detail/mul24_common.h` へ |

目標構成との差分:

- `exl3_mul24_grouped.hip` は `exl3/grouped.hip` へ移動した。
- `exl3_mul24_prefill.hip` は `exl3/prefill_rotate.hip` へ移動した。
- mul1 の `exl3.hip` は別 codebook のため今回は物理分割していない。
- `dispatch.hip` / `verify.hip` は設けていない。phase ごとの起動列は現状
  `optimized_dispatch.hip` にあり、P0 ではその所有・エラー処理を壊さないことを優先した。
  新しい API は `Exl3ExecutionPolicy` と `apply_execution_policy` までを用意している。

## 検証結果（commit `615dd65` との比較）

テスト（`HIP_VISIBLE_DEVICES=1`、直接起動）:

| テスト | 結果 |
|---|---|
| test_exl3_codec | PASS |
| test_exl3_dense | PASS |
| test_exl3_int8_decode | PASS |
| test_exl3_contracts（新規） | ALL PASS |
| test_exl3_workspace（新規） | ALL PASS |
| test_exl3_launch_plan（新規） | ALL PASS |
| test_exl3_verify_policy（新規） | ALL PASS |
| test_exl3_linear | PASS |
| test_exl3_int8_grouped | ok |
| test_exl3_int8_linear | ALL PASS |
| test_exl3_int8_prefill_rotate | bitwise PASS |
| test_exl3_int8_tp | ALL PASS |
| test_dflash2_exl3_head | rel L2 0.007154（ベースラインと同一） |

`test_exl3_grouped` は MoE fixture が無いため引き続き未カバー。

品質（teacher-forced、context 2048、64 位置、語彙 248320、既定経路）:

- `QUALITY_KL_MEAN=0`、`QUALITY_LOGIT_MAX_ABS_DELTA=0`、`QUALITY_TOP1_AGREEMENT=1`、
  `QUALITY_TOP5_OVERLAP=1`、`PPL_DELTA=0`。ベースライン dump と bit 一致。

性能（既定経路、2048 token、BF16 KV）:

- Prefill wall p50 約 1192 ms（ベースライン約 1200 ms、差は測定ばらつき）。

リソース（`-Rpass-analysis=kernel-resource-usage`、gfx1201）:

- `exl3_mul24_rotate_quantize_kernel`: SGPR 14 / VGPR 13 / LDS 528 / occ 16（一致）
- `exl3_mul24_gemm_kernel<1,1>`: 40 / 97 / 0 / 12（一致）
- `exl3_mul24_gemm_kernel<1,2>`: 46 / 145 / 0 / 9（一致）

## 完了チェックリスト

- [x] ベースラインが commit とともに保存済み
- [x] 公開ヘッダーは `exl3.h` 1 本（`detail/*.h` は内部）
- [x] mul1 / mul24 と Activation 精度を混同しない
- [x] I/O と stride の単位・制約を記述
- [x] mul24 workspace layout 算出が 1 箇所
- [x] Launch Plan が純粋計算として独立し golden 一致
- [x] Verify の tuning capacity と active rows を区別
- [x] Prefill / Decode / Grouped の実装単位を分離（Verify は policy で表現）
- [x] Grouped と flagged-row に回帰なし
- [x] Unit / Integration / Quality の回帰試験合格
- [x] GPU 性能・リソースの回帰試験合格（既定経路）
- [x] 旧 API / 旧ファイルの参照が残っていない
- [ ] `test_exl3_grouped` は fixture 不在で未カバー
- [ ] mul1 `exl3.hip` の物理分割と `dispatch.hip` / `verify.hip` は次段階
