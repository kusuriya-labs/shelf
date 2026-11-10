# EXL3 family P2 リファクタリング結果（内部構成・名前空間・テスト名）

## 目的

`exl3_family/` のファイル所有者、内部名前空間、CTest 名を一致させた。
公開 launch API・ABI・数値・性能は変更していない。

## Phase 1–2: ファイル所有者と名前空間

| 旧 | 新 |
|---|---|
| `src/.../linear/exl3_mul1.hip` | `src/.../linear/exl3_family/mul1/linear.hip` |
| `src/.../exl3_family/verify.hip` | `src/.../exl3_family/mul1/verify.hip` |
| `src/.../exl3_family/launch_plan.cpp`（mul1 部分） | `src/.../exl3_family/mul1/launch_plan.cpp` |
| `src/.../linear/exl3_mul24.hip` | `src/.../exl3_family/mul24/rotate_quantize.hip` |
| `src/.../exl3_family/decode.hip` | `src/.../exl3_family/mul24/decode.hip` |
| `src/.../exl3_family/prefill.hip` | `src/.../exl3_family/mul24/prefill.hip` |
| `src/.../exl3_family/prefill_rotate.hip` | `src/.../exl3_family/mul24/prefill_rotate.hip` |
| `src/.../exl3_family/grouped.hip` | `src/.../exl3_family/mul24/grouped.hip` |
| `src/.../exl3_family/workspace.cpp` | `src/.../exl3_family/mul24/workspace.cpp` |
| `src/.../exl3_family/launch_plan.cpp`（mul24 部分） | `src/.../exl3_family/mul24/split_plan.cpp` |
| `src/.../exl3_family/launch_plan.cpp`（共通） | `src/.../exl3_family/variant.cpp` |
| `include/.../exl3_family/decode_mul1.h` | `include/.../exl3_family/mul1/decode.h` |
| `include/.../exl3_family/launch_plan.h` | `include/.../exl3_family/mul1/launch_plan.h` |
| `include/.../exl3_family/decode_mul24.h` | `include/.../exl3_family/mul24/decode.h` |
| `include/.../exl3_family/workspace.h` | `include/.../exl3_family/mul24/workspace.h` |
| `include/.../exl3_family/epilogue.h` | 未参照のため削除 |

名前空間:

- mul1 固有 helper / host plan: `shelf::kernel::exl3_detail::mul1`。
- mul24 固有 helper / host workspace: `shelf::kernel::exl3_detail::mul24`。
- 共通 helper（`fwht128`、`fwht128_smem`、trellis、`resolve_group_plan`、
  `exl3_variant_supported`、`apply_execution_policy`）: `shelf::kernel::exl3_detail`。
- 公開 API と `exl3_mul24_gemm_splits` は `shelf::kernel` のまま。

主な改名:

- `Exl3Mul1Traits` → `mul1::Traits`、`Exl3Mul24Traits` → `mul24::Traits`
- `Exl3LaunchPlan` → `mul1::LaunchPlan`
- `make_mul1_launch_plan` → `mul1::make_launch_plan`
- `exl3_mul1_pick_n_tiles` → `mul1::pick_n_tiles`
- `Exl3Mul24WorkspaceLayout` → `mul24::WorkspaceLayout`
- `make_mul24_workspace_layout` → `mul24::make_workspace_layout`
- `apply_execution_policy` は `types.h` へ移動

`launch_gemm_exl3_w16a16_grouped` は公開 API の例外として維持し、mul24 dispatch を
含んだまま `mul1/linear.hip` に置く。

## Phase 3: テスト名

| 旧 | 新 |
|---|---|
| test_exl3_codec | test_exl3_mul1_codec |
| test_exl3_int8_decode | test_exl3_mul24_codec_decode |
| test_exl3_linear | test_exl3_mul1_linear |
| test_exl3_dense | test_exl3_mul1_dense |
| test_exl3_grouped | test_exl3_mul1_grouped |
| test_exl3_workspace | test_exl3_mul24_workspace |
| test_exl3_int8_linear | test_exl3_mul24_linear |
| test_exl3_int8_grouped | test_exl3_mul24_grouped |
| test_exl3_int8_prefill_rotate | test_exl3_mul24_prefill_rotate |
| test_exl3_int8_tp | test_exl3_mul24_tp |
| bench_exl3_int8_grouped | bench_exl3_mul24_grouped |
| exl3_codec_golden.h | exl3_mul1_codec_golden.h |
| exl3_int8_fixture.h | exl3_mul24_codec_fixture.h |

横断・統合のため維持: `test_exl3_contracts`、`test_exl3_launch_plan`、
`test_exl3_verify_policy`、`test_exl3_variant`、`test_exl3_dispatch_decision`、
`test_dflash2_exl3_head`。

## 検証

- フルビルド成功。
- EXL3 required テスト 15 本中 14 本 PASS。`test_exl3_mul1_grouped` のみ
  MoE fixture 不在で失敗（P0 から継続する既知の未カバー）。
- 実モデル prefill 2048 token: 1195.4 ms（P0 ベースライン約 1192 ms、差はばらつき）。
- teacher-forced 比較: KL=0、max abs delta=0、PPL 差分 0（bit 一致）。
- `git diff --check` 成功。
- 旧ヘッダー / 旧 source / 旧テスト名 / 旧 host シンボルの参照なし。
- `LINEAR_EXL3_INT8` などの永続化識別子と `SHELF_EXL3_INT8_*` は意図どおり残す。

## 未実施・注意

- `test_exl3_mul1_grouped` は外部 MoE fixture が必要で、この環境では未カバー。
- `src/apps/bench/exl3_int8.hip` と CLI `shelf-bench exl3-int8` は互換性維持のため
  リネーム対象外。
- `docs/research/` の過去記録は当時の事実として変更しない。
