# EXL3 / mul24 P1 リファクタリング ベースライン（P1-0）

## 前提

P0 手順書の Phase 0–7 完了後の状態を起点とする。P0 の結果は
`docs/research/20261110/exl3_refactor_p0_baseline.md` と
`docs/research/20261110/exl3_refactor_p0_result.md` に記録した。

## 環境

- P0 最終 commit: `37c42b1`（main）
- HIP 7.15.26333 / AMD clang 23.0.0git
- ROCm SDK: `/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`
- GPU: AMD Radeon AI PRO R9700（gfx1201）、driver 7.1.9.31600000
- CMake: Release、`SHELF_BUILD_TESTS=ON`、`SHELF_BUILD_BENCHMARKS=ON`、gfx1201
- 実行隔離: `HIP_VISIBLE_DEVICES=1`、`flock /tmp/r9700.lock`

## 現在のファイル構成

```text
include/shelf/kernels/optimized/linear/
  exl3.h                      # 唯一の公開ヘッダー
  detail/
    exl3_decode_device.h      # mul1 device helper（fwht128 float4, trellis decode）
    exl3_group_plan_device.h  # group plan resolver
    mul24_common.h            # mul24 共通定数・device helper（fwht128_smem）
    types.h                   # I/O 契約
    workspace.h               # mul24 workspace layout
    launch_plan.h             # mul1/mul24 launch policy
src/shelf/kernels/optimized/linear/
  exl3.hip                    # mul1（未分割）
  exl3/
    decode.hip                # mul24 rotate_quantize / gemm / rotate_out
    prefill.hip               # mul24 rowscale / recon / rotate_out_i32
    prefill_rotate.hip        # mul24 prefill 出力回転（hybrid）
    grouped.hip               # mul24 grouped
    workspace.cpp             # host-only layout
    launch_plan.cpp           # host-only policy
```

## 参照ホットスポット（`rg` 件数）

| 語 | 件数 |
|---|---:|
| exl3 | 1,164 |
| mul24 | 238 |
| fwht128 | 28 |
| decode_tile | 19 |
| splitk | 105 |
| flagged | 48 |
| quantize_dense_w8 | 5 |
| quantize_xrot_x8 | 6 |
| launch_quantize | 9 |
| fast_decode | 9 |
| int8_act | 40 |

`exl3_int8` は量子化層（`include/shelf/quantization/exl3/`、`src/shelf/quantization/exl3/`）と
永続化識別子に残る。P1 でも変更しない。

## テスト対象（P0 で追加した 4 本を含む）

`test_exl3_codec`, `test_exl3_dense`, `test_exl3_int8_decode`, `test_exl3_linear`,
`test_exl3_int8_grouped`, `test_exl3_int8_linear`, `test_exl3_int8_prefill_rotate`,
`test_exl3_int8_tp`, `test_exl3_contracts`, `test_exl3_workspace`,
`test_exl3_launch_plan`, `test_exl3_verify_policy`。

`test_exl3_grouped` は MoE fixture 不在で未カバー（P0 から継続）。

## 直交表（P1 で埋める範囲）

| 軸 | 必須ケース | 現状 |
|---|---|---|
| Phase | prefill / decode / verify | prefill・decode は E2E、verify は split 安定性テスト |
| Codebook | mul1 / mul24 | 両方 `test_exl3_linear` / `test_exl3_int8_linear` |
| Shape | 小M 1–16 / 大M 128–2048 | 単体テストと E2E |
| Backend | direct / recon+dense | `test_exl3_int8_linear` の dense 経路 |
| GEMM | FP16 / INT8 Activation | mul24 grouped の両方、dense は INT8 |
| Output | BF16 / F32 | `test_exl3_int8_linear` の両 dtype |
| Split-K | 1 / >1 / verify 固定 | `test_exl3_launch_plan` / `test_exl3_verify_policy` |
| Grouped | 1/2/4、空 expert、不均等 rows | `test_exl3_int8_grouped`（一部） |
| Special | mul1 ka/half、mul24 BITS、flagged | `test_exl3_linear` / `test_exl3_int8_linear` |

## Gate P1-0

- P0 最終 commit `37c42b1` を基準に固定した。
- P1-10 の回帰はこの commit のテスト結果・品質（KL=0）・性能（prefill 約 1192 ms）・
  リソース値と比較する。
