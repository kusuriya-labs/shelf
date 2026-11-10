# EXL3 mul1 実装分割（P3 相当）

## 目的

`src/shelf/kernels/optimized/linear/exl3_family/mul1/linear.hip`（977 行）に
集中していた dense GEMM・split-K reduce・rotation・recon・epilogue・grouped
dispatch を、責務単位のファイルへ移した。演算ロジックは変更していない。

## 分割結果

| 新ファイル | 主な内容 |
|---|---|
| `mul1/decode.hip` | dense GEMM、split-K reduce、launch orchestration、`launch_gemm_exl3_w16a16`、grouped、pre launch |
| `mul1/prefill.hip` | recon、rotate、`launch_rotate_exl3_rowsum`、`launch_recon_exl3_dense`、`launch_rotate_exl3` |
| `mul1/epilogue.hip` | `exl3_dense_epilogue_kernel` と `launch_epilogue_exl3_dense` |
| `mul1/verify.hip` | 既存の verify 入口（変更なし） |
| `mul1/decode.h` | tile decode helper、`bf16_value`、`kMul1*` 定数 |

行数の目安: decode 557、prefill 138、epilogue 75、verify 15、decode.h 260。

## 維持した境界

- 公開 API（`mul1.h` の `launch_*`）は名前・引数・レイアウト不変。
- launch plan（`mul1/launch_plan.cpp`）と FWHT / trellis の実装は不変。
- `launch_gemm_exl3_w16a16_grouped` は引き続き `decode.hip` に置く。
  `launch_plan<false>` と同一 TU に依存するため、今回の移動では分離しない。

## 検証

- フルビルド成功。
- `test_exl3_mul1_codec` / `_linear` / `_dense` / `_mul24_linear` / `_mul24_tp` /
  `_mul24_prefill_rotate` / `_mul24_grouped` / `_contracts` / `_launch_plan` /
  `_verify_policy` / `_variant` / `_dispatch_decision` / `test_dflash2_exl3_head` PASS。
- `test_exl3_mul1_grouped` は外部 MoE fixture 不在で未カバー（継続）。
- teacher-forced 比較: KL=0、max abs delta=0、PPL 差分 0。
- 実モデル prefill 2048 token: 1194.5 ms（P0 ベースライン約 1192 ms、差はばらつき）。

## 残る作業

- `launch_gemm_exl3_w16a16_grouped` の共通 grouped dispatch への移動。
- `optimized_dispatch.hip` に残る mul24 起動列の EXL3 側への委譲。
- `test_exl3_mul1_grouped` の合成 fixture 化。
- P1 の Fallback / 計測ラベル / 実験経路の正式クローズ。
