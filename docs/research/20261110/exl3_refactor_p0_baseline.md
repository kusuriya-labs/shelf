# EXL3 / mul24 P0 リファクタリング ベースライン（Phase 0）

## 目的

P0 リファクタリング着手前の状態を固定し、各 Phase の Gate で比較できる基準を残す。
測定原則は `docs/perf/methodology.md` を正本とする。

## 環境

- Git commit: `615dd65e832a2b710a61a25c660d6728e44dd2ed`（main）
- HIP: 7.15.26333-0000000 / AMD clang 23.0.0git
- ROCm SDK: `/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`
- GPU: AMD Radeon AI PRO R9700（gfx1201）、driver 7.1.9.31600000
- CMake: Release、`SHELF_BUILD_TESTS=ON`、`SHELF_BUILD_BENCHMARKS=ON`、
  `CMAKE_HIP_ARCHITECTURES=gfx1201`
- 実行隔離: `HIP_VISIBLE_DEVICES=1`、`flock /tmp/r9700.lock`

## 既存テスト（直接起動、`HIP_VISIBLE_DEVICES=1`）

| テスト | 結果 | 時間 |
|---|---|---:|
| test_exl3_codec | PASS | 5 ms |
| test_exl3_dense | PASS | 510 ms |
| test_exl3_grouped | **FAIL（既知の未カバー）** | 23 ms |
| test_exl3_int8_decode | PASS | 3 ms |
| test_exl3_linear | PASS | 5,082 ms |
| test_exl3_int8_grouped | ok | 459 ms |
| test_exl3_int8_linear | ALL PASS | 1,286 ms |
| test_exl3_int8_prefill_rotate | bitwise PASS | 121 ms |
| test_exl3_int8_tp | ALL PASS | 1,276 ms |
| test_dflash2_exl3_head | rel L2 0.007154 | 166 ms |

`test_exl3_grouped` は MoE モデルの fixture（`SHELF_QWEN4EXP_MODEL_DIR`）が
この環境に無いため失敗する。リファクタリングの回帰判定からは除外し、未カバーとして扱う。

## 性能ベースライン（既定経路、f4 off）

- Prefill 2048 token（`Qwen3.8-27B-EXL3-INT8-4bpw`、chunk 2048、BF16 KV）:
  wall p50 約 1200 ms（`docs/research/20261110/exl3_int8_rotate_f4.md` の交互 A/B と同条件）。
- Decode 相当の EXL3-INT8 単体（rows=16、n=k=5120、bits=4、split=2）:
  rotate_in 4.98 µs、gemm 38.5 µs、rotate_out 5.26 µs。
- f4 経路の詳細は別記録（`exl3_int8_rotate_f4.md`）にあり、P0 の対象外。

## カーネルリソース（`-Rpass-analysis=kernel-resource-usage`、gfx1201）

主要カーネルの SGPR / VGPR / LDS / occupancy（waves/SIMD）。

| カーネル | SGPR | VGPR | LDS (B) | occ |
|---|---:|---:|---:|---:|
| gemm_exl3_w16a16<4,pre,false,2,0> | 95 | 187 | 36096 | 6 |
| gemm_exl3_w16a16<2,pre,false,2,0> | 89 | 143 | 27264 | 8 |
| gemm_exl3_w16a16<1,pre,false,2,0> | 85 | 119 | 22848 | 10 |
| gemm_exl3_w16a16<4,false,false,1,0> | 92 | 256 | 27904 | 5 |
| exl3_int8_gemm_kernel<4,1> | 30 | 90 | 0 | 16 |
| exl3_int8_gemm_kernel<4,4> | 58 | 236 | 0 | 6 |
| gemm_mul24_grouped<true,2,2,1> | 58 | 180 | 21120 | 8 |
| gemm_mul24_grouped<false,2,2,1> | 56 | 141 | 25088 | 10 |
| recon_exl3_dense_kernel | 78 | 53 | 6144 | 16 |
| exl3_dense_epilogue_kernel | 16 | 27 | 0 | 16 |
| rotate_exl3_kernel<true> | 20 | 31 | 32 | 16 |
| exl3_int8_rotate_quantize_kernel | 14 | 13 | 528 | 16 |
| exl3_int8_rotate_out_i32_kernel<true> | 22 | 9 | 512 | 16 |

f4 版（`exl3_int8_rotate_quantize_f4_kernel`、`exl3_int8_rotate_out_*_f4_kernel`）は
LDS 0、VGPR 24〜33、occ 16 で、P0 の対象外だが比較用に記録する。

## 参照コマンド

```sh
CC=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++
$CC -DNDEBUG -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Iinclude -Isrc -O3 \
    -std=gnu++20 --offload-arch=gfx1201 -stdlib=libc++ \
    -Wno-unused-command-line-argument -Rpass-analysis=kernel-resource-usage -x hip \
    -c src/shelf/kernels/optimized/linear/exl3.hip -o /tmp/exl3.o
```

## Gate 0 判定

- ベースラインは commit `615dd65` で固定した。
- 既存テストのうち `test_exl3_grouped` のみ fixture 不足で失敗する（未カバー）。
- 以降の Phase は、このテスト結果・リソース値・既定経路の性能を基準に回帰を判定する。
