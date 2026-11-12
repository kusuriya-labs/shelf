# PLaMo linear 第2弾: large-M とコンパイラ解析

INT8 dense の公称 peak は 383 TOPS。[AMD 公式仕様](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) の非 sparse 値を使用する。
WMMA の 8192 ops / 8 cycles、128 SIMD、最大 2.92 GHz から 382.73 TOPS を得る。[RDNA4 builtin の公式 throughput 表](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html#instruction-throughput) と整合する。

## step の有効 INT8 演算率

body の sum(N*K)=2,327,838,720、head の N*K=275,251,200。演算量は `2*(M*body_sum_nk + output_rows*head_nk)`。head に prefill 全行を掛けない。
この演算量を LINEAR_EXL3_INT8 の event 合計で割る。量子化・復号・回転・flag の重複 GEMM と host/launch 区間を含む有効値であり、WMMA busy の実測値ではない。

| phase | c | M 帯 | samples | M 中央値 | linear ms 中央値 | 有効 TOPS 中央値 | 公称 peak 比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| before-exact | 64 | 1-64 | 14 | 64 | 9.323 | 35.68 | 9.31% |
| before-exact | 64 | 65-128 | 3 | 98 | 15.212 | 34.78 | 9.08% |
| before-exact | 64 | 129-255 | 16 | 168 | 18.522 | 44.30 | 11.57% |
| before-exact | 64 | 256-511 | 3 | 300 | 42.855 | 33.41 | 8.72% |
| before-exact | 128 | 65-128 | 4 | 128 | 16.583 | 40.19 | 10.49% |
| before-exact | 128 | 129-255 | 5 | 188 | 19.636 | 47.58 | 12.42% |
| before-exact | 128 | 256-511 | 10 | 375 | 45.474 | 39.85 | 10.40% |
| before-exact | 128 | 512-2048 | 1 | 537 | 55.927 | 45.94 | 12.00% |
| final2 | 64 | 1-64 | 14 | 64 | 9.320 | 35.72 | 9.33% |
| final2 | 64 | 65-128 | 3 | 98 | 15.155 | 35.30 | 9.22% |
| final2 | 64 | 129-255 | 16 | 168 | 18.493 | 44.46 | 11.61% |
| final2 | 64 | 256-511 | 3 | 300 | 41.192 | 34.76 | 9.08% |
| final2 | 128 | 65-128 | 3 | 127 | 16.127 | 41.00 | 10.70% |
| final2 | 128 | 129-255 | 3 | 218 | 22.619 | 47.18 | 12.32% |
| final2 | 128 | 256-511 | 14 | 364 | 42.950 | 41.20 | 10.76% |
| prefill | 64 | 1-64 | 18 | 64 | 9.081 | 36.14 | 9.44% |
| prefill | 64 | 65-128 | 8 | 124 | 12.680 | 48.08 | 12.55% |
| prefill | 64 | 129-255 | 11 | 193 | 17.880 | 56.39 | 14.72% |
| prefill | 64 | 256-511 | 5 | 304 | 22.699 | 63.90 | 16.69% |
| prefill | 128 | 65-128 | 6 | 126 | 13.382 | 49.41 | 12.90% |
| prefill | 128 | 129-255 | 9 | 214 | 18.375 | 58.43 | 15.25% |
| prefill | 128 | 256-511 | 10 | 366 | 24.695 | 71.13 | 18.57% |
| prefill | 128 | 512-2048 | 1 | 513 | 33.974 | 72.36 | 18.89% |
| rowscale | 64 | 1-64 | 22 | 64 | 9.281 | 35.90 | 9.37% |
| rowscale | 64 | 65-128 | 6 | 102 | 12.502 | 40.59 | 10.60% |
| rowscale | 64 | 129-255 | 8 | 194 | 16.292 | 57.31 | 14.96% |
| rowscale | 64 | 256-511 | 5 | 338 | 27.194 | 57.88 | 15.11% |
| rowscale | 64 | 512-2048 | 1 | 536 | 37.895 | 66.78 | 17.44% |
| rowscale | 128 | 65-128 | 7 | 126 | 13.164 | 49.83 | 13.01% |
| rowscale | 128 | 129-255 | 4 | 197 | 16.734 | 60.27 | 15.74% |
| rowscale | 128 | 256-511 | 13 | 308 | 26.731 | 56.27 | 14.69% |
| rowscale | 128 | 512-2048 | 1 | 571 | 38.569 | 70.74 | 18.47% |
| nocap | 64 | 1-64 | 22 | 64 | 9.329 | 35.72 | 9.33% |
| nocap | 64 | 65-128 | 4 | 99 | 12.452 | 39.87 | 10.41% |
| nocap | 64 | 129-255 | 12 | 170 | 14.690 | 54.32 | 14.18% |
| nocap | 64 | 256-511 | 4 | 358 | 27.995 | 60.51 | 15.80% |
| nocap | 128 | 65-128 | 3 | 127 | 13.521 | 48.99 | 12.79% |
| nocap | 128 | 129-255 | 7 | 187 | 15.270 | 59.41 | 15.51% |
| nocap | 128 | 256-511 | 14 | 364 | 27.679 | 60.94 | 15.91% |
| nocap | 128 | 512-2048 | 1 | 572 | 37.475 | 72.94 | 19.05% |

## baseline の静的資源

ROCm 7.14 の clang に `-Rpass-analysis=kernel-resource-usage -save-temps=obj` を指定し、device ELF と ISA を保存した。occupancy は compiler の waves/SIMD 上限であり、実行中の達成値ではない。動的 LDS は別に加算する。

| family / kernel | SGPR | VGPR | LDS B/block | scratch B/lane | SGPR/VGPR spill | waves/SIMD |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| decode (baseline) / `void exl3_mul24_gemm_kernel<4, 1, true>` | 28 | 90 | 0 | 0 | 0 / 0 | 16 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<4, 2, true>` | 40 | 138 | 0 | 0 | 0 / 0 | 10 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<4, 3, true>` | 50 | 164 | 0 | 0 | 0 / 0 | 9 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<4, 4, true>` | 56 | 192 | 0 | 12 | 0 / 2 | 8 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<5, 1, false>` | 38 | 115 | 0 | 0 | 0 / 0 | 12 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<5, 2, false>` | 46 | 163 | 0 | 0 | 0 / 0 | 9 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<5, 3, false>` | 56 | 213 | 0 | 0 | 0 / 0 | 7 |
| decode (baseline) / `void exl3_mul24_gemm_kernel<5, 4, false>` | 64 | 230 | 0 | 0 | 0 / 0 | 6 |
| prefill_wmma (baseline) / `void exl3_mul24_prefill_wmma_kernel<2u, 2u, 4u, true, 8u>` | 17 | 143 | 0 | 0 | 0 / 0 | 10 |
| prefill_wmma (baseline) / `void exl3_mul24_prefill_wmma_kernel<2u, 2u, 4u, true, 1u>` | 15 | 120 | 0 | 0 | 0 / 0 | 12 |
| prefill_wmma (baseline) / `void exl3_mul24_prefill_wmma_kernel<2u, 2u, 4u, false, 1u>` | 36 | 128 | 0 | 0 | 0 / 0 | 10 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 2u, 2u, 4u>` | 50 | 159 | 21120 | 0 | 0 / 0 | 9 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 2u, 2u, 5u>` | 52 | 186 | 21120 | 0 | 0 / 0 | 8 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 1u, 2u, 4u>` | 46 | 162 | 18752 | 0 | 0 / 0 | 9 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 1u, 2u, 5u>` | 46 | 204 | 18752 | 0 | 0 / 0 | 7 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 2u, 1u, 4u>` | 46 | 127 | 12928 | 0 | 0 / 0 | 10 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 2u, 1u, 5u>` | 52 | 156 | 12928 | 0 | 0 / 0 | 9 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 1u, 1u, 4u>` | 39 | 149 | 10560 | 0 | 0 / 0 | 9 |
| grouped (baseline) / `void gemm_mul24_grouped_kernel<true, 1u, 1u, 5u>` | 46 | 183 | 10560 | 0 | 0 / 0 | 8 |
| projection (baseline) / `void group_gemm<4, 1, true>` | 44 | 93 | 0 | 0 | 0 / 0 | 16 |
| projection (baseline) / `void group_gemm<4, 2, true>` | 54 | 141 | 0 | 0 | 0 / 0 | 10 |
| projection (baseline) / `void group_gemm<4, 4, true>` | 72 | 192 | 0 | 24 | 0 / 5 | 8 |
| projection (baseline) / `void group_gemm<5, 1, true>` | 50 | 118 | 0 | 0 | 0 / 0 | 12 |
| projection (baseline) / `void group_gemm<5, 4, false>` | 75 | 234 | 0 | 0 | 0 / 0 | 6 |
| projection (baseline) / `void group_gemm<8, 1, true>` | 44 | 108 | 0 | 0 | 0 / 0 | 12 |
| projection (baseline) / `void group_gemm<8, 2, true>` | 54 | 156 | 0 | 0 | 0 / 0 | 9 |
| projection (baseline) / `void group_gemm<8, 4, true>` | 72 | 223 | 0 | 0 | 0 / 0 | 6 |
| projection_prefill (accepted-prefill) / `void recon_group<4>` | 22 | 23 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (accepted-prefill) / `void recon_group<5>` | 28 | 48 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (accepted-prefill) / `void gemm_group<4, 2, true, true>` | 42 | 140 | 8448 | 0 | 0 / 0 | 10 |
| projection_prefill (accepted-prefill) / `void gemm_group<4, 4, true, false>` | 49 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (accepted-prefill) / `void gemm_group<4, 4, false, false>` | 52 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (accepted-prefill) / `void gemm_group<5, 4, false, false>` | 58 | 242 | 16896 | 0 | 0 / 0 | 5 |
| projection_prefill (rowscale-prefill) / `rowstat` | 16 | 17 | 64 | 0 | 0 / 0 | 16 |
| projection_prefill (rowscale-prefill) / `rowslot` | 11 | 24 | 4096 | 0 | 0 / 0 | 16 |
| projection_prefill (rowscale-prefill) / `rowfinish` | 18 | 10 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (rowscale-prefill) / `void recon_group<4>` | 22 | 23 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (rowscale-prefill) / `void recon_group<5>` | 28 | 48 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 2, true, true, true>` | 93 | 142 | 8448 | 0 | 0 / 0 | 10 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 4, true, false, true>` | 107 | 170 | 16896 | 0 | 43 / 0 | 8 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 4, false, false, true>` | 107 | 171 | 16896 | 0 | 43 / 0 | 8 |
| projection_prefill (rowscale-prefill) / `void gemm_group<5, 4, false, false, true>` | 107 | 160 | 16896 | 0 | 52 / 0 | 9 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 2, true, true, false>` | 42 | 140 | 8448 | 0 | 0 / 0 | 10 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 4, true, false, false>` | 49 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (rowscale-prefill) / `void gemm_group<4, 4, false, false, false>` | 52 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (rowscale-prefill) / `void gemm_group<5, 4, false, false, false>` | 58 | 242 | 16896 | 0 | 0 / 0 | 5 |
| projection_prefill (final-nocap-prefill) / `rowstat` | 16 | 17 | 64 | 0 | 0 / 0 | 16 |
| projection_prefill (final-nocap-prefill) / `rowfinish` | 18 | 10 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (final-nocap-prefill) / `void recon_group<4>` | 22 | 23 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (final-nocap-prefill) / `void recon_group<5>` | 28 | 48 | 0 | 0 | 0 / 0 | 16 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 2, true, true, true>` | 93 | 142 | 8448 | 0 | 0 / 0 | 10 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 4, true, false, true>` | 107 | 170 | 16896 | 0 | 43 / 0 | 8 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 4, false, false, true>` | 107 | 171 | 16896 | 0 | 43 / 0 | 8 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<5, 4, false, false, true>` | 107 | 160 | 16896 | 0 | 52 / 0 | 9 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 2, true, true, false>` | 42 | 140 | 8448 | 0 | 0 / 0 | 10 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 4, true, false, false>` | 49 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<4, 4, false, false, false>` | 52 | 206 | 16896 | 0 | 0 / 0 | 7 |
| projection_prefill (final-nocap-prefill) / `void gemm_group<5, 4, false, false, false>` | 58 | 242 | 16896 | 0 | 0 / 0 | 5 |

## ISA

全 opcode と WMMA を含む basic block を JSON に保存した。WMMA の距離は ISA 上の命令数であり、実行時の cycle 間隔ではない。GFX12 の s_wait_loadcnt / s_wait_dscnt 等も集計する。

| kernel | mul24 | dot4 | alignbit | WMMA | global b32/b64/b128 | scratch load/store | WMMA命令距離 中央値 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `void group_gemm<4, 2, true>` | 64 | 64 | 24 | 16 | 48/16/0 | 0/0 | 4 |
| `void group_gemm<4, 4, true>` | 64 | 64 | 24 | 32 | 64/32/0 | 4/4 | 5 |
| `void group_gemm<5, 4, false>` | 64 | 64 | 64 | 32 | 56/32/0 | 0/0 | 3 |
| `void recon_group<4>` | 16 | 16 | 6 | 0 | 4/0/0 | 0/0 | 0 |
| `void recon_group<5>` | 17 | 16 | 16 | 0 | 6/0/0 | 0/0 | 0 |
| `void gemm_group<4, 2, true, true>` | 0 | 0 | 0 | 16 | 2/4/10 | 0/0 | 2 |
| `void gemm_group<4, 4, true, false>` | 0 | 0 | 0 | 32 | 2/8/12 | 0/0 | 2 |
| `void gemm_group<4, 4, false, false>` | 128 | 128 | 48 | 32 | 34/8/4 | 0/0 | 2 |
| `void gemm_group<5, 4, false, false>` | 128 | 128 | 128 | 32 | 50/8/4 | 0/0 | 2 |
| `rowstat` | 0 | 0 | 0 | 0 | 1/0/0 | 0/0 | 0 |
| `rowslot` | 1 | 0 | 0 | 0 | 2/0/0 | 0/0 | 0 |
| `rowfinish` | 0 | 0 | 0 | 0 | 1/0/0 | 0/0 | 0 |
| `void recon_group<4>` | 16 | 16 | 6 | 0 | 4/0/0 | 0/0 | 0 |
| `void recon_group<5>` | 17 | 16 | 16 | 0 | 6/0/0 | 0/0 | 0 |
| `void gemm_group<4, 2, true, true, true>` | 0 | 0 | 0 | 16 | 34/4/10 | 0/0 | 2 |
| `void gemm_group<4, 4, true, false, true>` | 0 | 0 | 0 | 32 | 66/8/12 | 0/0 | 2 |
| `void gemm_group<4, 4, false, false, true>` | 128 | 128 | 48 | 32 | 98/8/4 | 0/0 | 2 |
| `void gemm_group<5, 4, false, false, true>` | 128 | 128 | 128 | 32 | 114/8/4 | 0/0 | 2 |
| `void gemm_group<4, 2, true, true, false>` | 0 | 0 | 0 | 16 | 2/4/10 | 0/0 | 2 |
| `void gemm_group<4, 4, true, false, false>` | 0 | 0 | 0 | 32 | 2/8/12 | 0/0 | 2 |
| `void gemm_group<4, 4, false, false, false>` | 128 | 128 | 48 | 32 | 34/8/4 | 0/0 | 2 |
| `void gemm_group<5, 4, false, false, false>` | 128 | 128 | 128 | 32 | 50/8/4 | 0/0 | 2 |
| `rowstat` | 0 | 0 | 0 | 0 | 1/0/0 | 0/0 | 0 |
| `rowfinish` | 0 | 0 | 0 | 0 | 1/0/0 | 0/0 | 0 |
| `void recon_group<4>` | 16 | 16 | 6 | 0 | 4/0/0 | 0/0 | 0 |
| `void recon_group<5>` | 17 | 16 | 16 | 0 | 6/0/0 | 0/0 | 0 |
| `void gemm_group<4, 2, true, true, true>` | 0 | 0 | 0 | 16 | 34/4/10 | 0/0 | 2 |
| `void gemm_group<4, 4, true, false, true>` | 0 | 0 | 0 | 32 | 66/8/12 | 0/0 | 2 |
| `void gemm_group<4, 4, false, false, true>` | 128 | 128 | 48 | 32 | 98/8/4 | 0/0 | 2 |
| `void gemm_group<5, 4, false, false, true>` | 128 | 128 | 128 | 32 | 114/8/4 | 0/0 | 2 |
| `void gemm_group<4, 2, true, true, false>` | 0 | 0 | 0 | 16 | 2/4/10 | 0/0 | 2 |
| `void gemm_group<4, 4, true, false, false>` | 0 | 0 | 0 | 32 | 2/8/12 | 0/0 | 2 |
| `void gemm_group<4, 4, false, false, false>` | 128 | 128 | 48 | 32 | 34/8/4 | 0/0 | 2 |
| `void gemm_group<5, 4, false, false, false>` | 128 | 128 | 128 | 32 | 50/8/4 | 0/0 | 2 |

旧4bit MT4は静的 ISA に scratch_store_b32 3個、scratch_store_b64 1個と対応する load を含む。8 waves の指定を満たすための spill を確認した。
旧 group_body は128 K groupあたり64個の mul_u32_u24 と64個の dot4で64個のINT8値/laneを復号する。
新 direct も同じ復号量である。初回と次 group 用のコードを別々に生成するため、関数全体の静的カウントは128個ずつになる。cached GEMMからは復号命令がなくなり、一時展開 kernel が担う。
旧4bit MT4は global b64 で activation fragmentを32回読む。新MT4はLDSから ds_load_b128 で2回分のWMMA Aを読み、global入力をworkgroup内で共有する。cached Bもglobal_load_b128で読む。
旧MT4のWMMA間の静的命令距離中央値は5、新MT4は2。発行burstは密になるが、これは実行cycleの測定ではない。
s_wait_loadcntはtrellis / activationの利用前に現れる。新kernelのs_wait_dscntとbarrier waitにはGEMMと出力FWHTのLDS依存が含まれる。関数全体のwait個数をstall時間とは扱わない。

## compiler A/B

baseline は通常の静的 library、candidate だけ別 TU にして flag を指定する。同じ process で AB/BA を交互に7回測定し、出力と padding の bit 一致を確認する。全体 flag は変更しない。

| variant | M / N×K / bit / count | before p50 µs (SD) | candidate p50 µs (SD) | 比 |
| --- | --- | ---: | ---: | ---: |
| baseline-ab | 192 / 3072×2560 / 4 / 3 | 112.61 (2.58) | 112.83 (0.33) | 1.002 |
| baseline-ab | 192 / 10240×2560 / 4 / 2 | 347.94 (10.06) | 350.24 (9.23) | 1.007 |
| baseline-ab | 192 / 2560×10240 / 4 / 1 | 266.06 (4.59) | 266.02 (0.70) | 1.000 |
| baseline-ab | 192 / 512×2560 / 5 / 1 | 73.75 (5.21) | 73.68 (0.23) | 0.999 |
| baseline-ab | 64 / 10240×2560 / 4 / 2 | 170.73 (3.07) | 170.05 (0.72) | 0.996 |
| clause32 | 192 / 3072×2560 / 4 / 3 | 113.26 (2.38) | 112.94 (0.78) | 0.997 |
| clause32 | 192 / 10240×2560 / 4 / 2 | 350.15 (12.01) | 352.89 (10.99) | 1.008 |
| clause32 | 192 / 2560×10240 / 4 / 1 | 265.90 (6.59) | 266.02 (1.17) | 1.000 |
| clause32 | 192 / 512×2560 / 5 / 1 | 73.72 (3.04) | 73.68 (0.04) | 0.999 |
| clause32 | 64 / 10240×2560 / 4 / 2 | 171.80 (2.62) | 170.58 (0.86) | 0.993 |
| decode-expert | 1 / 3072×2560 / 4 / 3 | 37.26 (2.63) | 37.16 (0.58) | 0.997 |
| decode-expert | 1 / 2560×3072 / 4 / 1 | 35.12 (2.88) | 35.26 (0.63) | 1.004 |
| decode-expert | 1 / 10240×2560 / 4 / 2 | 119.58 (5.89) | 119.26 (0.42) | 0.997 |
| decode-expert | 1 / 2560×10240 / 4 / 1 | 74.20 (5.97) | 74.38 (0.59) | 1.002 |
| decode-expert | 1 / 512×2560 / 5 / 1 | 29.90 (1.30) | 29.54 (0.92) | 0.988 |
| decode-expert | 1 / 107520×2560 / 8 / 1 | 615.21 (0.97) | 614.63 (1.02) | 0.999 |
| decode-nt | 1 / 3072×2560 / 4 / 3 | 37.34 (1.42) | 30.54 (0.96) | 0.818 |
| decode-nt | 1 / 2560×3072 / 4 / 1 | 35.54 (2.33) | 29.12 (0.82) | 0.819 |
| decode-nt | 1 / 10240×2560 / 4 / 2 | 120.34 (3.70) | 62.96 (0.71) | 0.523 |
| decode-nt | 1 / 2560×10240 / 4 / 1 | 73.70 (1.23) | 48.32 (1.76) | 0.656 |
| decode-nt | 1 / 512×2560 / 5 / 1 | 29.60 (0.22) | 28.92 (0.16) | 0.977 |
| decode-nt | 1 / 107520×2560 / 8 / 1 | 614.83 (0.81) | 454.61 (0.29) | 0.739 |
| expert | 192 / 3072×2560 / 4 / 3 | 112.59 (2.43) | 112.45 (0.70) | 0.999 |
| expert | 192 / 10240×2560 / 4 / 2 | 348.02 (11.13) | 352.38 (9.69) | 1.013 |
| expert | 192 / 2560×10240 / 4 / 1 | 266.56 (7.37) | 260.64 (0.61) | 0.978 |
| expert | 192 / 512×2560 / 5 / 1 | 73.50 (3.27) | 72.97 (0.53) | 0.993 |
| expert | 64 / 10240×2560 / 4 / 2 | 169.99 (2.60) | 169.71 (0.73) | 0.998 |
| max-ilp | compiler error | — | — | clang++: error: unable to execute command: Segmentation fault (core dumped) |
| metric100 | 192 / 3072×2560 / 4 / 3 | 113.78 (2.24) | 114.04 (0.54) | 1.002 |
| metric100 | 192 / 10240×2560 / 4 / 2 | 351.21 (12.78) | 355.99 (9.50) | 1.014 |
| metric100 | 192 / 2560×10240 / 4 / 1 | 266.35 (4.83) | 265.99 (1.86) | 0.999 |
| metric100 | 192 / 512×2560 / 5 / 1 | 73.64 (3.13) | 73.56 (0.43) | 0.999 |
| metric100 | 64 / 10240×2560 / 4 / 2 | 171.08 (2.61) | 171.02 (0.61) | 1.000 |
| mid-nt | 64 / 3072×2560 / 4 / 3 | 60.92 (2.61) | 58.62 (1.57) | 0.962 |
| mid-nt | 64 / 2560×3072 / 4 / 1 | 53.80 (1.56) | 51.18 (0.55) | 0.951 |
| mid-nt | 64 / 10240×2560 / 4 / 2 | 172.84 (2.95) | 162.90 (2.17) | 0.942 |
| mid-nt | 64 / 2560×10240 / 4 / 1 | 113.50 (2.71) | 102.20 (0.83) | 0.900 |
| mid-nt | 64 / 512×2560 / 5 / 1 | 47.46 (2.37) | 47.30 (1.01) | 0.997 |
| mid-nt | 64 / 107520×2560 / 8 / 1 | 902.53 (21.31) | 847.65 (18.30) | 0.939 |
| mid-nt | 128 / 107520×2560 / 8 / 1 | 1316.82 (9.29) | 1251.42 (36.08) | 0.950 |
| occ1 | 192 / 3072×2560 / 4 / 3 | 112.14 (3.64) | 113.94 (1.01) | 1.016 |
| occ1 | 192 / 10240×2560 / 4 / 2 | 350.52 (9.41) | 387.34 (4.69) | 1.105 |
| occ1 | 192 / 2560×10240 / 4 / 1 | 265.14 (6.96) | 275.50 (0.91) | 1.039 |
| occ1 | 192 / 512×2560 / 5 / 1 | 73.60 (3.99) | 73.52 (0.78) | 0.999 |
| occ1 | 64 / 10240×2560 / 4 / 2 | 170.98 (3.54) | 170.80 (0.67) | 0.999 |
| occ6 | 192 / 3072×2560 / 4 / 3 | 111.93 (2.97) | 107.09 (1.11) | 0.957 |
| occ6 | 192 / 10240×2560 / 4 / 2 | 348.28 (10.10) | 335.74 (5.86) | 0.964 |
| occ6 | 192 / 2560×10240 / 4 / 1 | 264.36 (8.62) | 242.66 (0.69) | 0.918 |
| occ6 | 192 / 512×2560 / 5 / 1 | 73.70 (2.90) | 73.66 (0.18) | 0.999 |
| occ6 | 64 / 10240×2560 / 4 / 2 | 170.49 (3.15) | 170.55 (0.90) | 1.000 |
| occ7 | 192 / 3072×2560 / 4 / 3 | 112.77 (2.53) | 112.37 (0.67) | 0.996 |
| occ7 | 192 / 10240×2560 / 4 / 2 | 349.16 (10.97) | 371.24 (2.23) | 1.063 |
| occ7 | 192 / 2560×10240 / 4 / 1 | 264.94 (7.67) | 263.44 (1.14) | 0.994 |
| occ7 | 192 / 512×2560 / 5 / 1 | 73.84 (4.12) | 73.76 (1.12) | 0.999 |
| occ7 | 64 / 10240×2560 / 4 / 2 | 170.37 (3.38) | 170.07 (0.80) | 0.998 |
| unroll1 | 192 / 3072×2560 / 4 / 3 | 113.35 (2.20) | 507.93 (1.59) | 4.481 |
| unroll1 | 192 / 10240×2560 / 4 / 2 | 354.36 (7.78) | 2264.56 (64.49) | 6.391 |
| unroll1 | 192 / 2560×10240 / 4 / 1 | 266.76 (4.15) | 1453.95 (36.04) | 5.450 |
| unroll1 | 192 / 512×2560 / 5 / 1 | 73.69 (3.09) | 266.48 (0.25) | 3.616 |
| unroll1 | 64 / 10240×2560 / 4 / 2 | 172.33 (2.35) | 1489.74 (41.14) | 8.645 |
| unroll2 | 192 / 3072×2560 / 4 / 3 | 113.72 (2.31) | 383.25 (3.30) | 3.370 |
| unroll2 | 192 / 10240×2560 / 4 / 2 | 361.89 (6.32) | 1650.05 (32.78) | 4.560 |
| unroll2 | 192 / 2560×10240 / 4 / 1 | 265.61 (8.55) | 1108.18 (22.27) | 4.172 |
| unroll2 | 192 / 512×2560 / 5 / 1 | 73.80 (3.29) | 215.80 (0.18) | 2.924 |
| unroll2 | 64 / 10240×2560 / 4 / 2 | 170.02 (2.93) | 600.03 (16.29) | 3.529 |

| variant | 設定 |
| --- | --- |
| clause32 | `-mllvm -amdgpu-max-memory-clause=32` |
| decode-expert | `-mllvm -amdgpu-expert-scheduling-mode` |
| decode-nt | `__builtin_nontemporal_load` |
| expert | `-mllvm -amdgpu-expert-scheduling-mode` |
| metric100 | `-mllvm -amdgpu-schedule-metric-bias=100` |
| mid-nt | `__builtin_nontemporal_load` |
| occ1 | `amdgpu_waves_per_eu min=1` |
| occ6 | `amdgpu_waves_per_eu min=6` |
| occ7 | `amdgpu_waves_per_eu min=7` |
| unroll1 | `-mllvm -unroll-count=1` |
| unroll2 | `-mllvm -unroll-count=2` |

## large-M 試作

入力と scale を swizzle 付き LDS に二重 buffer で置く。INT8 fragment の2回分を128 bitで運び、GEMMと出力回転を融合する。
cached は trellis を lane / K-pair 順の INT8 へ展開し、direct は元の trellis を使う。いずれも per-128-K の scale と累積順を維持し、旧 per-group reference と BF16 出力・padding の bit 一致を確認する。
exact版は256行以上の従来RowScale経路と丸めが変わる。M320の875位置/方向でtop1低下基準を超えたため不採用。最終候補はRowScale保持版である。

数値の全 p50・分散・資源・opcode は [JSON](plamo3_exl3_linear2_compiler.json) を参照。

| variant | M / N×K / bit | dense before µs | direct µs (SD) | cached µs (SD) |
| --- | --- | ---: | ---: | ---: |
| mt1-gfast | 320 / 512×2560 / 5 | 105.84 | 61.76 (0.26) | 40.18 (0.05) |
| mt1-gfast | 320 / 2560×10240 / 4 | 378.53 | 420.19 (0.56) | 308.81 (0.73) |
| mt1-gfast | 320 / 10240×2560 / 4 | 316.62 | 378.29 (1.65) | 271.98 (4.31) |
| mt1-gfast | 512 / 512×2560 / 5 | 173.26 | 78.24 (0.36) | 48.04 (0.13) |
| mt1-gfast | 512 / 2560×10240 / 4 | 495.69 | 670.47 (2.16) | 422.57 (8.57) |
| mt1-gfast | 512 / 10240×2560 / 4 | 413.85 | 606.43 (12.56) | 363.01 (10.67) |
| mt2 | 320 / 512×2560 / 5 | 105.24 | 64.02 (0.07) | 41.26 (0.02) |
| mt2 | 320 / 2560×10240 / 4 | 374.99 | 351.41 (0.40) | 290.43 (0.41) |
| mt2 | 320 / 10240×2560 / 4 | 316.83 | 256.89 (0.48) | 254.73 (0.42) |
| mt2-gfast | 320 / 512×2560 / 5 | 105.80 | 63.64 (0.10) | 41.30 (0.07) |
| mt2-gfast | 320 / 2560×10240 / 4 | 375.87 | 333.81 (0.96) | 277.06 (0.77) |
| mt2-gfast | 320 / 10240×2560 / 4 | 316.07 | 257.02 (0.40) | 240.32 (1.38) |
| mt2-gfast | 512 / 512×2560 / 5 | 172.62 | 71.50 (0.27) | 46.88 (0.11) |
| mt2-gfast | 512 / 2560×10240 / 4 | 498.45 | 465.49 (1.26) | 361.77 (4.57) |
| mt2-gfast | 512 / 10240×2560 / 4 | 414.49 | 395.57 (2.29) | 320.71 (9.09) |
| mt4-barrier | 320 / 512×2560 / 5 | 106.12 | 47.40 (0.12) | 48.32 (0.12) |
| mt4-barrier | 320 / 2560×10240 / 4 | 380.80 | 260.20 (0.92) | 262.72 (0.57) |
| mt4-barrier | 320 / 10240×2560 / 4 | 321.68 | 191.36 (2.40) | 218.92 (1.30) |
| mt4-barrier | 512 / 512×2560 / 5 | 173.70 | 52.34 (0.15) | 52.98 (0.18) |
| mt4-barrier | 512 / 2560×10240 / 4 | 480.35 | 331.56 (2.08) | 328.64 (8.69) |
| mt4-barrier | 512 / 10240×2560 / 4 | 419.37 | 288.78 (3.82) | 284.12 (0.32) |
| mt4-gfast | 320 / 512×2560 / 5 | 106.00 | 48.78 (0.12) | 48.52 (0.08) |
| mt4-gfast | 320 / 2560×10240 / 4 | 376.07 | 263.29 (0.17) | 279.75 (1.55) |
| mt4-gfast | 320 / 10240×2560 / 4 | 315.90 | 195.32 (0.23) | 229.78 (0.74) |
| mt4-gfast | 512 / 512×2560 / 5 | 173.30 | 53.38 (0.16) | 53.08 (0.09) |
| mt4-gfast | 512 / 2560×10240 / 4 | 496.99 | 326.91 (0.15) | 332.45 (0.93) |
| mt4-gfast | 512 / 10240×2560 / 4 | 414.91 | 288.73 (0.88) | 298.97 (1.94) |
| mt4-lb512 | 320 / 512×2560 / 5 | 106.38 | 47.58 (0.16) | 48.32 (0.16) |
| mt4-lb512 | 320 / 2560×10240 / 4 | 382.22 | 254.82 (6.04) | 260.90 (0.32) |
| mt4-lb512 | 320 / 10240×2560 / 4 | 321.94 | 194.38 (0.44) | 219.10 (6.63) |
| mt4-lb512 | 512 / 512×2560 / 5 | 173.60 | 52.26 (0.02) | 53.20 (0.15) |
| mt4-lb512 | 512 / 2560×10240 / 4 | 481.39 | 328.31 (1.28) | 333.24 (7.95) |
| mt4-lb512 | 512 / 10240×2560 / 4 | 418.61 | 289.98 (2.06) | 285.24 (0.81) |
| mt4-no-pipe | 320 / 512×2560 / 5 | 105.82 | 46.96 (0.09) | 48.46 (0.12) |
| mt4-no-pipe | 320 / 2560×10240 / 4 | 377.67 | 257.68 (0.13) | 257.00 (1.38) |
| mt4-no-pipe | 320 / 10240×2560 / 4 | 317.11 | 190.04 (0.62) | 224.94 (0.36) |
| mt4-no-pipe | 512 / 512×2560 / 5 | 173.18 | 52.08 (0.05) | 53.16 (0.12) |
| mt4-no-pipe | 512 / 2560×10240 / 4 | 498.19 | 328.43 (2.02) | 331.67 (0.40) |
| mt4-no-pipe | 512 / 10240×2560 / 4 | 413.61 | 281.43 (1.84) | 280.38 (5.24) |
| mt4-no-swizzle | 320 / 512×2560 / 5 | 105.48 | 67.38 (0.10) | 68.74 (0.02) |
| mt4-no-swizzle | 320 / 2560×10240 / 4 | 377.21 | 552.73 (0.30) | 578.49 (1.38) |
| mt4-no-swizzle | 320 / 10240×2560 / 4 | 316.35 | 428.13 (0.68) | 481.01 (0.83) |
| mt4-no-swizzle | 512 / 512×2560 / 5 | 173.32 | 71.86 (0.16) | 73.12 (0.15) |
| mt4-no-swizzle | 512 / 2560×10240 / 4 | 496.57 | 678.57 (1.54) | 710.59 (0.80) |
| mt4-no-swizzle | 512 / 10240×2560 / 4 | 415.43 | 651.45 (1.81) | 694.07 (1.48) |
| mt4-nt | 320 / 512×2560 / 5 | 105.52 | 46.40 (0.06) | 47.30 (0.07) |
| mt4-nt | 320 / 2560×10240 / 4 | 376.08 | 243.40 (0.18) | 230.22 (0.41) |
| mt4-nt | 320 / 10240×2560 / 4 | 316.50 | 191.00 (0.38) | 208.74 (0.63) |
| mt4-nt | 512 / 512×2560 / 5 | 173.52 | 51.70 (0.11) | 52.34 (0.06) |
| mt4-nt | 512 / 2560×10240 / 4 | 495.78 | 321.68 (1.11) | 337.80 (1.30) |
| mt4-nt | 512 / 10240×2560 / 4 | 411.26 | 282.16 (1.32) | 267.98 (1.30) |
| mt4-waves8 | 320 / 512×2560 / 5 | 106.30 | 69.92 (0.04) | 48.62 (0.10) |
| mt4-waves8 | 320 / 2560×10240 / 4 | 375.64 | 256.90 (0.20) | 256.08 (1.50) |
| mt4-waves8 | 320 / 10240×2560 / 4 | 315.78 | 189.60 (0.20) | 224.74 (1.30) |
| mt4-waves8 | 512 / 512×2560 / 5 | 173.16 | 75.96 (0.14) | 53.48 (0.22) |
| mt4-waves8 | 512 / 2560×10240 / 4 | 494.96 | 323.60 (0.45) | 329.50 (0.51) |
| mt4-waves8 | 512 / 10240×2560 / 4 | 412.52 | 282.58 (1.84) | 280.78 (2.60) |
| mt8-gfast | 320 / 512×2560 / 5 | 105.68 | 83.02 (0.18) | 70.74 (0.11) |
| mt8-gfast | 320 / 2560×10240 / 4 | 376.41 | 244.84 (0.95) | 276.70 (0.15) |
| mt8-gfast | 320 / 10240×2560 / 4 | 316.85 | 243.48 (0.72) | 275.96 (0.65) |
| mt8-gfast | 512 / 512×2560 / 5 | 172.46 | 91.04 (0.15) | 75.16 (0.19) |
| mt8-gfast | 512 / 2560×10240 / 4 | 497.45 | 427.11 (0.59) | 450.17 (0.28) |
| mt8-gfast | 512 / 10240×2560 / 4 | 415.45 | 320.59 (0.43) | 332.47 (0.43) |

非一時的 load は単体の一部で短縮したが、継続投入で安定した優位を示さず、通常 build に採用しない。
`mt4-expert` は M=320, N=10240, K=2560 の direct 出力で 3,216,714 要素の bit 不一致を検出し、中断した。品質不合格として不採用。
`max-ilp` は MachineScheduler のコンパイル中に Segmentation fault で失敗し、実行していない。
旧 projection の occupancy=6 は MT4 の spill を減らすが、新 selector では該当 body の M>=129 を新 kernel に置換するため採用しない。
unroll-count=1/2 は内側ループの最適化を失い大幅に悪化。LDS swizzle の除去も大幅に悪化。通常 build の全体 flag は変更しない。
