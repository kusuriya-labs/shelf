# EXL3 W16A16 kernel の ISA 調査

## 対象

- `src/shelf/kernels/optimized/linear/exl3.hip`
- `gemm_exl3_w16a16_kernel`（8 warp block + mul1 affine epilogue / packed pair 復号）
- build: `397e763` 相当、`-O3 --offload-arch=gfx1201`

ISA は `clang++ --save-temps` で device 側の `.s` を出して読んだ。

## kernel metadata

| 項目 | 値 |
| --- | ---: |
| wavefront | wave32 |
| threads / block | 256（8 warp） |
| VGPR | 92（rocprofv3 表示は 96） |
| SGPR | 32 |
| LDS / block | 14,400 B |

VGPR は 92 × 256 = 23,552 / block。64K VGPR では 2 block/CU まで。
LDS も 14.4 KiB × 2 = 28.8 KiB で 2 block/CU まで。したがって
**16 warp/CU = 8 warp/SIMD** で、3 block/CU には VGPR を 85 以下へ下げる必要がある。

## 命令 mix

全 2,168 命令（`state_from_ring` の modulo / 64-bit shift を残した状態）。

| 分類 | 数 | 割合 |
| --- | ---: | ---: |
| VALU | 1,022 | 47.1 % |
| wait / delay | 580 | **26.8 %** |
| LDS | 284 | 13.1 % |
| SALU / control | 262 | 12.1 % |
| VMEM | 20 | 0.9 % |

主な命令:

| 命令 | 数 |
| --- | ---: |
| `s_wait_dscnt` | 198 |
| `s_wait_alu` | 191 |
| `s_delay_alu` | 174 |
| `ds_load_b32` | 130 |
| `ds_bpermute_b32` | 88 |
| `v_mul_lo_u32` | 86 |
| `v_and_b32_e32` | 84 |
| `v_lshrrev_b64` | 64 |
| `v_add_co_u32` / `v_add_co_ci_u32_e64` | 45 / 46 |
| `v_cndmask_b32_e64` | 47 |
| `v_mul_hi_u32` | 16 |
| `v_sad_u8` / `v_sad_hi_u8` | 32 / 32 |
| `v_wmma_f32_16x16x16_f16` | 8 |

## 読み取れること

1. **MMA は主コストではない。**
   1 k chunk あたり `v_wmma` は 8 個（8 ks）。全 2,168 命令の 0.4 %。
   16×16×16 の WMMA 自体はボトルネックではない。

2. **wait / delay が 26.8 %。**
   `s_wait_dscnt`（LDS load 待ち）198、`s_wait_alu`（VALU 依存）191、
   `s_delay_alu` 174。命令 issue ではなく **依存 stall** が支配している。

3. **`state_from_ring` の runtime 除算と 64-bit shift が ISA に出ている。**
   `lo % word_count` / `hi % word_count` が `v_mul_hi_u32`、`v_add_co_u32`、
   `v_add_co_ci_u32_e64`、`v_cndmask_b32_e64` の列になり、
   `(pair >> shift)` が `v_lshrrev_b64` 64 個になっている。

4. **LDS の往復が多い。**
   `ds_load_b32` 130（ring の読み出しが主）、`ds_load_b128` 18（A fragment）、
   `ds_store_b32` 42（tile の u32 store）。tile を LDS に置いて A fragment を
   読み直す transpose が、decode と WMMA の間に `__syncwarp()` を挟んでいる。

5. **`ds_bpermute_b32` 88。**
   FWHT の 4 値 × 5 step × 2 row と row sum の reduction。

## 実験: 除算と 64-bit shift を消す

`state_from_ring` を次のように変えた。

- `% word_count` を条件減算へ（`lo`/`hi` は `2*word_count` 未満）。
- `(pair >> shift)` を `__funnelshift_r` へ。

ISA の変化:

| 項目 | 変更前 | 変更後 |
| --- | ---: | ---: |
| 全命令 | 2,168 | **1,980**（-8.7 %） |
| VALU | 1,022 | 893 |
| wait / delay | 580 | 528 |
| `v_mul_hi_u32` | 16 | 0 |
| `v_lshrrev_b64` | 64 | 0 |
| `v_alignbit_b32` | 0 | 64 |

性能（同じ GPU 1、`tg --context 128 --tokens 16 --warmup 1`、3 run）:

| 版 | tok/s |
| --- | ---: |
| 変更前 | 9.04 / 9.05 / 9.06 |
| 変更後 | 9.08 / 9.10 / 9.10 |

**命令数を 8.7 % 減らしても、decode は 0.5 % しか改善しなかった。**
この結果から、律速は命令スループットではなく、decode の依存 chain と
LDS / barrier / shuffle の stall であると判断した。変更は revert した。

## 次の候補

1. **decode と WMMA の software pipeline**。double buffer はすでにあるので、
   次の tile の復号を現在の WMMA の後ろに置き、独立命令で stall を埋める。
   ただし WMMA は 8 命令しかないため、期待できるのは LDS/依存 stall の隠蔽。
2. **A fragment を LDS 経由にしない**。現在は 16×16 tile を LDS に transpose して
   A fragment を読む。ring から直接 A fragment を生成できれば `ds_store` /
   `ds_load` / `__syncwarp` を削れるが、window がレーン間に分散するため
   shuffle が必要で、以前の shuffle 実験は LDS より遅かった。
3. **VGPR を 85 以下にして 3 block/CU**。現在は 2 block/CU。VGPR を削るには
   accumulator か address のレジスタを減らす必要がある。
4. **row sum / FWHT の shuffle 削減**。`ds_bpermute` 88 の一部。
5. rocprofv3 の PMC は gfx1201 で 0 が返り、PC sampling も出力が得られなかった。
   次の判断は静的 ISA と A/B 測定で行う。
