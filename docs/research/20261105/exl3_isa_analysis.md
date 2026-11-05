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
| VGPR | 92（割り当て粒度 24 で 96） |
| SGPR | 32 |
| LDS / block | 14,400 B |

## occupancy の再計算

`rocminfo` の gfx1201 は **2 SIMD/CU、Max Waves/CU = 32、wave32**。
RDNA4 の SIMD あたり 1,536 VGPR、割り当て粒度 24 とすると、96 VGPR は

```text
floor(1536 / 96) = 16 waves/SIMD
2 SIMD/CU -> 32 waves/CU
```

で **ハードウェア上限**に一致する。VGPR は occupancy を制限していない。
LDS も 14,400 B × 4 block = 57.6 KiB で、64 KiB/CU に 4 block 収まる。
つまり **資源上限は 4 block/CU（32 waves/CU）** であり、
「VGPR を 85 以下にしないと 3 block/CU にならない」という以前の見積は誤りだった。

実際の occupancy は **grid 数**で決まる。decode の rows=1 では
grid = (n/128) × 1 block、1 block = 8 warp。

| shape | blocks | waves | waves / 2048 slots |
| --- | ---: | ---: | ---: |
| n=1024 | 8 | 64 | 3.1 % |
| n=5120 | 40 | 320 | 15.6 % |
| n=17408 | 136 | 1,088 | 53.1 % |
| n=248320 (lm_head) | 1,940 | 15,520 | 100 %（資源上限で多重 wave） |

n=5120 / n=1024 は **CU あたり 0.6 / 0.125 block** しかなく、資源上限の 4 block/CU に
まったく届いていない。n=17408 でも 2.1 block/CU で、上限の半分程度。
VGPR 削減ではなく、**独立した warp を増やす**（K 分割や出力 tile の細分化）か、
1 warp 内の ILP を上げるのが筋になる。

`rocprofv3` の PMC は gfx1201 で 0 が返り、PC sampling も出力が得られなかった。
この環境では resident/active waves の実測ができないため、上記は resource と grid からの
計算値である。`wait` 命令数も待機サイクル数ではないので、stall の内訳は
段階別マイクロベンチで確認する必要がある。

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

1. **独立した Trellis 系列のインターリーブ**。1 warp 内で 2 / 4 / 8 個の tile を
   同時に復号し、`load_state → hash → sad → store` の依存待ちに別系列の命令を挟む。
   VGPR 増加と ISA の並びを確認し、2 / 4 / 8 を比較する。
2. **FWHT 初期段の register-local 化**。4 値を同一 lane に置いて初期 2 段を VALU で
   処理し、`ds_bpermute` を減らす。現在の 128 点 FWHT は 4 値 × 5 step なので、
   どこまでローカル化できるかを確認する。
3. **grid を増やす分割**。n=5120 / 1024 は block 数が足りず occupancy が 16 % / 3 %。
   `rows == 1` 固定ではなく、row tile 数と n から必要な split 数を決める
   batch-aware な K 分割にする。
4. **decode と WMMA の software pipeline**。double buffer を使い、次の tile の復号を
   現在の WMMA の後ろに置く。WMMA は 8 命令なので、主目的は LDS/依存 stall の隠蔽。
5. **A fragment の LDS 経由をやめる**。ring から直接 A fragment を作れれば
   `ds_store` / `ds_load` / `__syncwarp` を削れるが、window がレーン間に分散するため
   shuffle が必要。以前の shuffle 実験は LDS より遅かったので、系列インターリーブ後に再評価する。
6. **段階別マイクロベンチ**。decode のみ / FWHT まで / A fragment まで / WMMA までを
   同一 shape で測り、tok/s では見えない差を切り分ける。
7. **128 / 256 threads 構成の比較**。8 warp block を 4 warp block にして
   block 数を増やした場合の効果を測る。ただし総 warp 数は n/16 で変わらないため、
   K 分割なしでは occupancy の上限は同じ。
