# BF16 small-M の参照実装調査

実験 ID: `SMALLM-BF16-REF-1`

目的は、Shelf の BF16 small-M（M=1〜64）を作り直す前に、`../references/` の
実装が行 tile、K 分割、縮約順、batch 不変性をどう扱っているかを横並びで確認することである。
この文書は調査記録である。案 A の block-per-output-row と案 B（M=9..16 の WMMA）は
[bf16_small_rows.md](bf16_small_rows.md) で実装・計測した。非投機 decode が
横ばいで、DFlash2 の accept もプロンプト依存だったため、production selector には
接続していない。

## 1. 比較対象

| 参照 | 場所 | 位置づけ |
| --- | --- | --- |
| Strata | `../references/Strata/src/kernels/cuda/native_bf16.cu` | CUDA の BF16 GEMV。block-per-output-row、1〜8/16 token |
| Strata | `../references/Strata/src/kernels/cuda/bf16_gemv.cu` | warp-per-row と row ごとの split |
| Strata | `../references/Strata/src/kernels/cuda/verify_kernels.cu` | GDN a/b を含む multi-token の exact-order 演算 |
| llama.cpp | `../references/Qwen-QSA-llama.cpp/ggml/src/ggml-cuda/mmvf.cu` | 上流の `mul_mat_vec_f`。block-per-output-row、1〜8 token |
| llama.cpp | `../references/Qwen-QSA-llama.cpp/ggml/src/ggml-cuda/mmvf-multi.cu` | Qwen 向けの複数 segment 融合 GEMV。1〜4 token、K≤3072 |
| Radiance | `../references/radiance/libr4d/r4d_gemm_bf16_nt_m64.hip` | WMMA の skinny BF16 GEMM。M≤16 も同じ kernel |
| Radiance | `../references/radiance/libr4d/r4d_gemm_bf16_nt_m16.hip` | M≤16 の forwarding。M≤64 WMMA を使う |
| Radiance | `../references/radiance/libr4d/r4d_gemm_bf16_nt_tiled.hip` | prefill 用。m64 と byte-identical |
| Shelf | `src/shelf/kernels/optimized/linear/bf16.hip` | 現在の `ExactRows` / `ExactRowsDot2` |
| 旧 Shelf | `../references/phaseshift-fp8` / `phaseshift-mxfp4` | `ExactRows` の前身。新しい設計はない |
| hipfire | `../references/hipfire` | 主に量子化 GEMV。BF16 small-M の直接の参考にはならない |

## 2. 実装ごとの形

### 2.1 Strata: block-per-output-row の multi-token GEMV

`native_bf16.cu` の `bf16_f32_mmvf_kernel` と `bf16_f32_mmvf_multi_kernel` は、
**1 block が 1 出力行**を担当し、block の全 thread が K を分担する。

- block size は 32〜256。`mmvf_block_size(n_in)` が
  `(n_in + 2*block - 1) / (2*block)` を最小にする値を選ぶ。
- `NT` 個の token を register に保持し、重みは 1 回だけ読む。
- 各 token の accumulator は single-row 版と同じ順序（pair 昇順、2 つの順序付き FMA）で
  更新するため、**single-row 呼び出しと bit 一致**する。
- warp reduce の後、block size > 32 では shared memory で warp をまとめる。
- `STRATA_SMALL_UNR` の fast path は `n_in == 2560` の定数 trip count で、5 pass 分の
  weight / activation load を先に発行する。FMA の順序は変えない。
- `STRATA_MMVF_ROWS=1` では `bf16_f32_mmvf_rows_kernel` が **RPB=4 出力行**を 1 block で
  まとめ、activation の再読み込みを減らす。これも single-row と bit 一致する。
- PDL の prefetch は先頭 8 pair を activation の待ち前に読む。

`bf16_gemv.cu` は素朴な比較用として次を持つ。

- `bf16_gemv_naive_kernel`: 1 thread が 1 出力行を連続に読む。coalesce しない。
- `bf16_gemv_warp_kernel`: 1 warp が 1 出力行。`n_out >= 64` でこれを使う。
- `bf16_gemv_split_kernel`: 1 block が 1 出力行で、`threads_per_row` が K を分担する。
  `n_out` が小さいときに使う。
- `bf16_gemv_pair_warp_kernel`: 2 つの重み行列を 1 回の activation 読みで計算する。
  llama.cpp の `mul_mat_multi` と同じ形で、bit 一致する。

### 2.2 Strata: GDN a/b の multi-token

`verify_kernels.cu` の `gdn_ab_body` は、GDN の alpha / beta を 1 warp が 1 出力行として
`MAX_T` token 分の accumulator を持つ。`float4` で重みを読み、`n == 2560` では
10 pass の定数 trip count に展開する。token ごとの FMA 順序は single-token 版と同じである。

この形は Shelf の `in_proj_a` / `in_proj_b`（out=16〜32、K=2048〜2560）に直接対応する。
Shelf の現状は 1 演算ずつ `LINEAR_BF16` を呼ぶため、2 本を同じ launch にまとめる余地がある。

### 2.3 llama.cpp: `mul_mat_vec_f`（`mmvf.cu`）

- **1 block が 1 出力行**。block size は 32〜256 で、2 要素ずつ読む前提の iteration 数が
  最小になる値を選ぶ。
- `ncols_dst = 1..8` を template にし、token ごとに `sumf[ncols_dst]` を持つ。
- weight は 1 回読み、activation は token 数分読む。
- warp reduce + shared memory の block reduce。
- `ncols_dst == 1` のときだけ fusion（gate / bias）を受ける。
- AMD の BF16 分岐は、`bf16_mma_hardware_available(cc)` が真なら `ne11 <= 3`、
  偽なら `ne11 <= 8` で mmvf を選ぶ。gfx12 は BF16 WMMA を持つため、
  上流の判断では 4 token 以上は GEMM 側へ送る。

### 2.4 llama.cpp: `mmvf-multi.cu`（Qwen 固有）

Qwen の small-out 線形を 1 launch にまとめる専用実装である。

- 入力は F32 の activation 1〜4 列。K ≤ 3072。
- 1 block = 8 warp。block は `RPB` 出力行を持ち、warps が K を分担する。
- 各 lane は自分の float4 を先にすべて load し、その後 FMA する。
  重みは BF16 なら 8 byte、F32 なら 16 byte で読む。
- 複数 segment（router、shared expert gate、GDN alpha/beta、indexer q/k）を
  1 つの `mmvf_multi_args` にまとめ、segment ごとの epilogue
  （none / sigmoid / softplus）を適用する。
- K の分割数は 8 固定で、block 内の warp 順に reduce する。

Shelf の `router`、`shared gate`、`GDN a/b`、QSA indexer の small-out 線形は
この形に近い。特に GDN a/b は 2 本、indexer は 32 本あり、launch 統合の効果が大きい。

### 2.5 Radiance: WMMA の skinny BF16 GEMM

`r4d_gemm_bf16_nt_m64.hip` は、M=1〜64 を **WMMA** で処理する。

- 1 block = `WV` 個の 16 列 tile × `SK` 個の K split wave。
- 各 wave は 1 個の 16×16 tile を担当し、K の連続区間を計算する。
- split の縮約は LDS で **wave 順**に行うため、scheduling に依存しない。
- `MT` は block が持つ 16 行 tile 数。M=1〜16 も `MT=1` で同じ kernel を使う。
- M≤16 の `r4d_gemm_bf16_nt_m16.hip` は m64 へ forward するだけである。
  コメントには「M≤64 の WMMA は band 全体で scalar split-K より速い」とある。
- 縮約木は scalar kernel と一致しないが、**行ごとの出力が M に依存しない**ことを
  契約にしている。M=48 の GDN a|b で生成文が変わることは許容している。
- `r4d_gemm_bf16_nt_tiled.hip` は prefill 用で、m64 と同じ `SK` 規則なら
  byte-identical になる。

### 2.6 現在の Shelf `ExactRows`

`src/shelf/kernels/optimized/linear/bf16.hip` の `gemm_bf16_exact_rows_body` は、

- 1 block = 256 thread = 8 warp。
- 1 warp が 1 出力行を担当し、block で 8 出力行を処理する。
- `kRows` 個の token accumulator を register に持つ。
- 各 lane は K を 8 要素ずつ読み、`fdot2`（`ExactRowsDot2`）または scalar FMA で積む。
- K の末尾は lane stride で処理する。
- warp shuffle で 1 出力行を畳む。

batch 不変性はあるが、次の弱点がある。

- M=16 では 16 accumulator × 8 warp になり、register pressure が高い。
- 1 warp が 1 出力行なので、出力幅が広いと activation を出力行数分読む。
- M=1 でも 8 warp の block が 8 出力行を担当するため、小さい `n_out` では
  block 数が足りない。

## 3. 実測した現在の崖

R9700、BF16、`shelf-bench gemm --dtype bf16`、warmup 50 / samples 20 / launches 20。
値は p50 µs。`selected=gemm` は Shelf の selector が選んだ経路である。

| shape | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5120:5120 | 27.1 | 30.6 | 33.8 | 58.1 | 125.5 | 88.3 | 198.2 |
| 2560:5120 | 15.4 | 18.7 | 20.9 | 32.4 | 60.1 | 72.2 | 85.6 |
| 17408:5120 | 282.6 | 284.5 | 287.0 | 319.4 | 549.2 | 1121.0 | 2042.5 |

`5120:5120` の M=16 は M=32 より遅い。weight は 26 MB で L3 に収まるため、
これは帯域ではなく accumulator 数と occupancy の問題である。
`17408:5120` は M=1 が DRAM 天井（178 MB / 282.6 µs = 630 GB/s）に近く、
M=16 で 549 µs まで落ちる。重み traffic は変わらないので、ここでも
`kRows=16` のコストが支配している。

## 4. 参照実装の一致点

1. **1 出力行を 1 block が担当し、K を block 内で分担する**形が、
   Strata と llama.cpp の共通解である。Shelf の 1 warp = 1 出力行より並列度が高い。
2. **M に依存しない縮約順**が batch 不変性の実体である。
   参照実装は「single-row と bit 一致」または「M を変えても同じ順」を守る。
3. M が小さいときは **scalar GEMV より WMMA** が選ばれることがある。
   Radiance は M≤16 も M≤64 の WMMA に送り、scalar split-K を棄却している。
   llama.cpp も BF16 MMA がある AMD では 4 token 以上を GEMM に送る。
4. **複数行列の融合**は small-out で効果がある。
   llama.cpp の `mmvf-multi` は router / shared gate / GDN a/b / indexer を
   1 launch にまとめる。Shelf の GDN a/b と QSA indexer は同じ形を取れる。
5. block size は固定ではなく、K と出力幅から選ぶ。
   Strata は iteration 数、llama.cpp は `n_in` から 32〜256 を選ぶ。

## 5. Shelf への設計候補

### 案 A: block-per-output-row の multi-token GEMV を追加する

- 1 block = 1 出力行、32〜256 thread。
- `NT` token を register に持ち、重みを 1 回読む。
- 各 token の FMA 順序を固定し、M に依存させない。
- M=1〜8 の small-out（GDN a/b、router、indexer q/k）に向く。
- 既存 `ExactRows` の M=16 崖を避けられる。
- 欠点: wide N では出力行数分の block が必要で、weight の読みは 1 回だが
  activation は token 数分読む。M=1 の wide N は既に DRAM 律速なので、
  この案の対象は small-out に限定する。

### 案 B: WMMA の row-count-independent small-M を追加する

- Radiance の m64 と同じ、`WV` 列 tile × `SK` split wave。
- `SK` は K だけから決め、M では変えない。
- M=1〜64 を同じ kernel で処理し、LDS で wave 順に縮約する。
- `SK` を固定すれば、M=1 の verify と M=8 の verify が同じ行順になる。
- 欠点: M=1 では WMMA の 15/16 を捨てる。ただし weight stream が支配的な
  wide N では問題にならない。small-out では案 A の方が有利。

### 案 C: 既存 `ExactRows` を残し、M の境界だけ直す

- M=1〜4 は現行の warp-per-row を維持し、M=8〜16 だけ案 A/B に切り替える。
- 変更は小さいが、M の境界で縮約順が変わる。
- MTP verify の golden を更新する必要がある。

## 6. 推奨

1. **M=1〜8 の small-out は案 A**（block-per-output-row、NT token、block size 32〜256）。
   GDN a/b、router、indexer q/k を対象にし、複数行列の融合も同時に検討する。
2. **M=9〜64 は案 B**（WMMA、MT row tile、SK は K のみから決める）。
   M=16 の崖をなくし、M=32/64 と同じ経路に寄せる。
3. wide N（17408:5120 など）の M=1 は現行の `ExactRows` が既に DRAM 天井付近なので、
   無理に置き換えない。M=8〜16 の occupancy 改善だけを狙う。
4. 縮約順を変える場合は、MTP verify の golden と greedy 出力の一致を
   `test_gemm_bf16_verify_exact` で取り直す。参照実装は
   「single-row と bit 一致」または「M に依存しない」を契約にしており、
   特定の縮約木を契約にはしていない。
5. 最初の計測は次の形で行う。
   `shelf-bench gemm --dtype bf16 --shapes 16:2048,32:2048,256:2048,512:2048,2560:5120,5120:5120,5120:17408,17408:5120 --rows 1,2,4,8,16,32,64`
   `--impl exact|wmma` と新しい variant を同じ process で交互に測る。

## 7. 未検証

- Strata の `STRATA_MMVF_ROWS` / `STRATA_SMALL_UNR` の RDNA4 での実測。
- llama.cpp の AMD BF16 分岐が gfx1201 で `bf16_mma_hardware_available` を
  どう判定するか。
- Radiance の m64 を gfx1201 の Shelf 形状へ適用したときの M=1〜16 の実測。
- GDN a/b の 2 本融合、indexer 32 GEMV の 1 launch 化の効果。
- MTP verify の golden を「M に依存しない」契約へ移行できるか。
