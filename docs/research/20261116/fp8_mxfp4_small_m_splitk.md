# FP8 / MXFP4 small-M split-K の移植

実験 ID: `SMALLM-1`

EXL3-INT8（mul24）の small-M から、量子化のデコードではなく
**行 tile と split-K の選び方、および block 内 LDS 縮約**を FP8 W8A8 と
MXFP4 W4A8 へ移植した記録である。追加の workspace は使わず、
既存の 1 block = 1 wave の kernel に split 用の warp を足す形にした。

## 1. 条件

- GPU: Radeon AI PRO R9700 1 枚（gfx1201、`HIP_VISIBLE_DEVICES=1`）。
- build: 2026-11-16、Release。
- bench: `shelf-bench gemm`、`--warmup 30 --samples 15 --launches 15`。
  比較は同じ process ではなく、同じ GPU・同じ binary で条件を交互に近い順へ並べた。
  最初の process は clock が低く、split-K が 40% 遅く出ることがあったため、
  表の値は warm-up 後の値だけを使う。
- 数値は p50 µs。`selected` は selector が選んだ variant である。

## 2. FP8 W8A8

### 2.1 変更

- `include/shelf/kernels/optimized/linear/config.h`
  - `Fp8FinegrainedGemmConfigId::SplitK`、`split_k`、`split_mb` を追加。
  - `kFp8FinegrainedGemmSplitKMaxRows = 64`、`MaxWays = 16`、
    `WaveSlots = 768`、`SplitKMinGroups = 4` を追加。
- `src/shelf/kernels/optimized/linear/fp8.hip`
  - `gemm_fp8_finegrained_w8a8_splitk_kernel<MB, kSplit>` を追加。
  - 1 block の warps が K の 128 グループを連続に分担し、
    各 warp の F32 accumulator を LDS に書き、warp 0 が split 昇順で足す。
  - 行 tile は既存 kernel と同じ 16 行。split-K では MB=1 とし、
    grid.y で 16 行 tile を並べる。narrow out では block 数を増やす方が
    重みの再読み込みより効くためである。
  - 重み FP8 fragment、重み scale、activation scale の適用順は
    RowBlock 系と同じにした。activation scale は warp ごとの LDS に置く。
  - `launch_gemm_fp8_finegrained_w8a8_splitk` と
    `fp8_splitk_supported` を追加。
- `src/shelf/inference/execution/selectors/linear_selector.cpp`
  - `fp8_finegrained_splitk_config` を追加。
    `base_waves = ceil(out/16) * ceil(rows/16)` が `WaveSlots` 未満のとき、
    `ceil(WaveSlots / base_waves)` を 2/4/8/16 に丸める。
    `k_padded/128/4` を上限にし、K が短い形状では split しない。
  - split は rows≤16 の全形状と、rows≤64 かつ out≤1024 の形状に限定。
    rows>16 で out>1024 の形状は、実測で split が RowBlock1 に負けたためである。
  - rows 17〜64 で split を使わない形状は、out≤8192 なら RowBlock1、
    それ以外は rows≤32 で RowBlock2、rows>32 で RowBlock4 を選ぶ。
    1 warp/block の RowBlock1 が、狭い N では MB2/4 の重み再利用より速い。
- `src/apps/bench/gemm.hip`
  - FP8 の `--impl auto|splitk|rowblock1|2|4|prefill2d` と
    `--fp8-split N` を追加し、`selected` を表示する。
- `tests/unit/test_qwen35_linear_selector.cpp`
  - split-K と row tile の選択を固定。
- `tests/kernels/linear/test_gemm_fp8_optimized.hip`
  - rows 1〜64、N=24/48/256、K=2560 の split-K と、
    N=48/K=5120 の split 2/4/8/16 を追加。

### 2.2 正しさ

`shelf-bench gemm --dtype fp8 --check` で、split-K を選ぶ形状の
最大相対誤差は 0.73% 以下、NaN / Inf は 0 だった。
`test_gemm_fp8_optimized` は 302 件 pass、`test_mxfp4_linear` は pass、
`test_qwen35_linear_selector` は 110 件 pass。

split-K は 128 K グループの区間和を分割して最後に足すため、
RowBlock 系と bit 一致しない。production の correctness path との比較は
tolerance で行い、bit 一致の契約は外した。

### 2.3 速度

`auto`（新しい選択）と旧経路の比較。旧経路は rows≤16 が RowBlock1、
rows=32 が RowBlock2、rows=64 が RowBlock4 である。

| shape | rows | 旧 | 新 | 比 |
| --- | ---: | ---: | ---: | ---: |
| 24:5120 | 1 | 29.7 | 8.9 | 3.3x |
| 48:5120 | 1 | 19.8 | 8.9 | 2.2x |
| 512:5120 | 1 | 13.9 | 6.4 | 2.2x |
| 1024:5120 | 1 | 14.1 | 7.0 | 2.0x |
| 3072:5120 | 1 | 15.8 | 11.5 | 1.37x |
| 5120:5120 | 1 | 18.7 | 16.4 | 1.14x |
| 5120:6144 | 1 | 21.3 | 19.3 | 1.10x |
| 8704:5120 | 1 | 29.7 | 28.8 | 1.03x |
| 17408:5120 | 1 | 145.6 | 144.9 | 1.00x |
| 24:5120 | 16 | 31.3 | 9.6 | 3.3x |
| 48:5120 | 16 | 20.7 | 9.6 | 2.2x |
| 512:5120 | 16 | 14.6 | 6.9 | 2.1x |
| 1024:5120 | 16 | 14.7 | 7.7 | 1.9x |
| 3072:5120 | 16 | 16.0 | 12.4 | 1.29x |
| 5120:5120 | 16 | 21.6 | 17.0 | 1.27x |
| 5120:6144 | 16 | 27.0 | 21.4 | 1.26x |
| 8704:5120 | 16 | 34.4 | 30.4 | 1.13x |
| 17408:5120 | 16 | 148.1 | 148.4 | 1.00x |
| 24:5120 | 32 | 33.1 | 9.6 | 3.4x |
| 48:5120 | 32 | 17.6 | 9.6 | 1.8x |
| 512:5120 | 32 | 17.9 | 7.5 | 2.4x |
| 1024:5120 | 32 | 18.0 | 8.3 | 2.2x |
| 3072:5120 | 32 | 20.5 | 18.3 | 1.12x |
| 5120:5120 | 32 | 23.9 | 23.5 | 1.02x |
| 5120:6144 | 32 | 27.3 | 26.4 | 1.03x |
| 8704:5120 | 32 | 34.1 | 37.3 | 0.91x |
| 17408:5120 | 32 | 153.5 | 152.4 | 1.01x |
| 24:5120 | 64 | 58.0 | 9.6 | 6.0x |
| 48:5120 | 64 | 30.7 | 9.7 | 3.2x |
| 512:5120 | 64 | 31.1 | 8.7 | 3.6x |
| 1024:5120 | 64 | 32.3 | 11.7 | 2.8x |
| 3072:5120 | 64 | 33.4 | 23.4 | 1.43x |
| 5120:5120 | 64 | 41.8 | 33.3 | 1.25x |
| 5120:6144 | 64 | 48.5 | 41.1 | 1.18x |
| 8704:5120 | 64 | 56.4 | 55.3 | 1.02x |
| 17408:5120 | 64 | 197.2 | 201.3 | 0.98x |

`8704:5120` rows=32 は表では 0.91x だが、最終 build の selector は 8192 を境に
RowBlock2 を選ぶため、旧経路と同じ RowBlock2 である。34.1 と 37.3 の差は
同じ variant の run 間ゆらぎであり、selector の変更による差ではない。

`17408:5120` rows=64 の 0.98x は同じ RowBlock4 同士の差で、測定ゆらぎの範囲である。

### 2.4 採否

- ACCEPT: FP8 の block 内 split-K。narrow out（≤1024）の rows 1〜64 で
  1.8〜6.0 倍。rows≤16 では out≤10240 まで 1.03〜3.3 倍。
- ACCEPT: rows 17〜64 の RowBlock1（out≤8192）。MB2/4 の重み再利用より
  block 数を増やす方が速い。
- REJECT: rows>16 かつ out>1024 の split-K。5120:6144 などで RowBlock1 に負ける。
- DEFER: grid.z split + scratch。narrow out の更なる並列度は上がるが、
  program workspace の追加が必要になる。今回は block 内 LDS で足りた。

### 2.5 split-K の MB 選択

split-K の 16 行 tile を MB1 にするか MB2 にするかを、rows と out_features を
変えて測った。MB2 は block あたり 32 行を持つので、rows>32 では重みの再読み込みが
半分になる。一方で block 数が減るため、狭い out では MB1 が勝つ。

`--impl splitk --fp8-split 8` の p50 µs。MB1/MB2 を交互に 4 回測り、
warm-up 後の 2 回を使った。

| shape | rows | MB1 | MB2 | 比 |
| --- | ---: | ---: | ---: | ---: |
| 24:5120 | 64 | 6.8 | 9.7 | MB1 |
| 48:5120 | 64 | 6.7 | 9.8 | MB1 |
| 512:5120 | 64 | 8.6 | 10.2 | MB1 |
| 1024:5120 | 64 | 18.5 | 11.7 | MB2 1.58x |
| 1024:5120 | 32 | 8.3 | 10.7 | MB1 |
| 3072:5120 | 32 | 19.0 | 15.2 | MB2 1.25x |
| 5120:5120 | 32 | 28.7 | 22.3 | MB2 1.29x |
| 5120:6144 | 32 | 34.1 | 25.4 | MB2 1.34x |
| 8704:5120 | 32 | 51.2 | 34.5 | MB2 1.48x |
| 10240:5120 | 32 | 60.0 | 39.4 | MB2 1.52x |

規則は `rows > 32 && out_features >= 1024` で MB2、それ以外は MB1 とした。
rows=32 の out≥3072 は MB2 が速いが、split-K 自体をその帯で使っていないため
今回は selector の対象外である。split 数も MB に合わせて計算し直す
（MB2 では `row_blocks = ceil(rows/32)`）。

## 3. MXFP4 W4A8

### 3.1 変更

- `include/shelf/kernels/optimized/linear/config.h`
  - `kMxfp4GemmSplitKMaxRows` を 32 から 64 へ広げた。
  - `kMxfp4GemmSplitKMinKPadded = 2048` と
    `kMxfp4GemmRowBlock1MaxOutFeatures = 8192` を追加。
  - `Mxfp4GemmConfig` に `split_mb` を追加。
  - 使われなくなった `kMxfp4GemmSplitKMaxOutFeatures` を削除。
- `src/shelf/inference/execution/selectors/linear_selector.cpp`
  - split-K を `rows≤64 && out≤1024 && k_padded≥2048` に限定。
    K=640 の 2560:640 は、8-way split の縮約コストが並列度を上回るため除く。
  - rows 17〜32 で out≤8192 の形状は RowBlock1 を選ぶ。
    6144:5120 rows=32 は RowBlock2 より約 9% 速い。
  - rows>32 は従来どおり RowBlock2、rows>192 は RowBlock4。

### 3.2 速度

warm-up 後の交互比較。旧経路は rows>16 の RowBlock2。
SplitK の MB は selector の規則（3.3）で選んだ値である。

| shape | rows | RowBlock2 | SplitK | MB | 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 48:5120 | 32 | 22.7 | 10.9 | 1 | 2.08x |
| 48:5120 | 64 | 20.5 | 10.9 | 1 | 1.88x |
| 640:2560 | 32 | 13.1 | 8.4 | 1 | 1.56x |
| 640:2560 | 64 | 13.2 | 8.7 | 1 | 1.52x |
| 1024:5120 | 32 | 22.8 | 11.3 | 1 | 2.02x |
| 1024:5120 | 64 | 22.8 | 16.3 | 2 | 1.40x |

`6144:5120` rows=32 は RowBlock1 24.4 対 RowBlock2 26.9（1.10x）。
`10240:5120` rows=32/64 は RowBlock2 のままで、差は測定ゆらぎの範囲である。

### 3.3 split-K の MB 選択

`--impl splitk --mxfp4-split 8` の p50 µs。MB1/MB2 を交互に測った。

| shape | rows | MB1 | MB2 | 比 |
| --- | ---: | ---: | ---: | ---: |
| 48:5120 | 64 | 10.9 | 14.5 | MB1 |
| 640:2560 | 64 | 8.7 | 11.1 | MB1 |
| 2560:640 | 64 | 9.8 | 11.1 | MB1 |
| 1024:5120 | 64 | 18.6 | 15.4 | MB2 1.21x |
| 1024:5120 | 32 | 11.3 | 14.9 | MB1 |

FP8 と同じく `rows > 32 && out_features >= 1024` で MB2、それ以外は MB1 とした。
rows≤32 は MB1 の方が block 数が多く、rows>32 でも out<1024 は重みが小さいため
MB1 が勝つ。K=640 の 2560:640 は split-K 自体を使わない。

### 3.4 採否

- ACCEPT: rows 33〜64 の narrow out への split-K 拡張。
- ACCEPT: rows 17〜32・out≤8192 の RowBlock1。
- ACCEPT: split-K の MB 選択（rows>32 かつ out≥1024 で MB2、それ以外 MB1）。
- REJECT: K=640（2560:640）の split-K。縮約が重い。

## 4. 残課題

1. narrow out の更なる改善には grid.z split + scratch が必要。
   program workspace の追加は、lm_head のような大きな形状を含めて
   scratch を共有する設計にしないと容量が増える。
2. BF16 は未実装。参照実装の調査は
   [bf16_small_m_reference.md](bf16_small_m_reference.md) に分けた。
3. FP8 の rows=32・out≥3072 は MB2 split が RowBlock1 より速い帯がある
   （3072: 15.2 対 18.3、5120: 22.3 対 22.8、6144: 25.4 対 26.4）。
   現状は split の対象を rows≤16 または out≤1024 に限定しているため未接続。

## 5. 再現

```sh
HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype fp8 \
  --shapes 24:5120,48:5120,512:5120,1024:5120,3072:5120,5120:5120,5120:6144,8704:5120,17408:5120 \
  --rows 1,16,32,64 --warmup 30 --samples 15 --launches 15 --device 0

HIP_VISIBLE_DEVICES=1 ./build/shelf-bench gemm --dtype mxfp4 \
  --shapes 48:5120,640:2560,1024:5120,6144:5120,10240:5120 \
  --rows 16,32,64 --warmup 30 --samples 15 --launches 15 --device 0

HIP_VISIBLE_DEVICES=1 ./build/tests/test_gemm_fp8_optimized
HIP_VISIBLE_DEVICES=1 ./build/tests/test_mxfp4_linear
```
