# Qwen4Exp TP=2 prefill の correctness fallback と MXFP4 kernel の spill

## 1. Qwen3.8-Flash-Next TP=2 prefill が 3 倍遅い件

### 症状

`models/Qwen3.8-Flash-Next-Quark-MXFP4` の 4096 token prefill が、TP=1 で 2.51 s、
TP=2 で 12.65 s（device 2,3）／12.84 s（device 0,2）だった。GPU の PCIe switch 配置を
変えても変わらないため、host expert plane の帯域競合ではない。

### 原因

`src/shelf/inference/execution/selectors/linear_selector.cpp` の BF16 shape allowlist に、
TP=2 で GDN の `in_proj_z` が取る `(out_features 3072, k 2560)` が無かった。
TP=1 の `(6144, 2560)` は登録済みなので、TP=1 だけが最適化経路を通っていた。

- GDN の value 幅は `linear_num_value_heads 48 × linear_value_head_dim 128 = 6144`。
  TP=2 では 24 head になり 3072 に半減する。
- allowlist miss により `LINEAR_BF16` が correctness の汎用インタプリタへ落ちる。
  1 回は 4096 × 3072 × 2560 のスカラ GEMM で約 310 ms、36 GDN 層 × 2 rank で約 11.2 s。

### 証拠

- stderr: `correctness fallback: LINEAR_BF16 rows=4096`
- `SHELF_LINEAR_DEBUG=1`、512 token、TP=2: `LINEAR_SELECT_MISS family=0 out=3072 k=2560`
  を 72 回（36 層 × 2 rank）
- rocprofv3 kernel trace、TP=2、4096 token:
  - `model_dispatch_kernel` 144 dispatch × 約 310 ms = 44.6 s（warmup と測定の 2 run、2 rank 合計）
  - `gemm_mxfp4_w4a8_grouped_kernel` 1152 dispatch = 2.68 s
  - その他の kernel は合計 2 s 未満
- fallback 先の kernel は grid 16384 × 256、VGPR 136、scratch 2816 B、LDS 43008 B。
  WMMA を使わず、1 thread が 1 出力を K 方向にスカラ FMA する。

### 修正

- allowlist に `{3072, 2560, kFamilyBf16}` を追加。
- `tests/unit/test_qwen35_linear_selector.cpp` に TP=2 の `in_proj_z` 形の選択テストを追加。
- `tests/kernels/optimized/test_gemm_bf16_wmma.hip` に N=3072 / K=2560 の
  最適化カーネルと CPU 参照の比較を追加（rows=1, 96）。

### 修正後

`--device 2,3 --tensor-parallel 2 --moe-host-experts 224`、4096 token、warmup 1、
3 run は 1722.8 / 1447.4 / 1277.3 ms（p50 1447.4 ms、2829.9 tok/s）だった。
1 run 目はキャッシュが温まらず遅く、run を追うごとに短くなる。TP=1 は 2389.5 ms
（同日、`--device 1 --moe-host-experts 448`、p50、run 間の幅 0.4 %）で、TP=2 の方が
1.65 倍速い。

## 2. MXFP4 kernel の spill

ISA は gfx1201、`-O3`、実際の build object から取得した。値は
VGPR / VGPR spill / private segment bytes。

| kernel | 修正前 | 修正後 |
| --- | --- | --- |
| `gemm_mxfp4_w4a8_prefill_2d_kernel<64, guarded, fold=false>` | 256 / 177 / 496 | 削除 |
| `gemm_mxfp4_w4a8_prefill_2d_kernel<64, fold=true>` | 114 / 0 / 0 | 118 / 0 / 0 |
| `gemm_mxfp4_w4a8_prefill_2d_kernel<64, fold=false>` | 141 / 0 / 0 | 155 / 0 / 0 |
| `gemm_mxfp4_w4a8_wmma_kernel<4, fold=false>` | 256 / 54 / 164 | 175 / 0 / 0 |
| `gemm_mxfp4_w4a8_wmma_kernel<2, fold=false>` | 256 / 83 / 208 | 109 / 0 / 0 |
| `gemm_mxfp4_w4a8_wmma_kernel<1, fold=false>` | 188 / 0 / 0 | 188 / 0 / 0 |

### prefill 2D の guarded 経路

`out_features % 64 != 0` のときだけ使う guarded 経路が 256 VGPR に達し、177 spill を
出していた。prefill 2D は 64 出力の完全 tile を前提にし、端数は row-block 系に任せる。
allowlist と launcher の両方で `out_features % 64 == 0` を要求し、`kGuarded` template
引数と分岐を削除した。非 fold の K ループは 32-K block ごとの積を別に持つため、
`kb` を 1 つずつ進める（fold 側は従来どおり全 unroll）。

### row-block の非 fold 経路

`fold=false` は 32-K block ごとに E8M0 を掛けるため live set が広く、MB 2/4 の全 unroll が
256 VGPR 上限に達していた。非 fold の MB 2/4 だけ K group の unroll を 1 に落とした。
MB 1 は 1 accumulator で元から spill が無いため、unroll 4 のままにした。

### A/B（`models/Qwen3.8-27B-Quark-AWQ-MXFP4`）

このモデルは prefill 2048 token で `wmma_kernel<4, fold=false>` を 96 回
（GDN の `in_proj_a` / `in_proj_b`、48 × 5120、48 層 × 2）使う。
大きい行列は prefill 2D が担当する。

| 計測 | 修正前 | 修正後 |
| --- | ---: | ---: |
| `rowblock4` 2048 × 48 × 5120、fold=0、p50 | 31.33 µs | **25.04 µs**（−20 %） |
| `rowblock4` 256 × 17408 × 5120、fold=0、p50 | 502.4 µs | **418.1 µs**（−17 %） |
| e2e prefill 256 token、p50（2 回） | 169.77 / 170.78 ms | **156.98 / 156.84 ms**（−7.7 %） |
| e2e prefill 2048 token、p50（2 回） | 934.28 / 936.98 ms | 935.27 ms（変化なし） |

256 token では prefill 2D の条件（rows >= 512）に入らず全行列が row-block になるため、
spill 除去がそのまま e2e に出る。2048 token は大きい行列が prefill 2D に移るので差が出ない。

### 契約の固定

`tools/check_kernel_spills.py` の対象に `src/shelf/kernels/optimized/linear/mxfp4.hip` を
追加し、`gemm_mxfp4_` で始まる kernel すべてを spill 0 / private segment 0 の契約にした。

## 3. 再現

```sh
# TP=2 prefill
./build/shelf-bench pp --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --device 2,3 --tensor-parallel 2 --moe-host-experts 224 \
  --prompt-tokens 4096 --prefill-chunk 4096 \
  --kv-cache-dtype octave:c5v5 --tokens-file /tmp/pf4096.psktok \
  --warmup 1 --runs 3

# 27B Quark AWQ の row-block4 / fold=0
./build/shelf-bench gemm --dtype mxfp4 --impl rowblock4 \
  --rows 256 --n 17408 --k 5120 --mxfp4-fold 0 --warmup 20 --samples 30

# e2e prefill
./build/shelf-bench pp --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --device 0 --prompt-tokens 256 --prefill-chunk 256 \
  --kv-cache-dtype bf16 --warmup 1 --runs 3

# ISA 契約
python3 tools/check_kernel_spills.py --build-dir build
```

## 4. 採否

- ACCEPT: BF16 allowlist への `{3072, 2560}` 追加。TP=2 prefill の主因を除去した。
- ACCEPT: prefill 2D の guarded 経路削除。spill 0 になり、端数形状は row-block で処理する。
- ACCEPT: row-block 非 fold の MB 2/4 を unroll 1。kernel 単体で 17-20 %、
  256 token prefill で 7.7 % 改善した。
- REJECT: TP=2 の prefill 遅化を PCIe / host offload とみなす説。device 配置を変えても
  12.6-12.8 s で変わらず、trace でも MoE は 2.68 s に過ぎなかった。
