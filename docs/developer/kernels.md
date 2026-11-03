# Kernels

model 非依存の最適化済み kernel は `src/shelf/kernels/optimized/`（public は
`include/shelf/kernels/optimized/`）に置く。Qwen35 が使う。
Qwen35 固有 kernel（correctness / dflash2）は
`src/shelf/models/qwen35/kernels/` に置く。
配置と分割の詳細は `src/shelf/models/qwen35/kernels/README.md` を参照。

```text
shelf/
├── kernels/optimized/          # model 非依存の gfx1201 向け高速実装
└── models/qwen35/kernels/
    ├── correctness/            # 遅いが正確な参照実装（E2E token exactness の基準）
    └── dflash2/                # DFlash2 固有 kernel
```

## 分割ルール

- `KernelId` は論理IDであり、ファイル構成を決めない。
- 1 ファイル = 1 実装 family（同じ最適化問題）。複数の `__global__` / launcher を置いてよい。
- dtype は variant。同じ family の BF16 / FP8 は同じファイルに置く。
- `1 KernelId = 1 file` / `1 __global__ = 1 file` / `1 dtype = 1 file` は原則NG。
- 細切れの `silu.hip` / `sigmoid.hip` / `scale.hip` / `mul.hip` / `residual_add.hip` /
  `swiglu.hip` は作らず、`elementwise.hip` にまとめる。
- attention は `attention_paged_bf16.hip` / `attention_paged_fp8.hip` ではなく
  `paged_decode.hip` / `paged_prefill.hip` に BF16 / FP8 variant を内包する。

## Correctness

`correctness/model_dispatch_correctness.hip` が唯一の `__global__` dispatch + KernelId switch +
launcher を持ち、`detail/*.inc` を include して 1 つの HIP translation unit を構成する。

```cpp
#include "detail/dispatch_env.h"
#include "detail/elementwise.inc"
#include "detail/normalization.inc"
#include "detail/linear.inc"
#include "detail/attention.inc"
#include "detail/gdn.inc"
#include "detail/embedding.inc"
#include "detail/output.inc"
```

これにより source 上は分割しつつ、次を満たす。

- KernelId switch は 1 箇所
- device helper の共有が容易
- RDC を要求しない
- correctness reference が 1 つに保たれる

`correctness/standalone/` は kernel 単体テスト用の参照実装
（`silu` / `swiglu` / `rmsnorm` / `gemm_bf16`）。E2E correctness と同じ階層に
二重実装を増やさない。可能なら `DispatchBinding` を作って
`model_dispatch_correctness` 経由で検証し、この階層を縮小する。

correctness path は production と同じ W8A8 contract で動く。activation quantize は
`Int8ActivationWorkspaceLayout` の preshuffle layout、同じ block scale、同じ丸めを
使う。`ValueBinding::row_stride` は要素単位であるため、byte pointer で行を進める
場合は dtype の要素サイズを掛ける。bf16 は `bf16` 型で load し、`uint16_t` を
数値として変換しない。

`test_activation_quantize_fp8_finegrained`（`gpu1;required`）は、production shape と
全入力 pattern で `ACTIVATION_QUANTIZE_FP8_FINEGRAINED` の correctness path が
generic kernel と codes / scales まで一致することを固定する。

`LINEAR_MXFP4_W4A8` の correctness path も同じ activation layout を使い、weight 側は
native layout（16 output x 32 K block、byte b = code[k=b] | code[k+16]<<4、
E8M0 scale は row-major）を読む。
`test_mxfp4_linear`（`gpu1;required`）は、E2M1 / E8M0 の decode と 32 K 刻みの rescale が
double 参照に対して bf16 出力で bit 一致すること、optimized kernel の全 variant が
同じ payload で correctness path と bf16 bit 一致すること、および weight scale group・
codes stride・k_padded の contract 違反が dispatch 側で `INVALID_PROGRAM` になることを
固定する。

## Optimized

optimized variantは `src/shelf/kernels/optimized/` に実装する。
対象platformはgfx1201のみであり、arch suffixは付けない。ソースは
`cmake/targets.cmake` に明示列挙する。

- correctness referenceはarch-independentで1つ。
- public launcher は semantic operator / optimization family ごとに置き、
  shape / dtype / algorithm variant の選択は launcher 内部に閉じ込め、runtime側へ漏らさない。
- WMMA kernelはgfx12 builtin（`_gfx12`）を使う。
- small-M（M=1..16）のbatch-invariant順序は `linear/bf16.hip` のexact-rows
  経路が担う。MTP verifyはこの順序に依存する。

### linear / MXFP4 W4A8

`src/shelf/kernels/optimized/linear/mxfp4.hip` は gfx1201 の FP8 WMMA
（`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`）を E2M1 weight で使う。

- weight は native layout（[canonical_soa_payload.md](canonical_soa_payload.md) §11）。
  `mxfp4_expand` は `v_perm_b32` だけで E2M1 -> E4M3 と sign を展開する。
  sign は selector の bit3 で negative table を選ぶため、shift / or が不要である。
  native layout の (k, k+16) 対が WMMA operand の 8 byte と一致するので、
  展開は byte 再配置なしに operand register へ書ける。
- 32 K block ごとに `acc = fma(c, weight_scale * activation_scale, acc)` を適用する。
  activation scale は group 128 なので 128 K（4 block）を 1 組として LDS へ stage する。
  weight scale を E4M3 operand 側へ fold しない（fold は加算再結合を変え、
  bit 一致が失われるため）。
- variant は `SplitK`（rows <= 32 かつ out_features <= 2560、wave で K を分割し LDS で
  reduce）、`RowBlock1 / 2 / 4`（activation 行 tile の数）、
  `Prefill2D`（rows >= 512 の prefill。1 block が 256 行 x 64 output、8 wave。
  128 K chunk ごとに weight fragment を展開して LDS へ stage し、4 行 tile で再利用する）。
  `rows == 1` の専用 variant は持たない（FP8 linear で decode1 経路を削除した方針に従う。
  rows == 1 は SplitK または RowBlock1 が担当し、DRAM 律速 shape では 590 GB/s 出る）。
  variant 選択と geometry allowlist は `selectors/linear_selector.cpp` が hold する。
- activation codes / scales は FP8 W8A8 と同一の `Int8ActivationWorkspaceLayout`
  （64 K fragment block、group 128 F32 scale）をそのまま読む。
- **E8M0 の行基準 fold**（`d = ref[row] - e8m0[row][blk]` で E2M1 仮数を 2^-d 下げて
  E4M3 operand に畳み、`2^(ref-127)` を epilogue で 1 回適用する）は **Prefill2D のみ**
  で使う。RowBlock / SplitK は rows == 1 で遅くなった（表参照コストが per-block rescale
  の削減を上回る）ため unfold のまま。variant 別の判定は
  `mxfp4_gemm_folds_row_ref(config.id)`。
- fold は `d <= 8` で E2M1 仮数を厳密に表現する（`d >= 9` は E4M3 subnormal の
  1.5 刻みを丸める）。payload は行単位の E8M0 最大値平面 `row_ref` と、全行の
  E8M0 幅が `kMxfp4FoldExactMaxDistance`(=8) 以内であることを示す `row_ref_exact` を
  持つ。**dispatch は `row_ref_exact` が false の payload で fold しない**
  （`optimized_dispatch.hip`）。これにより optimized kernel は payload が厳密帯域に
  収まる限り correctness path と bf16 bit 一致を保つ。実測（Qwen3.8-27B MXFP4 の
  3,904,000 行のうち幅 > 8 は 5 行、2 tensor）と判断は
  [../research/20261103/mxfp4_gate_a_prime.md](../research/20261103/mxfp4_gate_a_prime.md)。

## workgroup barrier のスコープ

LDS のやり取りだけを order する barrier は `detail::lds_barrier()`
（`src/shelf/kernels/optimized/detail/lds_barrier_device.h`）を使う。

`__syncthreads()` は workgroup scope の全 address space に対する acquire-release fence であり、
gfx1201 では両側に `global_inv scope:SCOPE_SE` が emit される。これは次に読む global
（KV stream、activation codes、重み）の L0/L1 行を捨てる。`lds_barrier()` は fence を
`local` address space に限定するため、その invalidate が消える。

- 選択基準: barrier の後に、同一 block の別の thread がその kernel 内で書いた global を
  読む経路が無いか。無いなら `lds_barrier()`。
- atomic・arrival counter・block 間 global 通信を扱う barrier は `__syncthreads()` のまま。
  現状 `src/shelf/kernels/optimized/` で該当するのは `radix_topn.hip` だけである。
- kernel 境界を跨ぐ global の可視性は launch 自体が保証するため、barrier の役割ではない。

## paged attention の不変条件

`paged_decode.hip` / `paged_prefill.hip` は次を満たす。

- 正規化の分母（softmax の指数和）が 0 のときは出力を書かない。
  full decode / split / reduce の全経路で同じ扱いとし、`0/0` を出力へ書かない。
  分母が 0 になるのは可視 token が 1 つも寄与しない場合（範囲外 page のみの
  split、committed prefix も ancestor も無い tree 行）である。
- bf16 prefill は V の LDS を二重バッファ（`v_page[2]`）で持ち、page loop の
  終端 barrier を置かない代わりに、staging が直前 page の PV と別バッファに
  書くことを前提にする。`p_tile` / `s_part` / `row_*` は同じ page 内で
  barrier を挟んで更新する。
- fp8 prefill は V を単一バッファで持つため、page loop の終端で barrier を取る。
- fp8 prefill の量子化契約は [quantization.md](quantization.md) の
  「fp8 KV の paged attention 数値契約」を正本とする。
