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
E8M0 scale は row-major）を読む。activation 平面は行基準 fold 形で、quantize / linear
とも [quantization.md](quantization.md) の activation contract に従う。
`test_mxfp4_linear`（`gpu1;required`）は、correctness path と optimized kernel の全 variant
が double 参照に対して bf16 出力の tolerance 内（bf16 exact >= 99.9%、rel_l2 < 2e-3）に
収まることを固定する。correctness path との bf16 一致率は診断情報として print するだけで
契約にはしない（reference 経路を optimized 側へ合わせ込む形を取らない）。
あわせて reference 側の rowref quantize と optimized 側の quantize が作る平面の bit 一致、
および weight scale group・codes stride・k_padded の contract 違反が dispatch 側で
`INVALID_PROGRAM` になることを固定する。

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
- activation 平面は行基準 fold 形（[quantization.md](quantization.md)）なので、K ループに
  activation scale は現れない。行基準 `2^ref` は epilogue で 1 回掛ける。
- weight scale を E4M3 operand 側へ fold すると 32 K block の rescale も消え、WMMA
  fragment が K 全体を 1 本の accumulator へ chain する。fold しないときは 32 K block
  ごとに `acc = fma(c, weight_scale, acc)` を適用する。
- variant は `SplitK`（rows <= 32 かつ out_features <= 2560、wave で K を分割し LDS で
  reduce）、`RowBlock1 / 2 / 4`（activation 行 tile の数）、
  `Prefill2D`（rows >= 512 の prefill。1 block が 256 行 x 64 output、8 wave。
  128 K chunk ごとに weight fragment を展開して LDS へ stage し、4 行 tile で再利用する）。
  `rows == 1` の専用 variant は持たない（FP8 linear で decode1 経路を削除した方針に従う。
  rows == 1 は SplitK または RowBlock1 が担当し、DRAM 律速 shape では 590 GB/s 出る）。
  variant 選択と geometry allowlist は `selectors/linear_selector.cpp` が hold する。
- activation codes / scales は FP8 W8A8 と同一の `Int8ActivationWorkspaceLayout`
  （64 K fragment block、scale row stride `align16(ceil(k_padded/128)*4)`）を読むが、
  scale 平面は column 0 の行基準だけを使う。
- **E8M0 の行基準 fold**（`d = ref[row] - e8m0[row][blk]` で E2M1 仮数を 2^-d 下げて
  E4M3 operand に畳み、`2^(ref-127)` を epilogue で 1 回適用する）は **Prefill2D のみ**
  で使う。RowBlock / SplitK は rows == 1 で遅くなった（表参照コストが per-block rescale
  の削減を上回る）ため unfold のまま。variant 別の判定は
  `mxfp4_gemm_folds_row_ref(config.id)`。rows == 1 では fold の損得はほぼゼロに変わった
  （group 平面で観測した fold 損は行基準形では消える）が、採用は Prefill2D のまま据え置く。
- fold は `d <= 8` で E2M1 仮数を厳密に表現する（`d >= 9` は E4M3 subnormal の
  1.5 刻みを丸める）。payload は行単位の E8M0 最大値平面 `row_ref` と、全行の
  E8M0 幅が `kMxfp4FoldExactMaxDistance`(=8) 以内であることを示す `row_ref_exact` を
  持つ。**dispatch は `row_ref_exact` が false の payload で fold しない**
  （`optimized_dispatch.hip`）。厳密帯域の外では fold が E4M3 subnormal の下位を丸める
  ため値そのものが変わる。実測（Qwen3.8-27B MXFP4 の 3,904,000 行のうち幅 > 8 は
  5 行、2 tensor）と判断は
  [../research/20261103/mxfp4_gate_a_prime.md](../research/20261103/mxfp4_gate_a_prime.md)。
- **grouped 形** `launch_gemm_mxfp4_w4a8_grouped`（同ファイル）は rank-3 expert 平面を
  group table（expert index と行数）で読む。1 workgroup = 16 output column × 1 group
  （最大 `kMxfp4GemmRowTile` 行）で、expert base を codes / scales / row_ref の batch
  stride だけずらして使う。activation 平面と出力は **group row** 番号でアドレスする
  （group g は行 [g*16, g*16+16) を占有）。MoE の geometry は out_features が 16 の倍数
  なので column guard は持たず、`group_rows[g]` を超える行は計算するが store しない。
  fold の可否は dense と同じ規則（`weight_ref` を渡すかどうか）で、呼び出し側が
  `row_ref_exact` から決める。contract と test は
  [qwen35moe.md](qwen35moe.md) / `test_moe_grouped_linear`。

### moe / routing・combine

`src/shelf/kernels/optimized/moe/router.hip` と `combine.hip` は standalone launcher で、
まだ `KernelId` も selector も持たない（graph 配線は未実装。
[qwen35moe.md](qwen35moe.md) を参照）。

- `launch_moe_router`: 1 workgroup = 1 token。f32 softmax → top-k（確率降順、同値は小
  index）→ top-k 合計で renormalize。expert 別件数を token tile 単位の histogram へ
  atomicAdd する（値は順序に依存しない）。
- `launch_moe_slot_scan` / `launch_moe_slot_scatter`: tile histogram から expert 別 slot
  範囲・group table・tile 別 base を決め、slot を expert 昇順 / token 昇順で group row
  へ写す。atomic は LDS の per-expert cursor だけで、block ごとに token 昇順を保持する
  ため結果は決定的である。scan は `row_token` を host 容量ぶん `-1` で埋める
  （activation 量子化は group 数ではなく容量ぶん row を launch する）。
  scan は 1 workgroup で、expert 別の合計は全 thread で LDS へ集約してから、prefix の計算を
  thread 0 が行う（thread 0 だけに依存ループを戻すと decode で 27 µs 級に遅い。計算順は
  expert index 昇順のままなので出力は変わらない）
- group は `kMxfp4GemmRowTile`(=16) slot までで、expert をまたがない。host 側の容量は
  `moe_group_count_upper_bound` / `moe_group_row_capacity` が単一の源。
- `launch_moe_combine`: grid は `(rows, ceil(features / kCombineThreads))` で、
  1 workgroup が 1 token の 128 feature 分を持つ（decode の rows=1 でも feature 方向に
  並列化する）。top_k 個の group row を **group row 昇順（= expert 昇順）** に並べ替えて
  f32 で蓄積し、最後に 1 回 bf16 へ丸める。feature ごとの accumulate 順序は workgroup 分割に
  よらないので、出力は分割前後で bit 一致する。shared expert を渡すと
  `gate_weight[t * gate_row_stride] * shared[t]` を routed 合計の後に足す。gate は 1 列でも
  row-major value の平面なので row stride を呼び出し側が渡す（packed 前提だと token 0 以外の
  shared expert の寄与が消えるか別データの bytes を読む）。`gate_weight` は活性化済み値で、
  活性化自体は既存の `launch_sigmoid_bf16_f32` が担う（combine は超越関数を使わない）。
- activation 平面の生成は `launch_activation_quantize_mxfp4_rowref` の `row_index`
  間接参照を使う（`row_index[m] < 0` の行は zero operand + scale 1.0 になる）。
  これにより grouped GEMM は常に fragment order の 16 行 tile を連続で読む。

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

- split-KV decode kernel（bf16 split / bf16 split reduce / fp8 split）が起動するのは、
  kv dtype ごとの capability 表に載っている `q_per_kv = q_heads / kv_heads` の geometry だけである。
  正本は `include/shelf/kernels/optimized/attention/paged_attention.h` の
  `paged_attention_bf16_split_supports(q_per_kv)`（4 / 6 / 8）と
  `paged_attention_fp8_split_supports(q_per_kv)`（4 / 6 / 8）で、kernel 側の gate と
  `plan_paged_attention` の split 選択の両方がそれを読む。planner は `split_capability` で
  kv dtype を選び、dtypeに無い instantiation を選ばない。capability の無い geometry は
  非 split decode kernel（1 workgroup / (row, q_head) で可視域を歩む）が受け持つ。
  kernel はこの条件を外れたときに `hipErrorInvalidValue` を返す（`hipErrorNotSupported` では
  ない）。dispatch 側は `hipErrorNotSupported` だけを correctness fallback へ訳すので、
  planner が capability を無視して splits を選べば実行は失敗として終わる。
- split 本体と reduce は `q_per_kv` ごとの別 instantiation で、partials の stride も
  `q_per_kv * (2 + head_dim)` である。片方だけを足すと stride がずれて**launch は成功したまま
  静かに誤った値**になる（`hipErrorInvalidValue` ですらない）。bf16 と fp8 の 8（MoE の 16/2）が
  それで、`tests/kernels/optimized/test_paged_attention.hip` の split16 比較（bf16 / fp8 双方）と
  `tests/unit/test_paged_attention_split_reduce_exact.hip` の q_per_kv 6 / 8 が対にする。

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
