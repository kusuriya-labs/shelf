# EXL3 dense prefill の選定表（R9700 / TP=1）

`d4fae4a2` と、その直後に `prefill_super64` の起動を列バンドへ分割した版について、
[kernel_optimization_and_fusion.md](../developer/kernel_optimization_and_fusion.md) の
①（計測と ISA 収集）を実施した結果である。数値は 1 回の run あたりに換算している。

## 測定条件

- GPU: R9700（gfx1201）× 1、TP=1、`--device 0`、`flock /tmp/r9700.lock` で排他
- コマンド: `shelf-bench pp --model-dir <Qwen3.8-27B-EXL3-INT8-4bpw> --mode forward
  --tokens-file /tmp/shelf-prefill-8192.psktok --prompt-tokens 2048 --prefill-chunk 2048
  --kv-cache-dtype bf16 --device 0 --warmup 1 --runs 1`
- 入力は 8192 ID の固定 corpus の先頭 2048。量子化は EXL3-INT8 4 bpw、KV は BF16
- 収集: `rocprofv3 --kernel-trace --output-format csv`。trace には warmup と本 run の
  2 反復が入るため、件数と時間を 1/2 して 1 run とした
- 分割後: wall 986.2 ms、カーネル合計 934.5 ms、カーネル外時間 51.7 ms（5.2%）

TP=2 で `--prefill-chunk` を併用すると稀にハングするため、内訳の取得は TP=1 で行う。

## 1. kernel 家族別（share 上位、分割後）

| kernel | 回数 | ms | share | VGPR | LDS B | wg | waves/SIMD | blocks/CU | 律速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exl3_mul24_prefill_super64_kernel<true,true>` | 608 | 513.05 | 54.9% | 232 | 34816 | 256 | 6 | 3 | ② 計算 / L2 |
| `shelf_gdn_recurrence_wmma_lossy_v32` | 48 | 154.47 | 16.5% | 248 | 14336 | 128 | 5 | 5 | ③ レイテンシ |
| `exl3_mul24_recon_x4_kernel<4>` | 400 | 46.09 | 4.9% | 40 | 0 | 256 | 16 | 8 | ② bytes 固定 |
| `exl3_mul24_gemm_kernel<4,4,true,true>` | 401 | 42.92 | 4.6% | 192 | 0 | 128 | 8 | 7 | ② L3 |
| `exl3_mul24_rotate_quantize_f4_kernel` | 400 | 32.89 | 3.5% | 40 | 0 | 256 | 16 | 8 | ① DRAM |
| `attention_paged_prefill_bf16_impl<0>` | 16 | 32.59 | 3.5% | 176 | 31744 | 256 | 8 | 3 | ②/③ |
| `exl3_mul24_rowprepare_kernel` | 400 | 12.19 | 1.3% | 32 | 512 | 256 | 16 | 8 | ② L3 |
| `exl3_mul24_rotate_out_i32_f4_kernel<true,true,true>` | 400 | 5.84 | 0.6% | 24 | 0 | 256 | 16 | 8 | ② L3 |

waves/SIMD と blocks/CU は
[kernel_optimization_and_fusion.md](../developer/kernel_optimization_and_fusion.md)
の算定式（`slots = ceil(VGPR/24)*24`、`min(16, 1536/slots)`、
`min(4*floor(47104/(32*slots)), 64)`、`131072/LDS`）による。
`prefill_super64` は VGPR 232（6 waves/SIMD = 24 waves/CU）と LDS 34816（3 blocks）の
両方が 3 blocks/CU を拘束している。4 blocks/CU には VGPR ≤ 168 と LDS ≤ 32768 が必要で、
VGPR を 168 以下に落とすには 4×4 の accumulator tile を半分にする必要がある
（試作では compiler が 196 VGPR 止まりで、spill 込みの E2E は +1.4% だった）。

## 2. EXL3 dense chain（1 linear = 6 launch × 400）

`use_dense && n % 256 == 0 && k % 64 == 0` の 400 linear はすべて
`rotate_quantize → rowprepare → recon_x4 → prefill_super64_fused → mul24_gemm(flagged)
→ rotate_out_i32_flagged` の 6 launch で並ぶ（trace で 800/800 がこの並び）。

linear 形状（1 run）: 17408×5120 ×128、5120×17408 ×64、5120×6144 ×64、
10240×5120 ×48、6144×5120 ×48、1024×5120 ×32、12288×5120 ×16。合計 99.6 TFLOP。

| 段 | 回数 | ms | chain share | uniq GB | GB/s | issued GB | GB/s | L2/L3 吸収 | TOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill_super64` | 608 | 513.05 | 78.4% | 46.24 | 90 | 411.14 | 801 | 789% | 194 |
| `recon_x4` | 400 | 46.09 | 7.0% | 36.49 | 792 | 36.49 | 792 | 0% | — |
| `mul24_gemm`（flagged） | 401 | 42.92 | 6.6% | 16.90 | 394 | 53.39 | 1244 | 216% | — |
| `rotate_quantize` | 400 | 32.89 | 5.0% | 18.00 | 547 | 18.00 | 547 | 0% | — |
| `rowprepare` | 400 | 12.19 | 1.9% | 12.06 | 989 | 12.06 | 989 | 0% | — |
| `rotate_out_i32`（fixup） | 400 | 5.84 | 0.9% | 5.99 | 1026 | 5.99 | 1026 | 0% | — |

bytes モデル: `rotate_quantize` = bf16 入力 + int8 codes + f32 scales、
`rowprepare` = codes の読み書き + scales、`recon` = trellis（`nk*bits/8`）+ dense（`nk`）、
`prefill_super64` = codes + 重み平面 × 行ブロック数（issued）または 1 回（uniq）+ BF16 出力、
`mul24_gemm` と `fixup` は flagged 行を `rows/8`（cap）と仮定した上限値。

読み方:

- `prefill_super64` は issued 801 GB/s で DRAM 上限 636 GB/s を超え、L2/L3 が 7.9 倍吸収する。
  **帯域律速ではなく 194 TOPS（実測ピーク 308 の 63%）の計算律速**。
  分割で起動回数が 400 → 608 に増え、codes の再読を含めて +14.5 ms（498.6 → 513.1 ms）。
- `recon` は 792 GB/s（DRAM 上限超）で L3 吸収込みの上限付近。削るには bytes を減らすしかない。
- `rotate_quantize` は 547 GB/s で DRAM 帯域帯（550–636）。① 帯域律速。
- `rowprepare` と `fixup` は 1 TB/s 前後で L3 常駐。

## 3. kernel 外時間

分割前は 97.8 ms/run（9.7%）あり、そのうち 84.3 ms が「1 回の起動が 1 ms を超える
カーネルの直後」に固定 262 µs で入っていた。`prefill_super64` の起動を列バンドに
分割して 1 回 0.84 ms 未満にすると、`super64 → mul24_gemm` の gap は
p50 261.5 µs → 13.0 µs、合計 67.8 ms → 16.6 ms になった。

| 遷移 | 回数/run | gap p50 | 合計/run |
| --- | ---: | ---: | ---: |
| `prefill_super64` → `mul24_gemm`（flagged） | 400 | 13.0 µs | 16.6 ms |
| `prefill_super64` → `prefill_super64`（バンド内） | 208 | 13.0 µs | 3.0 ms |
| `gdn_recurrence` → 直後 | 48 | 263.4 µs | 12.3 ms |
| `attention_paged_prefill` → `sigmoid` | 16 | 264.3 µs | 4.2 ms |
| `rowprepare` → `recon` / `recon` → `super64` / `mul24_gemm` → `fixup` | 400 ずつ | 4.2 µs | 7.0 ms |

残る空白は `gdn_recurrence`（12.3 ms）と `attention_paged_prefill`（4.2 ms）で、
どちらも 1 回の起動が 2〜3 ms ある。前者は系列方向の依存、後者はページ単位の構造のため
単純な分割ができない。機序の切り分けと分割の詳細は
[20261125/prefill_gap_262us.md](../research/20261125/prefill_gap_262us.md)。

## 4. 次のレバー（優先順）

1. **`gdn_recurrence` の VGPR**: 248 → 240 で slots 264 → 240、waves/SIMD 5 → 6（+20%）。
   share 16.5% で最大の非 GEMM カーネル。同時に 12.3 ms の空白も分割で消せるなら
   合計 4% 弱。
2. **`prefill_super64` の 194 TOPS**: ピーク 308 の 63%。occupancy を上げるには
   accumulator tile の半減（VGPR ≤ 168）が要り、試作では 196 VGPR + spill で改悪だった。
   tile 構成の変更そのものを測り直す必要がある。
3. **`attention_paged_prefill`**: 32.6 ms と 4.2 ms の空白。ページ構造の分割余地。
4. **`rotate_quantize` の bytes**: 547 GB/s の ① 帯域律速。producer フック
   （`exl3_swiglu_rotate_rows` 等）の適用範囲を広げて bf16 の書き戻しを消す方向のみ有効。

## 再現

```sh
export PATH=/opt/zen/.venv/bin:$PATH
flock /tmp/r9700.lock rocprofv3 --kernel-trace --output-format csv -o /tmp/prof/pp2048 -- \
  ./build/shelf-bench pp --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
  --mode forward --tokens-file /tmp/shelf-prefill-8192.psktok --prompt-tokens 2048 \
  --prefill-chunk 2048 --kv-cache-dtype bf16 --device 0 --warmup 1 --runs 1
```

trace の `Grid_Size_X` は workgroup 数を `Workgroup_Size_X` 倍した値で出るため、
block 数は割って求める。`Grid_Size_Y/Z` は block 数のままである。
