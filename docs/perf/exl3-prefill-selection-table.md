# EXL3 dense prefill の選定表（R9700 / TP=1）

`d4fae4a2` 時点の EXL3-INT8（mul24）dense prefill について、
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
- wall 1012.6 ms、カーネル合計 912.8 ms、カーネル外時間 97.8 ms（9.7%）

TP=2 で `--prefill-chunk` を併用すると稀にハングするため、内訳の取得は TP=1 で行う。

## 1. kernel 家族別（share 上位）

| kernel | 回数 | ms | share | VGPR | LDS B | wg | waves/SIMD | blocks/CU | 律速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exl3_mul24_prefill_super64_kernel<true,true>` | 400 | 498.57 | 54.6% | 232 | 34816 | 256 | 6 | 3 | ② 計算 / L2 |
| `shelf_gdn_recurrence_wmma_lossy_v32` | 48 | 150.69 | 16.5% | 248 | 14336 | 128 | 5 | 5 | ③ レイテンシ |
| `exl3_mul24_recon_x4_kernel<4>` | 400 | 45.99 | 5.0% | 40 | 0 | 256 | 16 | 8 | ② bytes 固定 |
| `exl3_mul24_gemm_kernel<4,4,true,true>` | 400 | 40.31 | 4.4% | 192 | 0 | 128 | 8 | 7 | ② L3 |
| `exl3_mul24_rotate_quantize_f4_kernel` | 400 | 32.59 | 3.6% | 40 | 0 | 256 | 16 | 8 | ① DRAM |
| `attention_paged_prefill_bf16_impl<0>` | 16 | 31.87 | 3.5% | 176 | 31744 | 256 | 8 | 3 | ②/③ |
| `shelf_swiglu_bf16` | 64 | 22.43 | 2.5% | 24 | 0 | 256 | 16 | 8 | ① DRAM |
| `shelf_residual_add_bf16` | 128 | 12.76 | 1.4% | 24 | 0 | 256 | 16 | 8 | ① DRAM |
| `exl3_mul24_rowprepare_kernel` | 400 | 11.74 | 1.3% | 32 | 512 | 256 | 16 | 8 | ② L3 |
| `shelf_silu_f32_bf16` | 48 | 11.14 | 1.2% | 32 | 0 | 256 | 16 | 8 | ① DRAM |
| `shelf_gdn_conv1d_bf16_f32` | 48 | 9.48 | 1.0% | 32 | 0 | 256 | 16 | 8 | ②/③ |
| `exl3_mul24_rotate_out_i32_f4_kernel<true,true,true>` | 400 | 5.75 | 0.6% | 24 | 0 | 256 | 16 | 8 | ② L3 |
| `shelf_rmsnorm_bf16_pf_oneplus` | 129 | 5.71 | 0.6% | 24 | 512 | 256 | 16 | 8 | ① DRAM |

waves/SIMD と blocks/CU は
[kernel_optimization_and_fusion.md](../developer/kernel_optimization_and_fusion.md)
の算定式（`slots = ceil(VGPR/24)*24`、`min(16, 1536/slots)`、
`min(4*floor(47104/(32*slots)), 64)`、`131072/LDS`）による。

## 2. EXL3 dense chain（1 linear = 6 launch × 400）

`use_dense && n % 256 == 0 && k % 64 == 0` の 400 linear はすべて
`rotate_quantize → rowprepare → recon_x4 → prefill_super64_fused → mul24_gemm(flagged)
→ rotate_out_i32_flagged` の 6 launch で並ぶ（trace で 800/800 がこの並び）。

linear 形状（1 run）: 17408×5120 ×128、5120×17408 ×64、5120×6144 ×64、
10240×5120 ×48、6144×5120 ×48、1024×5120 ×32、12288×5120 ×16。合計 99.6 TFLOP。

| 段 | 回数 | ms | chain share | uniq GB | GB/s | issued GB | GB/s | L2/L3 吸収 | TOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill_super64` | 400 | 498.57 | 78.5% | 46.24 | 93 | 411.14 | 825 | 789% | 200 |
| `recon_x4` | 400 | 45.99 | 7.2% | 36.49 | 793 | 36.49 | 793 | 0% | — |
| `mul24_gemm`（flagged） | 400 | 40.31 | 6.3% | 16.90 | 419 | 53.39 | 1325 | 216% | — |
| `rotate_quantize` | 400 | 32.59 | 5.1% | 18.00 | 552 | 18.00 | 552 | 0% | — |
| `rowprepare` | 400 | 11.74 | 1.8% | 12.06 | 1027 | 12.06 | 1027 | 0% | — |
| `rotate_out_i32`（fixup） | 400 | 5.75 | 0.9% | 5.99 | 1042 | 5.99 | 1042 | 0% | — |

bytes モデル: `rotate_quantize` = bf16 入力 + int8 codes + f32 scales、
`rowprepare` = codes の読み書き + scales、`recon` = trellis（`nk*bits/8`）+ dense（`nk`）、
`prefill_super64` = codes + 重み平面 × 行ブロック数（issued）または 1 回（uniq）+ BF16 出力、
`mul24_gemm` と `fixup` は flagged 行を `rows/8`（cap）と仮定した上限値。

読み方:

- `prefill_super64` は issued 825 GB/s で DRAM 上限 636 GB/s を超え、L2/L3 が 7.9 倍吸収している。
  **帯域律速ではなく、200 TOPS（実測ピーク 308 の 65%）の計算律速**である。
- `recon` は 793 GB/s（DRAM 上限超）で、L3 吸収込みの上限に近い。削るには bytes を減らすしかない。
- `rotate_quantize` は 552 GB/s で DRAM 帯域帯（550–636）。① 帯域律速。
- `rowprepare` と `fixup` は 1 TB/s 超で L3 常駐。融合しても pass が減るだけで帯域は減らない。

## 3. linear 境界の空白（kernel 外時間）

カーネル外時間 97.8 ms/run のうち、EXL3 chain の内部に 3 種類の遷移で固定 262 µs が入る。

| 遷移 | 回数/run | gap p50 | 合計/run |
| --- | ---: | ---: | ---: |
| `prefill_super64` → `mul24_gemm`（flagged） | 400 | 261.5 µs | 67.8 ms |
| `gdn_recurrence` → 直後 | 48 | 262.7 µs | 12.3 ms |
| `attention_paged_prefill` → `sigmoid` | 16 | 263.2 µs | 4.2 ms |
| `rotate_quantize` → `rowprepare` | 400 | 4.2 µs | 1.7 ms |
| `rowprepare` → `recon` | 400 | 4.2 µs | 1.7 ms |
| `recon` → `prefill_super64` | 400 | 4.2 µs | 1.7 ms |
| `mul24_gemm` → `fixup` | 400 | 4.3 µs | 1.7 ms |

上 3 行の合計は 84.3 ms（8.3%）。464 回のうち約 6 割が 262 µs、残りは 4.2 µs で、
p50 だけを見ると全回発生に見えるため注意する。残る 6.7 ms は layer 間など
EXL3 chain 外の遷移である。

- 262 µs が乗る 3 遷移に共通するのは「LDS を 14 KB 以上使う最大級の kernel の直後」である
  （super64 34816 B、gdn_recurrence 14336 B、attention 31744 B）。
  `0 → 34816` や `512 → 0` の遷移は 4.2 µs なので、LDS の増加や小さい解放では発生しない。
- ホスト側ではない。`hipLaunchKernel` の p50 は 3.0 µs、kernel 開始は launch 完了より
  中央値 245 ms 先行しており、queue は常に埋まっている。
- 発生機序は未特定である（仮説と実験は
  [20261125/prefill_gap_262us.md](../research/20261125/prefill_gap_262us.md)）。

## 4. 次のレバー（優先順）

1. **`prefill_super64` の occupancy**: VGPR 232 と LDS 34816 の両方が 3 blocks/CU を拘束している。
   4 blocks/CU には VGPR ≤ 184 かつ LDS ≤ 32768 が必要（+33% waves）。share 54.6% で最大。
2. **LDS/scratch を削って 262 µs 空白を消す**: 3 遷移の合計 84.3 ms/run（8.3%）。
   `prefill_super64` は scratch 544 B/thread も持つ。1 の変更と同時に検証できる。
3. **`gdn_recurrence` の VGPR**: 248 → 240 で slots 264 → 240、waves/SIMD 5 → 6（+20%）。
   share 16.5% で 2 番目に大きい。
4. **`rotate_quantize` の bytes**: 552 GB/s の ① 帯域律速。producer フック
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
