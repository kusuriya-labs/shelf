# Flash-Next EXL3 EP=2 の ISA・実効帯域サーベイ

`docs/developer/kernel_optimization_and_fusion.md` の手順 ① を、pair 融合後の build で
やり直した記録である。対象は `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`
を EP=2 / TP=1（`--device 0,1`）で動かしたときの prefill と decode である。

- measurement revision: pair 融合コミット `1903ab66` の working tree
- GPU: Radeon AI PRO R9700 × 2。1 GPU 1 process、`flock /tmp/r9700.lock`。sclk は未固定。
- trace: rocprofv3 1.4.1 `--kernel-trace --output-format csv`。trace は wall を膨らませる
  ため、時間配分と資源値の比較にのみ使う。
- ISA: 同 build の `.o` から `clang-offload-bundler` で gfx1201 を取り出し、
  `llvm-objdump --mcpu=gfx1201` の静的命令を数えた。`s_barrier` は静的出現数である。
- rocprofv3 CSV の `Grid_Size_X` は **grid × workgroup の総 thread 数**である
  （decode attention は 12 block × 256 thread = 3072、prefill は 128 × 12 × 256 = 393216）。
  grid の block 数は 256 で割って読む。
- 実効 GB/s は「rank あたり・1 回の処理で論理的に読む重み / KV / 状態の byte 数 ÷
  kernel duration」である。キャッシュ再利用、tile padding、再読は含まない。

## 1. 実効帯域（prefill 2048 token、rank あたり）

2048 token を 512 token の 4 chunk で処理した。router はほぼ一様なので、各 chunk で
local 256 expert の全重みが読まれると仮定する。expert 1 本の重みは gate/up が
1.65 MB、down が 0.83 MB（trellis 0.819 MB + suh/svh）。

| kernel | GPU ms | share | 論理 bytes | 実効 GB/s | 律速 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemm_mul24_grouped<2,1,3,true>`（EXL3 3bit gate/up） | 248.0 | 32.4 % | 81.2 GB | **327** | dequant ALU（②） |
| `gemm_mul24_grouped<2,2,3,false>`（EXL3 3bit down） | 97.8 | 12.8 % | 40.6 GB | **415** | dequant ALU（②） |
| `gdn_recurrence_wmma_lossy` | 45.1 | 5.9 % | 1.36 GB | 30 | 逐次 scan の compute |
| `hipBLASLt BBS`（hyper down/up） | 50.9 | 6.7 % | 5.03 GB | **99**（50.6 TFLOP/s） | skinny GEMM の tile |
| `attention_paged_prefill` | 19.8 | 2.6 % | 63 MB | 3.2（compute） | WMMA の compute |
| `hipBLASLt I8II`（EXL3 dense） | 36.3 | 4.7 % | 3.17 GB | **87**（89 TOPS） | INT8 GEMM の効率 |
| `exl3_mul24_rowfinish` | 23.2 | 3.0 % | 小 | — | 命令 / launch |
| `exl3_mul24_rowstat` | 18.8 | 2.5 % | 小 | — | 命令 / launch |
| `moe_slot_scan` | 17.4 | 2.3 % | index のみ | — | latency |
| `hyper_mix` | 16.7 | 2.2 % | 小 | — | launch |
| `moe_combine` | 14.9 | 2.0 % | 小 | — | launch / barrier |
| `exl3_mul24_rotate_out_i32_f4` | 14.7 | 1.9 % | 小 | — | 命令 |
| `rotate_quantize_f4` | 12.7 | 1.7 % | 小 | — | 命令 |
| `shelf_swiglu_bf16` | 11.5 | 1.5 % | 小 | — | launch |
| `rmsnorm` | 9.5 | 1.2 % | 小 | — | launch |
| `hyper_apply` | 8.8 | 1.1 % | 小 | — | launch |

## 2. 実効帯域（decode 1 token、context 512、rank あたり）

| kernel | ms/token | share | 論理 bytes | 実効 GB/s | 律速 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemm_mul24_grouped<2,1,3,true>` | 6.645 | 25.2 % | 396 MB | **60** | dequant ALU（②） |
| `shelf_attention_paged_bf16` | 3.106 | 11.8 % | 6.3 MB | **2** | mask 付き 1 block/(row,head) の逐次 scan（③） |
| `gemm_bf16_exact_rows_pair<1>` | 1.731 | 6.6 % | 771 MB | **445** | rows=1 の帯域 |
| `hyper_mix` | 1.612 | 6.1 % | 4.5 MB | **3** | launch |
| `exl3_mul24_gemm<5,1>`（5bit 線形） | 1.406 | 5.3 % | 未同定 | — | 命令 |
| `shelf_gemm_bf16_exact_rows_1`（hyper up） | 1.394 | 5.3 % | 636 MB | **456** | rows=1 の帯域 |
| `group_gemm<4,1,true>` | 1.271 | 4.8 % | 未同定 | — | 命令 |
| `moe_slot_scan` | 1.112 | 4.2 % | index のみ | — | latency |
| `gemm_mul24_grouped<2,2,3,false>` | 1.068 | 4.0 % | 198 MB | **186** | dequant ALU（②） |
| `exl3_mul24_gemm<4,1,true,true>` | 0.773 | 2.9 % | 未同定 | — | 命令 |
| `gdn_recurrence_wmma_lossy` | 0.681 | 2.6 % | 113 MB | **166** | 逐次 scan |
| `moe_router` | 0.649 | 2.5 % | 96 KB | 0.15 | latency |
| `hyper_apply` | 0.505 | 1.9 % | 2.2 MB | **4** | launch |
| `rmsnorm` | 0.446 | 1.7 % | 2 MB | **5** | launch |
| `rotate_out` / `rotate_quantize` | 0.688 | 2.6 % | 小 | — | launch |
| `moe_combine` | 0.238 | 0.9 % | 2.7 MB | 11 | barrier |

- decode の wall は無計測で 29.7 ms/token、trace 下の kernel 合計は 26.4 ms/token。
- `shelf_attention_paged_bf16` は trace 上 12 launch/token、1 launch 258.8 µs、
  grid は CSV で 3072 = 12 block × 256 thread（1 block / (row, q_head)）。
  実 batch は 1 row / visible 512 で、同じ GPU の mask 無し standalone は 32 µs
  （`shelf-bench paged-attention --rows 1 --visible 512 --q-heads 12 --kv-heads 1`）。
  **ここが decode で最も改善余地が大きい。** 1 block が可視域を逐次走るため、
  mask 付き経路の token ごとの可視判定と依存鎖がそのまま latency になる。
  §6 で mask を外して 33.4 µs まで短縮した。

## 3. ISA と資源（静的）

| kernel | instr | v_alu | gmem | LDS op | barrier | VGPR | LDS bytes | wg | grid | occupancy blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grouped<2,1,3,true>` | 7659 | 3860 | 95 | 463 | 16 | 112 | 17 408 | 256 | 1280×G | 6（register） |
| `grouped<2,2,3,false>` | 4904 | 2480 | 83 | 301 | 12 | 168 | 21 504 | 256 | 2560×G | 4（register） |
| `gdn_recurrence_lossy` | 5779 | 2318 | 308 | 404 | 18 | 224 | 35 840 | 256 | 12 288 | 3（register/LDS） |
| `attention_paged_prefill<0>` | 2549 | 901 | 89 | 110 | 14 | 176 | 31 744 | 256 | 4096 | 4 |
| `attention_paged_bf16`（decode） | 1309 | 420 | 50 | 49 | 8 | 64 | 12 800 | 256 | rows×heads×256 | 8 |
| `bf16 pair<1>` | 761 | 441 | 28 | 10 | 0 | 48 | 0 | 256 | 16 640 | 8 |
| `exl3_mul24_gemm<5,1>` | 1257 | 751 | 72 | 0 | 0 | 120 | 0 | 128 | 248 320 | 12 |
| `bf16 exact rows_1` | 373 | 221 | 14 | 5 | 0 | 48 | 0 | 256 | 10 240 | 8 |
| `hyper_mix` | 132 | 50 | 3 | 0 | 0 | 16 | 0 | 256 | 256 | 8 |
| `hyper_apply` | 107 | 38 | 4 | 0 | 0 | 16 | 0 | 256 | 256 | 8 |
| `moe_combine` | 419 | 91 | 11 | 26 | 4 | 8 | 512 | 128 | 128 | 8 |
| `moe_slot_scan` | 319 | 50 | 23 | 2 | 4 | 8 | 2048 | 256 | 256 | 8 |
| `moe_router` | 776 | 279 | 10 | 47 | 8 | 40 | 512 | 64 | 64 | 32 |
| `swiglu` | 922 | 455 | 15 | 0 | 0 | 24 | 0 | 256 | 256 | 8 |
| `exl3_mul24_rowfinish` | 247 | 87 | 15 | 0 | 0 | 24 | 0 | 256 | 126 976 | 8 |
| `exl3_mul24_rowstat` | 165 | 67 | 4 | 15 | 2 | 24 | 512 | 256 | 126 976 | 8 |
| `rotate_quantize_f4` | 381 | 180 | 4 | 25 | 0 | 40 | 0 | 256 | 126 976 | 8 |
| `tp_sum_hyper_apply` | 984 | 389 | 11 | 0 | 0 | 32 | 0 | 256 | 40 960 | 8 |

- occupancy は `docs/developer/kernel_optimization_and_fusion.md` §1 の
  `vgpr_r = ceil(VGPR/8)×8`、`budget_waves = min(4×floor(47104/(32×vgpr_r)), 64)`、
  `blocks = min(budget_waves/(wg/32), 131072/LDS, 64/(wg/32))` で算出した。
- EXL3 grouped は 16 / 12 本の `s_barrier` を持ち、register/LDS の双方で 4〜6 block に
  制限される。dequant の v_alu が支配的で、帯域より命令が先に詰まる。

## 4. 改善余地の大きい kernel（優先順）

`share × (1 − 現状 / 到達可能値)` の大きい順。到達可能値は「帯域律速なら 500 GB/s、
compute 律速なら実測 TFLOP/s の 2 倍」を仮置きした粗い上限である。

| 優先 | kernel | 現状 | 律速 | 改善の方向 | 見込み |
| ---: | --- | --- | --- | --- | --- |
| 1 | decode `shelf_attention_paged_bf16` | 2 GB/s、258.8 µs/layer | ③ mask 付き逐次 scan | within-qsa-budget で mask を外す（§6 で実施） | 258.8 → 33.4 µs、E2E +6.3 % |
| 2 | decode EXL3 grouped gate/up | 60 GB/s | ② dequant ALU（3860 v_alu、16 barrier） | MTP/batch で rows を増やす、dequant 命令削減、group fuse | share 25 %、10 % 改善で −0.66 ms |
| 3 | prefill EXL3 grouped gate/up | 327 GB/s | ② dequant ALU | 同上。500 GB/s まで 1.5 倍 | share 32 %、+30 % で −57 ms（prefill の 7 %） |
| 4 | prefill hyper（hipBLASLt BF16） | 99 GB/s / 50.6 TFLOP/s（26 %） | skinny GEMM の tile | N=320 / K=320 用の tile 選択、または専用 kernel | share 6.7 %、2 倍で −25 ms |
| 5 | decode `hyper_mix` + `hyper_apply` + `rmsnorm` | 3〜5 GB/s | ④ launch | down/inject/up の epilogue 融合、norm の融合 | 約 −3.6 ms（12 %） |
| 6 | decode MoE bookkeeping | 0.15〜11 GB/s | ④ latency | router + slot scan + scatter + combine の統合 | share 6.8 %、half で −1.3 ms |
| 7 | `gdn_recurrence_wmma_lossy` | decode 166 GB/s、prefill 30 GB/s | 逐次 scan（VGPR 224） | register 圧低減、行分割、conv 融合 | prefill share 5.9 % |
| 8 | EXL3 rowfinish / rowstat / rotate | 小 bytes | ② 命令 + ④ launch | prefill の dense 経路で融合 | prefill 合計 9 % |
| 9 | `bf16 exact rows` pair / hyper up | 445 / 456 GB/s | rows=1 の帯域 | 500 GB/s まで残り 10 %。MTP で rows を増やす方が有効 | 小 |

- 1 の根拠: trace の decode attention は 12 launch/token × 258.8 µs = 3.1 ms/token、
  grid 3072。同じ GPU の standalone kernel は rows=1 / visible 512 で 32 µs
  （12 layer で 0.38 ms）。mask=1 でも visible block が select budget 内なら結果は
  変わらないため、mask を外せる可能性が高い。grid の 256 倍次元の出所、数値一致、
  A/B の順で確認する。
- 2 / 3 は帯域ではなく命令が先に詰まる。EXL3 grouped の ISA は 7659 命令中 3860 が
  vector ALU で、16 本の `s_barrier` を持つ。帯域 636 GB/s に届かせるには ALU を
  減らす必要があり、単純な memory 改善では伸びない。
- 5 / 6 / 8 は bytes が小さいのに時間を使っている launch / latency 型である。
  1 launch 2 µs の固定費と、20 µs 級の小さな kernel が積み上がっている。
  融合の効果は帯域ではなく dispatch 数で見積もる。

## 5. 未測定

- MTP fused verify（rows=4）の kernel 別内訳と実効帯域。
- `exl3_mul24_gemm<5,1>` と `group_gemm<4,1,true>` の対象行列（5-bit 線形と projection
  group）の同定。weight slot の byte 数から逆算していない。
- 500 GB/s を目標にしたときの到達可能性の実測（EXL3 grouped の microbench、
  attention の mask 無し A/B）。

## 6. 適用: within-budget の dense mask 除去

`plan_paged_attention` に `mask_all_visible` を追加した。QSA block mask が付いていて、
可視 block 数が select budget（`qsa_top_k`）以下なら全 block が選ばれるので、dense
attention へ渡す mask を外す。planner の key にも 1 bit 足した。A/B 用の戻しは
`SHELF_DISABLE_QSA_DENSE_NO_MASK=1`。

- decode 1 launch の時間: **258.8 → 33.4 µs**（trace、ctx 512、EP=2）。
  standalone の mask 無しと同水準になった。
- E2E (`shelf-bench tg --mode greedy --context 512 --tokens 64 --warmup 2`) の
  interleaved A/B 3 ペア:
  - 戻し: 32.32 / 32.52 / 32.53 tok/s
  - 適用: 34.82 / 34.56 / 34.55 tok/s
  - 中央値 **+6.3 %**（30.9 → 29.1 ms/token）
- 生成 24 token は on / off で一致した。可視長が budget を超える場合は従来どおり
  mask を保持する（`test_paged_attention_plan_splits` に境界の check を追加）。

## 7. 適用: hyper up GEMM + gate の融合

`hyper_mix_kernel` は 1 CTA / row で動き（decode rows=1 では GPU 全体で 1 CTA）、
up GEMM の出力 10240 を一度 materialize してから読む。この 2 launch を
`hyper_up_mix_kernel` 1 launch に融合した。hidden index ごとに hc 本の dot を
1 warp で計算し、各 dot を BF16 に丸めてから sigmoid gate と 1/hc 平均を適用する
（丸めは unfused の up GEMM が store する値と同じ）。

- 配置: `src/shelf/kernels/fused/hyper_connection.hip` / `include/.../hyper_connection.h`
- matcher: `try_launch_linear_hyper_up_mix`（LINEAR_BF16 の直後が HYPER_MIX で、
  入力が一致し、両方が ExactRows のときだけ融合）。`SHELF_DISABLE_HYPER_UP_MIX=1`
  で戻せる。
- 正しさ: logits 17,879,040 値の **max abs diff 0.0**、生成 16 token も一致。
- 性能: kernel 合計は up 1.394 + mix 1.612 = 3.006 ms/token から
  up+mix 1.926 ms/token へ **−1.08 ms/token**。E2E interleaved A/B 3 ペアは
  off 34.95 / 34.75 / 34.61、on 34.91 / 35.10 / 35.16 tok/s（中央値 **+1.0 %**）。
  kernel の削減が wall にそのまま出ないのは、greedy の per-token sync の遊休に
  一部が吸収されるためである。
- hyper_mix / hyper_apply の hidden tile 分割（256 / 512 / 1024 / 1 CTA）は
  E2E で差がなかった（35.23〜35.40 tok/s）。1 launch あたり 45〜60 KB の
  latency 律速で、CTA 数を増やしても短くならない。tile 変更は採用しなかった。

## 8. 適用: MoE slot scan の並列化

`moe_slot_scan_kernel` は expert ごとの row / group の prefix と group 配列の構築を
`tid == 0` の 1 thread で行っていた。expert 数（512）と group 数の prefix を
256 thread の 2 要素/thread の block scan（row / 全 group / hit group / miss group の
4 本を同じ barrier で進める）に変え、group の書き込みも expert ごとに並列化した。
hit / miss の `expert_group_begin` は slot ごとの最小 index を `atomicMin` で求める。

- decode の `moe_slot_scan` 1 launch は **23.6 → 3.79 µs**（trace、ctx 512）。
  48 launch/token で約 0.95 ms/token の削減に相当する。
- E2E は同じ binary を逐次入れ替えた A/B で 35.17 → 35.38 tok/s（+0.6 %）。
  kernel の削減が wall にそのまま出ないのは、greedy の per-token sync の遊休に
  吸収されるためである。
- 正しさ: `test_moe_router` に hit / miss を含む split scan の検証を追加
  （group の被覆、slot 対応、expert 順、group row の初期化）。required 231/231。
  EP=2 の 2048 token prefill で offload なし / host 128 の生成 8 token が一致した。

## 9. レビュー指摘の評価

- **HyperConnection の並列性**: `hyper_mix` は rows=1 で 1 CTA だったが、up GEMM と
  の融合（§7）で hidden index ごとに 1 warp、320 CTA の kernel になった。
  `hyper_mix` / `hyper_apply` 単体の hidden tile 分割（256 / 512 / 1024 / 1 CTA）は
  E2E で差がなく（35.23〜35.40 tok/s）、1 launch 45〜60 KB の latency 律速だった。
- **EXL3 grouped の activation 回転**: §11 で producer + pre-rotated GEMM を実装し、
  prefill で採用した（microbench −17.4 %、E2E −9.1 %）。group fuse 3 / 4 は既存研究で
  VGPR 増加のため棄却済みなので、tile 拡大では追わない。
- **MoE slot scan**: §8 で並列化した（compact plan 専用経路は不要だった）。

## 10. 参考: prefill chunk と warm 帯域

`--max-num-batched-tokens` = prefill chunk を 2048 / 4096 / 8192 にして、8106 token の
2 本目（warm）prefill を測った（EP=2、bf16 KV、serve）。

| chunk | prefill ms | tok/s |
| ---: | ---: | ---: |
| 2048 | 2563.6 | 3162 |
| 4096 | 2472.2 | 3279 |
| 8192 | 2412.8 | 3360 |

chunk を 2048 → 8192 にすると **−5.9 %**。expert の重み traffic は chunk 数に比例する
ため、chunk を大きくすると同じ token 数でも expert を読む回数が減る。ただし
grouped GEMM は dequant ALU 律速で、traffic 半減ほどの短縮にはならない。
`tests/server/support.py` の `ComputeHarness` に `prefill_chunk` を追加した。

pre-rotation と row 前処理の融合後（7002 token、EP=2、bf16 KV、warm）に取り直すと、
budget と chunk を **両方** 8192 にしたときだけ短縮した。

| budget / chunk | prefill ms |
| --- | ---: |
| 4096 / 4096 | 1915.4 / 1924.6 |
| 8192 / 8192 | **1850.4 / 1866.2（−3.2 %）** |
| 8192 / 4096 | 1931.6 |
| 4096 / 8192 | 1936.1 |

片方だけ 8192 にしても効果がないため、`shelf-server` の
`--max-num-batched-tokens` と `--prefill-chunk` の既定を両方 8192 にした。
`run.sh` のように `--max-num-batched-tokens 4096` を明示している場合は、
その指定を外さないと 8192 chunk にならない。

## 11. 適用: EXL3 grouped の activation 回転を 1 回にする

grouped kernel は N tile の block ごとに activation の FWHT 回転と INT8 量子化をやり直していた。
gate は N=640 なので 5 block、down は N=2560 なので 10 block が同じ group row を回転する。
回転を producer で 1 回だけ行い、grouped GEMM は INT8 code と chunk scale を読む形にした。

- `launch_exl3_grouped_activate` が group row ごとに 1 warp で回転・量子化し、
  `codes[group_row][k]` と `scales[group_row][k / 128]` を書く。group row は
  `group_row_base` を通した full layout の row なので、hit / miss の split launch でも
  同じ plane を共有できる。
- grouped kernel は `pre_rotated_codes` が非 null のとき、tile block の回転を飛ばして
  code を shared に積むだけになる。`group_base` は `resolve_group_plan` から受け取る。
- code plane は grouped input value ごとの workspace range に置く。gate と up は同じ
  input を共有するので range も共有し、down は別 range になる。
  サイズは `rows_cap * k` bytes + `rows_cap * (k / 128) * 4` bytes。
  4096 token chunk・local expert 256 では gate/up が約 118 MB、down が約 30 MB。
- decode は producer の launch が増える分だけ損なので、routed rows が 128 未満では
  従来の fused 回転を使う。`SHELF_DISABLE_EXL3_GROUPED_PREROTATE=1` で全体を戻せる。

### microbench

`bench_exl3_mul24_grouped`（gate k=2560/n=640、down k=640/n=2560、iters 20、fuse=2）。
fused の gate+down と、producer + pre-rotated gate/down の合計を比べた。

| tokens | grouped rows | fused ms | producer + pre ms | 差 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.096 | 0.110 | +14.9 % |
| 8 | 80 | 0.354 | 0.327 | −7.7 % |
| 64 | 640 | 2.035 | 1.830 | −10.1 % |
| 1024 | 10240 | 4.276 | 3.655 | −14.5 % |
| 2048 | 20480 | 7.125 | 5.884 | −17.4 % |

全サイズで fused と pre-rotated の出力は bit 一致した。

### E2E

7002 token の warm prefill（EP=2、bf16 KV、serve、drafts 0、同一 binary を env で切替）。

| host experts | 無効 ms | 有効 ms | 差 |
| ---: | ---: | ---: | ---: |
| 0 | 2231.7 / 2238.5 | 2024.2 / 2037.1 | −9.1 % |
| 96 | 2611.8 | 2422.0 | −7.3 % |

- decode は 3162 token prefill + 58 token 生成で 37.3 → 37.2 tok/s（差なし）。
  MTP drafts 3 では 53.8 → 54.1 tok/s（差なし）。
- 生成 16 token は host experts 0 / 96 の両方で有効・無効が完全一致した。

### テスト

- `test_exl3_mul24_grouped` の全 case に pre-rotated plane との bit 一致比較を追加した。
- hit / miss を混ぜた split plan（`group_row_base` が非 identity）の比較も追加した。
  この test で producer の capacity を group 数として渡していた bug を見つけ、
  行数 (`activation_max_rows`) を渡すよう修正した。
- required 231/231 PASS。

## 12. 再サーベイ（pre-rotation 適用後）

§11 の build（`98b5af54`）で、2048 token を 512 token の 4 chunk で prefill した
trace を取り直した（EP=2、bf16 KV、`shelf-bench tg --mode forward --context 2048
--tokens 1 --prefill-chunk 512 --warmup 0 --device 2,3`）。数値は 2 rank 合計の
GPU 時間で、share は prefill 区間の合計に対する割合である。帯域は rank あたり。

| kernel | ms | share | 帯域 / 律速 |
| --- | ---: | ---: | --- |
| `gemm_mul24_grouped<2,1,3,true>`（gate + up） | 288.4 | 24.8 % | 551 GB/s（dequant ALU + DRAM） |
| `gemm_mul24_grouped<2,2,3,false>`（down） | 110.3 | 9.5 % | 725 GB/s（dequant ALU） |
| `gdn_recurrence_wmma_lossy` | 85.1 | 7.3 % | 逐次 scan、VGPR 224 / LDS 35 KB |
| `exl3_mul24_rowfinish` | 53.0 | 4.6 % | launch / 命令 |
| `exl3_mul24_rowstat` | 46.8 | 4.0 % | launch / 命令 |
| hipBLASLt BBS（hyper down/up） | 38.3 + 36.5 | 6.4 % | skinny GEMM |
| `exl3_mul24_rotate_out_i32_f4` | 38.3 | 3.3 % | 命令 |
| `exl3_grouped_activate`（producer） | 29.7 | 2.6 % | 帯域 690 GB/s |
| `hyper_mix` | 29.1 | 2.5 % | launch |
| `exl3_mul24_rotate_quantize_f4` | 27.2 | 2.3 % | 命令 |
| `moe_combine` | 26.9 | 2.3 % | barrier / launch |
| `attention_paged_prefill` | 26.0 | 2.2 % | WMMA compute |
| `exl3_mul24_recon` / `rowslot` | 20.5 + 12.2 | 2.8 % | 命令 |
| `shelf_swiglu_bf16` | 21.2 | 1.8 % | launch |

- grouped は回転を外したことで **551 / 725 GB/s** まで上がった。gate + up は
  DRAM 640 GB/s に対して ~86 % なので、同じ chunk のまま kernel 内部を詰める余地は
  小さい。次に効くのは rows を増やして expert weight の再読を減らす方向（chunk 8192）。
- いま残っている最大の cluster は **dense EXL3 の前後処理**
  （rowfinish + rowstat + rotate_out + rotate_quantize = 14.2 %）。いずれも 1 launch
  15〜21 µs で、GEMM 本体ではなく prologue / epilogue の launch と中間書き戻しが
  主体である。GEMM への融合、または rotate_quantize を producer に寄せる余地がある。
- 次点は **GDN recurrence 7.3 %**。VGPR 224 / LDS 35 KB で 3 block/CU に制限されて
  おり、scan の並列性を上げる余地がある。
- **hyper** は up + mix を multi-row prefill でも融合できれば `hyper_mix` 2.5 % と
  up GEMM の一部を削れる。LT は hipBLASLt が既に最速で、置換対象ではない。
- producer 2.6 % は down 側を SwiGLU に融合すれば 1 read と 1 launch を消せる。
- decode（同じ trace の最終 1 token、2 rank 合計）は gate 29.5 %、`hyper_up_mix`
  8.4 %、BF16 pair 7.4 %、dense EXL3 6.1 %、`group_gemm` 5.6 %、down 4.9 % で、
  §2 の順位から大きな変化はない。decode は pre-rotation の対象外（128 row 未満）。

次の実装候補は、share と launch 単価から **dense EXL3 の rowfinish / rowstat /
rotate_out の GEMM 融合**とする。採用判定は §11 と同じく、bit 一致または許容される
near-tie flip、interleaved A/B、required suite で行う。

## 13. 既定採用: dense EXL3 の row 前処理を 1 launch に融合

dense A 経路の `rowstat` + `rowslot` + `rowfinish` を、atomic で slot を詰める
`exl3_mul24_rowprepare_kernel` 1 launch に融合した。実装は
`src/shelf/kernels/fused/exl3/row_prepare.hip`、宣言は
`include/shelf/kernels/fused/exl3/row_prepare.h` に置く。既定は融合経路で、`SHELF_DISABLE_EXL3_ROW_FUSE=1` または
`SHELF_EXL3_ROW_FUSE=0` のときだけ従来の 3 kernel 経路に戻る。env 判定は
ホスト側で 1 回だけ行い、GPU kernel には分岐を入れない。

### specialized 測定

2048 token / 4 chunk / EP=2 の trace。既定側は旧 3 kernel だけ、opt-in 側は
`rowprepare` 1 kernel だけが走る。compact slot の counter は直前の
`rotate_quantize` が 1 store で 0 にするので、fill も別 launch にならない。

- row 前処理の GPU 時間: `rowstat` 46.817 + `rowslot` 12.214 + `rowfinish`
  52.452 = **111.483 ms** → `rowprepare` **47.147 ms**（**−57.7 %**）。
- prefill window の GPU 合計: **1159.15 → 1104.31 ms**（**−4.7 %**）。
- dispatch 数: 2484 + 2485 + 2484 = 7455 → **2486（−66.7 %）**。
- static instruction: 170 + 200 + 267 = **637** → **510（−20.0 %）**。
  `s_wait*` は 106 → 75（−29.2 %）。copy / requantize を 16 byte 単位にしたため
  unpack の `v_alu` は 228 → 243 に増えるが、動的命令数と GPU 時間は減る。
- 資源値: `rowprepare` は VGPR 32、SGPR 128、LDS 512 B、256 thread。
  32 VGPR × 1024 thread = 32768 < 65536、LDS 512 B なので **32 wave/CU の
  上限 Occupancy**（4 block/CU）。スレッド数やレジスタを増やしても Occupancy は
  上がらない。旧 `rowslot` だけが VGPR 40 / LDS 4096 B / 1024 thread だった。

### wall

| 条件 | 融合なし ms | 融合（既定）ms |
| --- | ---: | ---: |
| 7002 token、host experts 0 | 2035.6 / 2047.9 | **1987.7 / 2016.3** |
| 7002 token、host experts 96 | 2459.1 / 2464.0 | 2459.3 / 2432.6 |

host experts 0 では **約 2.0 %** 速い。host 96 では同等〜−1.3 % で悪化しない。
生成 16 token は両経路で一致したため、既定を融合経路にする。

## 15. GDN recurrence の occupancy 実験（棄却）と rows の再確認

### GDN 2 block/CU

GDN の state は `128 × 64 × 4 = 32 KB` の LDS を 1 block で使い、全体は
35 KB である。2 block/CU には state だけで 64 KB 必要なので、単純な LDS 削減では
届かない。それでも `kSp` を 68 → 64、lossy kernel の VGPR 上限を
`amdgpu_num_vgpr(128)` にして試した（2048 token prefill trace）。

| 項目 | 既定 | 実験 |
| --- | ---: | ---: |
| LDS | 35,840 B | 33,792 B |
| VGPR | 224 | **224（上限は実効せず）** |
| scratch | 0 | 0 |
| GDN kernel | 85.1 ms | **88.0 ms（+3.4%）** |

LDS はまだ 32 KB を超え、VGPR も 224 のままで 2 block/CU にならず、kSp の変更だけが
遅くなった。revert した。次の GDN レバーは value group を 64 → 32 に割って
state を 16 KB にし、block 数と 2 block/CU を同時に取る設計変更になる。

### MTP drafts

3162 token prefill + 64 token 生成、greedy、EP=2。

| drafts | decode tok/s |
| ---: | ---: |
| 2 | **61.7** |
| 3 | 54.1 |
| 4 | 51.9 |

run.sh の drafts 2 が最速で、増やすと遅い。

### concurrent batch

3162 token prompt、drafts 2、`max_num_batched_tokens` 8192。

| conc | aggregate tok/s | per-stream |
| ---: | ---: | ---: |
| 1 | 23.41 | 23.41 |
| 2 | 33.04 | 16.52 |
| 4 | **39.59** | 9.90 |

aggregate は conc 4 が最良で、server の既定 `--max-concurrent-requests 4` のままにする。
required 231/231 PASS。

## 16. 現行 build の再測定（GDN v32 試作 revert 後）

GDN v32 の試作を取り消した状態（row 前処理の既定融合と chunk 8192 を含む）で、
prefill / decode / trace / batch を取り直した。

### prefill（7002 token warm、EP=2、bf16 KV）

| 条件 | prefill ms | tok/s |
| --- | ---: | ---: |
| 8192 / 8192、host experts 0 | 1851.2 / 1859.9 | 3782.5 / 3764.8 |
| 8192 / 8192、host experts 96 | 2020.6 | 3465.3 |
| 4096 / 4096、host experts 0 | 1921.8 | 3643.4 |
| 8192 / 8192、row 融合 off | 1886.3 | 3712.1 |

chunk 8192 は 4096 より **−3.4 %**、row 融合は off より **−1.6 %**（1 回比較）。

### decode（3162 token prefill + 64 token 生成）

| MTP drafts | decode tok/s | prefill ms |
| ---: | ---: | ---: |
| 2 | **61.6 / 61.6** | 900.6 / 888.8 |
| 0 | 37.5 | 836.8 |

### trace（2048 token、4 × 512 chunk、2 rank 合計）

| kernel | ms | share | 資源 |
| --- | ---: | ---: | --- |
| grouped gate+up | 293.1 | 26.3 % | VGPR 128 / LDS 17,408 |
| grouped down | 112.3 | 10.1 % | VGPR 184 / LDS 21,504 |
| GDN recurrence lossy | 87.3 | 7.8 % | VGPR 224 / LDS 35,840 / scratch 0 |
| rowprepare（既定融合） | 47.0 | 4.2 % | VGPR 32 / LDS 512 |
| rotate_out_i32_f4 | 38.8 | 3.5 % | VGPR 32 |
| hipBLASLt BBS | 39.1 + 36.8 | 6.8 % | — |
| grouped producer | 30.2 | 2.7 % | VGPR 32 |
| hyper_mix | 29.4 | 2.6 % | VGPR 16 |
| rotate_quantize_f4 | 28.2 | 2.5 % | VGPR 40 |
| moe_combine | 27.4 | 2.5 % | VGPR 8 |

prefill window の GPU 合計は **1113.3 ms**。GDN は依然 7.8 % で、
`VGPR 224 / LDS 35,840 / scratch 0` のまま。rowprepare は `VGPR 32 / LDS 512`。

### concurrent batch

3162 token prompt、drafts 2、budget 8192、conc 4: **39.59 aggregate tok/s**
（per-stream 9.90）。

required **231/231 PASS**。この状態を次の変更の基準にする。

## 17. VGPR granule を使った grouped gate の occupancy 改善

compiler の gfx1201 VGPR capacity（1536 VGPR/SIMD、granule 24、16 wave/SIMD）を
grouped gate に適用した。gate `<2,1,3,true>` は VGPR 128 で granule 144 に丸められ、
`floor(1536 / 144) = 10 wave/SIMD` だった。LDS は 17,408 B で 3 block/CU が可能なので、
VPR を 120 以下にすれば `floor(1536 / 120) = 12 wave/SIMD` になる。

- `__launch_bounds__(256, 3)` だけでは VGPR は 128 のままだった。compiler の
  min-blocks は VGPR を 8 単位で見ており、実機 granule 24 の境界を跨げない。
- body を `__device__` 関数にし、gate wrapper を `amdgpu_num_vgpr(120)`、
  down wrapper を `amdgpu_num_vgpr(184)` に分けた。down は元の 184 を維持。
- 2048 token prefill trace:

| kernel | 変更前 | 変更後 |
| --- | ---: | ---: |
| grouped gate `<2,1,3>` | 293.1 ms, VGPR 128, scratch 0 | **265.6 ms, VGPR 120, scratch 0** |
| grouped down `<2,2,3>` | 112.3 ms, VGPR 184, scratch 0 | 110.8 ms, VGPR 184, scratch 0 |
| GPU 合計 | 1113.3 ms | **1077.6 ms（−3.2 %）** |

- 7002 token prefill は 1851.2 / 1859.9 → **1835.6 / 1846.5 ms**（−0.8 %）。
- decode MTP2 は 61.6 tok/s を維持。生成 16 token は一致。required 231/231。
- down は VGPR 120 にすると scratch 192 B が出て 112.3 → 122.4 ms に悪化したため、
  184 のまま 2 block/CU を維持する。

同じモデルで、rowprepare（VGPR 32）や moe_combine（VGPR 8）は 96 VGPR まで
wave を落とさずに使える。次はこれらの無料予算を unroll / 再利用に使う。

down の N tile も 1 にして 120 VGPR / 3 block/CU を狙えるか試したが、
1024 / 2048 token の pre-rotated down は **1.837 / 3.003 ms**（producer 込み
1.873 / 3.066 ms）で、既定の n_tiles=2 / VGPR 184 の **1.245 / 2.573 ms**
（1.261 / 2.636 ms）より **47.6 % / 16.7 % 遅い**。down は n_tiles=2 のままにする。

## 18. GDN recurrence の v32 分割（state 16 KB、3 block/CU）

GDN recurrence は `kWv=64`、`S[128][68]` = 34,816 B、LDS 合計 35,840 B で
1 block/CU に固定されていた。v 幅を 32 にすると payload は
`128 × 32 × 4 = 16 KB`、LDS 合計 19,456 B になり、LDS の 64 KB 予算で
3 block/CU が可能になる。

試作は 2 段階:

- 256 thread（8 warp、state を 2 k × 1 v、chunk は warp 0..3）: grid は 2 倍に
  なるが chunk 区間で 4 warp が遊ぶ。GDN 97.3 ms（VGPR 192 / LDS 19,456）、
  64-wide の 87.3 ms より遅い。
- 128 thread（4 warp、state を 2 k × 2 v、chunk は 4 warp 全部）:
  GDN **83.9 ms**（VGPR 224 / LDS 19,456 / scratch 0）。64-wide 比 **−3.9 %**。

128 thread 版は warp あたりの state tile を 2 k × 2 v に戻し、
`kt0 = warp * 2`、`vt0 = 0` とする。chunk の `(mt, nt)` は
`(warp & 1, warp >> 1)` で 32×32 を 4 warp が過不足なく分担する。
LDS は 19,456 B で 3 block/CU、VGPR 224（granule 240）は 6 wave/SIMD なので
拘束は LDS の 3 block。4 block には state を 16,384 B 以下にする必要があり、
bank conflict 回避の padding を入れると届かない。

E2E は neutral:

| 条件 | 64-wide | v32 |
| --- | ---: | ---: |
| prefill 7002 ms（5 回平均） | 1848.8 | 1856.1 |
| decode MTP2 tok/s | 64.1 / 61.5 | 61.6 / 64.1 |

生成 16 token は一致、required 231/231 PASS。`SHELF_GDN_V32=0` で
従来の 64-wide 経路に戻せる。head_v が 32 の倍数なら適用し、
`hv64`（`head_v=64`、vgroups=2）のテストを追加した。

## 19. GDN v32 の LDS を 14 KB に削って 4 block/CU にする

v32（`kSp=36`、`S[128][36]`）は LDS 19,456 B で 3 block/CU だった。state を
64 k 行ずつステージし、`dT` を `W` の三角求解で置き換える patch を A/B した。

- `S[kSstageRows=64][36]` にして、chunk 内の `kk==4` で barrier を置き、
  K 行 [64,128) を warp 2,3 が上書きする。state の正本は `Sacc`（レジスタ）に置く。
- `dT` を廃止し、前進代入の結果を `W[i][tid]` に書いて転置読みする。

| 項目 | 変更前 | 変更後 |
| --- | ---: | ---: |
| LDS | 19,456 B | **14,336 B** |
| VGPR | 224 | 248 |
| blocks/CU（LDS 拘束） | 3 | **4** |
| GDN ms（2048 token trace、2 rank 合計） | 87.06 | **83.31 / 83.50** |
| prefill window GPU ms | 1,142.9 | **1,139.7 / 1,144.1** |

- E2E は neutral: prefill 7002 token が 1,897.3 → 1,893.5 / 1,891.1 ms、
  decode MTP2 が 60.6 / 60.7 → 60.6 / 60.8 tok/s。
- VGPR 248 は granule 264 → 5 wave/SIMD で、4 block/CU（4 warp/SIMD）を
  拘束しない。
- `test_gdn_recurrence` PASS（hv64 含む）、required 231/231 PASS。

## 20. SwiGLU を MoE down producer に融合する

MoE の routed expert は `gate/up grouped GEMM → SwiGLU → down producer → down GEMM`
と進む。SwiGLU の出力が down 以外に消費されない層では、down producer が
`SwiGLU(gate, up)` を直接回転・量子化できる（`launch_exl3_grouped_swiglu_activate`）。

- 判定は **状態を持たない**。SwiGLU 側は「次の grouped down の入力が自分の出力で、
  他に消費者がない」ことを確認して dispatch を 1 つ skip し、down 側は
  program から producer を再発見して同じ判定を使う。
- ホスト `ctx` にマーカーを置く方式は、並行リクエストでマーカーが競合して
  非決定的に落ちたため採用しない（`limit 3` の純粋融合が 4 回中 2 回だけ失敗した）。
- SwiGLU の積は **BF16 に丸めてから** 回転に入れる。保存経路は SwiGLU が BF16 を
  書いて producer が読むため、丸めないと codes/scales が一致しない。dense の
  `producer_rotate_rows` も同じ位置で丸めている。
- down の INT8 codes 面は、allocator が gate/up の面を再利用することがある。
  その場合 producer が読む gate/up を自分で上書きして NaN になるため、
  **codes 面と gate/up 面が重ならないこと**を融合条件に含める。重なる層は
  SwiGLU を書き、通常の producer 経路に戻す。

2048 token prefill trace（2 rank 合計、EP=2）:

| 項目 | 融合 off | 融合 on |
| --- | ---: | ---: |
| `shelf_swiglu_bf16` | 37.92 ms / 1,438 | **22.34 ms / 1,058** |
| `exl3_grouped_activate_kernel<true>` | 0 | 6.77 ms / 380 |
| `exl3_grouped_activate_kernel<false>` | 30.85 ms / 1,151 | 26.51 ms / 771 |
| GPU 合計 | 1,162.2 ms | **1,144.1 ms（−1.5 %）** |

- 380 launch を削減。prefill 7002 token は 1,882.8 → 1,881.9 ms（neutral）、
  decode MTP2 は 60.6 / 63.1 tok/s（neutral）。
- `test_exl3_mul24_grouped` に「SwiGLU 出力を保存して回転」と「gate/up から直接回転」の
  codes / scales ビット一致検査を追加（group rows / token rows / split）。
- 生成 16 token 一致、required 231/231 PASS。`SHELF_DISABLE_SWIGLU_MOE_DOWN=1` で
  融合を無効にできる。

## 21. MOE_COMBINE への SIGMOID 取り込み

shared expert の gate は `SIGMOID（F32 1 列）→ MOE_COMBINE` と進む。SIGMOID の
出力が combine の gate 以外に消費されない層では、combine が生の logits から
sigmoid を計算して launch を 1 本消せる。

- combine の gate 入力が前段 SIGMOID の出力で、他に消費者がないことを確認し、
  SIGMOID dispatch を 1 つ skip する。combine 側は生の logits と row stride を
  受け取り、`1 / (1 + __expf(-g))` をカーネル内で適用する（要素演算カーネルと
  同じ式）。
- SIGMOID は F32 1 列のときだけ対象にする（shared gate の既定形）。

2048 token prefill trace（2 rank 合計、EP=2）:

| 項目 | 融合 off | 融合 on |
| --- | ---: | ---: |
| `shelf_sigmoid_f32` | 1.207 ms / 478 | **0 ms / 0** |
| `moe_combine_kernel` | 28.629 ms / 478 | 28.991 ms / 478 |
| GPU 合計 | 1,141.4 ms | 1,142.0 ms |

- 478 launch を削減。E2E は prefill 1,879.6 / 1,882.9 → **1,878.2 / 1,881.4 ms**、
  decode MTP2 は 60.5 → **60.9 tok/s**（いずれも neutral）。
- 生成 16 token 一致、required 231/231 PASS。
  `SHELF_DISABLE_MOE_COMBINE_SIGMOID=1` で無効化できる。

## 22. MTP と DFlash2 の Spec / Verify 計測

`SHELF_QWEN35_PERF_TIMING=1` で、単発 decode の draft / verify 内訳を取った。
条件は EP=2、device 0,1、bf16 KV ではなく `octave:c5v5`、prefix cache off、
prompt 150 token、生成 16 token、greedy。

| drafter | rounds | full accept | mean accepted | draft ms | verify ms | round ms | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP drafts 2 | 5 | 1.000 | 2.00 | 21.6 | 184.0 | 205.6 | **72.9** |
| DFlash2 BF16（incoai） | 3 | 0.667 | 4.00 | 5.9 | 86.3 | 132.9 | **112.2** |
| DFlash2 EXL3-INT8 4bpw | 3 | 0.667 | 4.00 | 5.5 | 86.9 | 131.4 | **113.0** |

- DFlash2 は MTP より **約 55 % 高速**。MTP は full accept 1.0 だが 2 draft/round で
  round 数が多く、verify 184 ms が round の 89 % を占める。
- DFlash2 は 4 draft/round 受け入れで round が 3 回に減り、verify は 1 round あたり
  約 28.7 ms。drafter の GPU 時間も 5.5 ms と小さい。
- EXL3-INT8 4bpw drafter は BF16 drafter と同 accept で **draft がわずかに速い**。

### verify の内訳（DFlash2 EXL3、decode window 193 ms、3 round）

| kernel | ms | share | 備考 |
| --- | ---: | ---: | --- |
| grouped gate `<2,1,3>` | 45.8 | 23.7 % | VGPR 120 / LDS 17,408 |
| `hyper_up_mix` | 25.5 | 13.2 % | 580 launch、44 µs/launch |
| GDN `wmma_verify_lossy` | 15.7 | 8.1 % | VGPR 224 / LDS 35,840、1 block/CU |
| grouped down `<2,2,3>` | 11.5 | 5.9 % | VGPR 184 / LDS 21,504 |
| `dflash2_topk_merge` | 10.7 | 5.5 % | 参照実装 1 thread/row |
| dense EXL3 `<5,1>` | 9.5 | 4.9 % | decode 線形 |
| BF16 exact rows pair `<7>` | 10.0 | 5.2 % | drafter の BF16 GEMM |

- `dflash2_topk_merge` は `top_k != 16` のとき参照実装（1 thread/row）が選ばれていた。
  最適化版（hierarchical、256 thread）に切り替えると GPU 時間は消えるが、proposal
  側の別 stream で verify と重なっているため round 時間は **neutral**（113.0 →
  114.4 tok/s、生成列一致）。壁時計のレバーではない。
- `DFLASH2_DEVICE_TOKEN_BRIDGE=0`（CLI は host token ids）。D2H 12.7 ms と
  proposal wait 12.5 ms が round の約 19 % を占める。serve 経路では bridge が
  有効なので、CLI の値はその分だけ不利。

### 改善レバーを 4 分類に割り当てる

`kernel_optimization_and_fusion.md` §2 の分類に、decode window の実測を当てはめる。
実効帯域は「1 verify forward あたりの静的 bytes × 3 round ÷ 実測時間」で出した。

| kernel | share | 実効 | 分類 | レバー |
| --- | ---: | ---: | --- | --- |
| grouped gate `<2,1,3>` | 23.7 % | 39 GB/s | ③ レイテンシ / 並列不足 | GEMM 統合、命令昇格、decode 用 M 分割 |
| `hyper_up_mix` | 13.2 % | 約 1.0 TB/s（L3） | ② L3 / pass | rows 方向の weight 再利用 |
| GDN `wmma_verify_lossy` | 8.1 % | 43 GB/s | ③（LDS 35,840 → 1 block/CU） | LDS ステージで 1 → 3 block/CU（v32 は 4） |
| grouped down `<2,2,3>` | 5.9 % | 下記 | ③ | gate と同じ |
| dense EXL3 `<5,1>` | 4.9 % | — | ③ | rows を増やす（MTP / batch） |
| `dflash2_topk_merge` | 5.5 % | — | ④/overlay | 壁時計には出ない（実測 neutral） |
| proposal D2H / wait | round 19 % | — | ④（CLI のみ） | device token bridge |

- grouped gate は 10 expert × n=1280 × k=2560 × 3.05 bpw × 48 layer × 3 = 1.80 GB を
  45.8 ms で流しており **39 GB/s**。① DRAM（550–636 GB/s）ではない。
- GDN verify は 36 layer × 96 block × 65,536 B × 3 = 680 MB を 15.7 ms で流して
  **43 GB/s**。同じく ③ で、拘束は LDS（35,840 B → 1 block/CU）。
- したがって §2 の優先順位「share 大 × ②③ × occupancy / 命令昇格 / 同期消去」では、
  **GDN verify の LDS 削減**が最初に検証すべきレバーになる。lossy 本体では
  64 行ステージで LDS 19,456 → 14,336 B、3 → 4 block/CU、−4.3 % を実測済み。
- verify は 64-wide のため、64 行ステージ後も LDS 18,432 B（`S[64][68]` 17,408 +
  extras）で 3 block/CU 止まり。4 block には 16,384 B 以下が必要で、padding の
  調整では届かない。v32 幅（kWv=32、128 thread）への移植が次の段になる。
- 次点は grouped gate の ③。7 行では wave が足りないため、prefill で負けた
  n_tiles=1 の判断を decode に持ち込まない（decode は n_tiles を増やす方向）。
  decode の pair-K 除去は gate 16.11 → 16.41 ms（+1.9%）で棄却済み。
- `dflash2_topk_merge` は参照実装（1 thread/row）を hierarchical に替えても
  round が neutral（113.0 → 114.4 tok/s）。proposal stream で verify と重なるため、
  ④/overlay として wall レバーから外す。

### 単発 MTP の不具合修正

`shelf-compute --mtp` は drafter 構築で
`linear output features do not match the weight column width` に落ちていた。
serve 経路と同じく、EP/TP では rank ごとの `local_kv_heads` で
`MtpDrafterInput` を作り、`drafter_config.local_kv_heads` と
`MtpGdnRankBinding` の `local_kv_heads` / `tp_rank` / `mtp_input` を設定する
（`9dbb1d42`）。

### GDN verify の LDS staging を採用（1 → 3 block/CU）

`recurrence.hip` の body に `kStage` を追加し、verify の serial 経路で 64 k 行ずつ
ステージする。`dT` は v32 と同じく `W` の前進代入で置き換え、`LdsSmem` も
`LdsSmem<kStage>` にして `S[128]` を `S[64]` に落とした。

| 項目 | 変更前 | 変更後 |
| --- | ---: | ---: |
| kernel | `wmma_verify_lossy` | `wmma_verify_lossy_stage` |
| VGPR | 224 | **192** |
| LDS | 35,840 B | **18,432 B** |
| block/CU | 1 | **3** |
| verify kernel（8 token trace） | 7.57 / 7.59 ms | **7.17 ms（−5.3 %）** |

interleaved A/B（DFlash2 EXL3、device 2,3、16 token）:

| run | verify GPU ms | round ms | tok/s |
| --- | ---: | ---: | ---: |
| off | 81.32 | 126.98 | 117.87 |
| **on** | 81.22 | 125.90 | **118.88** |
| off | 81.60 | 126.21 | 118.60 |
| **on** | **80.77** | 126.56 | 118.26 |

verify は平均 81.46 → 81.00 ms（−0.6 %）、decode は 118.24 → 118.57 tok/s（+0.3 %）。
生成 16 token は 4 run すべて一致。既定 on、`SHELF_GDN_VERIFY_STAGE=0` で戻せる。
`test_gdn_recurrence`（baseline / stage、chunk off 含む）PASS、required 231/231 PASS。

- 4 block/CU には届かない。verify は 64-wide で `S[64][68]` 単体が 17,408 B、
  extras 込みで 18,432 B となり、4 block の境界 16,384 B を超える。VGPR 192 は
  8 wave/SIMD で 4 block 可能なので、律速は LDS。次は verify の v32 化。

### 非投機と spec の最適化適用状況

| 最適化 | 非投機 decode | spec verify / commit |
| --- | --- | --- |
| grouped gate 120 VGPR | 入っている | 入っている（同じ kernel） |
| grouped down 184 VGPR | 入っている | 入っている |
| MOE_COMBINE sigmoid 融合 | 入っている | 入っている |
| GDN v32（128 thread、LDS 14 KB、4 block/CU） | 入っている | **入っていない**（verify は serial 64-wide、commit は別 body） |
| GDN verify の LDS ステージ（3 block/CU） | — | 今回追加（既定 on） |
| GDN commit の LDS ステージ | — | **未適用** |
| pre-rotation / SwiGLU producer | 入っていない（rows<128） | 入っていない（rows=7） |

GDN だけ verify / commit が別実装で、v32 と LDS ステージの恩恵を受けていなかった。
verify は今回埋めた。commit と verify の v32 化が残る。

### decode の grouped gate pair-K を外す案は棄却

decode（tokens ≤ 16）だけ pair-K を外し、`xbuf` を 272 → 144 値/row にして
LDS 17,408 → 13,312 B、VGPR 120 → 112、block/CU 3 → 4 を狙った。8 token の
DFlash2 decode window では gate が **16.11 → 16.41 ms（+1.9 %）**。

- pair-K の barrier 半減の損失が occupancy +1 block を上回る。decode window 全体は
  65.7 ms で変わらない。
- したがって gate は pair-K のままにする（`SHELF_EXL3_GROUPED_DECODE_PAIR` の
  分岐も追加しない）。

次の gate レバーは N tile 自体を 128 → 64 にすることだが、accum 8,192 B の削減に
対して grid が 2 倍になり、今回と同じ barrier/launch 側の損失が出る可能性が高い。
優先度は `hyper_up_mix`（13.2 %、4 launch/layer）の hc 融合に下げる。

### hyper up mix の rows 方向タイルを採用（−36 %）

`hyper_up_mix` は 1 warp が 1 `(row, i)` を担当し、`[hc, hidden, k]` の up weight を
**row ごとに読み直していた**。rows=7、hc=4、hidden=2560、k=320 で 26.4 GB を
25.5 ms で流しており、約 1.03 TB/s = L3 の上限（1263–1378 GB/s）に張り付いていた。
つまり ③ ではなく ② L3 / pass 律速で、削るべきは weight の再読。

1 warp が `i` を固定して 8 行タイルを持ち、weight を 1 回読んで各行に適用する
`hyper_up_mix_rows_kernel<8>` を追加した。行ごとの k 蓄積順と shfl 還元は従来と
同一なので、出力はビット一致する。

| 項目 | 従来 | rows タイル |
| --- | ---: | ---: |
| kernel 時間（8 token decode window） | 10.77 ms | **6.92 ms（−35.7 %）** |
| decode window 合計 | 65.7 ms | **61.6 ms（−6.2 %）** |
| VGPR | 40 | 112 |

interleaved A/B（DFlash2 EXL3、16 token、device 2,3）:

| run | verify GPU ms | round ms | tok/s |
| --- | ---: | ---: | ---: |
| off | 80.95 | 125.30 | 119.43 |
| **on** | **77.39** | **123.80** | **120.88** |
| off | 81.94 | 127.01 | 117.85 |
| **on** | **77.36** | **122.00** | **122.66** |

verify は平均 81.45 → 77.38 ms（−5.0 %）、round 126.16 → 122.90 ms（−2.6 %）、
decode は 118.64 → **121.77 tok/s（+2.6 %）**。生成 16 token は 4 run すべて一致。
rows ≤ 8 で既定 on、`SHELF_HYPER_UP_MIX_ROWS=0` で戻せる。rows ≥ 9（prefill、
混合 batch）は従来カーネルのままで、拡張は次の候補。required 231/231 PASS。

prefill（rows > 8）への拡張も試した。2048 token / 512 chunk の 4 chunk で
`hyper_up_mix` は rows タイル無し 4.00 ms、8 行タイル 3.94 ms（−1.6 %）、
GPU 合計は 1,129.7 → 1,131.4 ms で誤差。prefill では `hyper_up_mix` 自体が
0.35 % しかなく、拡張は採らない（rows ≤ 8 のまま）。prefill 側の hyper は
別 kernel の `hyper_mix`（同じ trace で約 29 ms）が本体で、こちらは未着手。

並行リクエスト（DFlash2、conc=4、32 token、device 2,3）でも rows タイルの全 rows 拡張を
A/B した。aggregate は 43.62 / 43.22（off）対 43.20 / 43.42（on）tok/s で **neutral**。
serve の fused verify は rows が 28 程度になり、従来カーネルでも同じ weight を L3 から
共有できるため差が出ない。prefill と同じく rows ≤ 8 のガードを維持する。
