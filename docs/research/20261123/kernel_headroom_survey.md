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

## 13. opt-in 採用: dense EXL3 の row 前処理を 1 launch に融合

dense A 経路の `rowstat` + `rowslot` + `rowfinish` を、atomic で slot を詰める
`exl3_mul24_rowprepare_kernel` 1 launch に融合した。wall は neutral だが、
launch 数・ISA・GPU 時間が減るため **opt-in**（`SHELF_EXL3_ROW_FUSE=1`）として残す。
既定は従来の 3 kernel 経路である。

### specialized 測定

2048 token / 4 chunk / EP=2 の trace。既定側は旧 3 kernel だけ、opt-in 側は
`rowprepare` と 4 byte fill だけが走る。env 判定はホスト側で 1 回だけ行い、
GPU kernel 内に分岐は入れていない。

- row 前処理の GPU 時間: `rowstat` 46.817 + `rowslot` 12.214 + `rowfinish`
  52.452 = **111.483 ms** → `rowprepare` 55.460 + fill 6.817 = **62.277 ms**
  （**−44.1 %**）。
- prefill window の GPU 合計: **1159.15 → 1117.88 ms**（**−3.6 %**）。
- dispatch 数: 2484 + 2485 + 2484 = 7455 → `rowprepare` 2486 + fill 2486 =
  4972。kernel launch だけでは **7455 → 2486（−66.7 %）**。
- static instruction: 170 + 200 + 267 = **637** → **482（−24.3 %）**。
  `v_alu` は 228 → 180（−21.1 %）、code size は 3204 → 2504 bytes（−21.8 %）。

### wall

| 条件 | 既定 ms | opt-in ms |
| --- | ---: | ---: |
| 7002 token、host experts 0 | 2034.2 / 2037.7 | 2033.0 / 2038.1 |
| 7002 token、host experts 96（前回測定） | 2427.8 | 2445.4 |

生成 16 token は既定 / opt-in で一致した。wall が動かないのは、この cluster が
prefill のクリティカルパスではなく、削減分が遊休に吸収されるためである。
したがって既定にはせず、launch / ISA / GPU 時間を削る opt-in として採用する。
