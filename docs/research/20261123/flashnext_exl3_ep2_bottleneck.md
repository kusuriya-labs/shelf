# Qwen3.8-Flash-Next EXL3-INT8-3.05bpw の推論ボトルネック調査

`docs/developer/kernel_optimization_and_fusion.md` の手順 ①（計測と ISA 収集 → 選定表 →
律速 4 分類 → レバー可否）を、`models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`
に対して実施した記録である。測定原則は [../../perf/methodology.md](../../perf/methodology.md)
を正本とする。ここに置く値は reference run であり、qualified baseline ではない。

## 1. 目的と範囲

- 対象 model: 48 層、hidden 2560、routed expert 512 / top-10、`moe_intermediate_size` 640、
  shared expert 640、GDN 36 層 + full attention 12 層、EXL3 INT8 3.05 bpw
  （expert は 3 bit と 4 bit の混在、lm_head / 一部 projection は 5 bit）。
- `run.sh` が指定する serving 構成（MTP serve、EP/TP、host expert offload）を起点に、
  decode と prefill の GPU 時間を kernel 単位へ分解し、律速とレバーを判定する。
- 最適化の実装と採否判定（手順 ②③）は本記録の範囲外である。

## 2. 結論

1. **`run.sh` の現行構成は起動しない。** `--tensor-parallel 2` は checkpoint の
   routed expert 形状（N=K=640）が EXL3 の 128 alignment を満たさず load 時に拒否される。
   この checkpoint で 2 GPU を使う構成は EP=2 / TP=1 だけである。
2. **調査時点の既定 build（`SHELF_HIP_GRAPH=ON`）では EP=2 の MTP serve が最初の round で
   失敗した。** `operation not permitted when stream is capturing` で停止する。
   原因は capture の `hipStreamCaptureModeGlobal` と、EP の 2 rank が同じ process で
   並行に HIP API を呼ぶことの競合である。§12 で `ThreadLocal` に変更して修正した。
3. **調査時点では EP=2 + `--moe-host-experts` が非有限値で失敗した。** 最初の prefill で
   `moe_router` が NaN を検出して停止した。原因は expert slot map の local expert 数を
   `global / tp_size` で作っていたことで、EP の axes overlap（TP=1 / EP=2）では
   実際の local expert 数（`global / ep_size`）の 2 倍になっていた。§12 で修正した。
4. **MTP の既定 verify（safe serial）は target-only より遅い。**
   EP=2 / c=1 で target-only 34.0、safe serial 31.3、fused verify 49.7 tok/s。
   `SHELF_MTP_FUSED_VERIFY=1` を設定しない限り MTP は逆効果である。
5. **この model の decode は DRAM 帯域律速ではない。** weight traffic は 1 token あたり
   実効 1.7〜3.2 GB で、実効 87〜116 GB/s（DRAM 636 GB/s の 14〜18 %）。
   律速は ③レイテンシ/並列不足 と ④launch（decode）である。
6. **decode の仕事は極端に細かい kernel に分割されている。** target-only 2772
   dispatch/token、fused MTP 1331 dispatch/token、うち 72 % / 67 % が 10 µs 未満。
   hyper connection の BF16 GEMV だけで 460（rows=1）/ 196（rows=4）dispatch/token、
   全体の 14〜17 % を占める。
7. **acceptance が低く、drafts を増やすと遅くなる。** この prompt では受理 1.1〜1.2 / round。
   fused verify で drafts=1 54.7、drafts=2 55.1、drafts=3 49.7、drafts=7 32.1 tok/s。
   drafts 1〜2 が最良で、`run.sh` の drafts=3 は約 10 % 損している。

## 3. 実行構成のブロッカー

### B1. TP=2 は load できない

```text
$ ./build/shelf-compute --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
    --device 2,3 --tensor-parallel 2 --moe-host-experts 128 ...
runtime create failed: qwen4exp_load.cpp:724: unsupported:
  EXL3 expert TP requires N/128 and K/128 to be divisible by tp_size:
  model.language_model.layers.0.mlp.experts.0.gate_proj.weight
```

`moe_intermediate_size` 640 を TP=2 で割った 320 が 128 の倍数でないためである。
`docs/developer/multi-gpu.md` §8 に記載済みの制約で、再現しただけになる。
この checkpoint の 2 GPU 構成は EP=2 / TP=1（axes overlap）のみである。

### B2. EP=2 の MTP serve は既定 build の graph capture で落ちる

HEAD `22e4f7e9` の `SHELF_HIP_GRAPH=ON` build、EP=2、`--mtp --mtp-drafts 3`、
`SHELF_MTP_FUSED_VERIFY=1`:

```text
mtp batch step failed: device_batch_context.hip:102: HIP error (prepare_batch_descriptor):
  operation failed due to a previous error during capture
```

EP=1 / TP=2 でも同様の報告が
[issue41_p0_verify_and_expert_swap.md](issue41_p0_verify_and_expert_swap.md) にある既知の
不具合である。本調査の MTP 測定はすべて `-DSHELF_HIP_GRAPH=OFF` の build
（`build-nograph`、HEAD `22e4f7e9` の source）で実施した。

### B3. EP=2 + host expert offload は NaN になる

`--expert-parallel 2 --moe-host-experts 128` で、offload は有効に働く
（rank あたり weight が 27.6 → 16.8 GiB へ減る）が、最初の prefill で停止する。

```text
context population failed at 0: executor.hip:2836: invalid state:
  host program execution failed: status=6 (a kernel reported a non-finite value: the run is stopped)
```

非有限値は `moe_router` が検出しており、router へ渡る hidden か logits のどちらかで
NaN が発生している。どちらかを本調査では特定していない。EP=2 では offload なしで
VRAM に収まるため実害は限定的だが、`run.sh` の指定値はそのままでは使えない。
host expert + EP の組み合わせは未検証の経路と判断する。

## 4. 測定条件

- GPU: Radeon AI PRO R9700 × 2（`--device 0,1`、一部 `--device 2,3`）。1 GPU 1 process、
  `flock /tmp/r9700.lock`。sclk は固定していない。
- build: HEAD `22e4f7e9`。graph ON は `build/`、graph OFF は `build-nograph/`。
- model: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`。
- 並列: `--tensor-parallel 1 --expert-parallel 2`。KV は bf16（`shelf-bench`）または
  `octave:c5v5`（MTP serve、`run.sh` と同一）。
- prompt は "This is a stable shared system document used for prefix caching. " の
  繰り返し + "Count from 1 to 20: 1, 2, 3,"。生成は greedy。
  serve の decode 値は compute 自身の `decode_tok_s`（最初と最後の token の区間）である。
- kernel trace は rocprofv3 1.4.1 `--kernel-trace --output-format csv`。
  trace は wall を 1.5〜1.7 倍に膨らませるため、**時間配分の比較にのみ**使う。
  E2E の速度は無計測の run を正とする。

## 5. E2E 測定

### 5.1 decode（EP=2、c=1、context 3162、61 token）

| 構成 | decode tok/s | ms/token | 備考 |
| --- | ---: | ---: | --- |
| target-only（MTP なし、serve） | 34.0 | 29.4 | build-nograph、graph 不要 |
| MTP safe serial（drafts 3） | 31.3 | 31.9 | 既定 verify。target-only より 8 % 遅い |
| MTP fused（drafts 3） | 49.7 | 20.1 | `SHELF_MTP_FUSED_VERIFY=1` |
| MTP fused（drafts 2） | **55.1** | 18.1 | 同 prompt・同 build |
| MTP fused（drafts 1） | 54.7 | 18.3 | 同 prompt・同 build |
| MTP fused（drafts 7） | 32.1 | 31.1 | MTP 統計行は出なかった（未解決） |
| `shelf-bench tg --mode greedy` ctx128 / 64 token | 33.65 | 29.7 | bf16 KV、warmup 2、HEAD |
| `shelf-bench tg --mode forward` ctx128 / 8 token | 33.6 | 29.8 | bf16 KV、`a7df92cb` |

- drafts 3 のとき `MTP_BATCH` は `rounds=27 accepted=33 emitted=60 draft_forwards=81`
  で、受理は 1.22 / round、draft あたり 0.41 である。この prompt は "1, 2, 3, ..." の
  単純な列であり、受理が 1.2 程度に留まるのは drafter（MTP block、`mtp_bits=3`）側の
  予測品質を疑う材料になる。ただし corpus 依存のため、この値だけで drafter の欠陥とは
  断定しない。
- drafts 7 は `MTP_BATCH` 行を出さず、速度も target-only 並みだった。drafts 7 の
  verify 行数（8）で fused 経路が選択されていない可能性があるが、未確認である。

### 5.2 prefill（EP=2、3162 token）

| 条件 | prefill ms | tok/s |
| --- | ---: | ---: |
| 1 本目（cold、hipBLASLt JIT を含む） | 1 693.6〜1 710.6 | 1 848〜1 867 |
| 2 本目（warm） | 1 023.7〜1 086.7 | 2 910〜3 178 |

- warm の 3 162 token は `--max-num-batched-tokens 4096` で 1 chunk である。
- cold の内訳は初回 shape の hipBLASLt JIT（合計約 1.0〜1.4 s）を含む。
  TTFT の初回だけの問題であり、定常 prefill の律速ではない。
- `SHELF_TP_TRANSPORT=host` にすると prefill は 1 257 tok/s（p2p 1 848 の 68 %）、
  decode は 40.8 tok/s（p2p 49.7 の 82 %）へ落ちる。rank 間 transport は無視できない。

## 6. kernel trace

### 6.1 target-only decode（rows=1、bf16 KV、ctx 512、8 token）

trace 下の wall は 49.9 ms/token、kernel 時間の合計は rank あたり 27.0 ms/token、
dispatch は 2 772 /token である（無計測の wall は 29.7 ms/token）。

| 家族 | ms/token | share | dispatch/token |
| --- | ---: | ---: | ---: |
| EXL3 GEMM / dequant | 12.47 | 46.2 % | 840 |
| BF16 GEMM（hyper 投影ほか） | 3.76 | 13.9 % | 460 |
| attention（paged / QSA） | 3.23 | 12.0 % | 36 |
| hyper connection（mix / apply / inject） | 2.46 | 9.1 % | 289 |
| MoE bookkeeping（router / slot scan / scatter / combine） | 2.16 | 8.0 % | 192 |
| elementwise | 0.94 | 3.5 % | 434 |
| GDN | 0.79 | 2.9 % | 72 |
| norm | 0.75 | 2.8 % | 244 |
| TP transport | 0.29 | 1.1 % | 146 |
| 10 µs 未満の kernel 合計 | 5.88 | 21.8 % | 1 990 |

主要 kernel 単体:

| kernel | ms/token | dispatch/token | 1 call µs | share |
| --- | ---: | ---: | ---: | ---: |
| `gemm_mul24_grouped_kernel<2,1,3,true>` | 6.70 | 94.0 | 71.3 | 24.8 % |
| `shelf_gemm_bf16_exact_rows_1` | 3.76 | 460.0 | 8.2 | 13.9 % |
| `shelf_attention_paged_bf16` | 3.13 | 12.0 | 260.9 | 11.6 % |
| `hyper_mix_kernel` | 1.59 | 97.0 | 16.4 | 5.9 % |
| `exl3_mul24_gemm_kernel<5,1,false,true>` | 1.41 | 71.0 | 19.9 | 5.2 % |
| `group_gemm<4,1,true>` | 1.28 | 81.0 | 15.8 | 4.7 % |
| `moe_slot_scan_kernel` | 1.13 | 48.0 | 23.5 | 4.2 % |
| `gemm_mul24_grouped_kernel<2,2,3,false>` | 1.08 | 47.0 | 23.1 | 4.0 % |
| `exl3_mul24_gemm_kernel<4,1,true,true>` | 0.77 | 80.0 | 9.7 | 2.9 % |
| `shelf_gdn_recurrence_wmma_lossy` | 0.69 | 36.0 | 19.1 | 2.5 % |
| `moe_router_kernel` | 0.66 | 48.0 | 13.7 | 2.4 % |
| `hyper_apply_kernel` | 0.50 | 48.0 | 10.4 | 1.9 % |

- `shelf_attention_paged_bf16` は ctx 128〜512 で 1 call 253〜261 µs とほぼ一定である。
  KV は 1 token あたり約 6 MB しか読まない。**短い context では 12 層で 3.0 ms/token を
  固定費として払っている**（③の典型）。長い context ではこの費が線形に増える。
- `shelf_gemm_bf16_exact_rows_1` は 460 dispatch/token。model の BF16/F16 行列は
  hyper connection の 4 行列（`mlp_hyper_connection` と `attn_hyper_connection` の
  up/down、合計 4 × 0.315 GB）と GDN の a/b 投影、PLE の投影である。1 層あたり
  9.6 launch になり、4 行列に対して複数 launch が割り当てられていると推定するが、
  dispatch 単位の同定は未実施である。1 launch 8.2 µs、実効は約 330〜360 GB/s で、
  DRAM 天井の 52〜57 % に留まる。
- 1 層あたり約 58 dispatch（2 772 / 48）で、elementwise と norm だけで 678/token ある。

### 6.2 MTP fused decode（verify rows=4、octave:c5v5、61 token）

trace 下の wall は 1 699 ms（27.9 ms/token）、kernel 合計は rank あたり 15.5 ms/token、
dispatch は 1 331 /token。無計測は 20.1 ms/token（drafts 3）である。

| 家族 | ms/token | share | dispatch/token |
| --- | ---: | ---: | ---: |
| EXL3 GEMM / dequant | 7.38 | 47.6 % | 415 |
| BF16 GEMM（hyper 投影ほか） | 2.59 | 16.7 % | 209 |
| hyper connection | 1.16 | 7.5 % | 133 |
| GDN（verify 変種） | 1.14 | 7.3 % | 31 |
| MoE bookkeeping | 1.04 | 6.7 % | 89 |
| attention（paged + octave split） | 0.96 | 6.2 % | 23 |
| other（rotate/recon 等） | 0.74 | 4.8 % | 51 |
| elementwise | 0.48 | 3.1 % | 198 |
| norm | 0.36 | 2.3 % | 113 |
| 10 µs 未満の kernel 合計 | 2.81 | 18.1 % | 892 |

- verify が 1 round で 4 行を処理するため、weight 1 回の読みで複数 token を進められる。
  EXL3 の 1 token あたり時間は 12.47 → 7.38 ms（1 emitted token あたり）へ下がる。
  これが fused verify の効果の本体である。
- それでも 1 emitted token あたり 1 331 dispatch あり、BF16 hyper 投影が 209 dispatch、
  2.59 ms（16.7 %）を占める。draft forward が重なるため、BF16 の比率は target-only
  より上がる。

### 6.3 warm prefill（3 162 token、1 chunk）

trace 下の wall は 1 050 ms、kernel 合計は rank あたり 833 ms（GPU busy 79 %）。

| kernel 家族 | ms（rank あたり） | share |
| --- | ---: | ---: |
| EXL3 grouped GEMM（3 bit gate/up） | 233.7 | 28.1 % |
| EXL3 grouped GEMM（down） | 96.7 | 11.6 % |
| GDN recurrence | 71.8 | 8.6 % |
| attention prefill | 37.1 | 4.5 % |
| TP copy | 32.5 | 3.9 % |
| hyper mix / apply | 40.8 | 4.9 % |
| MoE combine / slot scan | 45.0 | 5.4 % |
| EXL3 rotate / rowfinish / quantize | 47.8 | 5.7 % |
| hipBLASLt BF16 / INT8 | 78.0 | 9.4 % |

- prefill は expert の dequant を含む grouped GEMM が 40 % を占める。token 数が多いため
  1 dispatch が 1 ms 級になり、decode のような launch 支配ではない。
- GPU busy 79 % の残り 21 % は rank 間同期と descriptor 準備である。`SHELF_TP_TRANSPORT=host`
  で prefill が 32 % 落ちることから、transport の寄与は実在する。

## 7. 選定表（decode、target-only rows=1）

| kernel 家族 | share | 実効 GB/s（推定） | VGPR / LDS | wg / grid | occupancy 拘束 | 命令クラス（ISA 静的） | 律速分類 | 有効なレバー |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `gemm_mul24_grouped<2,1,3,true>` | 24.8 % | 約 116（EXL3 全体） | 112 / 17 408 | 256 / 1280×320 | register: 6 block（LDS は 7） | 7 659 命令、v_alu 3 860、`s_barrier` 16 | ② dequant ALU + ③ rows=1 | dequant 命令削減、行数拡大（MTP/ batch） |
| `gemm_mul24_grouped<2,2,3,false>` | 4.0 % | 同上 | 168 / 21 504 | 256 / 2560×320 | register: 4 block | 4 904 命令、v_alu 2 480、`s_barrier` 12 | ② + ③ | 同上 |
| `exl3_mul24_gemm<5,1,false,true>` | 5.2 % | 約 340（lm_head 0.2 GiB） | 120 / 0 | 128 / 10240×2 | register: 12 block | 1 257 命令、v_alu 751、barrier 0 | ① と ③ の境界 | split / N tile |
| `group_gemm<4,1,true>` | 4.7 % | — | 96 / 0 | 128 / 5120 | register: 15 block | 未計測 | ③ | 行数拡大 |
| `shelf_gemm_bf16_exact_rows_1` | 13.9 % | 約 350 | 48 / 0 | 256 / 10240 | register: 8 block | 373 命令、v_alu 221、barrier 0 | ③ と ① の中間 | 4 投影の 1 launch 化、vector 幅 |
| `shelf_attention_paged_bf16` | 11.6 % | 約 2（KV 6 MB） | 64 / 12 800 | 256 / 393 216 | register: 8 block（LDS は 10） | 未計測 | ③ 並列過多・固定費 | 短 context 用 kernel、page 列挙の削減 |
| `hyper_mix` / `hyper_apply` / `inject` | 9.1 % | elementwise | 16 / 0 | 256 / 大 | 非拘束 | 未計測 | ④ launch | 融合 |
| MoE bookkeeping 4 種 | 8.0 % | index のみ | 8〜40 / 0.5〜2 KB | 64〜256 / 小 | 非拘束 | slot scan 319 命令、router 776 命令 | ② 冗長 pass + ④ | router / scan / scatter / combine の融合 |
| `shelf_gdn_recurrence_wmma_lossy` | 2.9 % | 約 144（state 113 MB） | 224 / 35 840 | 256 / 12 288 | register: 3 block | 5 992（verify 変種）、v_alu 2 404、barrier 18 | ② | rows 分割（既存研究あり） |
| elementwise 全体 | 3.5 % | elementwise | 16〜24 / 0 | 256 / 大 | 非拘束 | 未計測 | ④ launch | 融合 |
| norm 全体 | 2.8 % | elementwise | 16〜24 / 0.5 KB | 32〜256 / 大 | 非拘束 | 未計測 | ④ launch | 融合 |

- occupancy は [kernel_optimization_and_fusion.md](../../developer/kernel_optimization_and_fusion.md)
  §1 の式で静的算定した。`budget_waves = min(4 × floor(47104 / (32 × vgpr_r)), 64)`、
  `blocks = min(budget_waves / (wg/32), 131072 / LDS, 64 / (wg/32))`。
- 実効 GB/s は rank あたり・1 token あたりの静的 bytes を trace の時間で割った値である。
  EXL3 は routed expert（local 256 のうち選択 5 個 = 0.459 GB）+ backbone の
  EXL3 行列（attn / GDN / shared の rank 分 0.793 GB）+ lm_head 0.199 GB = 1.45 GB。
  BF16 は hyper 投影 1.258 GB。全体では約 3.1 GB/token で、29.7 ms に対して 106 GB/s
  （DRAM 636 GB/s の 17 %）である。
- ISA は `llvm-objdump --arch-name=amdgcn --mcpu=gfx1201` で static 命令を数えた。
  `s_barrier` は静的出現数であり、実行回数ではない。

## 8. 律速分類と判断

- **decode は ① DRAM 帯域律速ではない。** 実効 87〜116 GB/s は 550〜636 GB/s 帯の
  1/5 以下である。「重み全体が L3 を超えるから DRAM 律速」という dense model の議論は、
  top-10/512 の MoE では成立しない。1 token が読む routed expert は全体の約 0.5 % である。
- **decode の主題は ③ と ④ である。** 1 token に 1 331〜2 772 dispatch、その 7 割が
  10 µs 未満、BF16 hyper 投影は 1 launch 8 µs で 460 本、attention は 1 層 260 µs の
  固定費である。1 kernel あたりの仕事が小さすぎて、wave を埋めても効率が出ない。
- **prefill は ② である。** grouped EXL3 GEMM の dequant（v_alu 主体、barrier 16 本）が
  40 % を占め、GPU busy 79 % で launch 支配ではない。token あたりの bytes は増えない。
  したがって prefill のレバーは転送削減ではなく、dequant 命令と同期の削減である。
- **transport は decode の 1 %、prefill の数 % だが、host 経路に落ちると 18〜32 % 失う。**
  `SHELF_TP_TRANSPORT=auto` が p2p を選べていることが前提条件になる。

## 9. レバー候補と見積り

Amdahl の式 `share × (1 − 1/倍率)` で、fused MTP の 20.1 ms/token を分母に置く。

| 候補 | 根拠 | 見積り |
| --- | --- | --- |
| L1: 既定 verify を fused にする | 31.3 → 49.7 tok/s（実測） | **−37 %**（設定のみ） |
| L2: drafts を 1〜2 にする | 49.7 → 55.1 tok/s（実測） | **−10 %**（設定のみ） |
| L3: BF16 hyper 投影の launch 統合 | 209 dispatch/token、2.59 ms。4 投影を 1 launch にすると dispatch −75 % | −1.2〜1.8 ms（6〜9 %） |
| L4: 短 context attention の固定費削除 | 非 MTP で 12 層 3.0 ms/token、KV は 6 MB | 非 MTP −2.0 ms（7 %）、MTP −0.6 ms（3 %） |
| L5: MoE bookkeeping の融合 | 89 dispatch/token、1.04 ms（slot scan 24 µs × 22） | −0.3〜0.5 ms（2〜3 %） |
| L6: elementwise / norm の融合 | 311 dispatch/token、0.84 ms | −0.3〜0.4 ms（2 %） |
| L7: EXL3 grouped の dequant 命令削減 | 47.6 %、v_alu 3 860、barrier 16 | 10 % 改善で −0.7 ms（3.5 %） |
| L8: acceptance の改善 | 受理 1.2 / round。3.0 になれば tokens/round 2.2 → 4.0 | 理論値で最大 **+80 %**（未検証） |

- L2 は code 変更なしで今日適用できる。L1 も B2（graph capture）を修正すれば設定だけで
  有効になる。既定 build では MTP serve 自体が動かない。
- L8 は本調査で効果を確認していない。受理が低い原因（MTP block の 3 bit 量子化、
  draft の sampling、prompt 依存）の切り分けが先である。
- L3〜L7 は 1 token あたり合計 3〜4 ms（15〜20 %）の削減候補である。
  fused verify（L1）と drafts（L2）の設定効果の方が大きい。

## 10. 未測定・残課題

- graph ON build で MTP を動かす修正（B2）。本調査では graph OFF build のみ測定した。
- EP=2 + host expert offload の NaN（B3）の原因。offload を使わない構成では再現しない。
- drafts 7 で `MTP_BATCH` が出ない理由と、そのとき fused 経路が選ばれているか。
- acceptance の corpus 依存。自然文と 1 024 token 以上の出力での再測が必要である。
- attention が ctx 128〜512 で 260 µs 一定になる原因（grid 393 216、block の仕事量）。
- BF16 hyper 投影 460 dispatch/token の内訳（どの行列が何本か）。`SHELF_LINEAR_DEBUG`
  は miss しか出さないため、dispatch 単位の同定は未実施である。
- MTP fused の共有区間や c=2 以上の scaling。本調査の c=2 は prefill の直列化を含み、
  decode の scaling として解釈できない。
- 本記録の数値は 1 回の run が多く、5-run p50 の qualified baseline ではない。
  採用判定には [methodology.md](../../perf/methodology.md) §5 の interleaved A/B が必要である。

## 11. 再現

```sh
# 既定 build（HEAD 22e4f7e9）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --target shelf-bench shelf-compute -j 8

# target-only decode（bf16 KV）
flock /tmp/r9700.lock ./build/shelf-bench tg --mode greedy \
  --context 128 --tokens 64 --warmup 2 --device 0,1 \
  --tensor-parallel 1 --expert-parallel 2 \
  --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
  --kv-cache-dtype bf16

# MTP serve（graph OFF build が必要）
cmake -S . -B build-nograph -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSHELF_HIP_GRAPH=OFF
cmake --build build-nograph --target shelf-compute -j 8
SHELF_MTP_FUSED_VERIFY=1 SHELF_COMPUTE=./build-nograph/shelf-compute \
  python3 <MTP serve harness>

# kernel trace
rocprofv3 --kernel-trace --output-format csv -d /tmp/trace -o dec -- \
  ./build/shelf-bench tg --mode forward --context 512 --tokens 8 --warmup 0 \
  --device 0,1 --tensor-parallel 1 --expert-parallel 2 \
  --model-dir models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw \
  --kv-cache-dtype bf16
```

`<MTP serve harness>` は `tests/server/support.py` の `ComputeHarness` で
`ep=2`、`mtp=True`、`mtp_drafts=3`、`mtp_window=4096`、`mtp_sink=64`、
`kv_cache_dtype="octave:c5v5"` を指定し、3162 token の prompt を 1 本流す。

## 12. 追記: 修正と既定変更

§3 のブロッカーと §9 の L1 / L2 を実装した。

### F1. graph capture の排他 mode

`executor.hip` の `hipStreamBeginCapture` を `hipStreamCaptureModeGlobal` から
`hipStreamCaptureModeThreadLocal` へ変えた。EP=2 は 2 rank を同一 process で並行に
動かすため、Global mode では capture 中の他 rank の HIP 呼び出しが capture を
無効化し、`prepare_batch_descriptor` の `hipGetLastError` に前段のエラーが現れていた。
失敗は間欠的で、5 回中 1 回の再現だった。変更後は既定 build（graph ON）で
fused / safe serial の両方が完走し、fused は 51.1 tok/s（graph OFF の 49.7 より速い）。
DFlash2 の drafter は既に `ThreadLocal` を使っており、これに揃えた形になる。

### F2. expert slot map の EP 範囲

`qwen4exp_load.cpp` の slot map と host / VRAM の expert 数を
`global_experts / options.tp_size` から `global_experts / ep_size` へ変えた。
`load_moe` は expert ID を `ep_size` で分割し、行列は `expert_tp_size` で分割する。
axes overlap（TP=1 / EP=2）では `options.tp_size` が 1 のため、旧実装は local expert を
512 として slot map を作り、実際の 256 とずれていた。expert 128〜255 が存在しない
VRAM slot を指し、host plane の先頭を別 expert として読んでいた。
同じ不整合は TP>1 + host offload（matrix split）にもあった。

検証:

- `test_qwen4exp_load` に `SHELF_QWEN4EXP_EP` と slot map の範囲チェックを追加した。
  修正前は `MOE expert slot map matches the local expert range` が FAIL し、
  修正後は rank0 / rank1 とも PASS する。
- EP=2 + `--moe-host-experts 128` の E2E で、offload なしと同一の 16 token を生成した。
- offload 128 の decode は 19.0 tok/s（offload なし 33.5）である。EP=2 では VRAM に
  収まるため、throughput 目的では使わない。

### F3. MTP fused verify を既定に

`SHELF_MTP_FUSED_VERIFY` の未設定時を fused にした。`=0` で safe serial に戻る。
MTP reserve の見積りも同じ既定へ揃えた。

- 既定（fused、drafts 2、EP=2、3,162 token prompt）: decode 54.0 tok/s。
- `SHELF_MTP_FUSED_VERIFY=0`: 31.3 tok/s。
- `test_compute_mtp_prefix.py` 13/13、`test_compute_mtp_batch.py`（EP=2、2 request）
  10/10、required 231/231 が PASS した。

### F4. `run.sh`

`--tensor-parallel 2` を `--expert-parallel 2` に変え、`--moe-host-experts 128` を外し、
`--mtp-drafts` を 3 から 2 にした。また EP=2 / offload なしで `--max-seq-len 262144` は
KV を 135,760 token しか確保できず起動しないため、131,072 にした。
server を起動し、`/v1/models` と chat completion が通ることを確認した。

### F5. 同一入力 BF16 GEMM の pair 融合

`SHELF_BF16_DISPATCH_DEBUG` で decode 1 token あたりの BF16 GEMM を数えると 460 本で、
内訳は hyper down 97 / hyper up 97 / hyper block_inject 96 / GDN in_proj_a+b 72 /
MoE router 48 / shared expert gate 48 / PLE 2 だった。このうち **同じ入力を持つ
2 本**を 1 launch にまとめた。

| 融合 | 元 | 削減 |
| --- | --- | ---: |
| hyper `input_mix_weight_down` + `block_inject_weight` | 2 → 1 / site | 96 |
| GDN `in_proj_a` + `in_proj_b` | 2 → 1 / layer | 36 |
| MoE `router` + `shared_expert_gate` | 2 → 1 / layer | 48 |

- 実装は fused kernel `src/shelf/kernels/fused/bf16_linear_pair.hip`
  （`include/shelf/kernels/fused/bf16_linear_pair.h`、共有 body は
  `include/shelf/kernels/fused/detail/bf16_exact_rows_device.h`）。
  `grid.x = blocks(n0) + blocks(n1)` で 2 行列を 1 launch に収め、行列ごとに
  weight / output / stride / dtype を持つ。exact-rows の行単位の計算は変えていない。
- matcher は `optimized_dispatch.hip` の `try_launch_linear_bf16_pair`。
  program 上で隣接する 2 本の `LINEAR_BF16` が同じ入力・同じ rows・同じ k で、
  両方 `ExactRows` のときだけ 1 launch にする。prefill など `ExactRows` でない
  場合は 2 launch のまま。`SHELF_DISABLE_BF16_PAIR=1` で無効化できる。
- 融合のため hyper の inject 線形を down の直後へ、MoE の shared gate 線形を
  router の直後へ移動した。どちらも同じ入力だけを読む独立計算で、数値は変えない。
- 正しさ: `test_gemm_bf16_exact_rows` に pair の case を追加し、2 本の
  exact-rows launch と **byte-exact** を確認した（320:4:10240、24:24:2560、
  512:1:2560 rows=4、17:3:65 rows=2、±3e38・denormal 入力、F32/BF16 混在）。
- E2E: `shelf-bench tg --mode greedy --context 128 --tokens 64 --warmup 2`
  の interleaved A/B 3 ペアで **off 32.91 / 33.23 / 33.53 → on 33.51 / 33.83 / 34.06 tok/s**
  （中央値 33.23 → 33.83、**+1.8 %**、per-token 30.1 → 29.6 ms）。
  同じコードを fused/ へ移す前の build では +3.1 % だった。build の配置と
  マシン状態で絶対値が動くため、採否には移設後の 3 ペアを使う。
  生成 24 token は on / off で一致した。
- dispatch は 460 → 280 / token（−39 %）。残る hyper up / down の 194 本は
  down の結果を up が読む依存があり、1 launch にできない。
