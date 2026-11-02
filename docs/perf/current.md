# Current performance baseline

この文書は Shelf の最新の qualified performance baseline を記録する。
ここに載る値は下記 measurement revision での計測結果であり、それ以後に追加された
未計測の変更に対する性能を保証しない。

測定手順の原則は [methodology.md](methodology.md) を正本とする。

> **対応 format の注記**: 現在の runtime がロードできる量子化 format は
> BF16 / FP8 E4M3 のみである（[../developer/quantization.md](../developer/quantization.md)）。
> 本文書の計測は、当時の main performance target だった
> `models/Qwen3.8-27B-Quark-AWQ-MXFP4`（および TP 節の `models/Qwen3.8-27B-PSQ`）を
> 対象に行ったものであり、これらの model は現在の build ではロードできない。
> **HF FP8 / BF16 構成の広範な再計測は未実施**であるため、以下の数値は計測当時の記録として
> 扱い、現在の構成の性能値としては引用しない。
> 例外として **HF FP8 の E2E Prefill（TP2 / pp2048、2026-10-22 再計測）と
> decode（TP=1 / TP=2、2026-10-23 再計測）は再計測済み**
> （下記「HF FP8 E2E Prefill」「HF FP8 decode」節）。BF16 は未計測のままである。

## HF FP8 E2E Prefill（TP2 / pp2048、2026-10-22 再計測）

`shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --prompt-tokens 2048
--tp 2 --device 2,3 --arena-gib 16 --warmup 1 --runs 5` の `wall_ms_median`。
GPU2/3 ペア（計測時 GPU0 に第三者プロセス稼働のため回避）、外部プロセス非影響を
KFD 監視で確認。model revision `017b9c7a`、measurement revision = R&D 記録の
prefill-cta-swizzle 最終コミット、build Release / gfx1201、HIP graph 既定 OFF。

| rep | wall_ms_median | tok/s |
| --- | ---: | ---: |
| 1 | 539.03 | 3800.7 |
| 2 | 538.31 | 3805.7 |
| **中央値** | **538.67** | **3803** |

run 間のばらつきは 0.13%。アンカー計測（`shelf-compute --tp 2 … max-new-tokens 1`）:
`PREFILL_MS=548.8`、`GENERATED_IDS=47410`（既存トークン gate と一致）。
同条件の A/B シリーズ（GPU0/1 ペア）では 545〜551ms 帯（GPU 個体差 ±2%）。
この断面は C2-b 幾何・G3 scale fold・G4/G5/G7/G10 棄却・G6 ping-pong・G8 4行同時・
G9 CTA swizzle（G=4）採用後の状態（swiglu+quantize 融合は別分岐で採用・本断面では未投入）。
経緯と全実験の根拠は [../research/20261022/prefill_kernel_optimization/final_report.md](../research/20261022/prefill_kernel_optimization/final_report.md) を参照。

## HF FP8 decode（TP=1 / TP=2、2026-10-23 再計測）

`models/Qwen3.8-27B-FP8` の single-request greedy decode。
`shelf-compute` の `DECODE_TOKENS_PER_SEC`（生成128 token / decode 区間）、
5 run 中央値。prompt は 1697 token（計測用 id 列）。

TP=2 は GPU2/3 ペア + resident model host 2 台（target のみを保持、
drafter は compute が local load）。TP=1 は GPU0 の local load。

| arm | tok/s | spread | rounds | mean accepted |
| --- | ---: | ---: | ---: | ---: |
| TP=1 非投機 | 19.10 | 0.26% | — | — |
| TP=2 非投機 | 31.15 | 0.19% | — | — |
| TP=2 + DFlash2（drafts 7） | **67.16** | 0.21% | 36 | 2.528 |

- TP=2 非投機は TP=1 の **1.63×**（効率 81.6%）。
  TP=1 と TP=2 は GPU 個体が異なる（GPU0 対 GPU2/3）ため、この比には
  cross-GPU 差が含まれうる（±2% 想定）。
- DFlash2 は非投機 TP=2 の **2.16×**。
- prefill（1697 token、`PREFILL_MS` 中央値）: TP=1 1376.0 / TP=2 666.6。

### 正しさ

全 15 run（TP=1 非投機 / TP=2 非投機 / +DFlash2）で
greedy 128 token の生成列が完全一致した。
`GENERATED_IDS` の sha1 `cb1bae73ebc56dac846a48f952967d9a3045a325`。

### 計測条件

| 項目 | 値 |
| --- | --- |
| measurement revision | commit `c745c31`（計測時 working tree = この commit と同一。`c778ff5` + 未コミット変更として実施し commit した） |
| date | 2026-10-23 |
| model / drafter | `models/Qwen3.8-27B-FP8` / `models/Qwen3.8-27B-DFlash2`（3.6 GB） |
| build | Release / gfx1201 / HIP graph OFF（既定） |
| GPU | TP=1: HIP 0。TP=2: HIP 2+3（resident host が target rank weights を各 16.3 GB 保持） |
| options | `--max-new-tokens 128 --max-seq-len 2048 --arena-gib auto --prefix-cache 0 --temperature 0` |
| TP=2 attach | `--model-host "0=/tmp/g1.sock,1=/tmp/g2.sock"`（ephemeral arena 15.58 GiB、drafter は local load） |

TP=1 + FP8 + drafter は単カード 31.86 GB に target / drafter / KV が同居せず
OOM するため、TP=1 は非投機のみである。

### 計測コマンド

```sh
# TP=1 非投機
./build/shelf-compute --model-dir models/Qwen3.8-27B-FP8 \
  --input-ids-file build/tp2-bench-prompt-1920.txt --max-new-tokens 128 \
  --max-seq-len 2048 --arena-gib auto --device 0 --prefix-cache 0 --temperature 0

# TP=2（非投機 / +DFlash2 は drafter option のみが異なる）
./build/shelf-compute --model-dir models/Qwen3.8-27B-FP8 \
  --model-host "0=/tmp/g1.sock,1=/tmp/g2.sock" \
  --input-ids-file build/tp2-bench-prompt-1920.txt --max-new-tokens 128 \
  --max-seq-len 2048 --arena-gib auto --device 2,3 --tp 2 --prefix-cache 0 --temperature 0 \
  # --dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7
```

resident model host は次のように起動する（target のみ warmup）。

```sh
./build/shelf-model-host --device 2 --socket /tmp/g1.sock \
  --warmup-qwen35 models/Qwen3.8-27B-FP8 --warmup-tp-size 2 --warmup-tp-rank 0
./build/shelf-model-host --device 3 --socket /tmp/g2.sock \
  --warmup-qwen35 models/Qwen3.8-27B-FP8 --warmup-tp-size 2 --warmup-tp-rank 1
```

## 計測断面

| 項目 | 値 |
| --- | --- |
| model | `models/Qwen3.8-27B-Quark-AWQ-MXFP4`（計測当時の main performance target、HF 直接ロード。現在は非対応） |
| draft model | `models/Qwen3.8-27B-DFlash2` |
| model dir | `/opt/zen/work/shelf/models`（worktree 配下には `models/` が無いため、計測時は絶対パスで指定する） |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) / ROCm / HIP / Linux |
| measurement revision | commit `cc718024` |
| date | 2026-10-17 |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF |
| build directory | `build` |
| HIP graph | OFF（`SHELF_HIP_GRAPH` 既定） |
| arena | bench 24 GiB / compute 26 GiB |
| page-tokens | 16 |
| keep-alive | 有効（executor 既定。`SHELF_KEEPALIVE=0` で無効化） |
| device | 1 |

build condition で変わるのは `SHELF_HIP_GRAPH` の ON/OFF のみである。
Fusion / DeepFusion は一時的に build option が撤去されている。
本 baseline は既定値である HIP graph OFF で計測する。

model は HF 配布物を直接ロードする（ロード契約は
[../developer/quantization.md](../developer/quantization.md) の
「HF 由来 real-quantized safetensors の直接ロード」を参照）。
計測当時の `LINEAR_MXFP4` の Prefill2D は E8M0 fold（lossy）を通っており、
prompt ≥ 512 トークンの greedy 出力は fold 以前のビルドと一致しないことがある
（fold の経緯は `rnd/quantization/kernel_optimization_history.md` §7.95）。

## Target prefill（pp2048）

`--prompt-tokens 2048 --mode forward`、外側 3 rep × 内側 runs 3 / warmup 1 の中央値。
単位は gpu_tokens_per_sec。

| rep | gpu_ms_median | tok/s |
| --- | ---: | ---: |
| 1 | 698.65 | 2931.63 |
| 2 | 695.39 | 2945.36 |
| 3 | 698.66 | 2931.58 |
| **中央値** | | **2931.63** |

run 間のばらつきは 0.47%。

## Target non-spec decode（tg128・ctx2048）

`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`、3 rep 中央値。
単位は gpu_tokens_per_sec。

| rep | tok/s |
| --- | ---: |
| 1 | 29.94 |
| 2 | 29.92 |
| 3 | 29.87 |
| **中央値** | **29.92** |

run 間のばらつきは 0.23%。
`GREEDY_TOKEN_SUM=2818069` は全 rep で一致する。この系列は ctx2048 の prefill が
E8M0 fold 経由であるため fold 以前のビルドとは一致しない（正しさ契約節）。

## GPU-MCU decode backend

GPU-MCU backend はこの branch に含まれない。計測値・正しさ契約・再現手順は
`exp/gpu-mcu` ブランチの `docs/perf/current.md` を正本とする。

## TP=2 decode（`test_qwen35_tp_e2e`）

TP は `shelf-compute --tp 2 --device 0,1` から利用できる（単一 request と
`--serve-stdio`）。
`shelf-bench` は pp / quality のみ `--tp` を露出し、`shelf-server` の CLI は
露出しない（[../developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md) §10）。
下表の計測口は `build/tests/test_qwen35_tp_e2e` である。compute 経由
（HF FP8）の TP2 性能は上記「HF FP8 decode」節を参照。

この表は上記「計測断面」の measurement revision とは異なる revision の値である。

`SHELF_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ` を設定して 5 回走らせ、
decode の p50 を採る（PSQ model は現在の runtime では非対応）。

| 指標 | TP=1 | TP=2 |
| --- | ---: | ---: |
| decode_ms p50 | 535.196 | 370.994 |
| decode ms/token | 35.680 | 24.733 |
| decode tok/s p50 | 28.03 | **40.43** |
| run 間 spread | 0.25% | 2.55% |

2 GPU による加速は **1.442×**、効率は **72.1%**（40.43 / (28.03 × 2)）。

TP2 の 5 run は 370.305 / 370.703 / 370.994 ms と 378.825 / 379.765 ms の
2 クラスタに分かれた（差 +2.4%）。p50 は低クラスタ側にある。

### 計測条件

| 項目 | 値 |
| --- | --- |
| measurement revision | commit `f83c8e1e`（branch `exp/gpu-mcu`） |
| date | 2026-10-10 |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF / `build` |
| working tree | HEAD と同一（未コミットの実験変更なし） |
| prompt | 5 token（`304 17 283 2454 304`） |
| new tokens | 16（prefill step 1 + decode 15） |
| max_seq_len | 64 |
| max_scheduled_tokens / requests | 16 / 2 |
| page_tokens | 16 |
| arena | TP=1 24 GiB、TP=2 は rank あたり 20 GiB |
| device | TP=1 HIP 0、TP=2 HIP 0+1 |
| transport | `SHELF_TP_TRANSPORT` 未設定（report の `transport=hip-p2p`） |
| HIP graph | OFF（`SHELF_HIP_GRAPH`） |
| value_trace | decode 区間は無効（prefill 直後に解除） |

### 指標の定義

`decode_ms` は `ContinuousBatcher::step()` の decode ループ
（`while (batcher.has_pending())`）の wall 時間である。
`decode tok/s = decode_steps / decode_ms × 1000`。

**prefill step は既に token0 をサンプルする**（`token_budget_scheduler` の
`sample = final_chunk && max_new_tokens > 0`）。したがって decode 窓は
「生成 token 数 − 1」step 分しか含まない。生成 token 数で割ると分母が1過剰になり、
16 token なら **+6.7% の過大表示**になる。正は `TP_TIMING` の `decode_steps` である。

`prefill_ms` は value_trace が有効な区間を測っているうえ、prompt が 5 token で
固定オーバーヘッドが支配的であるため、**性能値として記録しない**。

### 正しさ

`test_qwen35_tp_e2e` 5/5 PASS。全 run で次の値が一致した。

- greedy 16 token の TP1 / TP2 の列が完全一致
- prefill logits の argmax が一致（`17`）、max abs diff `0.596`（上限 2.0）
- final hidden の相対差 `0.0356`（上限 0.10）
- `tp1 gate_rows=17408 / tp2 gate_rows=8704`、`kv_heads 4 / 2`（rank-local）

### 計測コマンド

```sh
SHELF_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ ./build/tests/test_qwen35_tp_e2e
```

既存の Gate 記録（`rnd/quantization/tp2_resource_regime.md`
の ctx64 decode 24.761 ms/token）とは **−0.11%** で一致する。ただし相手は
`new_tokens=32` の別 probe（削除済み）であり、条件は同一ではない。

## DFlash2 speculative decode（prose K7 256）

`--dflash2-drafts 7`、256 new tokens、3 rep 中央値。
単位は DECODE_TOKENS_PER_SEC。prompt は `real_prose/prompt.txt`。

> **計測条件の注記（2026-10-20 判明）**: `real_prose/prompt.txt` はカンマ区切り
> であり、direct mode の `read_input_ids`（`fscanf("%ld")`）はカンマで変換が
> 止まるため、この節の計測で compute に渡った
> prompt は**先頭1トークン（`[760]` = "The"）**である。正しさ契約の
> `GENERATED_IDS` sha `87dd2adfa49f` がこの条件の再現と完全一致することで
> 確認済み。絶対値と「31 tokens」ラベルは実態と一致しない。実際の31トークン prompt での
> 位置別 acceptance は
> `rnd/dflash2/acceptance_position_report.md`
> を参照。

| rep | tok/s | DFLASH2_ROUNDS | DFLASH2_ACCEPTED_DRAFTS | DFLASH2_MEAN_ACCEPTED |
| --- | ---: | ---: | ---: | ---: |
| 1 | 101.08 | 46 | 209 | 4.543 |
| 2 | 102.64 | 46 | 209 | 4.543 |
| 3 | 102.83 | 46 | 209 | 4.543 |
| **中央値** | **102.64** | | | |

run 間のばらつきは 1.70%（rep1 が低め。絶対値は中央値で判定する）。
`GENERATED_IDS` の sha1 先頭 12 桁 `87dd2adfa49f` は全 rep で一致する。
prompt は 31 tokens で Prefill2D（最小 rows 512）を通らないため、この計測に
E8M0 fold の数値影響はない。verify は role 固定の exact 経路である
（[../developer/dflash2.md](../developer/dflash2.md)）。

## DFlash2 batched verify throughput scaling

serve mode は複数 request の target verify を 1 batch に統合する。その scaling を
`test_dflash2_batch_perf` で測る。この表は上記「計測断面」の measurement revision とは
異なる revision の値である（label `gpu1;perf`、正しさテストと分離）。

全 request に同一 prompt を与え、batch 生成区間（`dflash2_spec_step_batch` →
`hipStreamSynchronize`）の wall 時間を測る。`ms/req` は wall / request 数、`scale` は
n=1 の `ms/req` を 1 とした比である。3 rep の中央値。

| n | tokens | wall ms | tok/s | ms/req | scale | rounds | mean accepted / slot |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 2463.0 | 52.0 | 2463.0 | 1.00 | 44 | 1.886 |
| 2 | 256 | 2836.2 | **90.3** | 1418.1 | 0.58 | 38 | 2.342 |
| 3 | 384 | 3919.4 | **98.0** | 1306.5 | 0.53 | 38 | 2.342 |
| 4 | 512 | 4683.2 | **109.3** | 1170.8 | 0.48 | 38 | 2.342 |

n=4 で集約スループットは n=1 の **2.10×**、request あたり時間は **0.48×** になる。

Phase 2B（batched proposal）適用前との比較（同一条件の 3 rep 中央値）:

| n | wall ms | tok/s | draft GPU ms |
| ---: | ---: | ---: | ---: |
| 1 | 2473.6 → 2463.0 | 51.7 → 52.0 | 535.8 → 533.9 |
| 2 | 3098.5 → 2836.2 | 82.6 → 90.3 | 940.5 → 659.0 |
| 3 | 4212.7 → 3919.4 | 91.2 → 98.0 | 1414.3 → 1084.1 |
| 4 | 5153.7 → 4683.2 | 99.3 → 109.3 | 1888.3 → 1347.5 |

n=4 で wall −9.1%、tok/s +10.1%、draft GPU −28.6%。n=1 は単一 request 経路へ委譲するため
変化しない。`rounds` と `mean accepted / slot` は適用前後で完全に一致する
（proposal が単一 request 経路と bit 一致している）。

`test_dflash2_batch_perf` は `DFlash2SpecBatchOutput` の timing から round 内訳も出す。
round wall に占める内訳は次のとおり（3 rep の中央値）。

| n | round ms | draft GPU ms | draft share | verify GPU ms | verify share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2365.2 | 533.9 | 22.6% | 1692.8 | 71.6% |
| 2 | 2715.5 | 659.0 | 24.3% | 1802.6 | 66.4% |
| 3 | 3736.5 | 1084.1 | 29.0% | 2281.1 | 61.1% |
| 4 | 4439.1 | 1347.5 | 30.4% | 2566.5 | 57.8% |

batched proposal は launch 数と host stall を減らすが、draft GPU 時間は依然 n とともに
増える（+271 ms/request、適用前は +450 ms/request）。drafter の weight traffic は
3.85 GB で、n=1 の proposal は約 12 ms（DRAM 640 GB/s 相当の 2 回読み）であるのに対し、
n=4 の batched proposal は約 35 ms かかる。row 数に比例して drafter の GEMM が
weight を再読込しているためで、GEMM config の row tile に依存する。3 rep で `rounds` /
`mean accepted` は完全に一致する（生成が deterministic である）。

### drafter linear の packed dot（Phase 2B 追加）

drafter の bf16 linear を `v_dot2c_f32_bf16`（bf16 pair dot）へ移した前後
（同一条件の 3 rep 中央値）:

| n | wall ms | tok/s | draft GPU ms | rounds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2461.8 → 2449.0 | 52.0 → 52.3 | 533.9 → 531.6 | 44 → 44 |
| 2 | 2832.4 → 2661.6 | 90.3 → 96.2 | 659.0 → 564.5 | 38 → 37 |
| 3 | 3916.1 → 3670.7 | 98.0 → 104.6 | 1081.2 → 961.6 | 38 → 37 |
| 4 | 4673.8 → 4276.6 | 109.3 → 119.7 | 1346.0 → 1096.4 | 38 → 37 |

n=4 で wall −8.7%、tok/s +9.5%、draft GPU −18.6%、draft share 30.4% → 27.2%。
`rounds` が 38 → 37 に減ったのは mean accepted が改善したためである
（n=4: 9.368 → 9.730）。packed dot は決定論的・非ロスだが byte-exact ではない
（target verify は `ExactRows` のままで decode と byte-exact を維持する）。

### 計測条件

| 項目 | 値 |
| --- | --- |
| measurement revision | commit `f636e52`（Phase 2B + packed dot） |
| date | 2026-10-19 |
| build | Release / gfx1201 / benchmarks ON / optional tests ON / `build`（worktree） |
| model | `models/Qwen3.8-27B-Quark-AWQ-MXFP4`（計測当時。現在は非対応） |
| drafter | `models/Qwen3.8-27B-DFlash2` |
| K / context / new tokens | 7 / 64 / 128 |
| arena | 30 GiB |
| device | 0 |
| HIP graph | OFF（`SHELF_HIP_GRAPH` 既定） |
| reps | 3（中央値） |

## BetterBench によるカテゴリ別比較（参照実装 vllm-mxfp4 の一カード）

外部 harness `vendor/BetterBench`（corpus v1.0、`--quick` = 1 warmup + 5 measured）で、
gfx1201 側の参照実装（`references/radiance_stack.md`）の
一カード表と同じ corpus・同じ sampling（temp 0.7 / top_p 0.95 / top_k 20、nonce で
prefix cache は cold）で計測した。2026-10-19、R9700 1 枚、`--device 1`、
DFlash2 `--dflash2-drafts 7`、KV fp8、`--max-concurrent-requests 8
--working-set-reserve-gib 4`、カテゴリ毎に server を再起動。

**参照側と同じ `--max-seq-len 65536`**（8 カテゴリ完走）:

| category | decode t/s | TTFT p50 ms（TTFT 修正前） | 参照（TP1・lazy） | 比 |
| --- | ---: | ---: | ---: | ---: |
| code | 115.6 | 135 | 145 | 0.80 |
| json | 150.0 | 136 | 184 | 0.82 |
| math | 114.6 | 134 | 169 | 0.68 |
| file_edit | 109.5 | 142 | 166 | 0.66 |
| chat | 83.0 | 139 | 81 | 1.02 |
| summarization | 82.8 | 142 | 138 | 0.60 |
| reasoning | 66.5 | 136 | 121 | 0.55 |
| prose | 44.2 | 119 | 79 | 0.56 |
| **加重合成**（code .30 / reasoning .20 / prose .15 / json .15 / file_edit .10 / summarization .10） | **96.3** | — | **137.6** | **0.70** |

`--max-seq-len 8192` でも各カテゴリは同等（code 110.9-120.2、json 150.0-150.2、
reasoning 59.7-66.7、prose 42.7-46.2、summarization 80.8、chat 97.4）で、
**context 長を 65536 にしても decode は劣化しない**。

- 参照側の加重合成は公開値 137.7 t/s と一致する（重みの検算）。参照側の **eager**
  一カード（こちらに GDN lazy snapshot が無いので本来の比較相手）は 114.7 t/s で、
  こちらは **−16%**。
- decode t/s の差は **1 round のコスト**で説明できる。round 時間はカテゴリにも
  位置にも依存せず 54-56 ms で、その **73% が target verify forward（8 行）の 40 ms**
  （15.5GB / 40ms = 387GB/s = streaming 実測 636GB/s の 61%。15.5GB は計測当時の
  MXFP4 model の per-round weight traffic）。`tok/round`
  （= 受理 + 1）は code 5.60 / json 6.27 / reasoning 3.67 / prose 2.57 で、
  **受理は参照側の公開値（SPEC=7 で code/json/file_edit 4.7-6.0 tok/update）と同等以上**。
  内訳と採否は `rnd/reference_gap_analysis.md` §5.6。
- こちらの `tok/update` は server がトークン単位で SSE を出すため計測できない
  （ITL として報告される）。decode t/s と TTFT は粒度に依らず比較可能。
- **TTFT（2026-10-19 修正後）**: prefill がサンプルした最初のトークンを投機 round の前に
  出すようにした（`ContinuousBatcher::step` は DFlash2 の prefill を含む step で round を
  実行しない）。8192 での TTFT p50 は code 82.9 / prose 65.6 / reasoning 82.8 ms で、
  修正前（134.8 / 116.4 / 134.4 ms）から **−38..−44%**。参照 eager の conc-1 TTFT p50
  93.8 ms より速い。decode t/s は同一（code 110.9→113.3、prose 46.2→42.9、
  reasoning は CV 35% の run 間ばらつき内）。
- **prefill スループット**（`shelf-bench pp --mode forward`、prompt は 8192 行ずつ
  chunk、**KV bf16**）: 2k 2973 / 4k 2908 / 8k 2753 / 16k 2452 / 32k 2006 / 64k 1472 PP t/s。
  参照は 2k 2738 / 8k 2721 / 16k 2795 / 32k 2666 / 64k 2494（fp8 KV）。
  **2-8k は同等以上、16k 以降で差が開く**。
- **KV dtype の効果**（`--kv-cache-dtype fp8`、同条件）:
  | prompt | bf16 KV | fp8（leg 前） | fp8（QK leg） | fp8（QK+PV leg） | fp8（+64 行 tile） |
  | --- | --: | --: | --: | --: | --: |
  | 2k | 2973 | — | — | **3028** | 3022–3049 |
  | 8k | 2774 | 2563 | 2667 | **2807** | 2822–2851 |
  | 16k | 2452 | 2164 | 2334 | **2531** | 2541–2564 |
  | 32k | 2006 | 1657 | 1865 | **2126** | 2167–2180 |
  | 64k | 1491 | 1120 | 1325 | **1588** | **1725–1728** |

  QK^T を fp8 WMMA に置き換え（K は raw byte staging、Q は tile 単位で fp8 量子化、
  `q_scale` を score に畳む）、PV も fp8 WMMA にした（V は raw fp8、P は
  row/page 単位の max で正規化して fp8、`v_scale` は P に畳み、page の寄与を
  `p_scale[row]` で掛けてから累積）。64k で leg 前 1120 → **1588（+42%）**、
  **bf16 KV（1491）を上回る +6.5%**。
- **query tile 64 行化**（fp8 prefill 経路）。32 行 tile では page の K/V staging と
  page あたり 3 回の `__syncthreads()` を 32 行でしか償却していなかった。tile を 64 行に
  すると warp 割当は 8 warp = 4 m-tile × 2 dim-half（各 warp が 128 dim）になり、
  `s_part` は 4×2、`p_tile` と行配列は 64 行分、LDS は 18,944 → **20,480 B**（1 block/CU の
  まま）。同一 device での A/B（`--device 2`、runs 3、fp8 KV）:
  | prompt | 32 行 tile | 64 行 tile | 差分 |
  | --- | --: | --: | --: |
  | 2k | 3047.6 | 3021.8–3048.8 | 測定誤差（±1%） |
  | 8k | 2824.4 | 2821.5–2850.7 | 0〜+0.9% |
  | 16k | 2513.7 | 2541.0–2564.1 | +1.1〜+2.0% |
  | 32k | 2097.7 | 2166.6–2179.7 | +3.3〜+3.9% |
  | 64k | 1601.1 | 1725.5–1727.5 | **+7.8%** |

  効果は page 数に比例して増える（barrier と staging を page 数ぶん償却するため）ので、
  64k で最大になる。64k 再現性は 0.1% 以内。
- **attention kernel の実効レート**（rocprofv3、64k、fp8 KV、同一 device・同一手順での
  前後比較）: leg 前 36.94s → leg 後 19.75s（42.5 TFLOP/s）→ **tile 64 行化後 16.84s
  （50.1 TFLOP/s、全体の 44.3%）**。per-launch 154.3 → **131.6ms（−14.8%）**、
  launch 数は不変で grid.x が半分（tile が 2 倍）。参照は fp8 で 150 TF（hot 8k chunk）
  としており、残差は依然 tiling / barrier 側にある。
- occupancy は律速ではない: fp8 prefill（32 行 tile 時点）の `__launch_bounds__` を
  2 → 4 blocks/CU にしても 64k は 1606 → 1610 t/s（+0.2%、測定誤差）。LDS 20,480 B は
  2 blocks/CU に収まらないため実効 1 block/CU で、tile 64 行化でもこの点は変わらない。
  内訳（rocprofv3、64k）: 8k は gemm 64.6% / attention 12.7% / gdn 12.3% だが、64k は
  **attention 44.3%（16.84s / 37.98s）/ gemm 40.9% / gdn 4.8%**。tile 64 行化で attention の
  比率は 48.3% → 44.3% に下がり、gemm が次の最大項目になった。
- **chunk サイズの効果**（64k prompt、同一 binary）: chunk 2048 = 1457 / 4096 = 1465 /
  8192 = 1472 PP t/s。**chunk を倍にしても +0.4-0.6%** で、64k のコストは chunk 数ではなく
  attention 側（O(n²) と leg）が支配している。参照の CHUNK も 8192 である。
- 診断は `SHELF_DFLASH2_SERVE_STATS=1` の `DFLASH2_STAGE`
  （round / draft_gpu / verify_gpu / …）と `DFLASH2_POSITION=`（256 トークン刻みの
  位置バケット: rounds / 受理 / round_ms）で取る。

再現手順は [methodology.md](methodology.md) の「カテゴリ別 serving 計測（BetterBench）」。

## fp8 KV の数値経路（2026-10-19 変更）

`--kv-cache-dtype fp8` の paged prefill attention は、2026-10-19 に QK^T / PV を
fp8 WMMA（e4m3）へ置き換えた。以降の fp8 KV の生成結果は、それ以前の fp8 KV の
結果と一致しない（bf16 KV とも一致しない。fp8 KV 自体が lossy なため）。

- ゲート: `test_paged_attention` の `fp8 PREFILL` ケース（同一の量子化 KV に対する
  attention 出力）。判定は行単位の相対 L2 と非有限値の有無で行う
  （[../developer/testing.md](../developer/testing.md) の「fp8 paged attention の単体テスト判定」）。
- 品質ゲート: `tools/quality_gate.py` と `shelf-bench quality` が、`vendor/BetterBench/corpus/v1`
  を chat template で token 化した corpus 上で BF16 KV と FP8 KV の perplexity / mean KL /
  top-1 一致率 / 生成 token 一致率 / 固定 end-task（reasoning + GSM8K）を測る
  （[../developer/testing.md](../developer/testing.md) の「quality regression」）。
  **2026-10-23 再計測では FP8 KV は数値品質ゲートを PASS する**
  （HF FP8、`tools/quality_gate.py run --skip-endtask --skip-gsm8k`、
  corpus 2203 token、R9700 1 枚）。

  | ctx | positions | ppl 差 | mean KL | top-1 | gate |
  | ---: | ---: | ---: | ---: | ---: | --- |
  | 128 | 2048 | -3.67% | 0.0130 | 0.9561 | PASS |
  | 1600 | 602 | -3.19% | 0.0167 | 0.9552 | PASS |

  採点位置を 1601 以降に固定した `--positions 600` の比較では、BF16 KV の
  ctx128 と ctx1600 の差が ppl +3.34% / mean NLL +0.0328 / KL 0.0060 であり、
  修正後の FP8 KV の差はこの範囲に収まる。prefill 性能は
  `pp --prompt-tokens 2048` の p50 が 2339.9 → 2327.8 tok/s（-0.52%、
  5 回交互測定、run 間変動 ±1%）。
  **2026-10-20 実測では FAIL していた**（HF FP8: ppl +28.12% @ctx128 /
  +29.96% @ctx1600、mean KL 0.223 / 0.697、top-1 0.816 / 0.723）。当時の原因は
  fp8 paged prefill kernel の欠陥（Q scale の bf16 切り捨て、PV の V 転置読み出し、
  P scale の denormal 逆数）であり、KV 保存形式の損失ではなかった。経緯と
  切り分けの実測は
  [../research/20261023/fp8_kv_prefill_quality_audit.md](../research/20261023/fp8_kv_prefill_quality_audit.md)
  を参照する。
- したがって **2026-10-19 以前に fp8 KV で取った生成品質・生成 ID の記録は、
  現在の build の数値とは別物**として扱う。bf16 KV（`shelf-bench tg` の既定）は
  影響を受けない。

## 正しさ契約

速度の数値は生成結果が一致している rep からのみ採用する。

- `shelf-bench tg` — `GREEDY_TOKEN_SUM=2818069` が全 rep で一致する。
  これは計測当時の fold 経由の系列であり、prompt ≥ 512 トークンでは fold 以前の
  ビルドと一致しないことがある（fold の経緯は
  `rnd/quantization/kernel_optimization_history.md` §7.95。一致した数値経路が要る場合は
  `SHELF_QWEN35_KERNEL_MODE=correctness`）。
- `shelf-compute`（DFlash2）— `GENERATED_IDS` の sha1 先頭 12 桁
  `87dd2adfa49f`、`DFLASH2_ROUNDS=46`、`DFLASH2_ACCEPTED_DRAFTS=209` が全 rep で一致。
  この sha は prompt `[760]`（"The" の1トークン）での軌道である
  （「DFlash2 speculative decode」節の計測条件の注記を参照）。
- `shelf-bench pp` — `--mode forward` のため生成結果を持たない。
- `test_qwen35_tp_e2e` — TP 節の measurement revision での値。現行 revision では未計測。

`GENERATED_IDS` の sha1 は `GENERATED_IDS=` 行の値部分（行末改行を含む）に対する
ものである。次のコマンドで再現できる。

```sh
grep '^GENERATED_IDS=' <output> | sed 's/^GENERATED_IDS=//' | sha1sum
```

## 計測コマンド

次のコマンドは計測当時の model dir（`Qwen3.8-27B-Quark-AWQ-MXFP4`）を指定している。
現在の build は BF16 / FP8 のみ対応のため、この model では再現できない。
加えて DFlash2 節の prompt 生成に使った `tools/reference/` の script は削除済みで、
この節は当時の計測の記録であり、現在実行できる手順ではない。

```sh
# pp2048
./build/shelf-bench pp --model-dir /opt/zen/work/shelf/models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1

# tg128
./build/shelf-bench tg --model-dir /opt/zen/work/shelf/models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --mode greedy --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1

# DFlash2 prose K7 256（prompt は当時に生成済みのものを使う）
./build/shelf-compute --model-dir /opt/zen/work/shelf/models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --dflash2-model-dir /opt/zen/work/shelf/models/Qwen3.8-27B-DFlash2 \
  --dflash2-drafts 7 --input-ids-file build/dflash2-gate51/real/real_prose/prompt.txt \
  --max-new-tokens 256 --max-seq-len 4096 --arena-gib 26 --temperature 0 \
  --dflash2-stats 1 --device 1
```

`real_prose/prompt.txt` は計測時に
`tools/reference/export_qwen38_dflash2_gate51_real.py`（削除済み。生成手順は
Git history を参照）の `--mode prompts` で生成したものである。
**カンマ区切り**の token id 列である（当時の `tests/rnd/dflash2`（削除済み）の
`read_prompt` と同じ）。
`shelf-compute` direct mode の `read_input_ids` はカンマ区切りを読めない
（上記「DFlash2 speculative decode」節の注記を参照）。
exporter の既定の `--language-dir models/Qwen3.8-27B` はホストに存在しないため、
当時は tokenizer を持つ model dir を指定した（計測当時の MXFP4 dir。
`tokenizer.json` は FP8 配布物と sha256 一致で同一）。

## Caveat

- GPU0 は別 workload と共有していることがある。再現時は `--device 1` 等で
  隔離された GPU を使う。
- 初回プロセスは GPU クロックのランプアップで pp が大きく遅く出るため、
  warmup で除外している。
- 絶対値は single-run ではぶれる。run 間安定性（中央値）で判定する。
- **絶対値は計測日の GPU 状態にも依存する。** 2026-09-28 計測の 2327.66 と
  2026-09-30 の 2300.9 は **−1.15%** 差があり、その間 GPU[1] の junction 温度は
  31°C → 57°C に上がっている。約 1% 程度の振れはコードではなく蓄熱・クロック
  による日時差と考えること。日付違いの値を直接比較しないこと。
- ここに無い性能値が必要な場合は、勝手に推測せず
  [methodology.md](methodology.md) の手順で新規計測し、日付・revision・commit とともに
  追記する。
