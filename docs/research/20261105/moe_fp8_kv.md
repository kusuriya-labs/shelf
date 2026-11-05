# MoE の fp8 KV cache（allowlist 登録と split-KV q_per_kv 8）

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（`(q_heads, kv_heads) = (16, 2)`、`head_dim 256`、
`page_tokens 16`）。目的は MoE で `--kv-cache-dtype fp8` を実用にすること。実験 ID `MOE-FP8-KV`、
結論 **ACCEPT**（§7）。

## 1. 着手時点の状態

selector の rule table から導出した到達可能経路（`docs/perf/methodology.md` §4 の方針）。

| kernel | bf16 KV (16/2/256) | fp8 KV (16/2/256) |
| --- | --- | --- |
| `PAGED_ATTENTION` decode | registered（rows 1..8192、visible 1..262144） | **rule 無し → correctness** |
| `KV_APPEND` | registered（rows 1..8192） | **rule 無し（`kv_heads 4` のみ）→ correctness** |
| split-KV | `q_per_kv` 4 / 6 / 8 | `q_per_kv` 4 / 6（8 の instantiation が無い） |

`kv_heads = 2` の fp8 rule も無いため、dense `Qwen3.8-27B-FP8 --tensor-parallel 2`（local kv_heads 2）
でも `KV_APPEND` は correctness 側だった。TP 経路には census 口（`--perf-stats-output`）が無く、
このTP側の影響は本記録では**未実測**として残す。

fallback のコスト感（診断値。`methodology.md` §2 のとおり p50 baseline には使わない）:
`shelf-bench paged-attention --variant correctness`（rows=1、visible 16384、bf16 KV）で
**1 launch 1,034,217 µs**。最適化版の同条件 700 µs 級、split なら 180 µs 級なので、
fp8 KV を選んだ MoE の decode は 1 層あたり秒オーダーで止まっていた計算になる。

## 2. 測定（kernel 単体、device 0、rows=1、`--check` 付き）

`shelf-bench paged-attention --variant optimized --out-dtype f32 --page-tokens 16
--q-heads 16 --kv-heads 2 --head-dim 256`。p50。

| visible | bf16 非 split | fp8 非 split | bf16 split16 | fp8 split16 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8.64 | 9.02 | — | — |
| 1024 | 48.44 | 54.72 | — | — |
| 2048 | 93.03 | 105.36 | — | 32.80 |
| 4096 | — | 199.74 | — | 52.56 |
| 16384 | 701.17 | 785.36 | 178.74 | 173.85 |
| 65536 | — | — | 763.73 | 739.16 |
| 262144 | — | — | 3217.00 | 3099.65 |

rows をまたいだ fp8（rows=1 の split16 / visible 64 の非 split）:

| 条件 | p50 |
| --- | ---: |
| split16 visible 4096 rows=8 | 160.78 µs |
| split16 visible 4096 rows=64 | 1369.37 µs |
| 非 split visible 64 rows=64 | 53.38 µs |
| 非 split visible 64 rows=2048 | 1428.02 µs |
| 非 split visible 64 rows=8192 | 5368.98 µs |

`kv-append`（fp8、`kv_heads 2`、`head_dim 256`、`page_tokens 16`、`--check` PASS）:
rows=1 で 6.4 µs、rows=64 で 5.8 µs、rows=2048 で 27.4 µs。

### 実測から見えた性質

- **fp8 の非 split は bf16 の非 split より遅い**（16384 で 785 vs 701 µs、12 %）。fp8 の
  可視域 traffic は bf16 の半分（rows=1 で 134 MB vs 268 MB）なので、fp8 非 split kernel は
  自分の roofline（636 GB/s で 211 µs）の 27 % でしか走っていない。split 経路は fp8 が bf16 と
  同等〜僅かに速い（16384 で 173.85 vs 178.74 µs）ので、**fp8 KV が実用になるのは split 側**である。
- visible 2048 以降は planner が splits=16 を選ぶため、fp8 は非 split より 3 倍速い
  （2048 で 105.36 → 32.80 µs）。

## 3. 実装

- `include/shelf/kernels/optimized/attention/paged_attention.h`:
  `paged_attention_fp8_split_supports` を 4 / 6 / 8 に拡張（bf16 と同じ集合）。
- `src/shelf/kernels/optimized/attention/paged_decode.hip`:
  `shelf_attention_paged_fp8_split_q8_full`（`attention_paged_split_fp8_body<8u, true>`）と
  `launch_attention_paged_fp8_split` の `q_per_kv == 8u` 分岐（full head / 非 full head）を追加。
  分岐は `q_per_kv == 8u && full_head` → `q_per_kv == 8u` の順で、後者を置かないと
  `full_head` の fallthrough で `<6u, true>` へ落ちる（§4）。
- selector rule 追加: `paged_attention_selector.cpp` に
  `{FP8_E4M3, F32 out, 16, 2, 256, 16, rows 1..8192, visible 1..262144}`、
  `kv_append_selector.cpp` に `{FP8_E4M3, 2, 256, 16, rows 1..8192}`。
- `shelf-bench tg` に `--kv-cache-dtype bf16|fp8` を追加（fp8 KV の decode を
  再現可能な口から測るため。`pp` / `quality` には既にあった）。

## 4. fail-first（pairing の張直し）

capability だけ 8 を通し、launcher の `q_per_kv == 8u` 分岐を両方外す（6u instantiation へ
fallthrough させる）tamper を入れて実行した。

- `test_paged_attention`: **3 FAIL**（`fp8 MOE DECODE qh=16 kvh=2 hd=256 page=64` の
  vis 8192 / 4096 / 64 の split16 比較、`max_rel=2.5e+01 / 7.3e+01 / 8.1e+34`）。
  restore 後は `49 passed, 0 failed, 0 skipped`。
- `shelf-bench paged-attention --kv-dtype fp8 --splits 16 --check`: **FAIL**（`max_rel=3.4e+01`、
  launch 自体は成功）。launch 成功のまま数値が誤る危険そのもので、planner 側では検出できない。

中間の誤解も残す: `q_per_kv == 8u && full_head` だけを外した tamper では**数値は正しかった**。
直後の `if (q_per_kv == 8u)` が `<8u, false>`（非 full head 版、`head_dim` を実行時に見る形）を
選ぶためで、instantiation の意味が同じだった。タイムアウト無しで見るなら p50 が
52.56 → 102.41 µs に変わる（full head 版の `kFullHead` 定数化が消える）。**分岐を 1 本だけ
消した tamper ではこの hazard は検出できない**。

## 5. E2E census と decode

`shelf-bench tg --mode greedy --context 64 --tokens 32 --warmup 1 --kv-cache-dtype fp8`
（device 0、`SHELF_QWEN35_PERF_STATS=1`）:

- `logical_dispatches 43360 == optimized_operations 43360`、**`correctness_fallbacks 0`**、
  `physical_kernel_launches 43392`（grouped 系の multi-launch ぶん。visible 64 < 2048 なので
  split は入らない）
- `PAGED_ATTENTION` / `KV_APPEND` は logical 320 = optimized 320、fallback 0
- `gpu_ms 495.81` / **64.54 tok/s**（bf16 KV の同一条件 497.27 ms / 64.35 tok/s と同等）
- anchor は bf16 の `GREEDY_TOKEN_SUM=2199` に対し fp8 で **14441**（生成列が変わる）。
  KV dtype の数値差なので、採否は §6 の ppl / KL で張る。

`--context 4096 --tokens 8 --prefill-chunk 2048`（fp8）:

- `logical_dispatches 10840 == optimized_operations 10840`、`correctness_fallbacks 0`、
  `physical_kernel_launches 10928`。差分 88 は grouped 8 + split attention 80（10 層 × 8 decode）で、
  **fp8 で split-KV が実際に選ばれている**証拠（bf16 と同じ内訳）
- `PAGED_ATTENTION` / `KV_APPEND` は logical 80 = optimized 80、fallback 0

decode 速度の A/B（同一条件を fp8 / fp8 / bf16 / bf16 の順で実行、device 0、p50):

| 条件 | fp8 run1 | fp8 run2 | bf16 run1 | bf16 run2 |
| --- | ---: | ---: | ---: | ---: |
| context 4096 / chunk 2048 / 8 token (tok/s) | 62.23 | 62.23 | 62.18 | 62.07 |

context 4096 の anchor は fp8 / bf16 とも `GREEDY_TOKEN_SUM=8`（8 token すべて id 1 の退化列）。
短生成なのでこの anchor は数値差の検出に使えない。

## 6. 数値の受け入れ

`shelf-bench quality` を teacher forcing で bf16 KV を baseline に fp8 KV を candidate として
compare した（MoE corpus は sha256 `9136d38b…` の 2191 token、`--positions 64`、device 0）。

| 条件 | baseline ppl (bf16) | candidate ppl (fp8) | delta | KL mean | top-1 | top-5 | logit max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context 128 / score start 129 | 75.1460 | 94.3507 | **+25.56 %** | 0.16576 | 0.8125 | 0.7781 | 6.949 |
| context 512 / score start 513 | 29.6400 | 27.4009 | -7.55 % | 0.07542 | 0.7969 | 0.8563 | 4.040 |
| context 2048 / score start 2049 / chunk 2048 | 4.5247 | 4.4925 | **-0.71 %** | 0.00695 | 0.9531 | 0.9625 | 3.163 |

context 128 の bf16 baseline ppl 75.146 は `docs/perf/current.md` の既存記録と一致（再現）。
同一条件の fp8 同士の compare（context 2048 / 64 positions）は `KL_MEAN=0`・`TOP1_AGREEMENT=1`
で、fp8 経路自体は決定論的。ppl の符号は context で揺れる（128 で増え、512 と 2048 では減る）ので、
採否の指標は KL と top-1 一致率を使う。

- context 2048 側（split-KV が走る領域）は、bf16 で測った split on / off の差
  （KL 0.00623・top-1 0.984・ppl 差 -2.70 %）や TP1 / TP2 の reduction 順序差
  （KL 0.00494・top-1 0.9375）と同じ帯にいる。
- context 128 側（fp8 非 split decode kernel と fp8 prefill leg の領域）は差が 1 桁大きい。
  これは (a) fp8 e4m3 の KV 量子化そのもの、(b) 非 split fp8 decode kernel の実装、
  (c) 低 confidence 領域（ppl 75）では logit の絶対差が ppl に効きやすい、のどれかを区別していない。
  同じ条件を dense `Qwen3.8-27B-FP8` で測って (c) / 共通特性かを切る（§6-1）。

### 6-1. dense `Qwen3.8-27B-FP8` との対比（同じ条件）

同じ `quality` 経路を dense（corpus sha256 `99fe6aa9…`、2203 token、`--positions 64`、device 0）で
測った。dense の `(q_heads, kv_heads)` は 24 / 4（`q_per_kv` 6）で、context 128 では fp8 非 split
decode kernel を通る。

| 条件 | baseline ppl (bf16) | candidate ppl (fp8) | delta | KL mean | top-1 | top-5 | logit max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense context 128 | 12.3659 | 11.3385 | -8.31 % | 0.00504 | 0.9375 | 0.9563 | 3.938 |
| dense context 2048 | 5.8388 | 6.1002 | +4.48 % | 0.00724 | 0.9219 | 0.9656 | 2.482 |

fp8 KV の短 context 劣化は **dense では起きない**（dense context 128 の KL 0.00504 は
reduction 順序の帯と同じ）。MoE の context 128（KL 0.16576）だけが 1 桁外れている。

### 6-2. dump からの行別分布（`--dump` 64 行）

行別の KL（baseline bf16 に対する candidate fp8）と logit 絶対差の中値。

| 条件 | KL 中値 | KL p90 | KL max | KL > 0.05 の行 | `|Δlogit|` 中値 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE context 128 | 0.1082 | 0.4487 | 0.8405 | 41 / 64 | 3.15 |
| MoE context 512 | 0.0654 | 0.1337 | 0.3331 | 43 / 64 | 2.55 |
| MoE context 2048 | 0.0048 | 0.0154 | 0.0352 | 0 / 64 | 0.93 |
| dense context 128 | 0.0013 | 0.0200 | 0.0386 | 0 / 64 | 0.86 |
| dense context 2048 | 0.0026 | 0.0168 | 0.0812 | 2 / 64 | 0.57 |

MoE context 128 は数行だけが飛んでいるのではなく、**64 行の大半で KL が高い**。離散的な
例外（1 行だけ大きい）ではない。

### 6-3. 切り分けの状態

- kernel の pairing / geometry 誤りは疑わないでよい。同じ geometry（16 / 2 / 256）で
  `shelf-bench paged-attention --kv-dtype fp8 --check` は rows=1 の可視 2048 / 4096 / 16384 で
  PASS（相対 2e-3）、rows=8 / 64 の split16 も PASS。非 split kernel
  （`v_decode_fp8::attention_paged_fp8_impl`）は `q_per_kv` を実行時に
  `q_heads / kv_heads` から導く形で、4 / 6 / 8 を固定していない。
- MoE の KL は context とともに 0.16576 → 0.07542 → 0.00695 と減る。dense の KL は
  context 128 / 2048 で 0.00504 / 0.00724 と flat。**MoE 側だけ短 context で 1 桁悪い**。
- ここで MoE の 512（非 split）と 2048（split）の差が 10 倍あるので、原因として
  (a) context の confidence（ppl 75 → 29.6 → 4.52）と router の離散性の組み合わせ、
  (b) fp8 非 split decode kernel と MoE geometry の組み合わせ、がまだ分離できていない。
  判別には context 1024（非 split）と 4096（split）の A/B を足して、KL が境界で飛ぶのか、context に滑らかに追随するのかを見る（§8）。
- (a) を直接検証するには選択 expert 列を host へ出す診断口が無い。plan の `group_count` と同じく
  production 側に口が無いため、現状は仮説どまり。

## 7. 採否

**ACCEPT（ただし fp8 KV の短 context 数値差を condition として明記）**

- 速度: fp8 KV は bf16 KV と同等（context 64 で 64.54 / 64.35 tok/s、context 4096 で
  62.23 / 62.18 tok/s）。rule 登録前は `PAGED_ATTENTION` / `KV_APPEND` が correctness
  fallback で、実用上の値は出ていなかった（§1）。
- 容量: `shelf-compute --model-dir <MoE> --max-seq-len 8192 --max-num-batched-tokens 2048
  --prefix-cache 0` の auto memory plan で、KV budget は bf16 が 527,968 tokens、fp8 が
  **1,039,728 tokens**（同一 arena capacity 30.78 GiB・target weights 20.53 GiB）。ちょうど 2 倍。
- 数値: 長 context（2048 以上）は bf16 に対して ppl -0.71 %・KL 0.00695・top-1 0.9531 で、
  reduction 順序の帯（TP1/TP2・split on/off）と同じ。
  **短 context は KL 0.16576（128）/ 0.07542（512）で top-1 一致 0.81 / 0.80**まで落ちる。
  同じ条件の dense は KL 0.005 に収まる。MoE で fp8 KV を短 context に使うときはこの差を前提にする。
- 採否の方向: 登録は撤収しない（撤収すると fp8 KV は MoE で使えないまま）。短 context
  差の原因切り分けは §8 の harness 限界（選択 expert を数える口が無い）を潰す方へ移す。

## 8. 未完了・次の一手

- **短 context 差の判別実験**: MoE quality を context 1024（非 split）と 4096（split）で
  bf16 / fp8 の A/B し、KL が split 境界で飛ぶのか、context に滑らかに追随するのかを見る
- MoE の短 context fp8 差を機構として検証する口: router の選択列を host へ出す診断口
  （`test_moe_router` の d2h 口を production 側へ移す）が無いため、現状は仮説どまり
- fp8 非 split decode kernel の roofline 不足（§2 の実測から可視 16384 で自分の floor の 27 %）。
  bf16 非 split より遅い（785 vs 701 µs）ので、split の閾値をまたぐと fp8 が bf16 に勝つ
- `arena create` 失敗経路の後始末が壊れている（§10）

## 9. harness と runtime の既知の問題（本次元で確認、本変更の原因ではない）

- `shelf-bench kv-append` は `--rows 4096` 以上で **segfault** する（bf16 / fp8 双方、本変更前から
  再現）。rows の上限実測は 2048 まで。rule の rows 上限は既存 bf16 rule と同じ 8192 のまま
  登録したので、この行の実測根拠は bf16 側から継承している（production の MoE prefill chunk は
  4096 まで）。
- `shelf-bench paged-attention --variant prefill --kv-dtype fp8 --check` は **geometry を問わず
  FAIL**（`24/4`・`12/2`・`16/4`・`16/2` すべて）。fp8 prefill kernel は Q を行単位 amax で
  e4m3 化した入力を要求するためで、bench がその入力を準備していない
  （`tests/kernels/optimized/test_paged_attention.hip` は自分で量子化して検証している:
  `fp8 PREFILL qh=4 kvh=2 hd=256`）。fp8 prefill の数値検証は unit test と E2E で張る。
  なお `test_paged_attention` へ `fp8 PREFILL qh=16 kvh=2 hd=256 page=16 rows=64` を追加した。
- `shelf-compute --model-dir <MoE> --max-num-batched-tokens 2048 --max-seq-len 8192`（auto arena）は
  bf16 / fp8 双方で `MEMORY_PLAN` の KV auto-fit 後に
  `insufficient memory: arena capacity exceeded: capacity 30.78 GiB used 30.72 GiB requested 0.06 GiB`
  で `runtime create failed` する。main（本変更なし）でも同じで、auto reserve の
  `working set reserve 0.06 GiB (row bucket 16)` 推定と KV fit の取り分が合っていない。
  本次元では触っていない。
- arena 確保に失敗した後始末が壊れている。`test_qwen35moe_lowering` を VRAM の空いていない
  device で実行すると `FAIL: arena create` のあとに
  `arena destructor: free failed: ... out of range: target device is out of range` と
  `munmap_chunk(): invalid pointer` で abort する（リソースライフサイクルルールが狙われる箇所）。
- GPU の空きが足りない状態で `gpu1` test を走らせると `insufficient memory` で FAIL する
  （本変更の FAIL ではなく、`test_qwen35moe_model_load` / `lowering` / `executor` が該当）。
  bench と同時実行しないこと。
