# MoE decode: RMS_NORM @2048 と PAGED_ATTENTION 16/2/256 の registered 化

`docs/research/20261104/moe_correctness_path_counters.md` の kernel 単位 counter で、linear
登録後の MoE decode に残る correctness path が `RMS_NORM` 2592 / `PAGED_ATTENTION` 320 /
`EMBEDDING_LOOKUP` 32（32 token 区間）だと分かった。この 2 つを計測して registered にした記録。

## 計測口

```sh
shelf-bench rmsnorm --variant optimized|correctness --rows ... --features 2048 --group-size 2048 \
  --weight-layout per_feature --input-dtype bf16 --output-dtype bf16 --weight-dtype bf16 \
  --mode one_plus [--check]
shelf-bench paged-attention --variant optimized|correctness --rows ... --visible ... \
  --q-heads 16 --kv-heads 2 --head-dim 256 --page-tokens 16 --kv-dtype bf16 --out-dtype f32 [--check]
```

`--check` は production correctness path との突き合わせ。GPU は R9700 device 0（device 0 / 1 idle）。

## RMS_NORM（hidden 2048・per-feature・one_plus）

| rows | optimized p50 µs | correctness p50 µs | 倍率 | `--check` |
| ---: | ---: | ---: | ---: | --- |
| 1 | 5.07 | 287.58 | 56.8× | PASS |
| 8 | 4.99 | 288.81 | 57.9× | PASS |
| 64 | 5.04 | 286.98 | 56.9× | PASS |
| 128 | 5.14 | 430.50 | 83.8× | PASS |

rows に依らず optimized は 5 µs 台。decode 81 本 / token なので correctness 側 23.3 ms → optimized
0.41 ms（推算）。rows 128 で correctness が 430 µs へ跳ねる（correctness path は rows で増える）。

## PAGED_ATTENTION（q_heads 16・kv_heads 2・head_dim 256・page 16・bf16 KV / f32 出力）

decode（rows=1、非 split）:

| visible | optimized p50 µs | correctness p50 µs | 倍率 | `--check` |
| ---: | ---: | ---: | ---: | --- |
| 64 | 8.39 | 4493.33 | 535× | PASS |
| 256 | 16.17 | 16478.96 | 1019× | PASS |
| 1024 | 49.97 | 65120.71 | 1303× | PASS |
| 4096 | 184.72 | 260755.85 | 1412× | PASS |
| 8192 | 353.96 | - | - | PASS |
| 16384 | 713.86 | - | - | - |
| 32768 | 1570.42 | - | - | - |
| 65536 | 3970.00 | - | PASS (8192 / 65536) | - |
| 131072 | 7868.30 | - | - | - |
| 262144 | 15509.71 | - | - | - |

prefill kernel（rows > 1）:

| rows | visible | optimized p50 µs | `--check` |
| ---: | ---: | ---: | --- |
| 64 | 512 | 179.14 | PASS |
| 128 | 2048 | 1385.81 | PASS |
| 512 | 2048 | 6149.12 | PASS |
| 2048 | 2048 | 23886.59 | PASS |
| 4096 | 8192 | 193160.43 | PASS |
| 8192 | 8192 | 391503.26 | PASS |

correctness 基準（rows 64 / visible 512）は 39330 µs で、optimized（179 µs）の 220 倍。
`kMeasuredMaxRows`（8192）まで `--check` を取ったので、rows は production の chunk 上限まで
registered にできる。

## split-KV は q_per_kv 4 と 6 の kernel しかない

`(16, 2)` は `q_per_kv = 8`。split-KV 経路の kernel は `q_per_kv` が 4 / 6 でしか起動しない
（`src/shelf/kernels/optimized/attention/paged_decode.hip` の bf16 split / bf16 split reduce /
fp8 split の 3 箇所が `q_per_kv != 4u && q_per_kv != 6u` で `hipErrorInvalidValue`）。

問題はこれが `hipErrorNotSupported` **ではない**こと。`try_launch_paged_attention` は
`hipErrorNotSupported`だけを `NotApplicable`（＝correctness fallback）へ訳し、その他は
`Status::hip_error` として実行を失敗させる。したがって selector へ `16/2/256` を足すだけでは、
`max_visible_tokens >= kPagedAttentionSplitMinVisible`（2048）のとき plan が `splits = 16` を選び、
**長い context で実行が落ちる**。

そこで kernel capability をヘッダへ 1 箇所書き、planner と kernel 両方から読む形にした。

- `include/shelf/kernels/optimized/attention/paged_attention.h`
  `inline constexpr bool paged_attention_split_supports(uint32_t q_per_kv)`（4 / 6）
- `paged_decode.hip` の 3 箇所の gate をこの関数へ置換
- `plan_paged_attention` は `paged_attention_split_supports(q_heads / kv_heads)` が真のときだけ
  splits を選ぶ。偽なら非 split decode kernel（1 workgroup / (row, q_head) で可視域を歩む）

`shelf-bench paged-attention` の `--variant optimized` は既定で非 split を通すので、上の表は
`q_per_kv = 8` で実際に使われる経路の実測である。

## dense 側は q_per_kv 4 / 6 で split が実際に走っている（確認実測）

`q_per_kv = 8` が必要なのは対象 checkpoint のうち grouped expert（MoE の 16/2）だけである。
dense は TP で q / kv が同じ割合で割れるので `q_per_kv` は shard 不変。

| checkpoint | `(q_heads, kv_heads)` TP1 / TP2 | q_per_kv |
| --- | --- | ---: |
| Qwen3.5-9B | 16/4 | 4 |
| Qwen3.8-27B-FP8、Qwen3.8-27B-Quark-AWQ-MXFP4 | 24/4、TP2 で 12/2 | 6 |
| Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 | 16/2（TP 非対応） | 8 |

DFlash2 の drafter（32/8 = 4、head_dim 128）は `PAGED_ATTENTION` selector を通らない
（`src/shelf/models/qwen35/kernels/dflash2/attention.hip` 自前 kernel）。

bench（rows=1、visible 16384、bf16 KV、f32 出力、`--check` 付き）:

| geometry | splits=1（非 split） | splits=16 | 倍率 | `--check` |
| --- | ---: | ---: | ---: | --- |
| 24/4 | 758.42 µs | 168.76 | 4.5× | 両方 PASS |
| 12/2 | 704.35 µs | 147.76 | 4.8× | 両方 PASS |
| 16/4 | 738.39 µs | 175.16 | 4.2× | 両方 PASS |

E2E でも split launch 本数で確認した。counter には split 情報が無いので、
`physical_kernel_launches - logical_dispatches` を読む（split は split kernel + reduce kernel
の 2 launchなので、split を選んだ attention 1 呼出ぶん差分が増える）。`shelf-bench tg`
Qwen3.8-27B-FP8 / TP1 / tokens 8:

| run | logical | physical | 差分 | `PAGED_ATTENTION` logical |
| --- | ---: | ---: | ---: | ---: |
| `--context 4096`（`kPagedAttentionSplitMinVisible` 2048 超） | 13224 | 13360 | 136 | 128 |
| `--context 1024`（閾値未満の対照） | 13224 | 13232 | 8 | 128 |

差分 136 - 8 = 128 = 16 full attention 層 × 8 decode step で、全 decode attention が split を
通っていることと一致する。`GREEDY_TOKEN_SUM` は 40712 / 36112（context が違うので値は比較しない）。

## 登録直後に見えた欠陥: splits の決定が 2 箇所あった

rule を載せた状態で `shelf-bench tg --context 4096`（MoE）を実行すると

```
HIP error (paged-attention optimized launch): invalid argument
```

で落ちた。切り分けのため `launch_paged_attention_physical` の失敗時に plan の中身を出す一時
計装をした（`SHELF_OP_SYNC=1` でも op index は出ない。launch 自体がエラーを返すため）。

```
PA_LAUNCH_FAIL i=0 variant=3 splits=16 rows=1 qh=16 kvh=2 hd=256 pt=16 pages=258 stride=258 ...
PA_LAUNCH_FAIL i=1 variant=7 splits=16 ...
```

`splits=16` が立っている。原因は **split を選ぶ決定が 2 箇所**あったこと。

- `plan_paged_attention`（`plan.splits` を決める。capability gate を最初に入れたのはここ）
- `append_decode_region`（`ctx.decode_attn_partials` と `max_visible_tokens` だけで `splits = 16u`
  を**独立に**計算し、`plan.splits` を読んでいなかった）

physical plan を作る側が独自の決定を持っていたので、gate は実行経路に効かなかった。kernel は
`q_per_kv = 8` を `hipErrorInvalidValue`（`hipErrorNotSupported` ではない）で弾くため、dispatch 側は
それを fallback に訳さず、実行失敗になった。

修正は決定の一本化。`plan_paged_attention` が `paged_attention_split_supports(q_per_kv)` を見て
`splits` を決め、`append_decode_region(a, region, ctx, out, splits)` へ渡す。region 単位で必要な
条件（tree 行は split を使わない）だけを受け手側に残した。

- `tests/unit/test_paged_attention_plan_splits.cpp`（`cpu;required`）で plan レベルの決定を固定
  （`q_per_kv` 4 / 6 は splits=16、8 は non-split、可視 < 2048・partials 無し・tree 行は non-split）
- fail-first は tamper で実演: plan の capability gate を外すと
  `q_per_kv 8 has no split kernel and stays non-split` など 2 件が失敗

### 実測（最終 binary）

| run | 結果 |
| --- | --- |
| MoE `tg --context 4096 --tokens 8` | 51.37 tok/s。`PAGED_ATTENTION` optimized 80 / fallback 0、`RMS_NORM` optimized 1048 / fallback 0 |
| 27B dense `tg --context 4096 --tokens 8` | 18.12 tok/s。`GREEDY_TOKEN_SUM=40712` は gate 前後で無変更。`physical_kernel_launches` 13360 = `logical_dispatches` 13224 + 136（attention op 毎に split + reduce の 2 本）で、dense の split 経路は維持されている |

`physical_kernel_launches` と `logical_dispatches` の差が split の有無を見る口になった。

## split-KV の q_per_kv=8 を probe して却下した

bf16 の split / split reduce 本体を `kQPerKV = 8` で instantiation し、gate を通して
`shelf-bench paged-attention --splits` で測った（rows=1、bf16 KV、f32 出力、`--check` 付き）。

| visible | splits=1（非 split） | splits=4 | splits=8 | splits=16 | `--check`（splits ≥ 2） |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 186.58 µs | 172.78 | 92.42 | 54.59 | **FAIL** |
| 16384 | 722.94 µs | 664.28 | 334.82 | 180.06 | **FAIL** |
| 65536 | 3982.16 µs | 3365.30 | 1741.56 | 764.28 | **FAIL** |

潜力は 3.4〜5.2 倍。しかし splits=16 / visible 4096 の不一致は `mm=4092 / 4096 要素`・
`max_rel=8.3`で、launch 失敗ではなく（bench は launch error を返す経路を持つ）**full head の
split 本体 `attention_paged_split_bf16_body<8u,true>` が `kQPerKV = 8` で構造的に誤っている**。
同じ bench で production が使う `q_per_kv = 4 / 6`（16/4・24/4・12/2）の splits 4 / 16 は
`--check` PASS なので、bench 側の検証ではなく kernel 側の実差である。よって 8u の split は入れて
いない。現状は非 split decode kernel が正しく、`PAGED_ATTENTION (16,2,256)` はその envelope で
registered 済み。

follow-up に残す条件: split 本体は `q_lds[kQPerKV*256]` と `acc_lds[kSplitWaves*kQPerKV*256]`
を static LDS に取る（kSplitWaves=4 なので 8u で 8 KB + 32 KB = 40 KB、6u で 30 KB）。単に
template を instantiate するだけでは壊れるので、reduce 側の `qh_per_wg` と同じ発想で q head を
workgroup に分割し、partials layout もそれへ合わせる必要がある。

もう 1 つ条件がある。`paged_attention_split_supports` は **kv dtype を見ていない**ので、
bf16 と fp8 で同一の capability 表を読んでいる。8u を bf16 だけへ追加すると、fp8 の geometry で
planner が splits を選び、fp8 split kernel が `hipErrorInvalidValue` を返して**実行失敗**になる
（`hipErrorNotSupported` でないため fallback にもならない。契約は
`docs/developer/kernels.md` の split-KV decode kernel 節）。8u を入れるなら capability を
dtype 別に分ける変更が前提である。probe 中は実際に capability を bf16 / fp8 の 2 本へ分割したが、
8u を却下したので元へ戻してある。

価値の出る範囲は visible ≥ 2048 の MoE decode に限られる。visible 4096 では attention が
1 呼出 186.6 µs × 10 層 = 1.87 ms / token で token 全体（19 ms 台）の 10 % 程度、
visible 65536 では 39.8 ms / token になって律速側へ寄る。

## なぜ q_per_kv=8 の split-KV を今実装しないのか（採否判断）

MoE decode の context 4096（`tg --context 4096 --tokens 8 --warmup 1 --device 1`、
`SHELF_QWEN35_PERF_TIMING=1`）の per-op GPU time。timed window 151.516 ms / 8 token：

| kernel | 件数 | total ms | mean µs | share |
| --- | ---: | ---: | ---: | ---: |
| `LINEAR_BF16` | 2363 | 65.694 | 27.80 | 43.4 % |
| `PAGED_ATTENTION` | 60 | 14.156 | 235.94 | 9.3 % |
| `LINEAR_MXFP4_W4A8_GROUPED` | 484 | 11.191 | 23.12 | 7.4 % |
| `MOE_COMBINE` | 241 | 9.460 | 39.25 | 6.2 % |
| `RMS_NORM` | 792 | 7.057 | 8.91 | 4.7 % |
| `MOE_PLAN_SCAN` | 242 | 6.478 | 26.77 | 4.3 % |
| 他（`ACTIVATION_QUANTIZE_MXFP4_ROWREF` 3.1 %、`SWIGLU` 2.6 %、`RESIDUAL_ADD` 2.5 %、`GDN_RECURRENCE` 2.5 %、`MOE_ROUTER` 2.3 %、`L2_NORMALIZE` 2.0 %） | | | | |

（`gpu_tokens_per_sec` 35.18、`gpu_ms` 227.3910 ms / 8 token。この breakdown を取るための
1 run の条件で、decode の現在値ではない。現在値は `docs/perf/current.md`）

`16/2` の split counterfactual（上記 probe の表）で visible 4096 は 186.58 → 54.59 µs
（3.4 倍）。したがって attention 14.156 ms は 4.16 ms になり、**timed window は 6.5 % 短くなる
（9.996 ms の節約）**。context 64（59.01 tok/s の計測点）では 8.39 µs × 10 層 = 0.084 ms で、
attention は token 時間の 0.5 %しかない。

帯域側の論点も同じ方向を向く。KV 交通は 2 (K, V) × 2 kv_heads × 256 dim × 2 B × 10 層 =
**20 KB / context token / decode token** で、weight traffic 4.4 GB/token と等しくなるのは
context 約 215k。context 4096 では KV は 83 MB で weight の 2 % しかないため、
この context での split-KV の得は帯域ではなく **latency / occupancy** に限られる。

したがってこの時点での順序は次の通り。

1. **Amdahl**: 最大項目は `LINEAR_BF16` 43.4 %、grouped GEMM + `MOE_COMBINE` +
   `MOE_PLAN_SCAN` で 17.9 %。attention は 9.3 % で、その最大得は 6.5 %。
2. **roofline の順序**: weight traffic 4.4 GB/token と DRAM 636 GB/s から roofline は
   145 tok/s。現状 59.01 tok/s（context 64）で残り 2.4 倍があり、attention を消しても
   この天井は動かない。
3. **実装コスト**: template を足す話ではない。split 本体は `q_lds[kQPerKV*kHeadLimit]`
   （8u で 8 KB）と `acc_lds[kSplitWaves*kQPerKV*kHeadLimit]`（32 KB）、
   `acc[kQPerKV][kElemsPerLane]`（8u で 64 float）を q head ごとに直接載せる形で、
   8u で static LDS 計 ~40 KB。q head を workgroup 間で分割する（reduce 側の
   `qh_per_wg` と対応する）設計変更と partials layout の整合が前提。**naive な
   instantiation は全要素系統的に誤り、原因の切り分けは未実施**。
4. **capability の前提**: `paged_attention_split_supports` は kv dtype を見ていないので、
   bf16 だけ 8u を足すと fp8 の geometry で planner が splits を選び、
   `hipErrorInvalidValue` で実行失敗する（fallback にならない）。dtype 別への分割が先に要る。

この判断は「入れない」ではなく「条件が揃うまで入れない」で、条件は上記 probe 節の 2 つ
（q head の workgroup 分割と partials layout、capability の dtype 分割）。context が
16k 〜 64k に達すれば attention は 7.14 ms（visible 16384）〜 39.7 ms（visible 65536）/ token
になり、この順序は逆転する。

## 登録

- `rmsnorm_selector.cpp`: `{BF16, BF16, BF16, PerFeature, mode 0, features 2048, group 2048, rows 1..kMeasuredMaxRows}`
- `paged_attention_selector.cpp`: `{BF16 KV, F32 out, q_heads 16, kv_heads 2, head_dim 256, page_tokens 16, rows 1..kMeasuredMaxRows, visible 1..262144}`
  （`kMeasuredMaxVisible16x2 = 262144` を追加。model の max seq len まで実測したため cliff は無い）
- 追加した selector 契約は `tests/unit/test_qwen35_rmsnorm_selector.cpp` と
  `tests/unit/test_qwen35_paged_attention_selector.cpp` に固定した（measured max まで
  `Optimized`、1 超えた rows / visible は `Correctness`、fp8 KV の `16/2/256` は未計測なので
  `Correctness`）。fail-first は rule 2 本を削って実演: rmsnorm 2 件・paged attention 3 件が
  失敗し、`paged_attention_split_supports(8u) == false` の assertion も planner の gate を
  外すと意味を失うため、capability の所在を test に置いてある。

## EMBEDDING_LOOKUP @2048（decode に残った最後の fallback）

kernel 単位 counter で残った最後の 1 本 / token が `EMBEDDING_LOOKUP` だった。
`embedding_selector.cpp` の rule は hidden 2560 と 5120 のみで、hidden 2048 が無かった。

```sh
shelf-bench embedding --variant optimized|correctness --storage bf16 --vocab 248320 --hidden 2048 \
  --rows 1,64,1024,2048
```

| rows | optimized p50 µs | correctness p50 µs | 倍率 |
| ---: | ---: | ---: | ---: |
| 1 | 3.87 | 145.04 | 37.5× |
| 64 | 3.93 | 144.18 | 36.7× |
| 1024 | 19.67 | 219.20 | 11.1× |
| 2048 | 14.34 | 250.05 | 17.4× |

decode 1 本ぶん（145 → 3.9 µs）なので速度への寄与は 17 ms/token に対して 0.8 % と小さい。
意味は**decode の correctness fallback がゼロになった**ことで、`SHELF_QWEN35_PERF_STATS=1` の
census が `logical_dispatches == optimized_operations`（43360 / 43360、fallback 0）を
受け入れ条件にできる。

`shelf-bench embedding` には `--check` が無いので、数値は
`tests/kernels/optimized/test_embedding.hip` に `bf16-hidden-2048` と `bf16-hidden-2048-pad`
（correctness path と CPU 参照の突き合わせ）を追加して固定した。selector 側の境界は
`tests/unit/test_qwen35_embedding_selector.cpp`（rows 1 / 2048 は Optimized、8193 は
Correctness）。



## 数値ゲート（teacher-forced A/B）

この当時 `shelf-bench quality` は MoE checkpoint を load できなかった（`qwen35_config.cpp` の
`read_qwen35_text_config` が dense 直行で、quality は `LoadedModel` 経由でない）。teacher
forcing は `shelf-compute` 側で base の生成列をプロンプトへ埋めて再現した。
（`quality` の LoadedModel 配線は後日 `402ef9e` で入り、MoE の teacher-forced ppl が
`docs/perf/current.md` に載った。）

- base = rule 2 本を削った build、cand = rule ありの build（同一 tree のそれ以外）
- prompt = `ids_moe.txt`（6 token）+ base が生成した 8 token = 14 token、
  `--max-new-tokens 8 --seed 42 --max-num-batched-tokens 64 --max-concurrent-requests 1`
- base `GENERATED_IDS=220,17,4007,13,271,760,1118,5286`
  cand `GENERATED_IDS=220,17,4007,13,271,760,1414,314`（先頭 6 本一致）
- `--dump-logits` 9 行の突き合わせ: top-1 一致 **7/9**

| row | 位置 | top1 base → cand | rel_l2 | margin base / cand |
| ---: | --- | --- | ---: | --- |
| 0 | prefill 末尾 | 220 → 220 | 0.05652 | 5.21 / 5.09 |
| 2 | decode 1 | 17 → 17 | 0.16834 | 0.89 / 0.65 |
| 3 | decode 2 | 4007 → 4007 | 0.17927 | 2.33 / 1.91 |
| 4 | decode 3 | 13 → 13 | 0.14087 | 0.11 / 0.74 |
| 5 | decode 4 | 271 → 271 | 0.11909 | 2.52 / 2.56 |
| 6 | decode 5 | 760 → 760 | 0.10553 | 0.08 / 0.35 |
| 7 | decode 6 | 1118 → 1414 | 0.08112 | 0.23 / 0.27 |
| 8 | 分岐後 | 5286 → 314 | 0.79589 | - |

- 入力列が同一だった row 0〜6 は top-1 がすべて一致（rel_l2 0.057〜0.18）
- row 7 は cand 側で top1 / top2 の差が 0.27 のほぼ同着で反転している。分岐はここで起きた
- row 8 以降は入力列が違うので数値差の比較対象ではない
- `DECODE_TOKENS_PER_SEC` はこの run で 17.82 → 56.52
- HF との絶対照合（`--hidden-taps` 層別比較）は依然未実施
