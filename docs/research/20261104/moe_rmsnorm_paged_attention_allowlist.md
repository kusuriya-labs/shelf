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

## split-KV の q_per_kv=8 対応は未実施

`q_per_kv = 8` の split kernel は存在しないので、長い context の MoE decode は非 split kernel
（1 workgroup / (row, q_head)）で歩く。262144 context で 15.5 ms / 呼出、KV 交通量から見た帯域下限
は 4.2 ms / token（10 層）なので 3.7 倍の余裕がある。split-KV の `q_per_kv = 8` 対応は残件。

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

## 数値ゲート（teacher-forced A/B）

`shelf-bench quality` は MoE checkpoint を load できない（`qwen35_config.cpp` の
`read_qwen35_text_config` が dense 直行で、quality はまだ `LoadedModel` 経由でない。
`docs/developer/qwen35moe.md` の残件そのまま）。teacher forcing は `shelf-compute` 側で
base の生成列をプロンプトへ埋めて再現した。

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
