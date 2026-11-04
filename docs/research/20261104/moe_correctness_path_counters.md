# correctness path に落ちた kernel を数える機構を 1 本に統合

MoE decode の最適化で「どれが correctness 経路か」を 3 箇所で別々に数えていて、互相に矛盾した
値が出ていた。律速特定の段取り自体が止まったので、機構を統合した。

## 統合前の実態（同じ build、linear allowlist 登録直後）

同じ run に対して 4 本の口があり、うち 1 本が「fallback ゼロ」と出していた。

| 口 | 有効化 | 見えていたもの | 実測値 |
| --- | --- | --- | --- |
| `perf_stats_dump` JSON | `SHELF_QWEN35_PERF_STATS=1` + `--perf-stats-output` | `correctness_fallbacks` / `fallback_reasons` | **0** / 空 |
| `KERNEL_TRACE_FALLBACK` | `SHELF_QWEN35_KERNEL_TRACE=1` | KernelId 毎の fallback 件数 | RMS_NORM 243・PAGED_ATTENTION 30・EMBEDDING_LOOKUP 3・SAMPLING 1・OUTPUT_GATHER 1・LINEAR_BF16 1（tokens=2 run、合計 279） |
| `OPFALLBACK` 行 | `SHELF_OP_DEBUG=1` | dispatch index・kernel 名・rows | 同上の内訳が op 単位で出る |
| `semantic_timing.csv` | `SHELF_QWEN35_PERF_TIMING=1` | dispatch 毎の GPU time と optimized / correctness | PAGED_ATTENTION correctness 51.3%・RMS_NORM correctness 24.9% |

JSON だけが 0 を出していた理由: `DecodePerfStats::correctness_fallbacks` を増やしていたのは
`linear_record_fallback`（linear selector が geometry を拒否した箇所）だけで、executor の
fallback 地点が呼んでいたのは別名の `record_correctness_fallback(KernelId)`（`optimized_dispatch.hip`
の atomic 配列）だった。同名異物が 2 つあり、JSON 側は linear 限定だった。

`SHELF_QWEN35_PERF_STATS` の 0 を信じて「linear は終わり」と判断しかけ、実測では 32 token 区間の
76 % が依然 correctness interpreter 側にいた。

## 統合の方針

決定地点は `execute_program_range` の 1 箇所以外に無い（TP 経路も同じ関数へ range で入る）。
そこに kernel 単位の計数を置き、他を全部削った。

残したもの（`include/shelf/inference/execution/decode_perf_stats.h`）:

- `KernelDispatchCounters { logical, optimized, fallback, degraded, fallback_rows_min, fallback_rows_max }`
- 更新は `record_logical_dispatch` / `record_optimized_dispatch` / `record_fallback_dispatch` の 3 つ
- 不変式 `logical == optimized + fallback`（kernel 毎）
- dump JSON は kernel 名つきの table と `fallback_rows` の min / max

削ったもの:

- `SHELF_QWEN35_KERNEL_TRACE` と、それのために置いてあった 12 個の `g_*_optimized_count`
  atomic・`g_fallback_counts`・`atexit` dump・`*_optimized_hit` / `*_optimized_count` accessor
  （`try_launch_optimized` から計装の boilerplate が消え、kernel id の dispatch は switch になった）
- `SHELF_OP_DEBUG` と `OPFALLBACK` printf（rows は `fallback_rows` に集約）
- `FallbackReason` enum と `fallback_reasons[]`（linear しか入っていなかった）
- linear 専用の `LinearFamilyRef` / `linear_record_hit` / `linear_record_fallback` / `m_bucket`
  （kernel 単位の table に置換される）
- `kernel_id_calls[]`（`logical` に置換）

`degraded` は別建てにした。optimized kernel 自身の inline generic 版へ下がった 3 箇所
（`ROPE` の `F32Bf16Generic`、`GDN_RECURRENCE` の `F32`、fp8 activation quantize の
`!specialized`）は correctness interpreter を実行していないので、fallback に混ぜると
「interpreter に落ちた件数」という意味を失う。同じ table の別列に置いた。

per-op GPU time（`semantic_timing.csv`）と per-op host wall（`SHELF_OP_DUMP`）は落とした質問が
別（時間の所在）なので残した。ただし path の最適化 / correctness 判定は executor の同じ 1 箇所で
決まる値なので、JSON と食い違うことは無い。

## 統合後の実測（MoE、`shelf-bench tg --mode greedy --context 64 --tokens 32 --warmup 1`）

`SHELF_QWEN35_PERF_STATS=1`。測定区間は reset 後の decode 32 token（prefill は含まない）。

```
logical_dispatches 43360   optimized_operations 40416   correctness_fallbacks 2944   physical_kernel_launches 43392
EMBEDDING_LOOKUP   logical=   32  optimized=    0  fallback=   32  fallback_rows=[1, 1]
RMS_NORM           logical= 4192  optimized= 1600  fallback= 2592  fallback_rows=[1, 1]
PAGED_ATTENTION    logical=  320  optimized=    0  fallback=  320  fallback_rows=[1, 1]
```

- `RMS_NORM` 2592 = 81 / token × 32（40 層 × 2 + final）。`optimized` 1600 は rule table に
  既に在る geometry（per-group の `features:group_size`）で、hidden 2048 の per-feature norm が
  `rmsnorm_selector.cpp` の rule に無い。層別の対応付けは計測していない
- `PAGED_ATTENTION` 320 = 10 / token × 32（hybrid の full-attention 層数）。`paged_attention_selector.cpp`
  の rule は `(q_heads, kv_heads, head_dim)` = `(16,4,256)`・`(24,4,256)`・`(12,2,256)` で、
  MoE の `(16,2,256)` が無い。stderr の `outside measured envelope` 警告と一致する
- `EMBEDDING_LOOKUP` 1 / token（decode の token 1 本 lookup）
- `LINEAR_BF16` / grouped / router / combine は fallback ゼロ（allowlist 登録が効いている）

この table が次の登録対象（`RMS_NORM` @2048、`PAGED_ATTENTION` 16/2/256）を決める根拠になった。
登録には `shelf-bench rmsnorm` / `shelf-bench paged-attention` の実測を使う。

## 受け入れ

`tests/unit/test_decode_perf_stats.cpp`（`cpu;required`）が、非 linear kernel
（`PAGED_ATTENTION`）の fallback が kernel 毎に出ること、`degraded` が fallback に混ざらないこと、
`logical == optimized + fallback`、dump JSON が kernel 名・`fallback_rows`・`degraded` を出すこと、
`fallback_reasons` のような第二の内訳を出さないことを見る。

fail-first は tamper で実演した。`record_fallback_dispatch` が `LINEAR_BF16` だけ数える旧挙動に
戻すと 6 件失敗（`paged_attention fallback counted per kernel, not only for linear` ほか）、
戻すと PASS。
