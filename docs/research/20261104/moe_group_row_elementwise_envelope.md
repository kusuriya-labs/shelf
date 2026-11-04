# MoE の group 行数が elementwise envelope を超えると結果が静かに壊れていた

`shelf-bench tg --context 2048 --prefill-chunk 1024`（MoE）が

```
host program execution failed: status=6 (a kernel reported a non-finite value: the run is stopped)
```

で落ちた。同じ context で `--prefill-chunk 512` は通る。production の `shelf-compute` 既定は
`--max-num-batched-tokens 2048` なので、**既定設定で MoE の prefill が壊れていた**可能性が高い
段階の話。解決の経緯と計測。

## 失敗時に内訳が出るようにした

`SHELF_QWEN35_PERF_STATS=1` の census は成功時にしか書かれていなかった。`shelf-bench tg` に
失敗時 dump（`StatsOnExit`）を足して、壊れた run の内訳を読んだ。

```
logical 1355  optimized_ops 1310  fallbacks 45
SWIGLU  logical=80  optimized=40  fallback=40  fallback_rows=[1024, 1024]
```

40 層のうち shared expert 側（token rows=1024）は optimized、**routed expert 側（group rows）が
correctness 経路に落ちていた**。`fallback_rows` に 1024 と出ているが、これは executor が
`ctx.actual_rows` を記録していたもので、実際の行数は group 行数だった。

## correctness interpreter は group 行数を数えられない

`src/shelf/kernels/correctness/detail/dispatch_env.h` の `value_rows()` は

```cpp
if (value.row_domain == ValueRowDomain::OUTPUT_ROWS) return env.context->num_outputs;
return env.context->actual_rows ? env.context->actual_rows : 1u;
```

で、`GROUPED_ROWS` の領域を持たない。つまり interpreter は group 平面のうち token 行数ぶん
（1024 行）だけ処理し、残りの 11264 行を stale のまま残す。grouped GEMM は group の全行
（padding 行を含む）を読むので、stale が expert 出力へ移り、combine 以降で non-finite になる。
router の `NUMERIC_ERROR` 契約がそれを検知して run を止めていた（止まらなければそのまま
出力されていた）。

fixtures が通っていた理由も同じで、arena の平面がゼロ埋めだったため stale が 0 だった。
`test_qwen35moe_executor` は elementwise が fallback していることに気づいていなかった。

## 行数の算術と kernel の上限

`grouped_row_capacity(rows, {experts, top_k}) = (rows * top_k / 16 + experts) * 16`。
MoE（experts 256・top_k 8）の bucket 別 capacity:

| token rows | group 行数 capacity |
| ---: | ---: |
| 512 | 8192 |
| 1024 | 12288 |
| 2048 | 20480 |
| 4096 | 36864 |
| 8192 | 69632 |

elementwise kernel は `blockIdx.y` が行インデックスなので **grid.y ≤ 65535** が上限。
`shelf-bench elementwise` で rows=69632 の optimized が 0.51 µs だった（= launch 失敗で仕事を
していない）。rows が 8192 を超えた瞬間に interpreter 側が壊れ、65535 を超えると kernel 自体が
起動しない、というのが実態だった。

## 計測

```sh
shelf-bench elementwise --variant optimized|correctness --profile swiglu --features 512 \
  --rows 1,5120,8192,12288,69632
```

| rows | optimized p50 µs | correctness p50 µs |
| ---: | ---: | ---: |
| 1 | 3.92 | 144.62 |
| 5120 | 16.69 | 254.73 |
| 8192 | 24.10 | 284.28 |
| 12288 | 34.12 | 392.43 |
| 69632 | 0.51（launch 失敗） | 1737.93 |

数値の一致は `tests/kernels/optimized/test_elementwise.hip` に swiglu f=512 の rows 12288 /
36864 と、fixture width の f=128 rows 5120 をケースとして固定した（bench 側に `--check` が
ないため）。

## 対応

1. `elementwise_selector.cpp` に **group 行数専用の bound** `kMeasuredMaxGroupRows = 36864`
   を置き、`{SwigluBf16, 512}` をそこまで registered にした。fixture の width 128 も同じ bound で
   registered（実測ケースを追加）。
2. executor に guard を入れた。`GROUPED_ROWS` に触れる dispatch は correctness 経路へ落とさず
   `Status::unsupported` を返す（メッセージに op index・kernel 名・group 行数）。あわせて census
   の `fallback_rows` は group 行数を記録するようにした。
3. `shelf-bench tg` は失敗時にも census / semantic CSV を書く。

## 実測（対応後）

| run | 結果 |
| --- | --- |
| `tg --context 2048 --prefill-chunk 512` | 54.43 tok/s（対応前から通っていた） |
| `tg --context 2048 --prefill-chunk 1024` | 52.04 tok/s、`GREEDY_TOKEN_SUM=1067350`（chunk 512 と同一） |
| `tg --context 4096 --prefill-chunk 1024 / 2048 / 4096` | 51.36 / 51.28 / 49.31 tok/s、grouped fallback ゼロ |
| `tg --context 8192 --prefill-chunk 8192` | `unsupported: grouped-row dispatch has no correctness path (op 24 SWIGLU rows=69632; raise the elementwise allowlist or lower the row budget)` |
| `shelf-compute --max-num-batched-tokens 1024 / 2048 / 4096`（prompt 2048 token） | 53.91 / 52.35 / 53.78 tok/s、`GENERATED_IDS=17237,1247,13,220`（3 条件で同一） |

chunk 512 と chunk 1024 で `GREEDY_TOKEN_SUM` が一致するので、chunking 自体は生成列を変えない。
壊れていたのは 8192 を超えた group 行数の elementwise だけだった。

対応前の binary（`linear allowlist 登録まで入った木`）で production 既定をそのまま実行すると

```
prefill step failed: ... status=6 (a kernel reported a non-finite value: the run is stopped)
```

になった（`shelf-compute --input-ids-file <2048 token> --max-seq-len 4096
--max-num-batched-tokens 2048`）。同じ flags で対応後は 52.35 tok/s、`--max-num-batched-tokens`
を 1024 / 4096 に変えても `GENERATED_IDS` は同一だった。

## 受け入れ

- `tests/unit/test_qwen35_elementwise_selector.cpp`: rows 12288 / 36864 は Optimized、36865 は
  Correctness、width 128 は Optimized。bound を 8192 へ戻すと 2 件失敗（tamper 実演）
- `tests/kernels/optimized/test_elementwise.hip`: 上記の大 rows ケース
- `tests/unit/test_qwen35moe_executor.hip`: bucket 毎の run で SWIGLU の fallback 増分が 0 である
  こと、未 registered の group geometry（inter 384）が `grouped-row dispatch` を名乗る
  `unsupported` で拒否されること。`moe_intermediate_size` を fixture パラメータ化した
  （128 を登録する前はこの test が guard で落ちた = fail-first）

## 残る制約

bucket R8192 の capacity 69632 行は elementwise kernel の grid.y 上限（65535）で表現できない。
rows を grid.z か row offset で分割する kernel 変更が無い限り、MoE の prefill chunk は 4096 token
まで。超過は起動時の `unsupported` として見える（暗黙の誤出力はしない）。
