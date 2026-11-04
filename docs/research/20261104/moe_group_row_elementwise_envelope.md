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

## grid.y 上限を grid.z で越えた（bucket R8192 / chunk 8192）

上の「残る制約」を解消した。動機は prefill chunk の上限ではなく、**上限が存在する理由が
kernel の address 計算だけ**だったこと。先に hardware の値を実測で固定する:

| launch | 結果（gfx1201、`gridprobe`） |
| --- | --- |
| `dim3(1, 65535, 1)` | launch / sync 成功、到達した最大 row = 65534 |
| `dim3(1, 69632, 1)` | **`invalid configuration argument`**（launch 自体が失敗） |
| `dim3(1, 65535, 2)` | 成功。`blockIdx.y` の最大は 65534、`blockIdx.z` は 0 / 1、block 数 131070 |

grid.y は 65535 で打ち切られる（grid.y 単独では 69632 行を表現できない）が、grid.z へ載せれば
1 launch のままでいける。

### 最初に入れた分割は誤りだった（行数の二重実行 + 範囲外書き込み）

最初に置いた helper は `grid.y = min(rows, 65535)` / `grid.z = ceil(rows / 65535)` で、
kernel は `blockIdx.y + blockIdx.z * gridDim.y` を読む。これは **grid.y × grid.z ≫ rows** になる。
`rows = 65536` なら 65535 × 2 = 131070 個の row slot を実行し、131070 - 65536 = 65534 行ぶんは
buffer の後ろへ**範囲外書き込み**だった（launch も sync も成功するので、数値は前にある正しい行だけ
見ると壊れない。テストは `mm=0` で PASS した）。実行時間に出た:

| rows | _naive 分割_ p50 | 均等分割 + row bound p50 |
| ---: | ---: | ---: |
| 49152 | 250.16 µs | 260.55 µs |
| 65535 | 332.35 µs | 332.04 µs |
| 65536 | **667.70 µs** | 332.16 µs |
| 69632 | **666.71 µs** | 352.20 µs |
| 98304 | **667.71 µs** | 494.80 µs |

naive では 65536 行を超えた瞬間に時間が 2 倍になり、その後 rows を増やしても **一定**
（98304 でも 667 µs）だった。work ではなく grid.y × grid.z の block 数が決まっていた証拠。
均等分割（`grid.z = ceil(rows / 65535)`、`grid.y = ceil(rows / grid.z)`）+ kernel 側の
`if (row >= args.rows) return;` に直した後は、行数に比例する
（69632 行で 213.9 MB / 352.20 µs = **607 GB/s**、65535 行で 606 GB/s。DRAM 実測 636 GB/s の 95 %）。

均等分割でも `rows` が割り切れないことがある（69633 行 → grid.y 34817 × grid.z 2 = 69634 slot）。
なので row bound は単なる保険ではなく**必須**。`RowBucket` の最大容量 69632 行はちょうど
2 で割れるが、境界の geometry はテストで押さえる。

### 変更

- `elementwise.hip` の 9 本すべての kernel が行を `elementwise_row()` =
  `blockIdx.y + blockIdx.z * gridDim.y` で引き、`row >= rows` の slot を捨てる
  （`ElementwiseArgs.rows` は既にあった欄。`launch_concat_bf16` の kernel は args struct を
  使わないので `rows` 引数を増やした）。launcher は `elementwise_grid(features, rows)`。
  `launch_concat_bf16` が自分で計算していた grid.x は `elementwise_grid_x` と同一式だったので
  ここも helper へ寄せる
- elementwise selector の group 行数 bound `kMeasuredMaxGroupRows` を 36864 → **69632**
  （最大の production bucket R8192 の容量）へ上げる。bound は「実測した geometry の上限」
  であり容量の正本ではないので、69632 行超は今までどおり allowlist 外（`Status::unsupported`）
- `elementwise_dispatch.hip` の `fill_geometry` は grid.y 部分を記録する（grid.z は launcher で
  決まる。この struct の grid 欄に読み手は無いので grid.z 欄は足さない）

### 実測（device 1）

`shelf-bench pp --prompt-tokens 8192 --runs 3 --warmup 1`（MoE。`pp` が grouped expert を
load できることが前提。`pp` の LoadedModel 配線は `docs/research/20261104/moe_bench_loaded_model_wiring.md`）:

| `--prefill-chunk` | group 行数 | pp tok/s |
| ---: | ---: | ---: |
| 1024 | 8192 | 3648.15 |
| 2048 | 16384 | 3759.43 / 3769.14 |
| 4096 | 32768 | 3683.91 |
| 8192 | 69632 | 3561.42（naive 分割）/ 3569.17（均等分割） |

**chunk 8192 は通るようになったが速くならない**（最良の 2048 より 5.3 % 低い）。elementwise の
二重実行を直しても pp は 3561 → 3569 tok/s しかならない（swiglu 1 本の 315 µs × 40 層 = 12.6 ms は
prefill 2290 ms の 0.55 %）。なのでこの変更の価値は throughput ではなく **capability**
（8192 token の 1 batch が拒否されなくなる）と、範囲外書き込みが消えたこと。既定値は変えない。

`shelf-bench tg --mode greedy --context 8192 --tokens 8 --prefill-chunk …`（device 1、各 2 run）:

| `--prefill-chunk` | 変更前 | 変更後 |
| ---: | --- | --- |
| 8192 | `unsupported: grouped-row dispatch has no correctness path (op 24 SWIGLU rows=69632; …)` | 52.61 / 54.93 tok/s（`GREEDY_TOKEN_SUM=732`） |
| 4096 | — | 55.09 / 54.86 tok/s（`GREEDY_TOKEN_SUM=710`） |

decode は chunk 8192 で速くならない（52.61 / 54.93 と 55.09 / 54.86 は run の揺れの範囲）。
`GREEDY_TOKEN_SUM` が違うのは prefill の chunk 大きさで数値が揺れる既知の理由で、同一 flags の
A/B 以外では比較しない。

### 受け入れ

- `tests/kernels/optimized/test_elementwise.hip`
  - `swiglu-moe-group-r69632`（f=512・rows=69632）と **`swiglu-moe-group-r69633`**
    （grid.z で割り切れない行数。row bound が無ければ範囲外書き込みになる行数）を追加
  - **test 側の穴を 2 つ塞いだ**:
    1. correctness 経路と同じ output buffer へ optimized を走らせていたので、optimized が
       書かなかった行は correctness の値のまま残り bit 比較で不一致にならなかった →
       optimized の launch 前に output を 0xA5 で poison する（全 case で guard=ok、つまり
       optimized は全行を書いている）
    2. 行 domain の外は検査していなかった → 出力 buffer の末尾に 4096 B の guard 領域を
       用意し、launch 後に 0xA5 のままかを 보는（`guard=TOUCHED` で FAIL）
  - case が失敗しても `mm` に反映されていなかったので exit code に効かなかった。`*mm_out` へ
    書く前に guard を数えるよう直した
  - **fail-first は tamper で実演**: (A) swiglu の `row >= args.rows` を消すと
    `r69633` だけが `guard=TOUCHED` で FAIL、(B) `elementwise_row()` を `blockIdx.y` だけにすると
    `r69632` / `r69633` が FAIL（`r12288` / `r36864` / `f128` は PASS）
- `tests/unit/test_qwen35_elementwise_selector.cpp`: 69632 は Optimized、69633 は Correctness。
  `elementwise_grid_y` / `elementwise_grid_z` が grid.y ≤ 65535 を保ったまま行数を覆うことを
  1 / 65535 / 65536 / 69632 / 131071 / 196606 / 1000000 行で確認（bound を 36864 へ戻すと
  69632 の期待が失敗する = fail-first）
