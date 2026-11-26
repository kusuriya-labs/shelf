# shelf-bench

ベンチマークアプリ。サブコマンドはカテゴリごと:

```text
E2E:
  pp                 prefill benchmark
  tg                 decode / token-generation benchmark
  batch              continuous-batching / prefix-cache benchmark

Linear:
  activation-quantize A8 activation quantization micro-benchmark
  gemm               linear GEMV/GEMM micro-benchmark
  exl3-int8          EXL3-INT8 decode linear micro-benchmark

Normalization:
  rmsnorm            RMSNorm micro-benchmark
  l2-normalize       L2 normalize micro-benchmark

Attention/KV:
  kv-append          KV append benchmark
  paged-attention    paged-attention benchmark
  rope               RoPE micro-benchmark

GDN:
  gdn-recurrence     GDN recurrence benchmark
  gdn-conv1d         GDN causal conv1d benchmark

Auxiliary:
  elementwise        Qwen elementwise benchmark
  embedding          embedding lookup benchmark
  sampling           greedy argmax sampling benchmark

Memory:
  tp-reduce          TP sum_hidden transport microbenchmark
  gpu-memory         global / LDS crossover sweep

Quality:
  quality            teacher-forced logits dump / quality comparison
```

`--model-dir` は pp / tg / batch / quality で必須。他は model 不要。

GEMV/GEMM are compute kinds. WMMA is an internal GEMM implementation
detail and is not a `--variant` value.

各サブコマンドの全 option は `shelf-bench <sub> --help` を参照。
ベンチマーク実行時は同一 GPU 上で他プロセスと競合しないよう `flock /tmp/r9700.lock` 等の排他ロックを使用する。測定原則は [../perf/methodology.md](../perf/methodology.md)、関連する環境変数の一覧は [env_vars.md](env_vars.md) を参照。

## pp

<!-- cli:bench-pp -->
| option | 説明 |
| --- | --- |
| `--prompt-tokens N` | (default 128) |
| `--prefill-chunk N` | (default 0 = up to 8192 tokens per batch) |
| `--ple-prefetch 0\|1` | prefetch the next PLE rows during prefill (default 1) |
| `--gap-keepalive US` | (default 0 = off; keep the CP awake) |
| `--gap-keepalive-flags N` | (default hipEventDisableSystemFence; -1 = hipEventCreate) |
| `--mode forward\|greedy` | (default forward) |
| `--device LIST` | comma separated GPUs, one per --tensor-parallel rank (default 0; e.g. --device 0,2) |
| `--tensor-parallel N` | tensor parallel degree 1 \| 2 \| 4 (default 1) |
| `--model-dir PATH` | (required) |
| `--page-tokens N` | (default 16) |
| `--kv-cache-dtype bf16\|fp8\|octave:FORMAT (default bf16)` |  |
| `--rope-factor auto \| 1 \| 2 \| 4 (default auto)` |  |
| `--arena-gib N` | (default auto = free VRAM minus 0.8 GiB) |
| `--moe-host-experts N` | routed experts per layer kept in host memory (default 0) |
| `--moe-cache-swap-budget N` | expert cache swap budget (default 96) |
| `--moe-cache-min-gain N` | minimum frequency gain to swap (default 150) |
| `--moe-cache-rebalance-steps N` | rebalance interval in steps (default 4) |
| `--moe-cache-rebalance-routes N` | rebalance interval in routes (0 = auto) |
| `--moe-cache-decay N` | halve expert frequency at this count (default 4096) |
| `--moe-cache-log-interval N` | expert cache hit-rate log interval (default 64) |
| `--moe-cache-oracle 0\|1` | log cache oracle decisions (default 0) |
| `--ple-hot-cache-gib N` | pinned host block cache for PLE (default 0) |
| `--lm-head-int8 0\|1` | quantize the BF16 lm head to INT8 W32 on load (default 1) |
| `--pcie-frac F` | share of PCIe the runtime may use (default 0.25) |
| `--pcie-gbps F` | assumed PCIe bandwidth in GB/s (default 25) |
| `--runs N` | (default 3) |
| `--warmup N` | (default 0) |
| `--tokens-file PATH` | PSKLDTOK corpus file used as the prompt |
| `--tokens-offset N` | corpus position to start the prompt (default 0) |
| `--perf-stats-output PATH` | (default none; per-kernel dispatch counters JSON, needs SHELF_QWEN35_PERF_STATS=1) |
| `--decode-backend TYPE` | hip (default) |
| `--output PATH` | (default none) |
<!-- /cli:bench-pp -->

`--tensor-parallel N` は `--device` の N GPU を使用する（N=1 / 2 / 4。例 `--device 0,1`、
非連続な `--device 0,2` も可。N 個の指定が必須）。`--arena-gib` は rank ごとの容量である。
Qwen3.8-27B の FP8 は単一 GPU に収まらないため `--tensor-parallel 2` が必須。
`models/Qwen3.8-Flash-Next-Quark-MXFP4` は rank あたり約 30 GiB の重みを持つため
`--tensor-parallel 4` を使う。`--mode forward` でも最終 chunk の最終行の logits を計算する
（lm_head 以降の dispatch は mode と TP で変わらない）。

`--tokens-file` を指定すると PSKLDTOK corpus の先頭（`--tokens-offset` 以降）を prompt に
使う。省略時は token id 1 の繰り返しになる。

`SHELF_QWEN35_PERF_STATS=1 --perf-stats-output PATH` で dispatch 統計を保存できる。
さらに `SHELF_QWEN35_PERF_TIMING=1`（TP=1 と `--perf-stats-output` が必要、最大 65536 event ペア）で
`semantic_timing.csv` に各 dispatch と EXL3-INT8 dense の各段階の GPU event 時間を保存できる。
親 dispatch と内側の段階は入れ子であり合算しない。測定自体の負荷があるため、速度は timing 無効時にも測る。
CPU 側の区間時間出力（`SHELF_PP_HOST_TIMING=1`）を含む診断用環境変数は
[env_vars.md](env_vars.md#8-診断計測デバッグ開発者切り分け用) を参照。

grouped expert checkpoint（Qwen3.6-35B-A3B MoE）も `--tensor-parallel 2` を受理する
（expert 軸分割で、rank が expert の contiguous 範囲を持つ）。rank local shape は
backbone kernel の allowlist に登録済みで、prefill の throughput は TP=2 が TP=1 の
1.64〜1.79 倍（bf16 / fp8 KV 双方。値は `docs/perf/current.md`）。MoE の prefill throughput
（TP=1）は
`--prompt-tokens 8192 --prefill-chunk 1024 / 2048 / 4096` で 3662 / 3807 / 3631 tok/s
（既定 2048 が最良。grouped expert の row bucket は 4096 token までなので、
`--prefill-chunk 8192` は `unsupported: grouped-row dispatch has no correctness path` になる。
値の根拠は `docs/perf/current.md`）。

## tg

<!-- cli:bench-tg -->
| option | 説明 |
| --- | --- |
| `--mode forward\|greedy` | (default forward) |
| `--context N` | (default 128) |
| `--tokens N` | (default 8) |
| `--prefill-chunk N` | (default 128) |
| `--warmup N` | (default 0) |
| `--compute-logits 0\|1` | (default 1) |
| `--device LIST` | comma separated GPUs, one per tensor-parallel * expert-parallel rank (default 0; e.g. --device 0,2) |
| `--tensor-parallel N` | tensor parallel degree 1 \| 2 \| 4 (default 1) |
| `--expert-parallel N` | expert parallel degree 1 \| 2 \| 4 (default 1) |
| `--model-dir PATH` | (required) |
| `--page-tokens N` | (default 16) |
| `--kv-cache-dtype bf16\|fp8\|octave:FORMAT (default bf16)` |  |
| `--rope-factor auto \| 1 \| 2 \| 4 (default auto)` |  |
| `--arena-gib N` | (default auto = free VRAM minus 0.8 GiB) |
| `--moe-host-experts N` | routed experts per layer kept in host memory (default 0) |
| `--moe-cache-swap-budget N` | expert cache swap budget (default 96) |
| `--moe-cache-min-gain N` | minimum frequency gain to swap (default 150) |
| `--moe-cache-rebalance-steps N` | rebalance interval in steps (default 4) |
| `--moe-cache-rebalance-routes N` | rebalance interval in routes (0 = auto) |
| `--moe-cache-decay N` | halve expert frequency at this count (default 4096) |
| `--moe-cache-log-interval N` | expert cache hit-rate log interval (default 64) |
| `--moe-cache-oracle 0\|1` | log cache oracle decisions (default 0) |
| `--ple-hot-cache-gib N` | pinned host block cache for PLE (default 0) |
| `--lm-head-int8 0\|1` | quantize the BF16 lm head to INT8 W32 on load (default 1) |
| `--pcie-frac F` | share of PCIe the runtime may use (default 0.25) |
| `--pcie-gbps F` | assumed PCIe bandwidth in GB/s (default 25) |
| `--output PATH` | (default none) |
| `--gap-keepalive US` | (default 0 = off; keep the CP awake) |
| `--gap-keepalive-flags N` | (default hipEventDisableSystemFence; -1 = hipEventCreate) |
| `--perf-stats-output PATH` | (default none; per-kernel dispatch counters JSON, needs SHELF_QWEN35_PERF_STATS=1) |
| `--tokens-file PATH` | PSKLDTOK corpus file used as the context |
| `--tokens-offset N` | corpus position to start the context (default 0) |
| `--decode-backend TYPE` | hip (default) |
<!-- /cli:bench-tg -->

greedy mode は `--compute-logits 1` が必要。

`--tensor-parallel N`（N=1 / 2 / 4）は backbone と各 routed expert の matrix 軸を
N GPU に分割する。`--expert-parallel M`（M=1 / 2 / 4）は routed expert の expert ID を
M GPU に分割する。qwen4exp では EP>1 かつ TP=1 で head 数が EP で割り切れる場合、
backbone も EP rank で分割する。`--device` には `N * M` 個の GPU を指定する。
TP の decode は rank0 の stream で測り、`tp=` / `ep=` / `devices=` を出力する。
EXL3 INT8 の routed expert は local の N / K が 128 の倍数でなければならないため、
checkpoint によっては TP>1 を起動できない。
`--tokens-file` / `--tokens-offset` は prefill する context を PSKLDTOK corpus から取る。

`--kv-cache-dtype fp8` は KV cache を fp8 E4M3 にする。grouped expert checkpoint（MoE）でも
bf16 / fp8 の双方で `KV_APPEND` と `PAGED_ATTENTION` が registered geometry であり、
long context では split-KV（splits 16）が選ばれる。fp8 KV の decode 速度は bf16 と同等で、
KV の容量は半分になる（実測は [../perf/current.md](../perf/current.md)）。

`--perf-stats-output PATH` は `SHELF_QWEN35_PERF_STATS=1` と併せたときだけ、測定区間の
dispatch 内訳を JSON で書く。中身は kernel 名ごとの `logical` / `optimized` / `fallback`
と、fallback が起きた rows 範囲（`fallback_rows`）で、**どの kernel が correctness path へ
落ちているかを見る口はこれだけ**である。`SHELF_QWEN35_PERF_TIMING=1` を添えると 同じ場所に
`semantic_timing.csv`（dispatch 毎の GPU time と optimized / correctness の別）も出る。

## quality

固定 token corpus を teacher forcing し、位置ごとの full-vocab logits を raw f32 で
dump する。`--compare` で baseline dump と突き合わせ、perplexity / mean KL /
top-1 一致率 / top-5 overlap を出力する。性能ではなく品質の測定であり、
`shelf-bench pp/tg` の数値とは混ぜない。

`--batch-size N`（既定 1、最大 64）は同じ token 列を N 個の独立した sequence に
与え、最初の sequence の logits を dump・比較する。batch 用 kernel の数値確認に使える。
N が 2 以上の場合は単一 GPU のみ対応する。

`--model-dir` は dense / grouped expert（MoE）/ Qwen4Exp のいずれでもよく、
`--tensor-parallel 1 / 2 / 4` が使える。Qwen4Exp は `--tensor-parallel 4` を使う。

corpus は `vendor/BetterBench/corpus/v1` から生成する（token corpus は commit しない）。

```bash
python3 tools/quality_gate.py corpus \
  --model-dir models/Qwen3.8-27B-FP8 --out /tmp/bb_corpus.psktok

# BF16 KV baseline
./build/shelf-bench quality --model-dir models/Qwen3.8-27B-FP8 \
  --tokens-file /tmp/bb_corpus.psktok \
  --kv-cache-dtype bf16 --dump /tmp/bf16.psq

# FP8 KV candidate
./build/shelf-bench quality --model-dir models/Qwen3.8-27B-FP8 \
  --tokens-file /tmp/bb_corpus.psktok \
  --kv-cache-dtype fp8 --dump /tmp/fp8.psq --compare /tmp/bf16.psq
```

`--positions 0` は corpus 末尾まで（上限 2048）を採点する。dump は 1 位置あたり
`vocab × 4 byte + 8 byte` であり、248320 vocab では約 1 MB/位置になる。
1 プロセスの logits は row ごとに host へコピーするため、dump 全体を VRAM に
保持しない。

dump 形式は 32 byte header（`PSQUAL01`、version、vocab、positions、kv dtype）+
位置ごとの `int32 target`、`uint32 position`、`float logits[vocab]` である。
byte order は同一ホスト内での比較を前提とする。

品質 gate の合否判定と end-task は `tools/quality_gate.py` が行う。
[../developer/testing.md](../developer/testing.md) の「quality regression」を参照。

## activation-quantize

```text
--variant correctness|e4m3-generic|e4m3-vec|e4m3-auto   (default e4m3-auto)
--check                                                CPU reference / correctness path と照合
```

FP8 E4M3 activation quantization (correctness 実装と optimized kernel の比較)。
`--check` は e4m3 系 variant で generic kernel と CPU reference を比較し、
model-dispatch correctness path の codes / scales も一致することを確認する。

## gemm

```text
--dtype bf16|fp8|mxfp4            (default bf16)
--variant correctness|gemm|auto   (default auto)
--rows 1,16,32,64,128             (comma list, default 1,16,32,64,128)
--n N                            (default 5120)
--k N                            (default 5120)
--shapes N:K,N:K,...             (explicit shape list)
--impl auto|...                  (dtype ごとの variant 強制)
--fp8-split N                    (fp8 split-K ways, default 8)
--fp8-split-mb N                 (fp8 split-K row tile 1|2, default auto)
--mxfp4-split N                  (mxfp4 split-K ways, default 8)
--mxfp4-split-mb N               (mxfp4 split-K row tile 1|2, default auto)
--mxfp4-fold 0|1                 (mxfp4 の row E8M0 fold)
--mxfp4-spread N                 (mxfp4 の合成 E8M0 指数幅)
--model-dir PATH                 (with --actual-qwen-shapes: read Qwen config)
--actual-qwen-shapes             (benchmark all deduped Qwen linear shapes)
--warmup N                       (default 100)
--samples N                      (default 50)
--launches N                     (default 20)
--seed N                         (default 1234)
--device N                       (default 0)
--verify-small-m                 (speculative verify rows 1,2,4,8)
--split N                        (rows を N launch に分割)
--check                          (verify selected path vs the correctness path)
--out PATH                       (append CSV rows)
```

`--impl` は dtype ごとに次を強制する。`auto` は production selector を使う。

```text
bf16   auto|wmma|wide|splitk|exact|smallrows
fp8    auto|splitk|rowblock1|rowblock2|rowblock4|prefill2d
mxfp4  auto|splitk|rowblock1|rowblock2|rowblock4|prefill2d
```

`auto` の場合、選択された variant を `selected_variant` として stdout/CSV に出力する。

CSV 列:

```text
dtype,requested_variant,selected_variant,shape_name,rows,n,k,
warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us
```

## rmsnorm

```text
--variant correctness|optimized|auto   (default optimized)
```

## kv-append

```text
--variant correctness|optimized|auto   (default optimized)
```

## paged-attention

```text
--variant correctness|optimized|auto   (default optimized)
```

## gdn-recurrence

```text
--variant correctness|optimized|optimized-verify   (default optimized)
```

## elementwise

```text
--variant correctness|optimized   (default optimized)
```

## embedding

```text
--variant correctness|optimized   (default optimized)
```

embedding lookup (bf16)。

## sampling

```text
--variant correctness|single|partitioned   (default correctness)
```

greedy argmax sampling。

## 例

```bash
./build/shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --prompt-tokens 512 --tensor-parallel 2 --device 0,1 --arena-gib 16
./build/shelf-bench tg --model-dir models/Qwen3.8-27B-FP8 --mode greedy --compute-logits 1
./build/shelf-bench pp --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --tokens-file /tmp/corpus.psktok --prompt-tokens 512 \
  --tensor-parallel 4 --device 0,1,2,3 --arena-gib 31
./build/shelf-bench tg --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --tokens-file /tmp/corpus.psktok --context 128 --tokens 64 \
  --tensor-parallel 4 --device 0,1,2,3 --arena-gib 31
./build/shelf-bench gemm --check --rows 1,16,128
./build/shelf-bench gemm --dtype fp8 --variant auto --actual-qwen-shapes --model-dir models/Qwen3.8-27B-FP8 --out gemm.csv
```

## exl3-int8

`rows <= 64` の EXL3-INT8 の入力回転・GEMM・出力回転と連結時間を GPU event で測る。
`--random 1` は重み・活性・符号ベクトルを決定的な乱数で埋め、`--dump FILE` は最終出力（bf16）を書き出す
（実装間の出力比較用）。
重みは合成データで、`--working-mib` 以上になるよう複数コピーを巡回する。
各値は warmup の後に取得した 5 回の中央値で、`min` と `max` はその範囲である。
`weight_GBps` は trellis のバイト数を時間で割った値で、実際の DRAM 転送量ではない。

```sh
./build/shelf-bench exl3-int8 --n 17408 --k 5120 --rows 8 --bits 4 --working-mib 256
```

`--split` を省略すると `--scratch-rows` で指定した容量を使って本番の分割数を選ぶ。
容量が実モデルと異なれば選択結果も異なる。GPU の分離と比較手順は
[計測方法](../perf/methodology.md) を参照。

## 統一ベンチマークスイート（`tools/bench/shelf_bench_suite.py`）

E2E 基礎性能・BetterBench 互換・凍結実タスク（`tools/bench/real_v1/`）・複数ターン agentic / サービング同時計測をまとめて実行し、Markdown レポートへ整形するラッパーである。4 トラックの定義・使い分け・JSON 契約は [../perf/methodology.md](../perf/methodology.md#9-統一-4-トラック体系と-shelf_bench_suitepy) を参照。

```text
サブコマンド:
  track-a   基礎性能（shelf-bench pp/tg）と投機固定チェック（1,024+ token、AB/BA 対応）
  track-b   BetterBench 互換（既定 20 pass/category・3 warmup、--quick は smoke 専用）
  track-c   凍結実タスクコーパス real-v1（5 領域 × 日英 × 3 入力長、全 30 課題）
  track-d   HTTP streaming closed-loop サービング計測（同時 1/4/8/16、agentic / prefill / decode / all）
  render    トラック別 JSON を C -> B -> A -> D の順で Markdown レポートへ整形
```

共通オプション（サブコマンドの前後に配置可能）:

- `--lock-file PATH`: 実行全体に `flock` の排他ロックを取得する
- `--output PATH`: 結果の保存先（省略時は標準出力）
- `--dry-run` / `--list`: 実測せずに実行計画または課題一覧を JSON で表示する

```bash
# 実行計画と課題一覧の確認
python3 tools/bench/shelf_bench_suite.py track-c --list

# Track A: 非投機 pp/tg + 投機固定チェック
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
python3 tools/bench/shelf_bench_suite.py track-a \
  --lock-file /tmp/r9700.lock \
  --shelf-bench ./build/shelf-bench \
  --model-dir models/Qwen3.8-27B-FP8 \
  --tokens-file /tmp/corpus.psktok \
  --spec-endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --spec-max-tokens 1024 \
  --output /tmp/track_a.json

# Track B: BetterBench 互換 full（20 pass/category・3 warmup）
python3 tools/bench/shelf_bench_suite.py track-b \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --conditions /tmp/conditions.json \
  --output /tmp/track_b.json

# Track C: 凍結 real-v1（全 30 課題）
python3 tools/bench/shelf_bench_suite.py track-c \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --output /tmp/track_c.json

# Track D: サービング同時計測（agentic / prefill / decode / all、外部凍結データセットは --dataset で指定）
python3 tools/bench/shelf_bench_suite.py track-d \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --concurrency 1,4,8,16 \
  --mode all \
  --prompt-tier mid_8k \
  --max-tokens 1024 \
  --output /tmp/track_d.json

# Markdown レポート生成（省略したトラックや欠測値は「未測定」と表示）
python3 tools/bench/shelf_bench_suite.py render \
  /tmp/track_a.json /tmp/track_b.json /tmp/track_c.json /tmp/track_d.json \
  --compare-b /tmp/other_track_b.json \
  --reviews /tmp/real_v1_reviews.json \
  --output /tmp/report.md
```

Track C で Qwen3.8 の追加の思考指示を外して比較する場合は `--reasoning-effort medium` を指定する。
Shelf Server の回答用予約も外す場合は `--no-answer-reserve` を追加する。
どちらも省略するとサーバーの既定値を使う。[思考方針と予算の設定](../developer/reasoning.md)

各サブコマンドの全オプションは `python3 tools/bench/shelf_bench_suite.py <サブコマンド> --help` を参照。
