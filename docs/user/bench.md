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

## pp

<!-- cli:bench-pp -->
| option | 説明 |
| --- | --- |
| `--prompt-tokens N` | (default 128) |
| `--prefill-chunk N` | (default 0 = up to 8192 tokens per batch) |
| `--gap-keepalive US` | (default 0 = off; keep the CP awake) |
| `--gap-keepalive-flags N` | (default hipEventDisableSystemFence; -1 = hipEventCreate) |
| `--mode forward\|greedy` | (default forward) |
| `--device LIST` | comma separated GPUs, one per --tensor-parallel rank (default 0; e.g. --device 0,2) |
| `--tensor-parallel N` | tensor parallel degree 1 \| 2 \| 4 (default 1) |
| `--model-dir PATH` | (required) |
| `--page-tokens N` | (default 16) |
| `--kv-cache-dtype bf16\|fp8\|octave:FORMAT (default bf16)` |  |
| `--arena-gib N` | (default auto = free VRAM minus 0.5 GiB) |
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
さらに `SHELF_QWEN35_PERF_TIMING=1` を指定すると、同じ directory の
`semantic_timing.csv` に各 dispatch と EXL3-INT8 dense の各段階の GPU event 時間を保存する。
この診断は TP=1 と `--perf-stats-output` が必要で、最大 65536 event ペアを記録し、
超過時は stderr に欠落を警告する。親 dispatch と内側の段階は入れ子であり合算しない。
測定自体の負荷があるため、速度は timing 無効時にも測る。
`SHELF_PP_HOST_TIMING=1` は sequence setup、chunk の `execute_batch` 呼び出し、
最後の event wait の CPU 時間を stderr に出す。TP=1 の測定区間では submit と
complete も分ける。complete は GPU 同期と commit を含み、submit も GPU 作業と重なり得る。

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
| `--device LIST` | comma separated GPUs, one per --tensor-parallel rank (default 0; e.g. --device 0,2) |
| `--tensor-parallel N` | tensor parallel degree 1 \| 2 \| 4 (default 1) |
| `--model-dir PATH` | (required) |
| `--page-tokens N` | (default 16) |
| `--kv-cache-dtype bf16\|fp8\|octave:FORMAT (default bf16)` |  |
| `--arena-gib N` | (default auto = free VRAM minus 0.5 GiB) |
| `--output PATH` | (default none) |
| `--gap-keepalive US` | (default 0 = off; keep the CP awake) |
| `--gap-keepalive-flags N` | (default hipEventDisableSystemFence; -1 = hipEventCreate) |
| `--perf-stats-output PATH` | (default none; per-kernel dispatch counters JSON, needs SHELF_QWEN35_PERF_STATS=1) |
| `--tokens-file PATH` | PSKLDTOK corpus file used as the context |
| `--tokens-offset N` | corpus position to start the context (default 0) |
| `--decode-backend TYPE` | hip (default) |
<!-- /cli:bench-tg -->

greedy mode は `--compute-logits 1` が必要。

`--tensor-parallel N`（N=1 / 2 / 4）は decode を N GPU に分割する。`--device` に N 個の
GPU を指定する。`models/Qwen3.8-Flash-Next-Quark-MXFP4` は `--tensor-parallel 4` を使う。
TP の decode は rank0 の stream で測り、`tp=` と `devices=` を出力する。
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
--dtype bf16|fp8                    (default bf16)
--variant correctness|gemv|gemm|auto  (default auto)
--rows 1,16,32,64,128             (comma list, default 1,16,32,64,128)
--n N                            (default 5120)
--k N                            (default 5120)
--model-dir PATH                 (with --actual-qwen-shapes: read Qwen config)
--actual-qwen-shapes             (benchmark all deduped Qwen linear shapes)
--warmup N                       (default 100)
--samples N                      (default 50)
--launches N                     (default 20)
--seed N                         (default 1234)
--device N                       (default 0)
--check                          (verify selected path vs the correctness path)
--out PATH                       (append CSV rows)
```

`--variant`:

```text
correctness   production correctness path
gemv          force Small-M GEMV compute kind (selector bypass)
gemm          force Big-M GEMM compute kind (selector bypass)
auto          production linear selector (default)
```

`auto` の場合、選択された compute kind (correctness/gemv/gemm) を
`requested_variant` と `selected_variant` として stdout/CSV に出力する。

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
