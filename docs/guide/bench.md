# shelf-bench 操作手順

現行の仕様は `docs/spec/bench.md` を参照。本書は運用手順（ビルド・実行・対処）。

* **prefill（`pp`）**・**decode（`tg`）**・**単一カーネル（`kernel <name>`）** の速度を測る道具。
* 入力は合成トークン列・合成バッファで、**モデルの答えは見ない**（正しさの検証は
  `docs/guide/validate.md`、計画上の位置づけは `draft_docs/directory_design.md` §5 のフェーズ 0）。
* `--help` は GPU を触らず exit 0（CPU ビルドでも動く）。**本体は HIP ビルドのみ**。

---

## 1. 手順（概要）

1. ビルドする（§2）
2. モデルディレクトリを用意する（`config.json` + `model.safetensors.index.json` + shard）。
   `kernel` のみはモデル不要
3. GPU の空きを確認し `--gpu` を決める（§5。既定 0 は埋まっていることがある）
4. 測るものを選んで実行する（§3 で pp / tg、§4 で kernel）
5. 出力を読む・CSV に残す（§7）、終了コードで成功/失敗を判定する（§8）

---

## 2. ビルド

```bash
# 計測用 = Release（**数値は必ずこれで取る**）
cmake --preset hip-rel          # 初回のみ（設定）
cmake --build --preset hip-rel  # → build/hip_rel/apps/shelf-bench

# 動作確認用 = Debug（計測には使わない）
cmake --preset hip
cmake --build --preset hip      # → build/hip/apps/shelf-bench

# CPU 版（--help のみ有効。本体は exit 1）
cmake --preset cpu
cmake --build --preset cpu      # → build/cpu/apps/shelf-bench
```

> **性能値は `hip-rel`（Release）で測定する**（2026-10-28 決定）。
> Debug は桁違いに遅く、そのままでは性能値にならない。Debug ビルドは動作確認にのみ使う。

---

## 3. 実行（pp / tg）

### 3.1 pp — prompt processing（prefill）

```bash
./build/hip/apps/shelf-bench pp \
    --model-dir /opt/zen/work/models/Qwen3.5-9B \
    --prompt-tokens 64 --runs 2 --warmup 1 --gpu 2
```

出力例（開発モデル `Qwen3.5-9B`・2026-10-28 実測、R9700 / gfx1201・**Release ビルド（hip-rel）**）:

```text
shelf-bench pp
schema_version=1
device=hip
gpu=2
model=/opt/zen/work/models/Qwen3.5-9B
prompt_tokens=64
prefill_chunk=64
runs=2
warmup=1
arena_mib=2298
wall_ms_median=198.746
wall_ms_min=198.473
wall_ms_p10=198.473
wall_ms_p90=198.746
wall_ms_stddev=0.137
wall_us_per_token=3105.412
wall_tokens_per_sec=322.02
run_wall_ms[0]=198.746
run_wall_ms[1]=198.473
```

* 主指標は `wall_ms_median` と `wall_tokens_per_sec`。`run_wall_ms[i]` は個票
* 入力トークンは合成列（`--seed` で再現）。各 run は `reset_states()` で独立

### 3.2 tg — token generation（decode）

```bash
./build/hip/apps/shelf-bench tg \
    --model-dir /opt/zen/work/models/Qwen3.5-9B \
    --context 64 --tokens 8 --warmup 2 --gpu 2
```

出力例（開発モデル `Qwen3.5-9B`・2026-10-28 実測・**Release ビルド（hip-rel）**）:

```text
shelf-bench tg
schema_version=2
device=hip
gpu=2
model=/opt/zen/work/models/Qwen3.5-9B
context=64
tokens=8
prefill_chunk=64
warmup=2
arena_mib=2298
mode=forward
context_prefill_ms=222.515
wall_ms_total=244.079
wall_us_per_token=30509.929
wall_tokens_per_sec=32.78
first_token_ms=30.435
step_ms_mean=30.499
step_ms_p50=30.520
step_ms_p90=30.559
step_ms_max=30.573
```

* コンテキストの prefill は**計測区間の外**（`context_prefill_ms` は参考値）。
  測定は `--tokens` ステップのデコードのみ
* `first_token_ms` と `step_ms_*` は 1 ステップあたりの壁時計時間の統計
* `--mode greedy` は毎ステップ logits の D2H と argmax を含む（現実に近いが重い）。
  純粋なステップ速度は既定の `--mode forward`

---

## 4. 実行（kernel）

```bash
./build/hip/apps/shelf-bench kernel gemm \
    --rows 1,16 --n 2048 --k 2048 --warmup 20 --samples 10 --launches 10 --gpu 2
```

出力例（1 シェイプにつき 1 まとまり）:

```text
shelf-bench kernel gemm
schema_version=3
kernel=gemm
device=hip
gpu=2
shape=1x2048x2048
rows=1
n=2048
k=2048
warmup=20
samples=10
launches=10
us_min=18.912
us_p10=19.036
us_p50=19.092
us_p90=22.436
us_p95=49.208
us_mean=22.716
us_stddev=8.921
throughput=439.38
throughput_unit=GFLOP/s
```

対象カーネルと固有オプションは `shelf-bench kernel --help` に一覧が出る。**現行 5 種**:

| name | 対象 | 主な固有オプション | throughput |
| --- | --- | --- | --- |
| `gemm` | linear（生産と同じ設定選択） | `--rows 1,16`（[1,64]）、`--n 2048`、`--k 2048`（8 の倍数） | GFLOP/s |
| `rmsnorm` | rms normalization | `--rows 1024`、`--features 2560` | GB/s |
| `elementwise` | `residual_add_bf16` | `--rows 1024`、`--features 2560` | GB/s |
| `kv-append` | paged KV append | `--rows` `--kv-heads` `--head-dim` `--page-tokens` `--num-pages` `--seq-len` | GB/s |
| `paged-attention` | paged GQA decode attention | 上記 + `--q-heads` | （空: 読み出し量は可視長に依存） |

* **モデルは不要**（合成バッファ 0 埋め。値はタイミングに影響せず NaN も生まない）
* 主指標は `us_p50`（1 発射あたりの µs）

---

## 5. GPU の選び方

埋まった GPU を指すと重みアップロードで失敗する:

```text
shelf-bench pp: upload: .../step_executor.cpp:36: malloc weight bf16: out of memory
```

```bash
rocm-smi                       # VRAM% / GPU% を見る
# GPU0 が埋まっていれば空きを指す
./build/hip/apps/shelf-bench pp ... --gpu=1
```

空き 1 GPU で開発モデル `Qwen3.5-9B` bf16 を動かす目安（2026-10-28 実測）: 重み payload 19.31 GB
+ アリーナ 2,299 MiB（rows=64 自動算出／最小必要 777〜784 MiB）。容量の内訳は `docs/spec/bench.md` §11。

---

## 6. CLI

| コマンド / オプション | 既定 | 説明 |
| --- | --- | --- |
| `pp` | — | prefill を測る（§3.1） |
| `tg` | — | decode を測る（§3.2） |
| `kernel <name>` | — | 単一カーネルを測る（§4） |
| `--model-dir DIR`（`--model`） | 必須 | モデルディレクトリ（`kernel` では不要） |
| `--prompt-tokens N`（pp） | 128 | 供給するトークン数 |
| `--context N`（tg） | 128 | prefill するコンテキスト長 |
| `--tokens N`（tg） | 8 | 測定するデコード歩数 |
| `--prefill-chunk N` | 0 | 1 ステップの行数。0 = `min(prompt|context, 64)` |
| `--runs N`（pp） | 3 | 計測回数（中央値を主指標にする） |
| `--warmup N` | 0（`kernel` の gemm/rmsnorm/elementwise は 100、KV 系は 30） | 破棄する回数 |
| `--samples N` / `--launches N`（kernel） | 50 / 20 | サンプル数 / 1 サンプル内の発射数 |
| `--mode M`（tg） | `forward` | `forward`（logits 不取得）\| `greedy`（毎歩 argmax） |
| `--gpu N` | 0 | GPU 番号 |
| `--seed N` | 1234 | 合成トークン列のシード |
| `--arena-mib N` | 0 | アリーナ（MiB）。**0 = グラフから自動算出**、正の値で固定 |
| `--output PATH` | なし | CSV を追記（ファイルが無ければヘッダを書く） |
| `--help` | — | 使い方を表示（GPU 非依存・exit 0） |

* 値の形式は `--key value` と `--key=value` の両対応。
  `--help` は値を取らないフラグ（`shelf-bench kernel --help` も動く）

---

## 7. 出力

### 7.1 stdout（`key=value`）

単位は行名に内包（`wall_ms_median` / `wall_us_per_token` / `us_p50` など）。
最初の行が `shelf-bench <コマンド>`、次が `schema_version`。

### 7.2 CSV（`--output PATH` のみ）

追記モード。ファイルが無い・空ならヘッダを書く。**1 コマンド = 1 行**
（`gemm` は 1 シェイプ = 1 行）。

**`schema_version` はコマンドごとに違うので、同じファイルへ混ぜないこと**:

| コマンド | schema_version | 列（先頭から） |
| --- | --- | --- |
| `pp` | 1 | `schema_version,device,gpu,model,prompt_tokens,prefill_chunk,runs,warmup,arena_mib,wall_ms_median,wall_ms_min,wall_ms_p10,wall_ms_p90,wall_ms_stddev,wall_us_per_token,wall_tokens_per_sec` |
| `tg` | 2 | `schema_version,device,gpu,model,context,tokens,prefill_chunk,warmup,arena_mib,mode,context_prefill_ms,wall_ms_total,wall_us_per_token,wall_tokens_per_sec,first_token_ms,step_ms_mean,step_ms_p50,step_ms_p90,step_ms_max` |
| `kernel` | 3 | `schema_version,device,gpu,kernel,shape,warmup,samples,launches,us_min,us_p10,us_p50,us_p90,us_p95,us_mean,us_stddev,throughput,throughput_unit` |

```bash
./build/hip/apps/shelf-bench pp ... --output /tmp/bench-pp.csv   # 追記（ヘッダは 1 回のみ）
```

---

## 8. 判定と終了コード

| コード | 意味 |
| --- | --- |
| `0` | 成功、または `--help` |
| `1` | 実行時エラー（デバイス・ロード・カーネル発射失敗、HIP 非対応ビルドでの本体実行） |
| `2` | 用法の誤り（未知コマンド・未知カーネル名・必須オプション欠落・数値不正・余分な位置引数） |

記録の目安:

* `pp` / `tg`: 同じ条件で **2 回以上**測り、`wall_ms_median`（`tg` は `step_ms_p50` も）を比較。
  1 回の値は `run_wall_ms[i]` / `step_ms_*` の散らばりと合わせて見る
* `kernel`: `us_p50` を主指標とし、`us_p95` と `us_stddev` で揺れを見る
  （今回のような小さな `samples` では p95 にノイズが載ることがある）
* 比較の起点として使う場合も、**フェーズ 0 で再測定**すること（本機の実測値は参考値）

---

## 9. トラブルシュート

| 症状 | 対処 |
| --- | --- |
| `malloc weight bf16: out of memory` | GPU が埋まっている。`rocm-smi` で確認し `--gpu` を変える |
| `arena exhausted`（+ `hint: ... raise --arena-mib above the current N MiB`） | アリーナは行数に比例する（バンプ割当でステップ内再利用なし）。自動算出でも足りない場合のみ `--arena-mib` を増やす。実効値は出力の `arena_mib` 行 |
| `this build has no HIP backend` | `cpu` ビルドのため本体は動かない。`hip` プリセットでビルドする（`--help` は cpu 版でも可） |
| `kernel: unknown name: ...` | 対象名は `shelf-bench kernel --help` で確認（exit 2） |
| CSV の列が食い違う | `pp` / `tg` / `kernel` の schema_version（1/2/3）を同じファイルに混ぜている。コマンド別に分ける |
| 実行が途中で詰まる・`HSA_STATUS_ERROR_MEMORY_FAULT` | 一度 GPU を解放して再実行。再現する場合は `docs/spec/bench.md` §13 の調査記録と照合して issue 化 |
| `prefill: ...` が極端に遅い | `--prefill-chunk` が小さい（既定は `min(prompt, 64)` で既に上限）。まず `--runs 1 --warmup 0` で素の値を取る |

---

## 10. 未実装（計画のみ）

* **GPU event 計測**（`gpu_ms_*`）— `stream_` が private で bench からイベントを記録できない。
  `HipStepExecutor` にイベント計測を足せば CSV に列を追加する
* **`--device cpu`** — 現行は HIP 専用（CPU 経路は `shelf-run` / `shelf-validate`）
* **JSON 出力** — 現行は stdout + CSV のみ（JSON は `docs/guide/validate.md` 側）
* 複数 GPU / TP / 常駐モデルホスト / roctx / `--gap-keepalive`
  （`phaseshift-bench` にあるもの。要件が出た時点で追加）
* `step_executor.cpp` の既存 nodiscard 警告（`docs/spec/bench.md` §13）
