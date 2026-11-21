# Measurement methodology

RDNA4 (gfx1201) 上で最適化を計測・採否判断するときに今後も使う測定原則の正本である。
ドキュメントの役割境界は次のとおり。

- `docs/perf/methodology.md`（本書）: 測定原則・比較手順の正本
- `docs/perf/current.md`: 現行ビルドで再現可能な性能基準値（新しい計測なしに数値を更新しない）
- `docs/research/`: R&D の実験経緯・採否判断・棄却根拠の正本
- [../user/bench.md](../user/bench.md): `shelf-bench` のサブコマンドとオプション一覧

個別 Gate の時系列ログや個別カーネルの試行錯誤はここへ書かない。

## 1. 隔離と環境固定

- **1 GPU = 1 bench process**。同一 GPU 上に複数 workload を重ねない
  （clock / 帯域の非対称汚染）。
- **GPU 排他ロック**: 共有ホスト上では `flock /tmp/r9700.lock` などの `/tmp/*.lock` を取得して実行する
  （`tools/quality/kld-vs-bf16.sh` も `/tmp/r9700.lock` を自動取得し、すでにロック内にある場合は `--already-locked` を渡す）。
- 複数 GPU を使う場合も、ある GPU では同時に 1 つだけ動かす。
- GPU0 は別 workload と共有していることがある。隔離された device を明示する。
- 同一コマンドで条件を揃える: seed / warmup / samples / launches / rows / dtype / model。

## 2. baseline / candidate

- **baseline と candidate は交互に測る**（run1: base→cand, run2: cand→base, ...）。
- **cross-GPU 比較は禁止**（個体差・clock state が乗る）。
- baseline は **production incumbent**（その時点の selector が実際に選ぶ経路）。
  「同じ dtype の類似 kernel」を代役にしない。
- **correctness / reference kernel を p50 baseline にしない**。参照実装は数桁遅く、
  比較に意味がない。
- fused candidate の比較対象は unfused production sequence（現行 selector の連結）。

## 3. warmup と再測定

- 5-run の p50 中央値 + **run 間安定性**で判定する。
- run 間で p50 が 30% 以上ぶれたら **FAIL / BLOCKED として調査**する
  （並列汚染の検出器でもある）。
- 数値を緩めて通さない。再測定で汚染を除く。
- micro kernel（4〜20µs 級）は idle clock ramp の影響が大きい。**preroll を入れる**
  （baseline 側にも同条件で）。
- 初回プロセスは GPU クロックのランプアップで遅く出るため warmup で除外する。

## 4. production reachability

- source tree にある / CMake に入る / public bench から叩ける ≠ production reachable。
- reachability は **runtime selector / matcher の rule table** から導出する。
- rule が無い経路は NON_PRODUCTION。release blocker に数えない。
- **「負けるカーネルを選ぶ rule」は promotion contract violation**（本物の FAIL）。
  勝つ側を選ぶよう rule を直すのが正解で、カーネルを微修正しても解決しない。

## 5. 測定 harness 自体を疑う

実測で見つかった harness 側の欠陥:

- device 選択が weight / workspace 確保より後 → `--device N` が効かず別 GPU で確保。
- CLI option の非対称 → usage error で perf が欠測。
- `--check` が no-op（受理されるが何も比較しない）。
- bench 入力がカーネルの前提を満たしていない。
- 複数 kernel バッチの GPU event 両端計測が wall-clock に対して非決定論的にずれる。
  その場合は per-op event か wall-clock を使う。

## 6. E2E と micro benchmark を混同しない

- micro benchmark は単体カーネルの ceiling と律速（帯域 / compute / launch）を切り分ける。
- E2E（pp / tg / speculative decode）は selector 連結を含む現実の性能を示す。
- 片方の改善を他方の結論として流用しない。
- **token 一致を確認する**。正しさを確認せずに速度だけを見ない。
- correctness path の出力と production incumbent の出力を混同しない。

## 7. launch floor と cache residency

- 小カーネルは launch 律速になりうる。1 dispatch あたりの固定費を先に疑う。
- decode は per-token weight traffic が L3 容量を超えるなら DRAM 帯域律速。
  L3 に収まる状態（state 等）では compute 律速に切り替わる。どちらかを取り違えない。
- 「ISA で説明できるから速くなる」は誤り。**必ず同一条件で A/B を取る**。

## 8. 報告

- 性能に関する主張には計測条件（revision / commit / GPU / コマンド）と数値を添える。
- 負結果も残す。採否判断と理由を記録する。
- 新しい計測をしていない数値を「更新」しない。
## 9. カテゴリ別 serving 計測（BetterBench）

`vendor/BetterBench`（外部 harness、`GGZ14/BetterBench`。gfx1201 側の参照実装と
対になる計測器）を OpenAI 互換 endpoint に向けて回す。token/s 1 本では見えない「カテゴリ別の受理数」と TTFT が出る。

```sh
pip install -e vendor/BetterBench
# server を起動（DFlash2 を有効化、KV fp8）
./build/shelf-server --model-dir models/Qwen3.8-27B-FP8 \
    --model Qwen3.8-27B-FP8 --port 8123 --compute ./build/shelf-compute \
    --max-seq-len 8192 --arena-gib 30 --device 1 --kv-cache-dtype fp8 \
    --max-concurrent-requests 4 \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7
BETTERBENCH_HOME=/tmp/bb betterbench run --endpoint http://127.0.0.1:8123/v1 \
    --model Qwen3.8-27B-FP8 --quick --decode \
    --out /tmp/bb/run.json --note gpu=R9700x1 --note quant=fp8 --note kv=fp8 --note spec=7
```

- corpus は `vendor/BetterBench/corpus/v1`（コード/json/math/summarization/file_edit/
  reasoning/chat/prose）。sampling 既定は temp 0.7 / top_p 0.95 / top_k 20、nonce により
  prefix cache は未ヒット（空）の状態で開始する。
- 参照側の公開表と比較するときは harness の版
  （`betterbench --version`）と corpus 版、`--max-seq-len`、pass 数を併記する。
- **update 粒度の差**: こちらは `shelf-compute` がトークン 1 個ごとに SSE を出すため
  BetterBench は "one token per update" と判定し、update gap の代わりに ITL を報告する。
  `tok/update` は比較に使えない。**decode t/s・TTFT・prefill は粒度に依らず比較できる**。
- **リクエスト数の上限**: per-request state は起動時に `max_concurrent_requests` 分
  pre-allocate されるため、1 リクエストあたりの arena 消費は 0 である。
  `--max-seq-len 65536` でもカテゴリをまたいで連続計測できる。
  KV 容量は arena 予算から導出される（§5.4）。カテゴリ毎の server 再起動は
  依然として行うが、理由は process 間で状態を持ち越さないためである。
- **DFlash2 の内訳診断**: `SHELF_DFLASH2_SERVE_STATS=1` で server 終了時に
  `DFLASH2_STAGE`（1 round 平均の round / draft_gpu / verify_gpu / …）、
  `DFLASH2_POSITION=`（256 トークン刻みの位置バケット: rounds / 受理 / round_ms）、
  `DFLASH2_VERIFY_ROWS_PER_ROUND` が出る。カテゴリ別 decode t/s の差が
  「受理（tok/round）」と「round 時間」のどちらで説明できるかをこれで切り分ける。
  `SHELF_BATCH_TRACE=1` は step ごとの `arena_used` を出し、
  per-request の増加がないこと（資源規律の回帰）を確認する。

## 10. モデル精度の比較

GDN decomposition、state の型、量子化などモデル出力に影響する変更は、
**GSM8K と HumanEval の両方**で teacher-forced logits を比較する。
片方だけでの判定は認めない。PPL の改善／悪化はモデルとコーパスに依存し、
単一コーパスの符号で採否を決められない。

### 10.1 corpus の作成

両方とも test split の入力列を連結し、Qwen tokenizer で 2048 トークンに切り、
PSKLDTOK 形式で保存する。

| データセット | repo | ファイル | 列 |
| --- | --- | --- | --- |
| GSM8K | `openai/gsm8k` | `main/test-00000-of-00001.parquet` | `question` |
| HumanEval | `openai_humaneval` | `openai_humaneval/test-00000-of-00001.parquet` | `prompt` |

```python
import hashlib, struct
from huggingface_hub import hf_hub_download
import pandas as pd
from transformers import AutoTokenizer

def make(repo, path, column, out):
    source = hf_hub_download(repo, path, repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(source)[column].tolist())
    tok = AutoTokenizer.from_pretrained("models/Qwen3.8-27B-Quark-AWQ-MXFP4")
    ids = tok.encode(text, add_special_tokens=False)[:2048]
    data = b"PSKLDTOK" + struct.pack("<IQ", 1, len(ids)) + struct.pack("<%di" % len(ids), *ids)
    open(out, "wb").write(data)
    print(out, hashlib.sha256(data).hexdigest())

make("openai/gsm8k", "main/test-00000-of-00001.parquet", "question", "/tmp/shelf-gsm8k.psktok")
make("openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", "prompt", "/tmp/shelf-humaneval.psktok")
```

corpus の SHA-256 を結果に併記する。

### 10.2 比較条件

- 1 GPU = 1 bench process。
- `--context 128 --positions 128 --prefill-chunk 4096 --kv-cache-dtype bf16`。
- baseline は環境変数なし（lossy が既定）、candidate は `SHELF_GDN_RECURRENCE_EXACT=1`。
- 両者で `--dump` を書き、candidate 側の `--compare` に baseline を渡す。

```sh
B=./build/shelf-bench
M=models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw
T=/tmp/shelf-gsm8k.psktok
$B quality --model-dir $M --tokens-file $T --context 128 --positions 128 \
    --prefill-chunk 4096 --kv-cache-dtype bf16 --dump /tmp/quality-lossy.bin
SHELF_GDN_RECURRENCE_EXACT=1 $B quality --model-dir $M --tokens-file $T \
    --context 128 --positions 128 --prefill-chunk 4096 --kv-cache-dtype bf16 \
    --dump /tmp/quality-exact.bin --compare /tmp/quality-lossy.bin
```

HumanEval と MXFP4（`models/Qwen3.8-27B-Quark-AWQ-MXFP4`）も同じ手順で実施し、
2 コーパス × 対象モデルの結果を並べる。

### 10.3 報告する値

- `QUALITY_KL_MEAN`、`QUALITY_TOP1_AGREEMENT`、`QUALITY_TOP5_OVERLAP`
- `QUALITY_PPL_BASELINE`、`QUALITY_PPL_CANDIDATE`、`QUALITY_PPL_DELTA_PCT`
- `QUALITY_LOGIT_MAX_ABS_DELTA`
- corpus の SHA-256、モデル、GPU、KV dtype、chunk

PPL の符号がモデル／コーパスで一貫しない場合は「精度向上」と結論しない。
top-1 と KL を主指標とし、PPL は参照値として扱う。
