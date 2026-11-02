# Measurement methodology

RDNA4 (gfx1201) 上で最適化を計測・採否判断するときに今後も使う測定原則の正本である。
`docs/research/20261022/optimization_findings.md` など R&D の記録は、測定原則について
本ドキュメントを正本として参照する。個別 Gate の時系列ログや個別カーネルの結果は
ここへ書かない（R&D 記録の正本は `docs/research/`）。

## 1. 隔離と環境固定

- **1 GPU = 1 bench process**。同一 GPU 上に複数 workload を重ねない
  （clock / 帯域の非対称汚染）。
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
    --max-concurrent-requests 8 \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7
BETTERBENCH_HOME=/tmp/bb betterbench run --endpoint http://127.0.0.1:8123/v1 \
    --model Qwen3.8-27B-FP8 --quick --decode \
    --out /tmp/bb/run.json --note gpu=R9700x1 --note quant=fp8 --note kv=fp8 --note spec=7
```

- corpus は `vendor/BetterBench/corpus/v1`（コード/json/math/summarization/file_edit/
  reasoning/chat/prose）。sampling 既定は temp 0.7 / top_p 0.95 / top_k 20、nonce により
  prefix cache は cold。
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

