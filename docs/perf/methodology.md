# Measurement methodology

RDNA4 (gfx1201) 上で最適化を計測・採否判断するときに今後も使う測定原則の正本である。
ドキュメントの役割境界は次のとおり。

- `docs/perf/methodology.md`（本書）: 測定原則・4 トラック統一仕様・比較手順の正本
- `docs/perf/current.md`: 現行ビルドで再現可能な性能基準値（新しい計測なしに数値を更新しない）
- `docs/research/`: R&D の実験経緯・採否判断・棄却根拠の正本
- [../user/bench.md](../user/bench.md): `shelf-bench` および統一ベンチマークスイートの使い方とオプション一覧

個別 Gate の時系列ログや個別カーネルの試行錯誤はここへ書かない。

## 1. 隔離と環境固定

- **1 GPU = 1 bench process**。同一 GPU 上に複数 workload を重ねない
  （clock / 帯域の非対称汚染）。
- **GPU 排他ロック**: 共有ホスト上では `flock /tmp/r9700.lock` などの `/tmp/*.lock` を取得して実行する
  （`tools/bench/shelf_bench_suite.py` は `--lock-file PATH` で実行全体に `flock` の排他ロックを取得でき、`tools/quality/kld-vs-bf16.sh` も `/tmp/r9700.lock` を自動取得し、すでにロック内にある場合は `--already-locked` を渡す）。
- 複数 GPU を使う場合も、ある GPU では同時に 1 つだけ動かす。
- GPU0 は別 workload と共有していることがある。隔離された device を明示する（単一 R9700 で native 測定する場合は `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` 等で対象 GPU を明示する）。
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

- 性能に関する主張には計測条件（revision / GPU / コマンド / sampling / 量子化 / KV 設定）と数値を添える。
- 負結果も残す。採否判断と理由を記録する。
- 新しい計測をしていない数値を「更新」しない。
- 欠測した条件・トラック・比較値は必ず「未測定」と記載し、推計値や別条件の数値で埋めない。

## 9. 統一 4 トラック体系と `shelf_bench_suite.py`

性能計測は目的別に **Track A / B / C / D** の 4 トラックへ分離し、`tools/bench/shelf_bench_suite.py` と凍結実タスクコーパス `tools/bench/real_v1/` を共通契約として運用する。

### 9.1 開発サイクル別の使い分け

| 開発フェーズ・目的 | 主に使うトラック | 確認する内容 |
| --- | --- | --- |
| 日常のカーネル・Graph・融合最適化 | **Track A** | 非投機 `pp`/`tg` と、`temperature=0`・1,024+ token の投機固定チェック（`--ab-compare` による AB/BA 交互実行と出力 SHA-256 一致） |
| バッチ・スケジューラ・サーバー改修 | **Track D** | HTTP streaming closed-loop 同時 1/4/8/16（`steady` / `mix`）の TTFT・E2E・集約 tok/s・完了率 |
| 節目・公開数値更新 | **Track C ＋ Track B** | 凍結 `real-v1` 全 30 課題（Track C）と BetterBench 互換 `full` 20 pass/category（Track B）を揃え、`render` で統合出力 |

### 9.2 共通実行契約・JSON スキーマ・投機 1,024+ token 基準

- **実行形式**: `python3 tools/bench/shelf_bench_suite.py <サブコマンド> [引数]`（サブコマンドは `track-a` / `track-b` / `track-c` / `track-d` / `render`）。
- **共通オプション**（サブコマンドの前後に配置可能）:
  - `--output PATH`: 結果の保存先（省略時は標準出力）。
  - `--lock-file PATH`: 実行全体に `flock` の排他ロックを取得する。
  - `--dry-run` / `--list`: 同義。実測を行わず実行計画または課題一覧を JSON で返す。
- **HTTP 系の既定値**: `--model` は `default`（Track A〜D）、`--endpoint` は `http://127.0.0.1:8080/v1`（Track B/C/D）、`--seed` は `42`（Track A/C/D）。
- **共通 JSON 構造**: トップレベルに `schema_version`（`1`）、`track`（`A` / `B` / `C` / `D`）、`metadata`（`timestamp_utc`, `git_revision`, `dirty`, `gpu`, `rocr_visible_devices`, `rocm_path`, `arguments`）を持つ。欠測値は `null` とし、試行失敗や A/B 不一致の判定は JSON のカウンタを参照する。
- **共通 streaming 計測**:
  - 最初と最後の非空の生成更新（思考チャネルを含む）の到着時刻差を用い、`decode_tok_s = (completion_tokens - 1) / 到着時間差(秒)` とする。
  - `completion_tokens` を取得できない場合は非空更新数を記録し、`completion_tokens_approximate=true`、`decode_tok_s=null`、`tokens_per_update=null` とする。
  - `output_sha256` は思考チャネルと回答チャネルを生成順に連結した UTF-8 テキストの SHA-256。`answer_text` は回答チャネルのみを保持する。
  - round 統計（`ms_per_round`, `accepted_drafts_per_round`, `committed_tokens_per_round`）は SSE 拡張 `speculative_stats` から取得する。DFlash2 診断ログからは `DFLASH2_ROUND_MS` / `DFLASH2_ACCEPTED_DRAFTS` / `DFLASH2_GENERATED_TOKENS` を `DFLASH2_ROUNDS` で割って算出し、合計カウンタがない場合は `DFLASH2_STAGE`（`round`）/ `DFLASH2_MEAN_ACCEPTED` / `DFLASH2_EMITTED_PER_ROUND` を読み取る（round 数が 0 の場合は `null`）。
- **投機デコード 1,024+ token 出力基準**: 256 token 程度の短出力ではドラフト受理率の試行間ブレが大きく、定常の受理特性を代表しない。そのため投機を含む計測（Track A の投機固定チェック、Track C の期待出力、Track D の `--max-tokens` 下限および `long_window_tok_s`）は **1,024〜2,048+ token 出力を標準**とし、Track C では繰り返しループ検出フラグを併用して異常な高受理・無限反復を分離する。

### 9.3 Track A（基礎性能・形状固定）

非投機の `shelf-bench`（`pp` / `tg` / `batch`）と、投機ありの日常固定チェックを 1 トラックにまとめる。

- **非投機（native）**: `--shelf-bench`（既定 `build/shelf-bench`）、`--model-dir`、`--device`（既定 `0`）、`--tensor-parallel`（`1|2|4`、既定 `1`）、`--tokens-file`（`pp`/`tg` 用 PSKLDTOK）、`--pp-lengths 128,512,2048,8192`、`--contexts 0,2048,8192`、`--batch-sizes 1,4,8,16`、`--tg-tokens 256`、`--repeats 5`、`--warmup 1`。
  - **`shelf-bench batch` の契約**: 共通 prefix を seed した後に各要求を逐次処理する（`batch_contract`: `max_scheduled_requests=1`, `timing=prefix_seed_and_serial_requests`, `parallel_tg_supported=false`）。並列 continuous batching の速度ではないため Track D と混同しない。正の prefix と suffix が必要なため context 0/1 は `unsupported=true`（`failures` にも計上）。
- **投機固定チェック**: `--spec-endpoint URL` または `--spec-cmd COMMAND` で追加する（投機のみ実行する場合は `--model-dir` を省略可能）。
  - `temperature=0`、**`--spec-max-tokens` の既定値・下限はともに `1024`**。実測 `completion_tokens` が 1,024 未満または未返却の場合は失敗として記録する。
  - `--spec-cmd` はシェル展開せず起動し、stdin に chat 要求 JSON を渡し、stdout から `output_text`, `completion_tokens`, `decode_tok_s`（および任意の round 統計、または stderr の DFlash2 診断ログ）を受け取る。
  - `ms_per_round` / `accepted_drafts_per_round` / `committed_tokens_per_round` と出力 SHA-256 を記録する。
- **AB/BA 交互比較**: `--ab-compare` に `--baseline-bench PATH` や `--baseline-spec-endpoint URL` / `--baseline-spec-cmd COMMAND` を組み合わせ、各条件を AB、BA の順で交互実行する。比率は spec が `decode_tok_s`、native が `wall_tok_s` の `candidate / baseline`。両方成功して SHA-256 が不一致の場合は `raw_candidate_over_baseline` を残し、採用用の `candidate_over_baseline` を `null` とする（native の `pp`/`tg` は全出力 ID を公開しないためハッシュ照合は `null`、`batch` は全要求の ID 列、`spec` は生成テキストを照合）。

### 9.4 Track B（外部比較: BetterBench 互換）

`vendor/BetterBench`（外部 harness、`GGZ14/BetterBench`）を OpenAI 互換 endpoint に向けて実行し、他エンジンと同条件で比較する。

- **既定条件と smoke 分離**: `vendor/BetterBench` の `decode` フェーズを既定 **20 pass/category・3 warmup**（`mode: full`）で実行する。`--quick` は 5 pass・1 warmup の **`quick_smoke` 扱い**として明確に分離し、公開比較の `full` と混在させない。
- **全 8 カテゴリ並記**: corpus は `vendor/BetterBench/corpus/v1` の全 8 カテゴリ（`code`, `file_edit`, `json`, `math`, `prose`, `reasoning`, `summarization`, `chat`）。単一代表値だけでなく、各カテゴリの `decode_tok_s`・`ttft_ms` の `median` / `iqr` / `min`〜`max`、失敗率・打切り率、`long_completed`（1,024 token 以上の成功試行数）、round 統計（`ms_per_round`, `accepted_drafts_per_round`, `committed_tokens_per_round`）を並記する。コピーしやすい高受理カテゴリによって見かけの速度が高く見える現象を、カテゴリ別分布と受理長・round 時間で可視化する。
- **診断ログと取り込み**:
  - `--betterbench-python PATH` で vendor CLI 実行用の Python を指定できる（スイート本体と `--import-json PATH` は標準ライブラリのみで動作するが、vendor CLI 実行には NumPy が必要）。
  - `--server-log [CATEGORY=]PATH`（繰り返し指定可能）で DFlash2 診断ログを取り込む。カテゴリ独立測定のログには `CATEGORY=` を付与し、無指定ログは `global_server_stats` に入る。
  - `--conditions PATH` で同条件比較用の構成 JSON（必須キー: `gpu`, `weights`, `quantization`, `kv`, `tokenizer`, `chat_template`, `thinking`, `cache`, `tp`, `drafter`, `eos`, `context_limit`, `corpus_sha256`, `server_flags`）を記録する。`render --compare-b` は双方が `full` であり、指定条件・BetterBench 版・コーパス版・モデル名・`corpus_hash`・pass 数・warmup・sampling・nonce 設定が一致した場合のみ比較値と速度比を表示する。
- **測定上の注意**:
  - **update 粒度の差**: `shelf-compute` はトークン 1 個ごとに SSE を送出するため、BetterBench は 1 token/update として ITL を報告する。`tok/update` は送信単位であり受理長ではなく比較に使えない。`decode_tok_s` と `ttft_ms` を主比較軸とする。
  - **server 再起動と診断環境変数**: per-request state は起動時に `max_concurrent_requests` 分 pre-allocate されるが、プロセス間で状態を持ち越さないためカテゴリ独立測定では server を再起動する。`SHELF_DFLASH2_SERVE_STATS=1` で終了時に `DFLASH2_STAGE` / `DFLASH2_POSITION=` / `DFLASH2_VERIFY_ROWS_PER_ROUND` を出力でき、`SHELF_BATCH_TRACE=1` で step ごとの `arena_used` の回帰を確認できる。

### 9.5 Track C（実タスク: 凍結 `real-v1`）

実運用に近い長文・日英・複数ターンを含む凍結コーパス `tools/bench/real_v1/` を用いる。

- **構成**: 5 領域（`code_edit`, `code_review`, `design_debug`, `tech_prose`, `multi_turn`）× 日英（`ja` / `en`）× 3 入力長（`short_2k` / `mid_8k` / `long_16k`）= **計 30 課題**。`multi_turn` は `user -> assistant -> user` の複数ターン履歴を送信する。
- **凍結資料と入力長**: `short_2k.txt` / `mid_8k.txt` / `long_16k.txt` と `tasks.json` の SHA-256 を `manifest.json` に固定している。資料単体の実測長（Qwen3.8-27B-BF16 tokenizer）は 1,947 / 7,898 / 16,267 token、chat template 適用後の参照入力長は 2,083〜2,171 / 8,031〜8,128 / 16,408〜16,490 token。
- **サンプリング・実行条件**: 全課題 `thinking=true`、`temperature=0.7`、`top_p=0.95`、`top_k=20`、`max_tokens=2048`、期待出力 `>= 1,024 token`（`min_expected_tokens=1024`）。既定は `--repeats 1`、`--warmup 0`、測定 seed は `--seed + repeat`（`--task ID` の繰り返し指定で個別課題の絞り込み可）。
- **ループ検出と集計**: 繰り返しループ検出フラグ（`loop_flagged`, `repeated_5gram_ratio`, `max_identical_lines`, `tail_period_tokens`）を記録し、`all` / `non_loop` / `long_output`（実測 1,024 token 以上）/ `long_non_loop` の 4 集計を出力する。総合マクロ平均（`macro_decode_tok_s`）は課題内中央値 → セル内算術平均 → 全 30 セル等重み平均とし、**必須セルが 1 つでも欠測している場合は `null`** とする。
- **品質照合**: 各課題は固定 rubric を持ち、採点済み JSON 配列（`{task_id, output_sha256, passed}`）を `render --reviews` に渡すことで出力ハッシュ照合済みの判定件数・合格件数を併記できる。

### 9.6 Track D（サービング / Continuous Batching）

OpenAI 互換 HTTP streaming endpoint に対し、closed-loop で同時要求を投入してサービング性能を測る。

- **条件**: `--concurrency 1,4,8,16`、`--requests N`（各同時数で既定は同時数 × 4）、`--mode steady|mix|both`（既定 `both`）、`--max-tokens 2048`（下限 `1024`）、`--temperature 0`。
- **モード**: `steady` は短資料を継続投入する。`mix` は長資料も混在投入し、最初の長資料は短要求の最初の更新または終了を待って投入する（同時数 1 では交互の逐次処理）。各要求に異なる接頭辞を付与して prefix cache の意図しないヒットを防ぐ。
- **指標**:
  - 遅延 4 指標（いずれも `p50` / `p95`）: `ttft_ms`、`e2e_ms`、`itl_ms`（`tokens_per_update == 1` の観測のみ）、`update_gap_ms`（複数 token/update の観測）。
  - スループットと完了率: 測定窓の集約スループット `aggregate_tok_s`（全要求の `completion_tokens` 合計 / 測定窓秒数。件数不明なら `null`）、全要求が成功かつ各実出力が 1,024 token 以上の窓限定の `long_window_tok_s`、要求別 decode 中央値 `decode_tok_s_median`、`completion_rate`、`long_completed`。

### 9.7 公開時の報告フォーマット（`render`）

`render` サブコマンドに各トラックの JSON を渡すと、公開時に並記すべき 4 断面を **Track C → Track B → Track A → Track D** の順で 1 つの Markdown レポートへ統合する（入力ファイルの順序は任意）。

1. **Track C（実タスク代表値）**: `real-v1` の `all` / `non_loop` / `long_output` / `long_non_loop` マクロ平均、セル別表、品質レビュー合格数。
2. **Track B（外部標準 8 カテゴリ別レンジと同条件比較）**: 全 8 カテゴリの median / IQR / min〜max・受理長・round 時間、`--compare-b` による同条件比較。
3. **Track A（基礎性能 & `ms/round`）**: 非投機 `pp`/`tg`/`batch`、投機固定チェックの `decode_tok_s`・`ms_per_round`・受理長、AB/BA 比較比率。
4. **Track D（サービング）**: `steady` / `mix` 別の同時数 1/4/8/16 の集約 tok/s・要求別 decode 中央値・TTFT/E2E/ITL（または update gap）・完了率。

省略されたトラック、欠測セル、同条件を確認できない比較値はすべて「未測定」と出力され、推計値で埋めない。

### 9.8 実行コマンド例

```sh
# Track A: 非投機 pp/tg/batch + 投機固定チェック（単一 R9700 の例）
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
python3 tools/bench/shelf_bench_suite.py track-a \
  --lock-file /tmp/r9700.lock \
  --shelf-bench ./build/shelf-bench \
  --model-dir models/Qwen3.8-27B-FP8 \
  --tokens-file /tmp/corpus.psktok \
  --device 0 --tensor-parallel 1 \
  --spec-endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --spec-max-tokens 1024 \
  --output /tmp/track_a.json

# Track A: AB/BA 交互比較（投機エンドポイントのみ比較する例）
python3 tools/bench/shelf_bench_suite.py track-a \
  --lock-file /tmp/r9700.lock \
  --ab-compare \
  --baseline-spec-endpoint http://127.0.0.1:8080/v1 \
  --spec-endpoint http://127.0.0.1:8081/v1 \
  --model Qwen3.8-27B-FP8 \
  --spec-max-tokens 1024 \
  --repeats 5 \
  --output /tmp/track_a_ab.json

# Track B: BetterBench 互換 full（20 pass/category・3 warmup）
python3 tools/bench/shelf_bench_suite.py track-b \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --conditions /tmp/conditions.json \
  --server-log /tmp/server_dflash2.log \
  --output /tmp/track_b.json

# Track B: smoke のみ（公開比較には使わない）
python3 tools/bench/shelf_bench_suite.py track-b \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --quick \
  --output /tmp/track_b_smoke.json

# Track C: 凍結 real-v1（全 30 課題）
python3 tools/bench/shelf_bench_suite.py track-c \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --seed 42 --repeats 1 --warmup 0 \
  --output /tmp/track_c.json

# Track D: サービング（steady / mix、同時 1/4/8/16）
python3 tools/bench/shelf_bench_suite.py track-d \
  --lock-file /tmp/r9700.lock \
  --endpoint http://127.0.0.1:8080/v1 \
  --model Qwen3.8-27B-FP8 \
  --concurrency 1,4,8,16 \
  --mode both \
  --max-tokens 2048 \
  --temperature 0 \
  --output /tmp/track_d.json

# render: C -> B -> A -> D の順で Markdown レポートを生成
python3 tools/bench/shelf_bench_suite.py render \
  /tmp/track_a.json /tmp/track_b.json /tmp/track_c.json /tmp/track_d.json \
  --compare-b /tmp/other_track_b.json \
  --reviews /tmp/real_v1_reviews.json \
  --output /tmp/report.md
```

実行前の計画確認や課題一覧の取得には各サブコマンドに `--dry-run`（または `--list`）を付ける。

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
