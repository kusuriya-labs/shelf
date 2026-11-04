# 残課題（homework）

Static YaRN（`--rope-factor`）の Target 対応は Gate 0〜4 まで完了し、main に入っている。
ここには**未実施・未接続の作業**だけを残す。実装済みの契約は
[developer/qwen35.md](developer/qwen35.md) §2.5、利用方法は
[user/compute.md](user/compute.md)、経緯と実測は
[research/20261023/yarn_context_extension.md](research/20261023/yarn_context_extension.md) を正本とする。

## 1. Gate 5 — Drafter（DFlash2）の長位置対応

- Target YaRN + DFlash2 の併用を許可した。Drafter は target の factor を継承せず、自身の
  checkpoint の RoPE（`rope_theta`、`rope_type=default`）に従う。
- `dflash2/detail/rope_device.h` の角度計算を `double(position) * inv_freq` へ変更し、
  262,144 以上の位置でも cos/sin の精度を保つ。
- 残作業:
  - DFlash2 checkpoint 自体が `rope_type=yarn` を持つ場合のプロファイル解決
    （現行 checkpoint は `default` のみ）。
  - Target 1x/2x/4x との組合せで acceptance rate と tokens/s を実測する。

## 2. Gate 6 — 2x の長文 E2E（品質・性能）

- 容量は実測済み: `--rope-factor 2 --kv-cache-dtype fp8 --tensor-parallel 2 --max-seq-len 524288` で起動・生成に成功し、
  rank あたり使用 23.11 GiB／arena 30.63 GiB。
- 必要な作業:
  - 524,288 トークン級の入力で prefill / decode の速度を測り、`docs/perf/current.md` に記録する。
  - 長文の情報検索・位置参照・品質劣化を評価し、結果を `docs/research/` に残す。
  - 品質評価は **BF16 KV** の短文比較（Native vs 2x）と、**FP8 KV** の長文（容量のための構成）を分離して扱う。
- 4x（1,000,000）は方針により**実装のみ**とし、E2E 検証は行わない。1M を「対応済み」と記載しない。

## 3. FP8 KV の品質再評価

- `docs/perf/current.md` の記録（2026-10-20）では FP8 KV は品質ゲート FAIL
  （HF FP8: ppl +28.12% @ctx128 / +29.96% @ctx1600）。
- 現行ビルドで再測し、YaRN 2x と組み合わせたときの劣化量を分離して記録する。
- 再測には `shelf-bench quality` / `tools/quality_gate.py` を使う。

## 4. 未接続・未検証の実装事項

| 項目 | 内容 |
| --- | --- |
| correctness の tree positions | `DeviceBatchContext::rope_positions` は書き込み経路が無く、correctness の `exec_rope` は常に `row_positions` にフォールバックする。spec verify で optimized が外れた場合に位置が食い違う可能性がある。inv_freq table と attention factor は Gate 3 で一致済み |
| HIP Graph | `SHELF_HIP_GRAPH` は既定 OFF。`graph_eligible` が `tp_schedules.empty()` を要求するため、検証は TP=1 でのみ意味を持つ。capture/replay と GPU ポインタ寿命の確認が未実施 |
| prefix cache のキー | キーはトークン列のみで rope 識別子を含まない。factor はプロセス起動時に固定・キャッシュはプロセスローカルなので現状は安全。factor を実行中に変える機構や永続化を入れる場合はキーへの追加が必要 |
| uint32 の上限 | 100 万トークン域の `max_seq_len + 1` / `row_positions` の加算順序とオーバーフロー境界の検証（単体テスト）が未実施 |
| server 回帰の前提 | `tests/server/test_server_context_boundary.py` は FP8 重み 27 GiB + 32K BF16 KV + workspace が 32 GiB のカードに収まらず、この環境では起動できない（arena に 3.26 GiB しか残らない）。テスト側の容量前提を見直すか、FP8 KV / 小さい `--working-set-reserve-gib` で実行する経路を用意する |

## 5. ドキュメント

- TP・PrefixCache・server の profile 同一性契約は [developer/qwen35.md](developer/qwen35.md) §2.5 に記載済み。
- Gate 5/6 の結果が出た時点で、現在形の契約は `docs/developer/`、性能値は `docs/perf/`、経緯は `docs/research/` に分離して更新する。

## 6. server の usage / メタ情報配信（Phase A〜C）

### 経緯（2026-10-23）

- 背景: 「残トークン予算などのメタ情報がハーネスに届かない」という調査から着手した。
  原因は server が stream 応答で usage を返さないことだった。OpenCode（AI SDK）と
  DeepSeek Harness（pi-ai）はどちらも既定で `stream_options: {"include_usage": true}` を
  送るが、server は `stream_options` を解析せず無視していた。
  - DeepSeek Harness: usage 非受信時、pi-ai はゼロ初期値のまま `done` を出し、ゼロ usage を
    「報告済み」として扱う。結果 `ContextMeter`（コンテキスト占有リング）が実態より大幅に小さい
    誤った占有率を表示し、`UsagePill` はゼロ判定で非表示になる。分母の `contextWindow` は
    `cordis.patch.yml` の設定依存（未設定なら既定 262144 でずれる）。
  - OpenCode: usage が全て 0 だとフッターの usage（トークン数と `% ctx`）を表示しない。
    分母は `opencode.json` の `limit.context`。
- worktree `.worktrees/feature-server-metadata`（ブランチ `feature-server-metadata`）で
  Phase A〜C を実装し `c495956` にコミットした。
  - Phase A: `stream_options.include_usage` を解釈し、指定時のみ `[DONE]` 直前に
    `choices: []` の usage チャンクを返す。`prompt_tokens` / `completion_tokens` /
    `total_tokens` に加え、prefix cache 復元量を `prompt_tokens_details.cached_tokens` で返す
    （compute の `done.restored_tokens` を wire に載せる）。非 stream 応答にも `cached_tokens` を
    追加。`stream_options` の未知キーと型違反は 400 で拒否する。
  - Phase B: `GET /v1/models` の各 model に `context_length`（`--max-seq-len`）を広告する。
  - Phase C: `GET /v1/meta` を追加。起動オプションの正本値（`max_seq_len` /
    `max_concurrent_requests` / kv 設定 / `arena_gib` / speculative 等）と compute の
    `prefix_cache`、`inflight` / `queued` を返す。`AsyncComputeClient` に in-flight /
    queued カウンタを追加した。
- docs は [user/server.md](user/server.md)（usage チャンク仕様・`/v1/meta` のフィールド表・
  OpenCode の `limit.context` と DeepSeek Harness の `contextWindow` を
  `GET /v1/meta` の `max_seq_len` と照合する手順）と
  [developer/server_status.md](developer/server_status.md)（Supported に
  `GET /v1/meta` と stream usage）に記載した。
- 検証: worktree のフルビルドが 428/428 目標で成功（exit 0）。E2E は
  `test_server_meta.py`（新規）15/15、`test_server_unsupported.py` 34/34、
  `test_server_models.py` 7/7、`test_server_stream.py` の usage アサート 11/11 が PASS。
  stream / chat の既存失敗は main ベースライン（当該コミット未反映）で同一ラベルが
  同じように失敗することを実測し、環境・モデル起因と切り分けた。
- マージ: `de4406c Merge branch 'feature-server-metadata'`（衝突なし、11ファイル +439/-12）。
  マージ後の main で staging を再生成し `test_server_models.py` 7/7、
  `test_server_meta.py` 15/15 を確認した。

### 対応結果（2026-10-23）

| 項目 | 結果 |
| --- | --- |
| `limit.context` の不整合 | compute に `{"op":"stats"}` と `kv_capacity_snapshot()` を追加し、`/v1/meta` に `kv_cache` を載せた。クライアント用 server を TP2 / FP8 KV / `--max-seq-len 131072` で起動し、`GET /v1/meta` の `max_seq_len=131072` と KV 容量 131104 token（8194 page）、`GET /v1/models` の `context_length=131072`、stream usage の `prompt_tokens=15` / `completion_tokens=3` / `cached_tokens=0` を実測した。`/opt/zen/work/opencode.json` の `limit.context: 131072` はこの値と一致していたため変更なし。DeepSeek Harness は `/opt/zen/.dsh/profiles/web/cordis.patch.yml` の `shelf` provider に `contextWindow: 131072` / `maxTokens: 8192` を追加した。起動例と照合手順は [user/server.md](user/server.md) の「client の context 設定」に記載した |
| 実機での表示確認 | **未実施**。usage チャンクと model 広告の wire は TP2 server で実測済みだが、OpenCode のフッターと DeepSeek Harness の `ContextMeter` / `UsagePill` は UI 操作が必要なため視覚確認は未実施。server は port 8123 で起動したままにしてある |
| 既存テスト失敗の解消 | harness に `SHELF_TEST_ARENA_GIB` / `SHELF_TEST_WORKING_SET_RESERVE_GIB` を追加し、対応 model の 27B-FP8 で server regression を実行した。core 9/9（DFlash2 の 2 件は環境変数不足で skip）、tools 3/3、unsupported 2/2、concurrency-cancel 4/4（`test_server_context_boundary.py` を除く）が PASS。`stream-multiple-chunks` / `single-compute-process` / `content-nonempty` / `finish-reason` も 27B で PASS した。`compute_pids()` は `SHELF_TEST_OWNER` で自テストが起動した分だけを拾うようにし、`test_compute_pids_scope.py` で固定した。CLI parity は server 終了後に実行するよう変更した。required acceptance は 139/139 PASS（skip 0）|
| `/v1/meta` に KV 空きが無い | `{"op":"stats"}` を追加し、`ContinuousBatcher::kv_capacity_snapshot()`（物理 page、active、`banker_safe`、`overcommit_pages`）と `queued_count()` を返す。`/v1/meta` の `kv_cache` に page / token 換算と `queued_requests` を載せ、`test_compute_service.py` と `test_server_meta.py` で固定した。5 秒で取得できない場合は `kv_cache: null` を返す |
| push・worktree | main に commit して `origin/main` へ push した。`feature-server-metadata` worktree は main にマージ済みのため削除した |

未実施は上記の表示確認のみである。`test_server_context_boundary.py` と
`test_agent_opencode.py` は 32K context を 32 GiB 1 枚に載せる前提のため、27B FP8 では
起動しない（[developer/testing.md](developer/testing.md) に記載）。Qwen3.5-4B はこの環境の
compute 出力が oracle（`3793, 248046` = `OK`）と一致せず `91`（`|`）を繰り返すため、
対応 model 外として追跡しない。

# TODO
## 機能
* 起動時の引数の名前を整理し、computeで使用可能なものはすべてE2Eで通す (命名をOpenAI API準拠にしたい)
* DFlash2 の drafter footprint 見積を詰める。session slot の見積 0.34 GiB に対し実測 0.094 GiB で、
  TP2 の KV auto-fit が約 1 GiB 過剰に確保している（`dflash2_footprint.cpp` の spec workspace 項）
* working set reserve の見積を正確にする。既定は auto（`executor_footprint.h` の
  `52 × Npad × hidden × 2`）だが、`create_model_executor` を plan（graph lowering +
  `build_program_set`）と create（arena 確保）へ分けられれば program workspace の
  実値を使える。TP2 では rank の shard 分だけ過大（見積 2.06 GiB / 実測 1.16 GiB）
* Decode(投機ONの場合、SpecとVerify)とPrefillを同じバッチ内で同時実行可能にして、pp=row2048が動作している場合でも、Decode出力を止めない
* exl3(mul1)の実装
* MXFP4 linear の A 側量子化は、採用判断（logit 変動ゲートへ切り替える可否）と本実装が未完了
  （docs/research/20261103/mxfp4_gate_a_prime.md §11〜§13）
* qwen35moe対応・最適化
* expert-parallelの実装
* Small-M GEMMの最適化

## pipeline parallel

PP の layer split・host hand-off・E2E は実装済み。以下は未対応または未検証。

* TP と PP の同時利用（現行は `--tensor-parallel 1` のみ）
* DFlash2 speculative decode との併用
* prefix cache との併用
* hidden taps / `--dump-logits` との併用
* MTP block との併用（compute の MTP は `--tensor-parallel 1` / `--pipeline-parallel 1` のみ）
* decode と prefill が同じ batch に混在する実行（機能 TODO の mixed batch と同じ。
  PP coordinator は現在 `unsupported` を返す）
* stage 間 overlap（現行は pinned host 経由の直列 hand-off。性能最適化は未着手）
* 3 stage 以上の実機検証（この環境は 2 GPU のため PP=2 のみ実測）
* stage 0 の KV page 数に全 stage を合わせる制約の解消
  （非対称 layer 分割または stage ごとの page 数指定）
* PP の prefill / decode 性能を warmup + 複数 run で計測し `docs/perf/current.md` へ記録
