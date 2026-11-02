# 残課題（homework）

Static YaRN（`--rope-factor`）の Target 対応は Gate 0〜4 まで完了し、main に入っている。
ここには**未実施・未接続の作業**だけを残す。実装済みの契約は
[developer/qwen35.md](developer/qwen35.md) §2.5、利用方法は
[user/compute.md](user/compute.md)、経緯と実測は
[research/20261023/yarn_context_extension.md](research/20261023/yarn_context_extension.md) を正本とする。

## 1. Gate 5 — Drafter（DFlash2 / EAGLE4）の長位置対応

- Target が YaRN のとき、Drafter は自身の checkpoint の RoPE（`rope_theta` のみ、`rope_type=default`）に従う。
  現状 2x/4x 起動時の speculative decoding は**起動時に拒否**しており、Drafter 側の YaRN 対応は未着手。
- 必要な作業:
  - Drafter の `rope_parameters` から独立に RoPE プロファイルを解決する（現状は非 `default` を拒否する実装）。
  - `float(position) * inv_freq` で角度を計算している箇所（`dflash2/detail/rope_device.h`、`eagle4/ops.hip`）の
    高位置精度を評価し、必要なら double へ変更する。
  - DFlash2 の Ring 位置・EAGLE4 の Tree 位置が長位置で整合することを検証する。
  - Target 1x/2x/4x との組合せで acceptance rate と tokens/s を測る。
- 未対応の組合せは「使用不可」と明示し、静かに誤った補助推論をしない。

## 2. Gate 6 — 2x の長文 E2E（品質・性能）

- 容量は実測済み: `--rope-factor 2 --kv-cache-dtype fp8 --tp 2 --max-seq-len 524288` で起動・生成に成功し、
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
| MTP | `lower_qwen35_mtp_to_primitives` は profile を受け取るようになったが production 経路に未接続（tests のみ）。「MTP 経路も整合する」完了定義は lowering テストの範囲に限定して扱う |
| prefix cache のキー | キーはトークン列のみで rope 識別子を含まない。factor はプロセス起動時に固定・キャッシュはプロセスローカルなので現状は安全。factor を実行中に変える機構や永続化を入れる場合はキーへの追加が必要 |
| uint32 の上限 | 100 万トークン域の `max_seq_len + 1` / `row_positions` の加算順序とオーバーフロー境界の検証（単体テスト）が未実施 |
| host の取得キャッシュ | model-host の Acquire キャッシュ fingerprint は `config.json` の内容を含まない。長時間稼働する host で設定変更が検出できない（launcher は毎回新規起動のため現状の実害は限定的） |
| server 回帰の前提 | `tests/server/test_server_context_boundary.py` は FP8 重み 27 GiB + 32K BF16 KV + workspace が 32 GiB のカードに収まらず、この環境では起動できない（arena に 3.26 GiB しか残らない）。テスト側の容量前提を見直すか、FP8 KV / 小さい `--working-set-reserve-gib` で実行する経路を用意する |

## 5. ドキュメント

- TP・Resident・PrefixCache・server の profile 同一性契約は [developer/qwen35.md](developer/qwen35.md) §2.5 に記載済み。
- Gate 5/6 の結果が出た時点で、現在形の契約は `docs/developer/`、性能値は `docs/perf/`、経緯は `docs/research/` に分離して更新する。

## 6. EAGLE4 の Tensor Parallel シャード

TP2 のシャード実装（重み分割・collective・rank ごとの drafter）は main に取り込み済みで、
serve 経路の TP2 + EAGLE-4 は動作する。契約は
[developer/eagle4.md](developer/eagle4.md) の tensor parallel 節、経緯と切り分けは
[research/20261024/eagle4_tp2_shard_crash.md](research/20261024/eagle4_tp2_shard_crash.md)
を正本とする。ここには残る作業だけを残す。

| 項目 | 内容 |
| --- | --- |
| シャード後の性能値 | `67886ff` の commit message にある TTFT 2.953 s（3 run の中央値）/ decode 65.5 tok/s（1 run）は `docs/perf/current.md` に未記録。同文書の EAGLE-4 行（62.36 tok/s）はシャード前の measurement revision `c745c31` の値である。5-run で再測し、条件とともに `docs/perf/` へ記録する |
| 27B の TP2 E2E | 27B の TP2 + EAGLE-4 は serve 経路の手動実測のみで、自動テストが無い。`test_eagle4_tp_forward` は小さい checkpoint の stage 比較であり、実 checkpoint の E2E は `test_dflash2_tp_e2e` 相当の test が無い |
| serve の tree 経路 | `shelf-server` が `--eagle4-tree-width` を露出していないため、TP2 では chain のみ到達可能である。tree は単一 rank でのみ検証済み |
| `eagle_ranks` の束縛失敗 | `continuous_batcher.cpp` は executor / context / stream / device のサイズが揃わないと `eagle_ranks` を空のまま渡し、single-rank の draft に静かに落ちる。起動時または job 構築時に error / log を出す |

candidate head の pool は rank ごとの local top-k の和のため、TP2 の候補集合は TP1 と
一致しない。`test_eagle4_tp_forward` は merge 結果の整合と「TP2 が TP1 より悪化しないこと」
だけを検証しており、既知の制約として許容している。

## 7. server の usage / メタ情報配信（Phase A〜C）

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

## 8. Qwen3.8-Flash-Next (qwen4exp) 対応

- 手順書の Gate 0〜5 まで完了し、`.worktrees/qwen4exp`（ブランチ
  `investigation/qwen4exp-main`）にコミット済み（required acceptance は全 PASS。
  コミット分割と件数の正本は
  [research/20261024/qwen4exp/gates.md](research/20261024/qwen4exp/gates.md)）。
  **main には未マージ**。
- ここには**未着手・未接続の作業**だけを残す。Gate の PASS 判定・誤差指標・差分調査票は
  [research/20261024/qwen4exp/gates.md](research/20261024/qwen4exp/gates.md)、
  設計判断と未解決の食い違いは
  [research/20261024/qwen4exp/source_audit.md](research/20261024/qwen4exp/source_audit.md)、
  現在の contract は [developer/qwen4exp.md](developer/qwen4exp.md) を正本とする。

### 8.1 main へのマージ（最優先）

- worktree は `investigation/qwen4exp-main` 上で完結している。`main` へどう統合するかは
  未決定。マージ前に required acceptance を main ベースで通し直す。

### 8.2 G5 の PASS 条件外の引き継ぎ（G6 で必要）

| 項目 | 内容 |
| --- | --- |
| entrypoint 接続 | `shelf-compute` / `shelf-bench` から qwen4exp を起動する経路が未実装。`qwen35::create_model_executor` は `Qwen35Model` と `lower_qwen35_to_primitives` に密結合しており、qwen4exp へは流用できない |
| mini runtime | `models/qwen4exp/runtime/` は state owner のみ。手順書 §10 の PASS 条件は小型 fixture（`lowering → build_program → correctness dispatch` 経由）で充足したが、`ContinuousBatcher` / `Executor` と統合した prefill → decode は未着手 |
| `PrefixCache` 統合 | 新 state（indexer key / PLE conv / PLE n-gram context）の保存・復元構造と owner は `source_audit.md` 付録 G に記録済み、実装は未。`PrefixCache` へは KV page と GDN state しか渡せていない |
| optimized dispatch | GR / MoE / QSA は現状 **correctness fallback** で動いている。`optimized_dispatch.hip` への `try_launch_*` の追加が未着手。**`kernels/optimized/moe_ops.hip` は router 出力が row_stride 契約（16B パディング）に未対応**（`kernels/correctness/detail/moe.inc` は修正済み）。接続時はまず row_stride 契約へ揃える |
| `PrimitiveStateKind::INDEXER_KEY_CACHE` | 未登録。`runtime/` の他エージェント作業との衝突回避で保留した。`IndexerKeyPool` は kind 登録を伴わない qwen4exp-local の所有バッファとして成立しており、`PrimitiveGraph` 経由で indexer key を state として扱うには登録が必要 |
| QSA の接続 | sparse attention kernel（`kernels/optimized/attention/qsa_sparse_attention`）と indexer key pool は実装済みだが、`PagedAttentionNode` への mask 入力（optional な 2 番目入力）と indexer key state の lowering 接続が未（`docs/developer/qwen4exp.md` gap #3 / #4） |
| PLE の GPU kernel 化 | `models/qwen4exp/ple/ple_forward` は **host F32**。projection / gate / conv の GPU 実行経路が未。実モデル 95.4 GiB table では必須 |
| `(24,2,256)` の optimized ルール | `paged_attention_selector.cpp` に無い（correctness fallback で機能成立）。**未計測 geometry に Optimized を宣言しない**方針を維持。追加は計測後に判断 |

### 8.3 G6（実モデル E2E）— 着手不可

- **`Qwen3.8-Flash-Next-FP8` checkpoint がローカルに存在しない**。
  FP8 の `quantization_config`（`quant_method` / `fmt` / `activation_scheme` /
  `weight_block_size`）と scale tensor の名前・shape が未確定。
  - これが主なブロッカー。取得後に HF FP8 の受理条件との一致を確認する
  - **`activation_scheme` を検査すべきか未判定**（手順書は検査を要求、現行 `hf_manifest.cpp`
    は未検査。実値が無いので推測で実装しない方針）
- Hugging Face Transformers `qwen4_exp` の参照 commit が未固定。
  インストール済み `transformers 5.14.1` に `models/qwen4_exp` が存在しない。
  正本は `docs/developer/qwen4exp.md` と公式 `config.json`
- **実モデル 512 expert / 48 層の VRAM 常駐・キャッシュ**が未検討（routed expert は約
  120.8 B パラメータで 32 GiB に常駐しない）。手順書 §1.4「必要 Expert だけを読み込む」
  と整合するキャッシュ方針が要る（`source_audit.md` の MoE weight 節に選択肢を記録）
- **PLE の SSD 経路 E2E**（`shard_0..127`、実 table 95.4 GiB）は synthetic fixture では
  検証していない。G2 は synthetic と fastpath 実測まで
- MTP（`docs/developer/qwen4exp.md` §11）は forward 正本が未確定（Gate 10）、
  vision tower は Gate 11 の対象

### 8.4 未解決の仕様差（再確認が必要）

| 項目 | 内容 |
| --- | --- |
| indexer の `q_layernorm` / `k_layernorm` | RMSNorm か LayerNorm か、weight mode が `ONE_PLUS` か `DIRECT` か §6.2 に記載が無く、§4.1 に倣って **RMSNorm + ONE_PLUS** で実装。参照実装の入手後に要再確認（oracle を差し替えて再検証できる構成） |
| Top-K 同点の正本順序 | 未確定のため `(score 降順, block id 昇順)` を契約として定義し、kernel と oracle で一致させた |
| PLE conv の重み方向 | `w[0]` = 最古 tap としたが、参照実装が不在で数値照合不能。実 checkpoint の `conv1d.weight` を使う経路で再確認する |
| G4a の GR / PLE conv | host float 実装のまま。GPU kernel 化と融合前後の比較が未 |

### 8.5 docs の後続更新

- `docs/developer/qwen4exp.md` の gap 表（§14）は round 3 時点に更新済みだが、
  上記が解消しつつ次第に再更新が必要（gap #1 / #3 / #4 が解消対象）
- G6 実施時: 現在形の契約は `docs/developer/`、性能値は `docs/perf/`、
  経緯は `docs/research/20261024/qwen4exp/` へ分離する
