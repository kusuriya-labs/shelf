# shelf server（任意コンポーネント / パス A）

`draft_docs/directory_design.md` §3.5 のパス A（OpenAI 互換の HTTP 面を Python で持ち、
エンジンは `shelf-compute` の JSONL で駆動する）の実装。

**実行ファイルのビルド（CMake）には結線していません**（Python 依存は「サーバを動かす
ときだけ」）。ただし pytest は ctest の `serve.python` として登録済みで、
`aiohttp` / `pytest` が無い環境では skip する。移植元は
`references/phaseshift-fp8/src/apps/server/`（自作品）。

## いま入っているもの

| ファイル | 内容 | 状態 |
| --- | --- | --- |
| `engine_client.py` | `shelf-compute` の JSONL クライアント。`done` を `Done` として返す（finish_reason / usage / seed） | 実装済み |
| `openai_errors.py` | `serve_protocol.md` §3 の code → HTTP 表と envelope（4 フィールド） | 実装済み |
| `server.py` | aiohttp の HTTP 層。`/healthz`・`/readyz`・`/metrics`・`/v1/models`・`/v1/chat/completions`（ストリーム/非ストリーム） | 実装済み |
| `tests/fake_engine.py` | モデル・GPU 不要のフェイクエンジン（error / stop / 遅い生成 / 即終了） | 実装済み |
| `tests/` | pytest 116 件（下記の表を 1 対 1 で固定） | 116 件通過 |

## 動かし方

```bash
# テスト（モデル・GPU・ポート不要）
python3 -m pytest apps/server/tests -q
# ctest から（pytest / aiohttp が無ければ skip）
ctest --test-dir build/cpu -R serve.python

# 実エンジンに繋ぐ（要 GPU・HIP ビルド）
cd apps/server
python3 server.py \
  --engine ../../build/hip-rel/apps/shelf-compute \
  --model ../../models/Qwen3.5-9B-FP8 \
  --model-name qwen3.5-9b-fp8 --port 8099

curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b-fp8","messages":[{"role":"user","content":"1+1は?"}],
       "max_tokens":16,"enable_thinking":false,"seed":42}'
```

## 受付時の検証（`serve_protocol.md` §2-2）

エンジンへ渡す前に 400 / 413 / 415 で返す。**未対応の項目は黙って捨てずに明示する**。

| 対象 | 規則 | code |
| --- | --- | --- |
| トップレベル | 未知フィールドを拒否 | `invalid_argument` |
| `messages` | 空でない配列・最大 4096 件・各要素はオブジェクト | `invalid_argument` |
| `messages[i]` | `role` は `system` / `user` / `assistant`、`content` は文字列のみ | `unsupported_parameter`（tool・マルチモーダル） |
| 入力の大きさ | 本文 16 MiB・本文の合計 4 MiB・`prompt + max_tokens <= max_context` | `request_too_large` / `context_length_exceeded` |
| `stream` | 真偽値のみ | `invalid_argument` |
| `stop` | 形（文字列 / 配列・最大 4 件）は検証するが**値は未対応** | `unsupported_parameter` |
| `n` | 1 のみ（複数生成は未対応） | `unsupported_parameter` |
| `chat_template_kwargs` | `enable_thinking` だけを受ける（真偽値） | `unsupported_parameter` |
| 数値 | `max_tokens` / `top_k` / `seed` は整数、`temperature` / `top_p` / `min_p` は有限値で範囲内 | `invalid_argument` |

## finish_reason / usage / seed（T02）

エンジンの `done` を**正**とする。HTTP 側で数え直したり推測したりしない。

* `finish_reason`: `stop` / `length` をそのまま写す（`cancelled` は応答を作らない）
* `usage`: `done.prompt_tokens` / `done.generated_tokens` / その和
* `seed`: エンジンが採番した値を返す（非ストリームは本文、SSE は最終 chunk）

## ストリームの文字列復元（T05）

BPE のトークン境界は文字境界と一致しない。SSE は**受け取ったトークン列全体**を
毎回 decode し、送出済み文字列との差分だけを送る（`IncrementalDecoder`）。
末尾が未完成の UTF-8（`U+FFFD`）の間は次トークンまで保留するので、日本語・
特殊トークンを含んでも**非ストリームの `decode(全 ID)` と一致**する。

## 中断（T03）

* `request_id` を照合した `cancel` をエンジンへ送る（別 ID の cancel で走っている生成を止めない）
* HTTP 切断を検出したら消費を止め、`finally` で `cancel` → terminal まで drain
* 子プロセスが終了したら待っている要求を `state_error`（500）で起こす
* 同時要求は `--max-inflight`（既定 4）。エンジン側も 4 スロットで連続バッチングする
  （`--max-batch`。JSONL には複数の `generate` が同時に流れる）
* `--prefix-cache-blocks N` で Prefix Cache（T16）を有効化する（既定 0 = 無効）。
  hit の生成列一致は実行間の決定性に依存する（9B は一致、27B/35B は prefill の
  非決定性が未解決。`draft_docs/n01_n03_2026-11-01.md` 参照）
* `--decode1-max-rows N`（N01。既定 4）: この行数以下の decode は 1 行ずつ
  decode1 で回し、1 並列と生成列を一致させる。`0` は全行バッチ（速度優先。
  一致は保証しない）
* `--no-eos`: EOS で止めず `max_tokens` まで生成する（数値一致の検証用）
* `--kv-in-host` / `--online-kv-pages N`（T14）: KV を pinned host に置き、
  直近 N ページを VRAM にミラーする（`--online-kv-pages` はスロットごとに
  窓を持ち、連続バッチングの 4 スロットでも host のみと同一の出力）
* Prefill 中もエンジンが制御行を読むので、cancel はチャンクの合間で効く

## 運用（T18）

外部公開に必要な最小限を入れてある。設計と実測は
`draft_docs/ops_2026-11-01.md` を参照。

| 項目 | 実装 |
| --- | --- |
| 認証 | `--api-key`（または環境変数 `SHELF_API_KEY`）。`Authorization: Bearer <key>` か `X-API-Key` を要求。`/healthz` `/readyz` は除外、`/metrics` は保護対象。失敗は 401 `invalid_api_key`（`WWW-Authenticate: Bearer`） |
| 同時要求制限 | `--max-inflight`（エンジンへ流す数。既定 4）+ `--max-queue`（待ち。既定 64）。合計を超えたら 429 `rate_limit_exceeded` |
| タイムアウト | `--request-timeout`（秒。既定 600、0 で無効）。非ストリームは 504 `timeout`、SSE は `error` を 1 つ流して `[DONE]` で閉じ、エンジンへは cancel を送る |
| readiness | `/readyz` は「受付可・停止中でない・エンジン子プロセスが生きている」の 3 条件。不可なら 503。`/v1/*` も同じ条件で 503 `service_unavailable` |
| エラー統計 | パス×status の要求数、code 別のエラー数、429 / timeout / 認証失敗、生成トークン数、レイテンシ合計と件数 |
| 監視 | `/metrics` に Prometheus テキストで公開（`shelf_ready` / `shelf_engine_alive` / `shelf_inflight` / `shelf_requests_total` / `shelf_errors_total` / `shelf_request_timeout_total` / `shelf_queue_rejected_total` / `shelf_auth_failures_total` / `shelf_tokens_generated_total` / `shelf_request_duration_seconds_*` / `shelf_uptime_seconds`） |
| 起動・終了 | SIGTERM / SIGINT で新規受付を止め（`shutting_down`）、`--shutdown-grace`（既定 10 秒）だけ処理中を待ってから HTTP とエンジンを閉じる。終了時に件数の要約を stderr へ出す |

```bash
python3 server.py --model ... --api-key "$SHELF_API_KEY" --request-timeout 300 \
  --max-inflight 4 --max-queue 32 --shutdown-grace 15
curl -s -H "Authorization: Bearer $SHELF_API_KEY" http://127.0.0.1:8080/readyz
curl -s -H "Authorization: Bearer $SHELF_API_KEY" http://127.0.0.1:8080/metrics
```

## 次の段（残り）

1. **T06**: chat template の実機確認 — 9B / 27B の tokenizer と思考モード ON/OFF は
   確認済み（テンプレート・語彙 248077・EOS 248046 が一致）。tool 呼び出しと
   reasoning の分離は参照 `phaseshift_chat` から未移植
2. **T07**: 実 HTTP → JSONL → GPU → HTTP の E2E 行列（通常応答・SSE・EOS・エラー・
   cancel は 1 モデルで確認済み。9B/27B・FP8/MXFP4・PP の組合せが残り）
3. **T18 の残り**: 分散配置（複数プロセス・共有メトリクス）と、API key の複数発行・
   scope（現状は単一キー）。ストリームの idle timeout（現状は総時間のみ）
