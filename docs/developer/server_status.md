# Server

Shelf Server (`shelf-server`) の開発者向け入口である。server の対応 surface と、
詳細 contract を持つ current doc へのリンクを示す。個別仕様は各 doc を正とする。

## Scope

OpenAI 互換の serving product であり、inference engine ではない。
`shelf-server` 自身が aiohttp で OpenAI Chat Completions の subset を提供し、
`shelf-compute --serve-stdio` へ直接 JSONL stdio で接続する。

```text
OpenAI client
    │ HTTP / SSE
    ▼
shelf-server
    ├── aiohttp
    ├── openai_protocol.py   request validation / wire encoding
    ├── chat_service.py      HF messages -> chat template -> compute -> response parser
    └── compute_client.py    asyncio native JSONL client
    │ persistent JSONL stdio
    ▼
shelf-compute --serve-stdio   (token IDs in / token IDs out)
```

起動順は model processor load → compute subprocess 起動 → `ping` → capability 取得 →
HTTP listen である。server が ready になるまで port を accept しない。

compute の親プロセス死亡時の扱い:

- 通常の SIGINT / SIGTERM / SIGHUP では server の cleanup が
  `compute.shutdown()` を実行し、JSONL shutdown で compute を停止する。
- server が SIGKILL などで cleanup を実行できない場合も、compute subprocess には
  Linux の `PR_SET_PDEATHSIG(SIGTERM)` が設定されるため、server の終了に追従して
  compute も終了する。
- `ping` に失敗した起動途中の compute も `shutdown()` で停止する。

## Supported

- `GET /v1/models`（各 model に `context_length` を含む）
- `GET /v1/meta`（起動オプションの正本値、`inflight` / `queued`、compute の `kv_cache`）
- `POST /v1/chat/completions`（`stream=false` / `stream=true` の SSE）
- stream の usage: `stream_options.include_usage` 指定時のみ `[DONE]` 直前に
  `choices: []` の usage チャンク（`prompt_tokens_details.cached_tokens` 付き）
- sampling: `temperature` / `top_p` / `top_k` / `seed` / `max_tokens` /
  `max_completion_tokens`
- tools: `tools`、`tool_choice="auto"` / `"none"`、`assistant.tool_calls`、
  `role=tool` の round-trip、OpenCode agent loop
- best-effort tool calling（生成は Transformers の Qwen response parser で解析する。
  parse に失敗した output を捏造・修復しない）
- reasoning（Qwen3.8 の thinking。既定で有効、`reasoning_effort` は
  `low` / `medium` / `xhigh`、応答は `reasoning_content` と `content` に分離する。
  thinking は request の推論予算から answer reserve を引いた地点で compute 側で閉じる）
- concurrent serving（複数 request が Continuous Batching へ到達する）
- request cancellation（HTTP client disconnect → JSONL cancel → ContinuousBatcher cancel）
- DFlash2 speculative decode（constraint なしの通常経路）

## Unsupported

以下は明示的に HTTP 400（`code=unsupported_parameter`）で拒否する。silent ignore はしない。

- `response_format`（Structured Output）
- `reasoning`（OpenAI の reasoning transport object）
- `text`
- `tool_choice="required"` / named function
- `strict:true`（tool 単位で拒否）
- `parallel_tool_calls=false`
- `n != 1` / `logit_bias` / `logprobs`

以下は route 自体を持たない。

- `POST /v1/responses` → HTTP 404 `code=unsupported_endpoint`
- `POST /v1/completions` など未登録 path → HTTP 404 `code=not_found`

## Prefix cache

prefix cache は compute runtime に実装されており、server はその protocol クライアントである。

- server は常に `prefix_cache_checkpoint_position = 0` を送る
  （prefix の再利用位置は compute 側の radix lookup に一元化されている）
- compute は `0` 以外を `invalid_argument` で拒否する
- `capabilities.prefix_cache` は compute 側で `PrefixCache::create` が成功したとき `true`
- `done` event は `prefill_tokens` / `restored_tokens` / `cache_checkpoint_tokens` を返す
  （`prefill_tokens = input - restored_tokens`、`cache_checkpoint_tokens` は常に 0）

詳細は [prefix_cache.md](prefix_cache.md) を参照。

## Compute metrics

`GET /v1/meta` は起動オプション由来の値に加えて、compute の KV 使用量を返す。

- `AsyncComputeClient.kv_stats()` が JSONL の control op `{"op":"stats"}` を送り、
  `{"event":"stats","kv":{...}}` を待つ。`ping` と同じ `_control_lock` で直列化する。
- compute は `ContinuousBatcher::kv_capacity_snapshot()`（物理 page 数、active request、
  KV Banker の安全性、overcommit）と `queued_count()`（sequence slot / KV admission 待ち）
  から応答を組み立てる。KV page の導入・解放は decode と同じ `PagedKVPool` を見る。
- 値は page 単位である。`free_tokens` は空き page を `page_tokens` で換算した値であり、
  page 境界の端数は含まない。
- 取得は短い timeout（既定 5 秒）で行い、間に合わない場合は直近に取得できた
  `kv_cache` を `stale: true` として返す。直近値も無ければ `kv_cache: null` を返す。
  server の起動値と `inflight` / `queued` は常に返すため、client は `null` を
  「メトリクス未取得」として扱う。

user 向けのフィールド表は [../user/server.md](../user/server.md) を参照。

## Architecture

責務境界と layer 構成は [architecture.md](architecture.md) を参照。

## Detailed contracts

- generation stop token: [qwen35.md](qwen35.md)
- reasoning（thinking / reasoning_effort / reasoning_content）: [reasoning.md](reasoning.md)
- sampling: [sampling.md](sampling.md)
- prefix cache: [prefix_cache.md](prefix_cache.md)
- server regression suite: [testing.md](testing.md)

## User guide

利用方法・起動 options・API 例は [../user/server.md](../user/server.md) を参照。

## Known limitations

- reasoning token 数の内訳（`completion_tokens_details.reasoning_tokens`）と、
  client からの thinking 予算指定、履歴の server 側永続化は未実装である。
  thinking 予算自体は server 既定値で compute 側に強制される
  （[reasoning.md](reasoning.md) 「thinking 予算」）。
- `Full JSON Schema semantics` は保証しない。tool の JSON Schema は chat template へ
  tool 説明として渡すだけで、生成時の schema enforcement は行わない。
- embeddings / vision / audio / realtime API / authentication / TLS termination /
  multi-model hot loading は対象外。
