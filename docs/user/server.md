# shelf-server

OpenAI互換のchat servingアプリ。model directoryを指定して起動すると、
`/v1` 配下でOpenAI Chat Completions APIのsubsetを提供する。

```bash
./build/shelf-server \
    --model-dir /path/to/Qwen3.8-27B-FP8 \
    --port 8123
```

起動すると次のエンドポイントが利用できる。

```text
http://127.0.0.1:8000/v1
```

対応範囲は次のとおり。

* `GET /v1/models`
* `GET /v1/meta`
* `POST /v1/chat/completions`（非stream / stream）

embeddings、vision、audio、Responses API、認証、TLS、複数modelは対象外。

依存は Python / `aiohttp` / `transformers==5.14.1` のみである。
LocalAI・gRPC・protobuf・XGrammar は要らない。

## 起動

```bash
./build/shelf-server \
    --model-dir /path/to/Qwen3.8-27B-FP8
```

起動順は model processor load → `shelf-compute` subprocess 起動 → `ping` →
capability 取得 → HTTP listen である。model load は server lifetime 中に 1 回だけ行われ、
request ごとに compute process を起動しない。server が ready になるまで port は accept しない。

起動完了時、stderr に以下が表示される。

```text
Shelf Server

Model:        shelf (Qwen3.8-27B-FP8)
Endpoint:     http://127.0.0.1:8000/v1
Context:      32768
Concurrency:  4
KV dtype:     bf16
KV capacity:  auto
Prefix cache: enabled (capabilities.prefix_cache=true)
Default output: 65536

Reasoning:    enabled (thinking on, effort xhigh, preserve_thinking on)
Efforts:      low, medium, xhigh

Supported:   Chat Completions, SSE, tools, reasoning, sampling, concurrency, cancel
Unsupported: Responses, structured output, strict tools, reasoning object

Ready.
```

## options

<!-- cli:server -->
| option | 説明 |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--model-dir MODEL_DIR` | model directory |
| `--model MODEL` | model name reported by the API |
| `--host HOST` | listen address (default 127.0.0.1) |
| `--port PORT` | listen port (default 8000) |
| `--compute COMPUTE` | shelf-compute path |
| `--max-seq-len MAX_SEQ_LEN` | max sequence length (default 32768) |
| `--max-concurrent-requests MAX_CONCURRENT_REQUESTS` | max in-flight requests (default 4) |
| `--arena-gib N\|auto` | GPU arena capacity in GiB; auto = free VRAM minus 1 GiB (default auto) |
| `--device DEVICE` | comma separated GPU devices, one per rank; count must match --tp (default 0) |
| `--tp {1,2}` | tensor parallel size forwarded to compute (default 1) |
| `--model-host MODEL_HOST` | resident model host socket spec forwarded to compute |
| `--kv-cache-dtype {bf16,fp8}` | KV cache dtype (default bf16) |
| `--page-tokens PAGE_TOKENS` | paged KV page size in tokens (default 16) |
| `--kv-cache-capacity-tokens KV_CACHE_CAPACITY_TOKENS` | physical KV token capacity (0 = derive from the arena budget) |
| `--working-set-reserve-gib WORKING_SET_RESERVE_GIB` | arena headroom kept free for the working set (default 4) |
| `--rope-factor {auto,1,2,4}` | static YaRN rope factor forwarded to compute (default auto) |
| `--default-max-output-tokens DEFAULT_MAX_OUTPUT_TOKENS` | default max_new_tokens when the request omits it (default 65536) |
| `--dflash2-model-dir DFLASH2_MODEL_DIR` | DFlash2 draft model directory (enables speculative decoding) |
| `--dflash2-drafts DFLASH2_DRAFTS` | speculative draft tokens per round (default 7) |
| `--verify-weights` | verify quantized payload CRC32 on load |
<!-- /cli:server -->

不正な値は起動時に拒否される。`--max-concurrent-requests` / `--max-seq-len` /
`--default-max-output-tokens` / `--page-tokens` / `--dflash2-drafts` は 1 以上である。
`--arena-gib auto` は compute と同じ memory plan を compute 側の stderr に出す。

## DFlash2 speculative decoding

`--dflash2-model-dir` を指定すると DFlash2 を有効化する。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

指定時の挙動:

- `--max-concurrent-requests` は指定値を尊重する。DFlash2 でも複数 request を受付できる。
  KV admission は通常 decode と同じ KV Banker が行うため、
  `--kv-cache-capacity-tokens` の自動拡張は行わない。不足時は一部の request が
  Queued になり、Active の完了後に admission される。
- tool calling / sampling / cancel は DFlash2 経路でも利用できる。

```bash
./build/shelf-server \
    --model-dir models/Qwen3.8-27B-FP8 \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2 \
    --port 8123 \
    --device 0 \
    --arena-gib auto \
    --max-seq-len 4096 \
    --kv-cache-dtype fp8
```

## 使い方

### chat completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "shelf",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

### streaming

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "shelf",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true,
    "temperature": 0,
    "max_tokens": 32
  }'
```

末尾に `data: [DONE]` が出る。stream chunk を連結した文字列は、同じ request の非 stream 結果と一致する。

`stream_options: {"include_usage": true}` を付けると、`[DONE]` の直前に
`choices: []` の usage チャンクが 1 つ追加される。usage は
`prompt_tokens` / `completion_tokens` / `total_tokens` と、prefix cache 復元量を
`prompt_tokens_details.cached_tokens` で返す。`stream_options` を付けない stream では
usage チャンクは出ない。OpenCode と DeepSeek Harness は既定で `include_usage` を送るため、
付けなくても両者には届く。

usage チャンクの例:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1780000000,
  "model": "shelf",
  "choices": [],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20,
    "prompt_tokens_details": {"cached_tokens": 0}
  }
}
```

### metadata

`GET /v1/meta` は server が起動オプションから持つ正本値と、現在の in-flight 状態を返す。
`limit.context`（OpenCode）や `contextWindow`（DeepSeek Harness）を
`--max-seq-len` と一致させる際の照合元として使う。

```bash
curl http://127.0.0.1:8000/v1/meta
```

```json
{
  "model": "shelf",
  "max_seq_len": 32768,
  "max_concurrent_requests": 4,
  "default_max_output_tokens": 65536,
  "arena_gib": "auto",
  "device": "0",
  "tensor_parallel": 1,
  "kv_cache_dtype": "bf16",
  "kv_cache_capacity_tokens": 0,
  "working_set_reserve_gib": 4,
  "speculative": null,
  "prefix_cache": true,
  "inflight": 0,
  "queued": 0,
  "kv_cache": {
    "page_tokens": 16,
    "total_pages": 2049,
    "used_pages": 0,
    "free_pages": 2049,
    "capacity_tokens": 32784,
    "free_tokens": 32784,
    "active_requests": 0,
    "queued_requests": 0,
    "banker_safe": true,
    "overcommit_pages": 0
  }
}
```

| フィールド | 意味 |
| --- | --- |
| `model` | `--model` で報告する model 名 |
| `max_seq_len` | `--max-seq-len`。コンテキスト上限の正本 |
| `max_concurrent_requests` | `--max-concurrent-requests`。最大 in-flight 数 |
| `default_max_output_tokens` | `--default-max-output-tokens` |
| `arena_gib` | `--arena-gib`（`"auto"` または整数 GiB） |
| `device` / `tensor_parallel` | `--device` / `--tp` |
| `kv_cache_dtype` / `kv_cache_capacity_tokens` | KV cache 設定（0 = arena 予算から導出） |
| `working_set_reserve_gib` | `--working-set-reserve-gib` |
| `speculative` | 有効時の drafter（`{"method": "dflash2", "drafts": 7}` 等）。無効時は `null` |
| `reasoning` | reasoning の既定（`enabled` / `default_reasoning_effort` / `reasoning_efforts`） |
| `prefix_cache` | compute の `capabilities.prefix_cache` |
| `inflight` | 現在 generate 中の request 数 |
| `queued` | in-flight 上限待ちの request 数（compute 内部の KV admission は含まない） |
| `kv_cache` | compute の KV 使用量。page 単位の実測値。取得できないときは `null` |

`kv_cache` は compute の `stats` op を 1 回問い合わせた結果である。page 単位で丸めているため
`free_tokens` は「あと何 token 受け付けられるか」の上限ではなく、空き page を token に換算した値である。

| `kv_cache` のフィールド | 意味 |
| --- | --- |
| `page_tokens` | KV page の token 数（`--page-tokens`） |
| `total_pages` / `used_pages` / `free_pages` | 物理 KV page の総数・使用中・空き |
| `capacity_tokens` / `free_tokens` | page を token に換算した容量と空き |
| `active_requests` | KV を確保済みで decode 中の request 数 |
| `queued_requests` | compute に submit 済みで、sequence slot または KV admission を待つ request 数 |
| `banker_safe` | KV Banker の不変条件を満たしているか |
| `overcommit_pages` | active request の最大要求が物理 page を超えた量（通常は 0） |

`kv_cache` は HTTP 応答を待たせないよう短い timeout で取得する。compute が prefill 中などで
間に合わない場合は `null` を返し、server の起動オプション由来のフィールドはそのまま返す。

### client の context 設定

`/v1/meta` の `max_seq_len` が正本である。クライアント側の context 上限を次の 2 箇所で揃える。

- OpenCode: `opencode.json` の `providers.shelf.models.shelf.limit.context`
- DeepSeek Harness: `cordis.patch.yml` の `llm-pi-ai` にある `shelf` provider の
  model entry の `contextWindow`（出力上限は `maxTokens`。OpenCode の `limit.output` と揃える）

32 GiB の GPU 1 枚では 27B の FP8 重みに KV を足せないため、131072 token を配るときは
TP=2 と FP8 KV を使う。

```bash
./build/shelf-server \
    --model-dir models/Qwen3.8-27B-FP8 \
    --tp 2 --device 2,3 --port 8123 \
    --kv-cache-dtype fp8 --max-seq-len 131072
```

起動後に `curl -s http://127.0.0.1:8123/v1/meta` で `max_seq_len` を確認し、両クライアントの
値をその値へ合わせる。`--max-seq-len` を変えたらクライアント側も同じ値に更新する。

### sampling

受け付ける request parameter:

```text
model / messages / stream
stream_options（include_usage のみ）
max_tokens / max_completion_tokens（両方指定で値が異なる場合は reject）
temperature / top_p / top_k / seed
tools / tool_choice
reasoning_effort（low / medium / xhigh）
chat_template_kwargs（enable_thinking / preserve_thinking）
```

`top_k` は Shelf extension である。指定がない値の既定は
`temperature=0`（greedy）/ `top_p=1.0` / `top_k=0` / `seed=0` である。

### reasoning（thinking）

Qwen3.8 の thinking は**既定で有効**である。生成した思考は
`message.reasoning_content`（stream では `delta.reasoning_content`）で返り、
回答は `content` に分離される。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "shelf",
    "messages": [{"role": "user", "content": "12と19を足すと？"}],
    "reasoning_effort": "xhigh",
    "temperature": 0,
    "max_completion_tokens": 1024
  }'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "12 + 19 を計算する。10 + 10 = 20、2 + 9 = 11、合計 31。",
      "content": "31"
    },
    "finish_reason": "stop"
  }]
}
```

streaming では `delta.reasoning_content` の chunk が続いたあと、境界を越えると
`delta.content` の chunk に切り替わる。1 つの chunk に両方は入らない。

```text
data: {"choices":[{"delta":{"reasoning_content":"12 + 19 を"}}]}
data: {"choices":[{"delta":{"reasoning_content":"計算する。"}}]}
data: {"choices":[{"delta":{"content":"31"}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
```

| parameter | 既定 | 意味 |
| --- | --- | --- |
| `reasoning_effort` | `"xhigh"` | `low` / `medium` / `xhigh`。それ以外は HTTP 400 |
| `chat_template_kwargs.enable_thinking` | `true` | `false` で thinking 無効（回答のみ） |
| `chat_template_kwargs.preserve_thinking` | `true` | 履歴の reasoning を Prompt に含めるか |

- thinking を無効にする場合、`reasoning_effort` は同時に指定できない（HTTP 400）。
- `max_completion_tokens` は思考も消費する。思考中に上限へ達した要求は
  `finish_reason="length"` となり、`content` が空でも正常応答である。
- 履歴を送るときは、assistant message の `reasoning_content` をそのまま返すと
  thinking を保ったまま継続できる。server は会話を保存しないため、client が送らない
  過去の思考は復元されない。
- reasoning token 数の内訳と Thinking の token budget 指定は未実装である。

詳細な contract は [../developer/reasoning.md](../developer/reasoning.md) を参照。

### tools

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "shelf",
    "messages": [
      {"role": "user", "content": "大阪の現在の気温をget_weatherを使って確認して"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128
  }'
```

応答は OpenAI 形式の `tool_calls` になる。function arguments は JSON object ではなく
JSON string である。

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"大阪\"}"
            }
          }
        ]
      }
    }
  ]
}
```

Shelf Server は tool を実行しない。実際の tool 実行は client 側（OpenCode 等）が行う。
client が tool 結果を `role: "tool"` message として返すと、Shelf はそれを読んで次の
assistant 応答を生成する。

`tool_choice` は `auto` / `none` のみ受け付ける。`none` は tool 定義を model へ渡さない。

tool calling は best-effort である。生成された tool call は Transformers の Qwen response
parser で解析し、schema の enforce や生成後の修復は行わない。parser が tool-call-like な
出力を解析できない場合は parse failure として扱い、生の markup を content として捏造しない。

streaming 中の tool call にも対応する。OpenAI 形式の `delta.tool_calls` として返り、
model 固有の `<tool_call>` 等の markup は client へ漏れない。

### concurrency と cancel

`--max-concurrent-requests N` は最大 in-flight request 数であり、同時に 1 executor step で
scheduler が扱える最大 request 数でもある。physical KV capacity
（`--kv-cache-capacity-tokens`）が不足する場合、一部の request は Queued となり、
KV Banker が Active 完了後に admission する。`--kv-cache-capacity-tokens 0`（default）は
arena 予算と `--max-concurrent-requests` から導出する。すなわち
`min(arena の残りで入るページ数, max_concurrent × (max_seq_len + 1))` で、`max_seq_len + 1`
すら入らない構成・明示指定が予算を超える構成は**起動時に拒否**する
（`--working-set-reserve-gib` が起動後に必要な余白、DFlash2 有効時は drafter 重みも加算される）。

stream client が connection を切断すると、in-flight generation は cancel される
（`finish_reason="cancelled"`）。cancel は request-specific であり、他の in-flight
request の KV / GDN / token stream には影響しない。

## サポート外の API

以下は明示的に HTTP 400（`code=unsupported_parameter`）で拒否する。silent ignore はしない。

```text
response_format
reasoning
text
tool_choice="required" / named function
tools[].function.strict=true
parallel_tool_calls=false
n != 1 / logit_bias / logprobs
```

以下は route 自体を持たない。

```text
POST /v1/responses   -> HTTP 404 {"error": {"code": "unsupported_endpoint"}}
その他の未登録 path -> HTTP 404 {"error": {"code": "not_found"}}
```

## Prefix Cache

prefix cache は compute runtime が提供する。共有 prefix の paged KV と GDN state を
page 単位で保持し、共有 prefix への extension や multi-turn の会話で prefill を省略する。

- server は常に `prefix_cache_checkpoint_position = 0` を送る（prefix の再利用位置は
  compute 側の radix lookup に一元化されている）
- raw JSONL client が `0` を超える値を送った場合は `invalid_argument` で拒否する
- `capabilities.prefix_cache` は prefix cache の作成成功時 `true`
- `done` の `restored_tokens` に復元 token 数が入り、`prefill_tokens` は
  `input - restored_tokens` になる
- 同一プロンプトの再実行は restore しない（最長一致が input 全長に達するため）。
  DFlash2 経路でも prefix cache を使い、drafter ring を持つ entry だけを restore する

詳細は [../developer/prefix_cache.md](../developer/prefix_cache.md) を参照。

## エージェントからの利用

agent frontend は OpenAI 互換 client として接続する。ユーザーが起動するのは
`shelf-server` だけである。

### OpenCode

OpenCode の project config（`opencode.jsonc`）に OpenAI-compatible provider を追加する。

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "shelf/shelf",
  "providers": {
    "shelf": {
      "name": "Shelf",
      "env": ["SHELF_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {"baseURL": "http://127.0.0.1:8000/v1"},
      "models": {"shelf": {
        "name": "Shelf",
        "limit": {"context": 32768, "output": 8192}
      }}
    }
  }
}
```

`limit` は server の `--max-seq-len` に合わせる。正本は `GET /v1/meta` の
`max_seq_len` である。OpenCode は既定で 32000 の output
budget を要求するが、server は明示 `max_tokens` を残り context window に丸めるため、
`limit` を指定しないと出力が途中で切れることがある。
`limit.output` は `--max-seq-len` から実際の prompt 長を引いた残りに収まる必要がある。

`limit.context` が一致していれば、OpenCode のフッターの usage（トークン数と
`% ctx`）と auto-compaction の判定が server の実値と一致する。usage は
`stream_options.include_usage` を OpenCode が既定で送るため、server 側の
stream 応答に usage チャンクが載っていればそのまま表示される。

`SHELF_API_KEY` は任意の dummy 値でよい（Shelf は認証を行わない）。
model は `shelf/shelf` を選ぶ。

Codex CLI は Responses API を使うため対象外である。

### DeepSeek Harness

DeepSeek Harness（`dsh-llm-pi-ai`）は openai-completions プロトコルの
custom gateway として接続する。`$DSH_HOME/profiles/<profile>/cordis.patch.yml` に
route を追加する。

```yaml
- id: llm-pi-ai
  config:
    providers:
      shelf:
        apiKeyEnv: SHELF_API_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:8000/v1
        models:
          - id: shelf
            contextWindow: 32768
```

`contextWindow` は server の `--max-seq-len`（正本は `GET /v1/meta` の
`max_seq_len`）に合わせる。これがないとコンテキスト占有リングの分母が
既定値（262144）になり、占有率が過小表示になる。usage は pi-ai が既定で
`stream_options.include_usage` を送るので、server の stream 応答に usage チャンクが
載っていればプロバイダ報告値として扱われる。設定は次リクエストから反映され、
再起動は不要である。

## runtime dependency

```bash
pip install 'aiohttp' 'transformers==5.14.1'
```

server / compute の責務境界は
[../developer/architecture.md](../developer/architecture.md) を参照する。
ユーザーは `shelf-compute` を個別に起動する必要はない。

## エラー挙動

- 起動: 不正な option 値、model directory 不在、`shelf-compute` 不在は、
  stderr へ理由を表示して non-zero で終了する。
- request: unsupported な parameter、不正な message role、不正な tool arguments、
  未知の model name は HTTP 400（model は 404）であり、生成を開始しない。
- context: prompt が `--max-seq-len` を超える場合は HTTP 400 となる。明示 `max_tokens` は
  残り context に丸められ、超過を理由とした 400 は無い。
- cancel: stream client の切断などで cancel された generation は `finish_reason="cancelled"`
  となる。

## テスト

server / compute の E2E は `tests/server/` の Python suite で検証する。model directory は
`SHELF_MODEL_DIR` で指定する。

```bash
SHELF_MODEL_DIR=models/Qwen3.8-27B-FP8 \
python3 tests/server/run_server_regression.py --all
```

`tests/server/run_server_regression.py` が canonical な server regression runner である。
non-zero return は失敗として扱う。`--list` で group と test の一覧を表示できる。
suite の詳細は [../developer/testing.md](../developer/testing.md) を参照する。
