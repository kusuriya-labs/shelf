# shelf-server

OpenAI互換のchat servingアプリ。model directoryを指定して起動すると、
`/v1` 配下で OpenAI Chat Completions API と Responses API の subset を提供する。

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
* `POST /v1/responses`（非stream / stream）

embeddings、vision、audio、認証、TLS、複数modelは対象外。

依存は Python / `aiohttp` / `transformers==5.14.1` / `jinja2` である。
構造化出力には `xgrammar==0.2.9` が必要である。未導入でも通常の request は利用できる。

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
Prefix cache: enabled (capabilities.prefix_cache=true)
Inference budget: 32768 tokens (thinking 24576, answer reserve 8192)
  prompt > 262144: 65536 tokens (thinking 49152)

Reasoning:    enabled (thinking on, effort xhigh, preserve_thinking on)
Efforts:      low, medium, xhigh

Supported:   Chat Completions, Responses, SSE, tools, reasoning, sampling, concurrency, cancel
Structured output: json_object, json_schema, strict tools (requires xgrammar)

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
| `--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS` | token budget per scheduled step; also sets the executor row bucket (default 2048) |
| `--max-concurrent-requests MAX_CONCURRENT_REQUESTS` | max in-flight requests (default 4) |
| `--inference-budget-tokens INFERENCE_BUDGET_TOKENS` | max generated tokens per request (default 32768) |
| `--large-context-tokens LARGE_CONTEXT_TOKENS` | prompt tokens above which the larger inference budget applies (default 262144) |
| `--large-context-inference-budget-tokens LARGE_CONTEXT_INFERENCE_BUDGET_TOKENS` | max generated tokens for a request whose prompt is longer than --large-context-tokens (default 65536) |
| `--arena-gib N\|auto\|-N` | GPU arena capacity in GiB; auto = free VRAM minus 0.5 GiB, -N = total VRAM minus N GiB (default auto) |
| `--device DEVICE` | comma separated GPU devices, one per rank; count must match --tensor-parallel (default 0) |
| `--tensor-parallel {1,2}` | tensor parallel size forwarded to compute (default 1) |
| `--kv-cache-dtype {bf16,fp8}` | KV cache dtype (default bf16) |
| `--page-tokens PAGE_TOKENS` | paged KV page size in tokens (default 16) |
| `--working-set-reserve-gib WORKING_SET_RESERVE_GIB` | arena headroom kept free for the working set (0 = size it from the row bucket, default 0) |
| `--rope-factor {auto,1,2,4}` | static YaRN rope factor forwarded to compute (default auto) |
| `--dflash2-model-dir DFLASH2_MODEL_DIR` | DFlash2 draft model directory (enables speculative decoding) |
| `--dflash2-drafts DFLASH2_DRAFTS` | speculative draft tokens per round (default 7) |
| `--dflash2-tree-budget N\|auto` | 木検証の request あたり行数。既定の `auto` はその round の生成中 request が 1 本だけで投機中なら 10 行、それ以外は chain の 8 行。`0` は木検証を無効化。auto は TP=1・draft 数 7 用 |
| `--dflash2-max-sessions DFLASH2_MAX_SESSIONS` | requests that may speculate at once; the rest decode without speculation (0 = min(`--max-concurrent-requests`, 8)) |
| `--mtp` | checkpoint の MTP block を drafter として使う（複数 request を同時に処理する） |
| `--mtp-drafts MTP_DRAFTS` | MTP の draft tokens per round（default 4） |
| `--verify-weights` | verify quantized payload CRC32 on load |
<!-- /cli:server -->

不正な値は起動時に拒否される。`--max-concurrent-requests` / `--max-seq-len` /
`--inference-budget-tokens` / `--page-tokens` / `--dflash2-drafts` は 1 以上である。
`--large-context-tokens` は 0 以上で、0 は大きい予算を無効化する。
`--large-context-tokens` が 0 より大きい場合、
`--large-context-inference-budget-tokens` は `--inference-budget-tokens` 以上でなければならない。
`--arena-gib auto` は compute と同じ memory plan を compute 側の stderr に出す。
DFlash2 で TP=1・draft 数 7 以外の場合、tree budget の省略時は警告を出して
chain を使う。auto で stochastic sampling を使う request は通常 decode する。

### prefix cache

prefix cache は常に有効である。active KV pool の物理 page を refcount 付きで共有し、
save は page を登録するだけ、restore は block table に page を map するだけで、
KV のコピーも shadow pool も作らない。cache は KV pool の全 page を追跡できるため、
`--max-seq-len` を削って prefix cache の領域を作る必要はない。

モデル固有の GDN state と DFlash2 ring だけは shadow slot を持つ。
Qwen3.8-27B / TP=2 / DFlash2 ではこの予約は約 0.45 GiB/rank である。
起動時に compute の stderr に出る `MEMORY_PLAN tp prefix cache` で確認する。

DFlash2 の単一 GPU 実行では、cache miss の非整列 prompt に対し、入力の prefix を
保存するため追加の prefill を行う場合がある。その場合は最初の token までの時間と
一時 KV 使用量が増える。境界が 1024 token を超える場合、prefix hit の場合、または
TP の場合はこの追加保存を行わず、自然な prefill 終端が page 境界と重なったときに保存する。

## 推論予算

request 1 個あたりの生成予算（推論予算）は `--inference-budget-tokens`（default 32768）である。

- client が `max_completion_tokens` / `max_tokens` を指定した場合、その値と推論予算の
  小さい方が生成上限になる。予算を超える指定は 400 ではなく clamp される。
- 未指定の場合は推論予算そのものが上限になる。
- prompt の token 数が `--large-context-tokens`（default 262144）を超える request は、
  予算が `--large-context-inference-budget-tokens`（default 65536）に拡大される。
  判定基準は当該 request の prompt token 数であり、会話の履歴や server 全体の占有ではない。
  この段階が発火するには `--max-seq-len` が `--large-context-tokens` より大きいことが
  必要である（`--max-seq-len 262144` では prompt が閾値を超えられない）。
- thinking が有効な request では、予算のうち answer reserve を残した地点で thinking を
  閉じる。reserve は `予算/4`（512 以上、予算/2 以下）なので、
  32768 なら thinking 24576 / answer reserve 8192、65536 なら thinking 49152 /
  answer reserve 16384 である。
- 生成上限は `--max-seq-len` から prompt を引いた残りでも上限を受ける。

細部は [../developer/reasoning.md](../developer/reasoning.md) と
[../developer/stop_conditions.md](../developer/stop_conditions.md) を参照。

## DFlash2 speculative decoding

`--dflash2-model-dir` を指定すると DFlash2 を有効化する。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

指定時の挙動:

- `--max-concurrent-requests` は指定値を尊重する。DFlash2 でも複数 request を受付できる。
  KV の物理容量は起動時に arena の余りから確定し、request ごとに拡張しない。
  KV admission は通常 decode と同じ KV Banker が行うため、不足時は一部の request が
  Queued になり、Active の完了後に admission される。
- tool calling / sampling / cancel は DFlash2 経路でも利用できる。
- verify は単一 request を含め既定で lossy chunk を使うため、非投機 greedy と
  生成列が一致しない場合がある。`SHELF_GDN_VERIFY_CHUNKED=0` で serial に戻せる。
  `SHELF_GDN_VERIFY_CHUNKED_EXACT=1` は exact chunk を選ぶが、bit 一致は保証しない。
- `--dflash2-adaptive` は投機を自動で停止・再試行する実験的な方式で、投機を停止する batch 負荷の
  上限は `SHELF_DFLASH2_POLICY_MAX_SESSIONS` で上書きできる（既定 12）。

TP=1 で drafter の block size が 8、sliding window が 256 以上の場合、prefill の
context 構築は既定で最大 256 行ずつ処理する。`SHELF_DFLASH2_PREFILL_ROWS=8` で
従来の処理に戻せる。`256` は明示的に既定を選択し、それ以外の値は起動エラーになる。
TP>1 または上記の drafter 条件を満たさない場合は従来の処理に戻る。

Qwen3.8-27B-DFlash2、同時 request 1 件では追加 scratch は 16.953 MiB。KV の自動予算は
その分減るため、収容できる context の上限も下がる。実際の容量は起動 log の
`KV budget` で確認する。測定では長い prompt の TTFT が短縮した一方、入力によっては
平均受理数が低下した。行数による数値差と測定結果は
[性能記録](../perf/dflash2-prefill-rows-default.md)を参照。

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

## MTP speculative decoding

`--mtp` を指定すると、checkpoint の MTP block（`mtp.*`）を drafter として使う。
`--dflash2-model-dir` とは排他的である。

- server の MTP は **1 request ずつ直列**に処理する。`--max-concurrent-requests` を
  2 以上にしても、生成中の request が完了するまで次を開始しない。
- tool calling / sampling / cancel は MTP 経路でも利用できる。cancel は prefill 中なら
  次の chunk 境界、生成中なら次の spec round で反映される。
- 思考 budget（`reasoning_budget_tokens`）は MTP では無視し、通常の生成を行う。
- `--mtp-drafts` は 4 前後が目安。大きすぎると verify の固定費が増える。

```bash
./build/shelf-server \
    --model-dir models/Qwen3.8-27B-exl3 \
    --mtp --mtp-drafts 4 \
    --device 0,1 --tensor-parallel 2 \
    --max-seq-len 524288 --rope-factor 2 \
    --kv-cache-dtype fp8 \
    --port 8123
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
`--max-seq-len` と一致させる際の照合元として使う。`kv_cache` のページ数と token 数は
arena、model、prefix cache の設定に依存するため、次の例の値は一例である。

```bash
curl http://127.0.0.1:8000/v1/meta
```

```json
{
  "model": "shelf",
  "max_seq_len": 32768,
  "max_concurrent_requests": 4,
  "inference_budget": {
    "tokens": 32768,
    "large_context_tokens": 262144,
    "large_context_inference_budget_tokens": 65536,
    "answer_reserve_tokens": 8192,
    "thinking_budget_tokens": 24576
  },
  "arena_gib": "auto",
  "device": "0",
  "tensor_parallel": 1,
  "kv_cache_dtype": "bf16",
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
    "overcommit_pages": 0,
    "stale": false
  }
}
```

| フィールド | 意味 |
| --- | --- |
| `model` | `--model` で報告する model 名 |
| `max_seq_len` | `--max-seq-len`。コンテキスト上限の正本 |
| `max_concurrent_requests` | `--max-concurrent-requests`。最大 in-flight 数 |
| `inference_budget` | 推論予算。`tokens`（`--inference-budget-tokens`）、`large_context_tokens`、`large_context_inference_budget_tokens`、既定予算での `answer_reserve_tokens` / `thinking_budget_tokens` |
| `arena_gib` | `--arena-gib`（`"auto"` または数値 GiB） |
| `device` / `tensor_parallel` | `--device` / `--tensor-parallel` |
| `kv_cache_dtype` | KV cache の dtype |
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
| `stale` | 直近に取得した値を返しているか。最新の取得に失敗したとき `true` |

`kv_cache` は HTTP 応答を待たせないよう短い timeout で取得する。compute が prefill 中などで
間に合わない場合は、直近に取得できた値を `stale: true` として返す。直近値も無い場合は
`null` を返し、server の起動オプション由来のフィールドはそのまま返す。

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
    --tensor-parallel 2 --device 2,3 --port 8123 \
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
temperature / top_p / min_p / top_k / seed / stop
tools / tool_choice
reasoning_effort（low / medium / xhigh）
chat_template_kwargs（enable_thinking / preserve_thinking）
n=1 / frequency_penalty=0 / presence_penalty=0
```

`stop` は非空文字列または 1〜4 個の非空文字列の配列を受理する。最初に一致した文字列より前の `content` を返し、生成を cancel して `finish_reason="stop"` とする。stream では一致しうる末尾を保留する。`reasoning_content` と、tools を指定した場合の tool call の markup・引数には適用せず、tool call の前後をまたぐ一致も扱わない。stop の一致時の `completion_tokens` は一致までにサーバが受信した生成 token 数であり、非表示の stop や reasoning を含む。cancel 前に compute が先行生成した token は含まれない場合がある。prefix cache の hit 数は cancel 時には返らない。

Chat では `user`、`metadata`、`service_tier`、`parallel_tool_calls=true` を受理するが、使用しない。`store` は `false` のみ受理し、`true` は拒否する。`frequency_penalty` と `presence_penalty` は未指定または 0 のみ受理する。

`top_k` / `min_p` は Shelf extension である。指定がない値の既定は
`temperature=0`（greedy）/ `top_p=1.0` / `min_p=0.0` / `top_k=0` / `seed=0` である。

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

`tool_choice` は `auto`、`none`、`required`、関数名指定を受け付ける。`none` は tool 定義を model へ渡さない。
非 strict の `required` と関数名指定では思考を閉じ、tool call の開始部分を prompt に補って生成する。
その開始部分は prompt token に含み、生成上限には含まない。強制指定では生成完了まで stream 出力を保留する。
関数名指定は指定した名前以外を返さない。`required` で未定義の関数名が生成された場合はエラーとする。
非 strict の引数の構文と意味はモデルの生成次第で、schema に対する保証はない。生成上限に到達して未完了の
markup は tool call に変換せず、`length` を返す。tools 未指定や存在しない関数名は 400 とする。
Chat での関数名指定は `{"type":"function","function":{"name":"get_weather"}}`、
Responses では `{"type":"function","name":"get_weather"}` とする。正常に完了した tool call の
`finish_reason` は `tool_calls`、出力上限に達した場合は tool call の有無にかかわらず `length` である。

非 strict の tool calling は best-effort である。生成された tool call は Transformers の Qwen response
parser で解析し、schema の enforce や生成後の修復は行わない。parser が tool-call-like な
出力を解析できない場合は parse failure として扱い、生の markup を content として捏造しない。

streaming 中の tool call にも対応する。OpenAI 形式の `delta.tool_calls` として返り、
model 固有の `<tool_call>` 等の markup は client へ漏れない。

### concurrency と cancel

`--max-concurrent-requests N` は最大 in-flight request 数であり、同時に 1 executor step で
scheduler が扱える最大 request 数でもある。physical KV capacity は起動時に arena の余りから
確定し、実行中に拡張しない。容量が不足する場合、一部の request は Queued となり、
KV Banker が Active 完了後に admission する。`max_seq_len + 1` すら入らない構成は
**起動時に拒否**する（`--working-set-reserve-gib` が起動後に必要な余白、
DFlash2 有効時は drafter の weight と session pool 領域も確保計算に入る）。

`--max-num-batched-tokens N`（既定 2048）は 1 executor step の token 予算で、
executor の row bucket も決める。prompt が N を超えても prefill は chunk 分割で進む。
N を下げると program workspace が減り、その分だけ KV capacity が増える
（context を大きく取る構成では N が実質的な容量ダイヤルになる）。

client が connection を切断すると、stream / 非 stream のどちらでも in-flight generation は
cancel される（`finish_reason="cancelled"`）。MTP の prefill は次の chunk 境界（`--max-num-batched-tokens`）、
生成は次の spec round で反映する。cancel は request-specific であり、他の in-flight
request の KV / GDN / token stream には影響しない。

## Responses API

`POST /v1/responses` はテキストと function calling に対応し、`stream: true` では
イベント名と sequence number を含む SSE を返す。`input` は文字列、または
`message`（user / assistant / system / developer）、`function_call`、
`function_call_output`、`reasoning` の item 配列である。message の content は
文字列または `input_text` / `output_text` の配列とし、`instructions` は system
message として扱う。tool は `type: function`、`name`、`description`、
`parameters` を同じ object に置く。tool の実行と結果の再送はクライアントが行う。

`max_output_tokens`、`temperature`、`top_p`、`seed`、`reasoning.effort`
（`low` / `medium` / `xhigh`）、`tool_choice`（`auto` / `none` / `required` / 関数名指定）、
`parallel_tool_calls: true`、`truncation: disabled`、空の `include` または
`include: ["reasoning.encrypted_content"]` を受け付ける。後者を指定しても暗号化された思考は返さない。
`reasoning.summary: "auto"` は受理するが summary の生成は指定できない。
Responses では `metadata`、`user`、`prompt_cache_key`、`client_metadata`、
`parallel_tool_calls: true` を受理するが使用しない。
`store` は `false` のみ受理し、`true` は拒否する。履歴はリクエストごとに送る。usage の `input_tokens_details.cached_tokens` は
compute の prefix cache の復元数である。非 stream の `output_tokens_details` は
reasoning token 数を含む。stream では reasoning token 数を取得できない。
出力上限に達すると `status: "incomplete"`、`incomplete_details.reason: "max_output_tokens"` を返し、
stream では `response.incomplete` で終了する。正常終了時は `response.completed` で終了する。

`previous_response_id`、`conversation`、画像・音声・ファイル、`custom` tool、
`parallel_tool_calls: false`、
`truncation: auto`、その他の `include`、暗号化された入力、未対応の項目は HTTP 400 で拒否する。
`web_search` と `namespace` tool はモデルに渡さず除外するため、検索や namespace 内の tool は使えない。
`GET /v1/responses` と保存済み response の取得には対応しない。

OpenAI Python SDK の例（認証は行わないため API key は任意の値）:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for event in client.responses.create(model="shelf", input="こんにちは", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

## サポート外の API

未知のトップレベルキーは HTTP 400（`code=unsupported_parameter`）で拒否する。明示的に受理して無視するキーは、Chat と Responses の各節に列挙したものに限る。

Chat Completions では、非 0 の `frequency_penalty` / `presence_penalty`、`top_logprobs`、`prediction`、`modalities`、`audio`、`web_search_options` も未対応。次の一覧も Chat Completions に適用する。Responses の未対応項目は上記の節に記載する。

```text
reasoning
text
parallel_tool_calls=false
n != 1 / logit_bias / logprobs
```

以下は route 自体を持たない。

```text
その他の未登録 path -> HTTP 404 {"error": {"code": "not_found"}}
```

## Prefix Cache

prefix cache は compute runtime が提供する。共有 prefix の paged KV は active KV pool の
page を refcount 付きで共有し、GDN state だけを page 単位で保持する。
共有 prefix への extension や multi-turn の会話で prefill を省略する。

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

### Codex CLI

Codex CLI を Responses API の custom provider として接続する例を示す。
Codex の `prompt_cache_key` と `client_metadata` は受理するが推論には使用しない。
`include: ["reasoning.encrypted_content"]` と `reasoning.summary: "auto"` は受理するが、
暗号化 reasoning と summary は返さない。履歴中の `encrypted_content: null` は受理し、
暗号化された実データは拒否する。`web_search` と `namespace` は除外するため、
それらに依存する作業は実行できない。function tool はクライアントが実行し、
その結果を次のリクエストに含める。`custom` tool は未対応である。

```toml
model_provider = "shelf"
model = "shelf"

[model_providers.shelf]
name = "Shelf"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
supports_websockets = false
```

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

## 構造化出力

Chat Completions は `response_format` の `json_object` と `json_schema` を受理する。
Responses は `text.format` の `json_schema` を同じ制約経路で受理する。
thinking 有効時は `</think>` まで自由に生成し、その後の回答に制約を適用する。
単一 token の close tag を持たないモデルでの thinking 併用、tools、stop との
併用は HTTP 400 で拒否する。stream と非 stream、temperature > 0 に対応する。
JSON Schema は `type`、`properties`、`required`、`items`、`enum`、
`additionalProperties`（boolean）、`title`、`description` の subset を受け付ける。
`enum` は scalar 型のみで、すべての値が指定された型と一致する必要がある。
未対応キーワードは HTTP 400 とする。生成上限に達した場合は JSON が未完成となりうる。

Chat の `tools[].function.strict: true` と Responses の function tool の `strict: true` は、
引数を同じ schema subset で制約する。parameters の最上位は `additionalProperties: false` の
object、parameter 名は英数字、`_`、`-` に限る。`required` / 関数名指定は文法で強制し、
`auto` は tool 開始後のみに制約を適用する。thinking 時は close token の後から有効になる。
string 内の tool markup による区切り衝突を避けるため、生成される文字列中の生の `<` は
禁止し、`<` を含む string enum は 400 とする。XML parameter の enum は XGrammar が
値の外側に許す空白を除いて schema の enum 値へ対応付ける。非 strict の型変換は行わない。浮動小数点に変換できないほど大きい値は
不正な JSON を返さず tool parse failure とする。
strict tools の stream は完成した tool call をまとめて送る。生成上限に達した未完成の call は
返さない。strict tools と stop の併用、および response_format と tools の併用は 400 とする。

制約付き request は DFlash2 / MTP が有効でも通常 decode を使用し、投機との併用は未対応である。

## PLaMo 3 Translate

実験的対応である。挙動や対応範囲は今後変わる可能性がある。

PLaMo 3 Translate の BF16 checkpoint は翻訳用 chat template を使う。2B の
EXL3-INT8（mul24）checkpoint も `--model-dir` に指定できる。
量子化モデルにも配布 tokenizer などのファイルが必要。
サーバーの
`--max-seq-len` の既定値は 2048、上限も 2048 token である。これは prompt と
生成を合わせた上限で、長い sliding-window 文脈への対応を意味しない。
reasoning は無効である。

```bash
./build/shelf-server --model-dir /path/to/plamo-3-translate-0.8b --port 8123
curl http://127.0.0.1:8123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"shelf","messages":[{"role":"user","content":"今日は良い天気ですね。"}],
       "temperature":0,"chat_template_kwargs":{"source_lang":"ja","target_lang":"en",
       "context":"天気についての会話","glossary":[{"source":"天気","target":"weather"}]}}'
```

`target_lang` は必須。`source_lang`、`context`、`glossary` は任意で、
日本語への翻訳では `form` に `polite` または `plain` を指定できる。
配布 tokenizer の custom code を使用するため、必要な Python 依存も導入する。
PLaMo の利用には [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en)
が適用される。商用利用には事前登録と年間売上高の条件があり、再配布や
出力の利用にも条件があるため、使用前に原文を確認する。
