# Reasoning（Thinking）contract

Shelf Server の Chat Completions は、Qwen3.8 系 chat template の thinking を
既定で有効にし、思考と回答を別の field で返す。この文書が reasoning の current contract の
正本である。API 例は [../user/server.md](../user/server.md) に置く。

```text
HTTP request
  -> ChatRequest (reasoning options)
  -> openai_messages_to_hf()
  -> codec.prompt_ids()            chat template へ enable_thinking / reasoning_effort / preserve_thinking
  -> 推論予算と thinking 予算の解決  (prompt token 数で決まる。下記「thinking 予算」)
  -> Compute.generate / generate_stream   (max_new_tokens / reasoning_budget_tokens / reasoning_close_token)
  -> GenerationRegionParser        REASONING / ANSWER の分割（request ごとに 1 個）
  -> codec.parse_answer()          ANSWER のみを tool parser へ
  -> OpenAI wire serializer
```

## 入力

| 入力 | 既定 | 意味 |
| --- | --- | --- |
| `reasoning_effort` | `"xhigh"` | `low` / `medium` / `xhigh` のみ受理 |
| `chat_template_kwargs.enable_thinking` | `true` | `false` で thinking 無効 |
| `chat_template_kwargs.preserve_thinking` | `true` | 履歴の reasoning を chat template へ渡すか |

- `reasoning_effort` は受けた値をそのまま chat template へ渡す。`high` などを `xhigh` へ
  読み替えない。
- `reasoning_effort` の型不正、未対応値は HTTP 400（`param=reasoning_effort`、
  `code=unsupported_parameter`）。`null` は省略と同じである。
- `chat_template_kwargs` は allow-list である。`enable_thinking` / `preserve_thinking` 以外の
  key は HTTP 400（`param=chat_template_kwargs.<key>`、`code=unsupported_parameter`）。
  値が boolean でない場合も 400。
- `reasoning_effort` と `enable_thinking=false` の同時指定は矛盾として HTTP 400。
- `reasoning`（OpenAI の別 transport）は引き続き HTTP 400。

## Prompt

`codec.prompt_ids()` は `apply_chat_template()` に 3 つの引数を明示的に渡す。既定値は
実モデルの chat template の既定と一致するため、省略時と同じ Token 列になる。

- thinking 有効時、生成開始位置は assistant turn の `<think>` の内側である。開始 tag は
  Prompt 側にあり、生成列には現れない。
- thinking 無効時、Prompt 側で thinking block が閉じており、生成開始位置は回答側である。
- 生成可能長は `max_seq_len - len(prompt_ids)` で決まる（Prompt は実レンダリング Token 列）。
- prefix cache の key はこの Prompt Token 列である。`reasoning_effort` や履歴が違えば別の
  Prompt になり、同じ prefix として扱われない。
- `tools` は従来どおり chat template へ渡す。effort 指示は template が system 部へ書くので
  あり、server が独自の System Prompt を足すことはない。

### 起動時 probe

server は起動時に `codec.reasoning_capability()` で実ロード chat template を検査する。

- thinking 有効の Prompt が `<think>` を含み close tag を含まないこと
- thinking 無効の Prompt が close tag を含むこと
- `low` / `medium` / `xhigh` の描画結果が同一でないこと

満たさない場合は、reasoning 設定を無視して起動するのではなく、理由を stderr へ出して
終了 code 2 で停止する。

## 生成の分割

境界は**生の Token ID** で判定する。`</think>` の Token 列は
`codec.reasoning_close_ids()` が tokenizer から取得する（実モデルでは単一の control token）。
tokenizer の encoding が decode で往復しない場合は、レンダリング済み Prompt 側の当該
Token 列を正本にする。

状態は `REASONING` と `ANSWER` の 2 つで、close tag を検出した時点で 1 回だけ遷移する。

- 境界 tag 自体はどちらの領域の文字列にも含めない。
- thinking 無効時は最初から `ANSWER` である。
- `ANSWER` の先頭空白は strip する。`REASONING` は生成された文字列のまま返す。
- 末尾が不完全な UTF-8 である間は出力を保留し、完結した時点で送る。
  トークンごとに単純 decode して連結するのではなく、領域ごとに累積 Token 列を decode する。
- close tag に到達せず EOS / `length` で終わった場合、`reasoning_content` のみを返し、
  `content` は空文字列、`tool_calls` は空である。回答を捏造しない。
- streaming と非 streaming は同じ Parser を使う。最終的に
  `join(delta.reasoning_content) == message.reasoning_content`、
  `join(delta.content) == message.content` が成立する。

実装は `src/apps/common/shelf_chat/reasoning.py`（tokenizer 非依存の状態機械）と
`codec.new_region_parser()` / `codec.split_generation()` にある。

## thinking 予算

thinking は生成 Token 数で区切られ、request の推論予算のうち answer reserve を残した地点で
強制的に閉じる。client から変更できる field は無い（OpenAI API に対応 field が無い）。
予算の値と決定規則は [../user/server.md](../user/server.md) 「推論予算」に正本として置き、
ここには compute 側の contract だけを書く。

compute の `generate` は次の 2 つの field を受理する。

| field | 型 | 意味 |
| --- | --- | --- |
| `reasoning_budget_tokens` | int >= 0 | thinking に許す最大 Token 数。0 は予算なし |
| `reasoning_close_token` | int | thinking を閉じる単一 Token ID。`reasoning_budget_tokens > 0` と同時に必須 |

server は thinking 有効の request に対し、有効な生成上限からこれを算出して送る。
`reasoning_budget_tokens` が `max_new_tokens` を超える場合は compute が 400（`invalid_argument`）で拒否する。

強制終了の位置は commit 時点で決まる。

- 非 speculative: `commit_sampled_token` が、未クローズの thinking Token 数が予算に達した位置の
  Token を `reasoning_close_token` へ**置換**する。
- speculative (DFlash2): verify の受理数（`num_accepted_drafts`）と commit 数は変えず、
  correction / bonus token の位置を `reasoning_close_token` へ置換する。
  残 budget が 0 の slot は draft を作らない single-target step になり、その 1 Token が close token になる。
- model が自分で close tag を生成した場合、そこで thinking は閉じられ、以降置換は起きない。

close tag が単一 Token でない model では予算を課さない。server は起動時の
`codec.reasoning_capability()` で `close_token` を検査し、単一 Token でない場合は
compute へ budget field を送らない（起動ログに理由を出す）。


## Tool calling置換は terminal ではない。置換後の Token が stop token と一致することはないため、
停止条件の優先順位は変わらない。



tool parser へ渡すのは `ANSWER` 領域の文字列だけである。

- tool parser は `prefix=""` で生成し、Prompt の `</think>` を start anchor として使わない。
  これにより thinking 有効時（close tag が Prompt に存在しない）でも同じ Parser を使える。
- `REASONING` 内の `<tool_call>` は tool call として解釈しない。
- 応答形式（`tool_calls` の wire 形式、arguments が JSON string、`finish_reason="tool_calls"`）は
  thinking 無効時と同じである。
- parse に失敗した markup を content へ昇格させない挙動も従来どおりである。

## wire

非 streaming:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "条件を整理すると…",
      "content": "答えは42です。"
    },
    "finish_reason": "stop"
  }]
}
```

- `reasoning_content` は空でないときだけ message に現れる。thinking 無効時は field 自体が
  無く、従来の応答形と同じになる。
- tool call 時も `content: null` と `tool_calls` の形を保ち、`reasoning_content` を併せて返す。

streaming:

- `delta.reasoning_content` と `delta.content` / `delta.tool_calls` を別 chunk で送る。
  1 つの delta に `reasoning_content` と `content` を同時に入れない。
- 最後は従来どおり `finish_reason` 付き chunk と `data: [DONE]`。
- `stream_options.include_usage` の usage chunk は従来どおりである。

`finish_reason` は reasoning の有無で変わらない（`stop` / `length` / `tool_calls`）。
thinking 中に `max_completion_tokens` へ達した要求は `length` であり、回答が空でも正常応答である。

## usage

`completion_tokens` は reasoning を含む生成 Token 数である。

thinking には別枠の予算を与える。request の推論予算を `B`（リクエスト毎の最大推論予算）とすると、

```text
answer_reserve = min(max(512, B / 4), B / 2)
thinking_budget = B - answer_reserve
```

を compute 側へ `reasoning_budget_tokens` / `reasoning_close_token` として送る。
既定 `B = 32768` なら thinking 24576 / reserve 8192、`B = 65536` なら thinking 49152 /
reserve 16384 である。`B < 1024` では予算を課さない（compute へ送らない）。

`completion_tokens_details.reasoning_tokens` は未実装である。境界は Token 単位で確定して
いるため実装可能だが、計数定義（境界 tag を数えるか）を contract として固定してから追加する。

## 会話履歴

`openai_messages_to_hf()` は assistant message の `reasoning_content` を chat template へ渡す。

```json
[
  {"role": "user", "content": "問題を解いて"},
  {"role": "assistant", "reasoning_content": "まず条件を整理する", "content": "結果はAです。"},
  {"role": "user", "content": "その根拠は？"}
]
```

- `preserve_thinking=true` のとき、template は履歴の thinking を Prompt に含める。
  `false` のときは含めない。
- `reasoning` という別名も受理する。両方ある場合は値が一致することを要求し、矛盾は 400。
- server は会話履歴を保存しない。client が過去の `reasoning_content` を送らない場合、
  server はそれを復元できない。

## エラー

| 状況 | 応答 |
| --- | --- |
| `reasoning_effort` が未対応値・型不正 | 400 `unsupported_parameter` |
| `reasoning_effort` + `enable_thinking=false` | 400 `invalid_request` |
| `chat_template_kwargs` の未対応 key | 400 `unsupported_parameter` |
| `chat_template_kwargs` の値が boolean でない | 400 `invalid_request` |
| assistant 履歴の reasoning が文字列でない | 400 |
| close tag 未到達で終了 | 200（`reasoning_content` のみ、`content` 空） |

## 制限

- reasoning token 数の内訳（`completion_tokens_details.reasoning_tokens`）と、
  client からの thinking 予算指定は未実装である。server 既定値による強制は上記の通り。
- streaming の decode は累積 Token 列の再 decode である。長い reasoning では P1 で
  増分 decoder を検討する。
- sampling の既定値は reasoning 有無で変えない（`temperature=0` / `top_p=1.0` / `top_k=0`）。
  Qwen3.8 の推奨 sampling を既定にすることは未実施である。

## 構造化出力との併用

単一 token の close tag を持つモデルでは、compute は thinking 中に mask を適用せず、
自然な close token または予算による強制 close token の出力直後に request を mask 待ちへ
移す。server は close token を matcher に渡さず、初期 mask を返す。以後は回答 token ごとに
matcher を進める。thinking 予算が 0 でも同様に自然な close を待つ。thinking 中に上限や EOS
で終了した場合は回答がなく、JSON の完成は保証されない。制約付き request は投機しない。
