# Reasoning 対応の実装記録

## 変更ファイル

| ファイル | 変更 |
| --- | --- |
| `src/apps/common/phaseshift_chat/reasoning.py` | 新規。tokenizer 非依存の `GenerationRegionParser`（REASONING / ANSWER の状態機械）と `split_regions()` |
| `src/apps/common/phaseshift_chat/codec.py` | `prompt_ids()` に reasoning 引数、`reasoning_close_ids()` / `new_region_parser()` / `split_generation()` / `parse_answer()` / `probe_reasoning()` を追加。tool parser を `prefix=""` に変更 |
| `src/apps/common/phaseshift_chat/__init__.py` | 公開 API の更新 |
| `src/apps/server/serverlib/openai_protocol.py` | `ChatRequest` に reasoning 設定、`parse_reasoning_effort()` / `parse_chat_template_kwargs()`、履歴 `reasoning_content`、`reasoning_content` の wire 出力 |
| `src/apps/server/serverlib/chat_service.py` | region parser による分割、`ReasoningDelta`、`AssistantResult.reasoning_content`、stream / 非 stream の共通化 |
| `src/apps/server/phaseshift_server.py` | 起動時 template probe、起動ログ、`/v1/meta` の `reasoning`、request log |
| `cmake/apps.cmake` | `reasoning.py` を server staging の依存へ追加 |
| `tests/server/test_reasoning_parser.py` | 新規。GPU / model 不要の Parser 単体試験 |
| `tests/server/test_reasoning_request.py` | 新規。Prompt 行列・入力検証・履歴・tool 解析（model のみ、GPU 不要） |
| `tests/server/test_server_reasoning.py` | 新規。非 streaming E2E |
| `tests/server/test_server_reasoning_stream.py` | 新規。streaming E2E と混在並列 |
| `tests/server/test_server_reasoning_tools.py` | 新規。thinking + tool calling E2E |
| `tests/server/run_server_regression.py` | `reasoning` group を manifest へ追加 |
| `tests/server/*.py`（既存 11 本） | 非 thinking を目的とする request へ `chat_template_kwargs.enable_thinking=false` を明示 |
| `docs/developer/reasoning.md` | 新規。current contract |
| `docs/developer/server_status.md` | 対応範囲の更新 |
| `docs/user/server.md` | API 例・parameter・制限の更新 |

## 実装の要点

- 分割は `reasoning.py` の 1 実装だけである。server は「到着 Token を feed する」か
  「全 Token を feed する」かの違いしか持たない。
- 領域ごとに累積 Token 列を decode する。末尾が不完全な UTF-8 の間は出力を保留する。
- `done.generated_ids` が streaming で観測した列と一致しない場合は、最終列で分割し直し、
  既に送った文字列との差分だけを送る（`_region_tail()`）。差分が取れない場合は
  `compute_error` として明示的に失敗する。文字列を遡って修正しない。
- tool parser は ANSWER 領域だけを受け取る。`prefix=""` により Prompt 依存の anchor 切り詰めを
  廃止した。`<tool_call>` が REASONING に現れても tool call にしない。
- 起動時に template を probe し、thinking / effort を解釈できない場合は起動を拒否する。
- `decode_stable()` / `monotonic_delta()` / `_parse_final()` は region parser に置き換わり、
  重複実装を残さないため削除した。

## 削除・置換した旧経路

| 旧 | 新 |
| --- | --- |
| `_stream_text` の累積 decode + `monotonic_delta` | `GenerationRegionParser.feed` |
| `_stream_tools` の `incremental_ok` fallback（`_parse_final`） | `_region_tail` の resync |
| `parse_assistant_message(prompt_ids, generated_ids)` | `split_generation` + `parse_answer(answer_text)` |
| `new_response_parser(processor, prompt, prompt_prefix)` | `new_response_parser(processor)`（`prefix=""` 固定） |

## 後回しにした事項

- `completion_tokens_details.reasoning_tokens`（計数定義を決めてから追加）
- streaming の増分 decoder（長い reasoning での decode コスト）
- Thinking の token budget / 強制 answer 遷移
- Qwen3.8 推奨 sampling（`temperature=1.0` / `top_p=0.95` / `top_k=20`）の既定化
- `reasoning` transport object（OpenAI Responses 系）の対応
- 履歴の server 側永続化

## 残存リスク

- reasoning が `max_seq_len` を消費するため、長い thinking では残り生成枠が減る。挙動は
  `max_seq_len - len(prompt_ids)` の既存規則どおりで、reasoning 用の別枠はない。
- streaming の decode は累積 Token 列の再 decode であるため、reasoning が長いほど
  Host CPU 時間が増える。P0 では正しさを優先した。
