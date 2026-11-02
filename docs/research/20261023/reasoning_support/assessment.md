# Reasoning 対応の事前調査（ZIP との差分・実テンプレート・境界 Token）

- 実施日: 2026-10-23
- 対象: PhaseShift Server / Qwen3.8-27B-FP8
- 指示書: `PhaseShift_Reasoning.md`（ZIP 基準 commit `3157cf2`）

## 1. 環境

| 項目 | 値 |
| --- | --- |
| 作業ブランチ基点 | `95e4c99`（ZIP 基準より新しい main） |
| Python | 3.12 |
| transformers | 5.14.1 |
| aiohttp | 3.14.3 |
| GPU | AMD Radeon AI PRO R9700（gfx1201）1 枚、34.2 GB |
| model | `models/Qwen3.8-27B-FP8`（HF fp8） |
| chat template | `models/Qwen3.8-27B-FP8/chat_template.jinja` |
| tokenizer | `Qwen2Tokenizer`（byte-level BPE、`clean_up_tokenization_spaces=false`） |
| compute | `build/phaseshift-compute`（既存ビルド、変更なし） |

## 2. ZIP との差分（重複実装の回避）

ZIP の記載と実ソースを照合した。**ZIP の行番号に合わせた改変は行っていない。**

| 項目 | ZIP の記載 | 実ソース | 採用した方針 |
| --- | --- | --- | --- |
| `chat_service.prepare()` | `enable_thinking=False` 固定 | 同じ | request の reasoning 設定を渡す |
| `stream_options.include_usage` | P1（未実装） | **既に実装済み**（usage chunk、`prompt_tokens_details.cached_tokens`） | 再実装しない |
| `_stream_text` / `_stream_tools` | 累積 Token 列を毎回全 decode | 同じ。`decode_stable` / `monotonic_delta` を使う | 共通の region parser に置き換える |
| `new_response_parser(processor, prompt, prompt_prefix)` | `start_anchor="</think>"` 前提 | 同じ（Prompt を decode して anchor で切り詰める） | `prefix=""` に変更し、ANSWER 領域だけを渡す |
| `probe_tool_calling()` | 記載なし | 既に実装済み（起動時 capability 相当） | reasoning 用の probe を追加し、tool probe を prefix 非依存に直す |
| `parse_assistant_message()` | content / tool_calls のみ | 同じ | `parse_answer(answer_text, tools)` に置き換え |
| `/v1/meta` | 記載なし | 既に実装済み（kv_cache を含む） | reasoning の既定を追加 |
| `docs/user/server.md` | 未対応 | usage chunk / meta を既に記載 | reasoning の節を追加 |

ZIP が P1 としていた usage chunk は既に本流へ入っているため、P0 の範囲を
「入力検証・Prompt・分割・wire・履歴」に絞った。

## 3. chat template の実測（T01–T09）

message は `[{"role": "user", "content": "2+2?"}]`。`tokenize=True` の
生成可能 Prompt 長（token 数）と末尾を実測した。

| ケース | `enable_thinking` | `reasoning_effort` | `preserve_thinking` | tools | Prompt 長 | 末尾 |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | 省略 | 省略 | 省略 | 無 | 56 | `<|im_start|>assistant\n<think>\n` |
| T02 | true | xhigh | true | 無 | 56 | 同上（T01 と同一 Token 列） |
| T03 | true | medium | true | 無 | 14 | 同上。system 部に effort 指示が入らない |
| T04 | true | low | true | 無 | 44 | 同上。low 用の指示が入る |
| T05 | false | 省略 | true | 無 | 16 | `…<think>\n\n</think>\n\n`（Prompt 側で閉じる） |
| T06 | true | xhigh | true | 有 | 305 | `…<think>\n`（tools の system 部の後） |
| T07 | true | xhigh | true | 有 | 305 | 履歴に reasoning を保持して描画 |
| T08 | true | xhigh | false | 有 | 269 | 履歴の thinking block を描画しない |
| T09 | true | high | true | 無 | 例外 | `TemplateError: Unexpected reasoning effort high` |

確認した事実:

- 既定（引数省略）と `xhigh` / `preserve_thinking=true` の明示は同一の Token 列になる。
- `medium` は effort 指示を追加しない（template の仕様）。`xhigh` と `low` は追加する。
- 同じ入力で 2 回描画した Token 列は一致する（決定論的）。
- 未対応 effort は template 自体が例外を投げる。server 側でも 400 にする。
- thinking 無効時は Prompt 側で thinking block が閉じ、生成は回答側から始まる。

## 4. 境界 Token（重要）

```
tokenizer.encode("<think>",  add_special_tokens=False) -> [248068]
tokenizer.encode("</think>", add_special_tokens=False) -> [248069]
```

- どちらも**単一の control token** である（`added_tokens_decoder` に登録済み）。
- `decode([248069], skip_special_tokens=True)` は `</think>` を返す（この token は
  `special=false` なので decode で消えない）。
- 生成列には close tag が**1 token** として現れる。したがって境界判定は文字列ではなく
  Token ID で行える。複数 Token 列の場合にも備え、実装は「Token 列の最初の一致」で判定する。
- tokenizer の `encode()` が decode で往復しない環境に備え、レンダリング済み Prompt から
  同じ Token 列を探す fallback を実装した。

## 5. decode の増分性

byte-level BPE であるため、累積 Token 列の decode は次を満たすことを実測した
（日本語・絵文字・ZWJ 結合絵文字・コードブロックの 5 サンプル）。

- 完結した rune まで含む decode 結果は、直前の結果の prefix である（単調）。
- 不完全な UTF-8 の間は末尾が U+FFFD になる（保留すればよい）。
- 最終的な decode 結果は一括 decode と一致する。

保留した token 数の実測はサンプルにより 0〜13。したがって「末尾 U+FFFD を保留する」
増分 decode で streaming と非 streaming の文面を一致させられる。

## 6. compute の出力契約

既存の回帰試験と本作業の E2E で確認した。

- `done.generated_ids` は thinking を含む生成列全体で、最後に stop token（`<|im_end|>`）を含む。
- streaming の `token_id` 列と `done.generated_ids` の先頭は一致する。
- `</think>` の直後でも生成は止まらず回答へ進む（E2E で reasoning と answer が両方得られた）。
- stop token は `generation_config.json` の `eos_token_id`（248046 / 248044）であり、
  reasoning の close tag（248069）とは独立である。

**Compute、KV cache、Scheduler、Drafter の変更は不要だった。**

## 7. tool parser の前提

Transformers 5.14.1 の `ResponseParser` は `prefix=""` を受け付け、prefix を消費せずに
implicit region から開始する。したがって ANSWER 文字列だけを渡せる。

- 旧実装は Prompt を decode して `start_anchor="</think>"` で切り詰めていた。
- thinking 有効時は Prompt に close tag が無いため anchor が見つからず、Prompt 全体
  （system の tool 説明と履歴）が parser に流れ込む。これは採用できない。
- 実測で `prefix=""` + ANSWER 文字列により既存の Qwen tool 形式（`<tool_call>` /
  `<function=...>` / `<parameter=...>`）が同じ結果で解析できた。固定文字列の
  単体試験と E2E の双方で確認した。
