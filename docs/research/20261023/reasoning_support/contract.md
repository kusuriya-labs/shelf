# Reasoning 対応の設計判断

現在有効な contract は [../../../developer/reasoning.md](../../../developer/reasoning.md) が正本である。
ここには、その contract を選んだ判断と、採用しなかった案だけを残す。

## D1: 境界は Token ID で判定する（ACCEPT）

`</think>` は実 tokenizer で単一の control token（248069）である（assessment §4）。
文字列 split ではなく Token 列の一致で境界を決める。

- ACCEPT: 生成 Token 列をそのまま走査し、close の Token 列に一致した位置で 1 回だけ遷移する。
  `skip_special_tokens` による文字列の消去・正規化に依存しない。
- REJECT: decode 後の文字列から `</think>` を `str.split` する。special token の消去設定や
  空白正規化に依存し、「生成列に現れた境界」と「decode 後の文字列」が一致する保証がない。
- REJECT: 開始 tag の生成を待つ設計。thinking 有効時は開始 tag が Prompt 側にあり、
  生成列には現れない。

## D2: 複数 Token の close にも備える（ACCEPT）

単一 Token を前提に固定 ID を埋め込まない。`tokenizer.encode()` の結果を Token 列として保持し、
複数 Token の場合も分割 chunk をまたいで一致を検出する。encode が decode で往復しない
tokenizer では、レンダリング済み Prompt から同じ Token 列を取得する fallback を使う。

## D3: ANSWER の先頭空白は strip する（ACCEPT）

template は close tag の後ろに改行を置くため、生成される回答は `\n\n` で始まることが多い。
reasoning は生成されたまま返し、ANSWER のみ先頭空白を落とす。streaming と非 streaming で
同じ規則を使うため、両者の文面は一致する。

## D4: stream と非 stream で 1 つの Parser を共有する（ACCEPT）

`GenerationRegionParser` を唯一の分割実装とし、streaming は到着 Token を、非 streaming は
全 Token を同じ Parser に投入する。wire の等価性は
`join(delta) == message` で検証する（test_report の U17 / E02）。

## D5: tool parser は ANSWER 専用にし、`prefix=""` で生成する（ACCEPT）

- ACCEPT: `parse_response(text, TOOL_RESPONSE_TEMPLATE, prefix="")` に ANSWER 文字列だけを渡す。
  transformers 5.14.1 の `ResponseParser` は prefix を消費せず implicit region から開始する。
- REJECT: `start_anchor` を消すだけ。Prompt を渡す呼び出しが残ると、anchor が無いため
  Prompt 全体（system の tool 説明・履歴）が parser に流れ、content に混入する。
- REJECT: `start_anchor` を `</think>` 以外（`<|im_start|>assistant`）へ変える。thinking 無効時は
  Prompt に `<think>\n\n</think>\n\n` が残るため、その文字列が content として解釈される。
- `TOOL_RESPONSE_TEMPLATE` の `start_anchor` は template 仕様上必須なので残すが、
  prefix を渡さないため評価されない。

## D6: reasoning 既定は server が決め、codec の既定は template と一致させる（ACCEPT）

`codec.prompt_ids()` の既定を `enable_thinking=true` / `reasoning_effort="xhigh"` /
`preserve_thinking=true` にした。実 template の既定と同じ Token 列になるため、
「引数省略」と「明示」で Prompt が変わらない（U01）。server は request の値をそのまま渡す。

## D7: 起動時 probe で template 能力を確認する（ACCEPT）

実ロード template が thinking と effort を解釈できない場合、reasoning 設定を黙って無視せず
起動を拒否する（終了 code 2）。`model` という表示名ではファミリ判定しない。

- REJECT: 警告だけ出して起動し、`reasoning_effort` を無視して生成する。
- REJECT: 起動後に request が来るまで気づかない。

## D8: 出力は `reasoning_content` のみ（ACCEPT）

公式モデルカードの例と互換性の高い `reasoning_content` を唯一の出力名とする。
`reasoning` を同時に返して重複させない。入力（assistant 履歴）では `reasoning` 別名を
受理するが、両方ある場合は値の一致を要求する。

## D9: reasoning token 数の内訳は P0 では返さない（DEFER）

境界は Token 単位で確定しているため実装可能だが、境界 tag を数えるかどうかの計数定義を
contract として固定してから `completion_tokens_details.reasoning_tokens` を追加する。
`completion_tokens` は従来どおり reasoning を含む総数である。

## D10: streaming の decode は領域ごとの累積 decode（ACCEPT）

領域（reasoning / answer）ごとに累積 Token 列を decode し、直前の出力との差分を送る。
不完全な UTF-8 の末尾は保留する。Token ごとの単純 decode 連結は行わない。
長い reasoning での二乗的な decode コストは既知の課題であり、増分 decoder は P1 とする。

## D11: Prompt 長・prefix cache は既存契約を変えない（ACCEPT）

Prompt は実レンダリング Token 列、生成可能長は `max_seq_len - len(prompt_ids)` である。
`reasoning_effort` や履歴が違えば Prompt Token 列も変わるため、prefix cache の key は
自然に区別される。cache の実装には触れていない。
