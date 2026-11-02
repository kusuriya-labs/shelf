# quality gate fixture

## corpus

teacher-forced perplexity / KL 用の token corpus は commit しない。
`tools/quality_gate.py` が `vendor/BetterBench/corpus/v1` の 8 category
（chat / code / file_edit / json / math / prose / reasoning / summarization）を
その順で読み、各 case を model の chat template（`enable_thinking=False`）で
token 化して 1 本の PSKLDTOK に連結する。

- 生成: `python3 tools/quality_gate.py corpus --model-dir <model> --out <path>`
- `run` は work dir に自動生成する
- corpus の token 数と SHA256 は report に残す
- BetterBench corpus と tokenizer の版が変わると token 列も変わるため、
  絶対 perplexity の比較ではなく同一 run 内の BF16 KV / FP8 KV 差分を gate にする

## reasoning.json

BF16 KV と FP8 KV の end-task 差分を見るための固定問題集である。各 case は
prompt と最終回答（数値）を持ち、回答は数値の完全一致で採点する。
絶対 accuracy ではなく、同一 prompt 集合に対する BF16 と FP8 の差分を gate にする。

問題を追加・変更したら、同じ gate で BF16 / FP8 の両方を測り直す。

## gsm8k.json

openai/gsm8k の test split（全1319問）を変換した end-task fixture である。
各 case は `prompt`（question 原文）と `expected`（回答末尾の `####` の後の数値）を
持ち、ルートの `instruction` は `run` が prompt の末尾に連結して使う。

採点は生成文の最後の `####` の後ろに最初に現れる数値（カンマ除去）を
抽出し、`expected` と float の完全一致で判定する。

- 既定は先頭 50 問。`--gsm8k-limit 0` で全件、`--skip-gsm8k` で除外
- `answer_format` が `gsm8k` の case のみがこの採点経路を使う
- BF16 からの後退の既定許容は 2 問（`--max-gsm8k-regression`）

絶対 accuracy ではなく、同一 prompt 集合に対する BF16 と FP8 の差分を gate にする。
