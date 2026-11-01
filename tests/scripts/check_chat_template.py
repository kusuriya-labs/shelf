#!/usr/bin/env python3
"""Qwen chat template の実機確認（T06。モデル・GPU 不要、tokenizer のみ）。

確認する契約:
  * 9B / 27B で tokenizer の語彙・特殊トークン ID が一致する
  * 思考モード ON / OFF でテンプレートの末尾が変わる
    （ON: `<think>\\n` で開く、OFF: `<think>\\n\\n</think>\\n\\n` で閉じる）
  * system / user / assistant の role が正しく描画される
  * tokenize=False の文字列と tokenize=True の ID が同じ内容を指す
  * `server._ids_from_template_output` が実 `BatchEncoding` から ID を取り出せる

使い方:
  python3 tests/scripts/check_chat_template.py \
      models/Qwen3.5-9B models/Qwen3.8-27B-FP8
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "apps", "server"))

import server as srv  # noqa: E402

SPECIAL = {
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|endoftext|>": 248044,
    "<think>": 248068,
    "</think>": 248069,
}

failures: list[str] = []


def check(cond: bool, what: str) -> None:
    if cond:
        print(f"PASS  {what}")
    else:
        print(f"FAIL  {what}")
        failures.append(what)


def template(tokenizer, messages, *, thinking, tokenize):
    kwargs = {} if thinking is None else {"enable_thinking": thinking}
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=tokenize, **kwargs)


def check_model(path: str) -> dict:
    from transformers import AutoTokenizer

    name = os.path.basename(path.rstrip("/"))
    print(f"\n== {name} ({path})")
    tok = AutoTokenizer.from_pretrained(path)

    check(len(tok) > 200000, f"{name}: 語彙が 200k 超（{len(tok)}）")
    check(tok.eos_token_id == SPECIAL["<|im_end|>"],
          f"{name}: tokenizer の eos は <|im_end|>（{tok.eos_token_id}）")
    for token, expected in SPECIAL.items():
        got = tok.convert_tokens_to_ids(token)
        check(got == expected, f"{name}: {token} = {expected}（実際 {got}）")

    # ---- 思考モード ON / OFF ----
    user = [{"role": "user", "content": "1+1は?"}]
    on_text = template(tok, user, thinking=True, tokenize=False)
    off_text = template(tok, user, thinking=False, tokenize=False)
    default_text = template(tok, user, thinking=None, tokenize=False)
    check(on_text.rstrip().endswith("<think>"), f"{name}: ON は <think> で開く")
    check("</think>" in off_text, f"{name}: OFF は </think> で閉じる")
    check(off_text.rstrip().endswith("</think>"), f"{name}: OFF の末尾は </think>")
    check(default_text == on_text, f"{name}: 未指定の既定は ON と同じ")

    on_ids = srv._ids_from_template_output(template(tok, user, thinking=True, tokenize=True))
    off_ids = srv._ids_from_template_output(template(tok, user, thinking=False, tokenize=True))
    check(SPECIAL["<think>"] in on_ids, f"{name}: ON の ID に <think> が入る")
    check(SPECIAL["</think>"] in off_ids, f"{name}: OFF の ID に </think> が入る")
    check(on_ids != off_ids, f"{name}: ON / OFF で ID 列が変わる")
    plain = tok.decode(on_ids, skip_special_tokens=True)
    check(plain.strip() != "", f"{name}: skip_special_tokens でも本文が残る")
    check("<|im_start|>" not in plain, f"{name}: decode が特殊トークンを漏らさない")

    # ---- role の整合性 ----
    convo = [
        {"role": "system", "content": "あなたは簡潔に答えます。"},
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "こんにちは。"},
        {"role": "user", "content": "元気?"},
    ]
    convo_text = template(tok, convo, thinking=False, tokenize=False)
    check("<|im_start|>system" in convo_text, f"{name}: system role が描画される")
    check(convo_text.count("<|im_start|>assistant") == 2,
          f"{name}: assistant の履歴と生成プロンプトが 2 回")
    check(convo_text.count("<|im_start|>user") == 2, f"{name}: user が 2 回")
    check("あなたは簡潔に答えます。" in convo_text, f"{name}: system の本文が入る")

    # ---- 文字列と ID の一致 ----
    convo_ids = srv._ids_from_template_output(template(tok, convo, thinking=False, tokenize=True))
    decoded = tok.decode(convo_ids)
    for piece in ["<|im_start|>system", "あなたは簡潔に答えます。", "<|im_start|>user"]:
        check(piece in decoded, f"{name}: decode(IDs) に {piece!r} が含まれる")

    return {"name": name, "vocab": len(tok), "eos": tok.eos_token_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen chat template の実機確認（T06）")
    parser.add_argument("models", nargs="+", help="モデルのディレクトリ（tokenizer を含む）")
    args = parser.parse_args(argv)

    present = [p for p in args.models if os.path.isdir(p)]
    missing = [p for p in args.models if not os.path.isdir(p)]
    for p in missing:
        print(f"SKIP  {p} が無い")
    if not present:
        print("モデルが 1 つも無い")
        return 2

    contracts = [check_model(p) for p in present]

    # ---- モデル間の一致 ----
    print("\n== モデル間の一致")
    eos = {c["eos"] for c in contracts}
    vocab = {c["vocab"] for c in contracts}
    check(len(eos) == 1, f"eos が全モデルで一致（{sorted(eos)}）")
    check(len(vocab) == 1, f"語彙数が全モデルで一致（{sorted(vocab)}）")

    print()
    if failures:
        print(f"{len(failures)} 件失敗")
        return 1
    print("chat template の契約はすべて一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
