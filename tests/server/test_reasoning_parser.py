#!/usr/bin/env python3
"""Reasoning region split: token-level state machine (no GPU, no model).

The parser is exercised with a byte-level fake tokenizer so the tests cover
multi-byte runes split across tokens without loading a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "apps" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phaseshift_chat import reasoning  # noqa: E402
from support import Checker  # noqa: E402

CLOSE = 1001
CLOSE_MULTI = (1002, 1003)


class ByteDecoder:
    """Decode token IDs as UTF-8 bytes; IDs >= 1000 are control tokens."""

    def __call__(self, ids) -> str:
        data = bytes(int(i) for i in ids if int(i) < 1000)
        return data.decode("utf-8", errors="replace")


def tokenize(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def run(ids, *, thinking=True, close=(CLOSE,), chunk_sizes=None):
    parser = reasoning.GenerationRegionParser(
        ByteDecoder(), close, thinking=thinking)
    deltas = []
    if chunk_sizes is None:
        deltas.extend(parser.feed(ids))
    else:
        position = 0
        for size in chunk_sizes:
            deltas.extend(parser.feed(ids[position:position + size]))
            position += size
        deltas.extend(parser.feed(ids[position:]))
    deltas.extend(parser.finalize())
    return parser, deltas


def collect(deltas, region):
    return "".join(delta.text for delta in deltas if delta.region == region)


def main() -> int:
    checker = Checker("reasoning-parser")

    parser, deltas = run([], thinking=True)
    checker.check("empty-generation", parser.reasoning == "" and parser.answer == ""
                  and parser.closed is False, repr(parser.result()))

    ids = tokenize("まず条件を整理する。\n") + [CLOSE] + tokenize("\n\n答えは42です。")
    parser, deltas = run(ids)
    checker.check("closed", parser.closed, repr(parser.result()))
    checker.check("reasoning-text",
                  parser.reasoning == "まず条件を整理する。\n", repr(parser.reasoning))
    checker.check("answer-leading-space-stripped",
                  parser.answer == "答えは42です。", repr(parser.answer))
    checker.check("boundary-not-in-text",
                  "</think>" not in parser.reasoning + parser.answer,
                  repr(parser.reasoning + parser.answer))
    checker.check("deltas-concat",
                  collect(deltas, reasoning.REASONING) == parser.reasoning
                  and collect(deltas, reasoning.ANSWER) == parser.answer,
                  repr(deltas))

    ids = tokenize("thinking") + [CLOSE] + tokenize("answer")
    one_shot, _ = run(ids)
    token_at_a_time, deltas = run(ids, chunk_sizes=[1] * len(ids))
    checker.check("incremental-equals-batch",
                  (one_shot.reasoning, one_shot.answer)
                  == (token_at_a_time.reasoning, token_at_a_time.answer),
                  f"batch={(one_shot.reasoning, one_shot.answer)} "
                  f"incremental={(token_at_a_time.reasoning, token_at_a_time.answer)}")
    checker.check("incremental-deltas-concat",
                  collect(deltas, reasoning.REASONING) == one_shot.reasoning
                  and collect(deltas, reasoning.ANSWER) == one_shot.answer,
                  repr(deltas))

    ids = tokenize("only reasoning, no close")
    parser, deltas = run(ids, chunk_sizes=[3] * 8)
    checker.check("unclosed-is-reasoning-only",
                  parser.reasoning == "only reasoning, no close"
                  and parser.answer == "" and not parser.closed,
                  repr(parser.result()))
    checker.check("unclosed-answer-not-fabricated",
                  collect(deltas, reasoning.ANSWER) == "", repr(deltas))

    ids = [CLOSE] + tokenize("answer only")
    parser, _ = run(ids)
    checker.check("empty-reasoning", parser.reasoning == ""
                  and parser.answer == "answer only", repr(parser.result()))

    ids = tokenize("think") + [CLOSE] + tokenize("   \n\nanswer")
    parser, _ = run(ids)
    checker.check("answer-lstrip", parser.answer == "answer", repr(parser.answer))

    emoji = "😀"
    raw = emoji.encode("utf-8")
    ids = tokenize("絵文字:") + [CLOSE] + [raw[0]] + [raw[1], raw[2], raw[3]] \
        + tokenize("です")
    parser, deltas = run(ids, chunk_sizes=[1] * len(ids))
    checker.check("utf8-split-answer", parser.answer == "😀です", repr(parser.answer))
    checker.check("utf8-split-no-replacement",
                  "\ufffd" not in parser.answer, repr(parser.answer))
    answer_deltas = [d.text for d in deltas if d.region == reasoning.ANSWER]
    checker.check("utf8-held-until-complete",
                  all("\ufffd" not in text for text in answer_deltas),
                  repr(answer_deltas))

    ids = tokenize("think") + list(CLOSE_MULTI) + tokenize("answer")
    parser, _ = run(ids, close=CLOSE_MULTI, chunk_sizes=[1] * len(ids))
    checker.check("multi-token-close", parser.reasoning == "think"
                  and parser.answer == "answer" and parser.closed,
                  repr(parser.result()))

    ids = tokenize("th") + [CLOSE_MULTI[0]] + [CLOSE_MULTI[1]] + tokenize("ans")
    parser, _ = run(ids, close=CLOSE_MULTI, chunk_sizes=[1] * len(ids))
    checker.check("split-close-detected-once",
                  parser.reasoning == "th" and parser.answer == "ans",
                  repr(parser.result()))

    textual = tuple(tokenize("</think>"))
    ids = tokenize("think text") + list(textual) + tokenize("answer")
    parser, _ = run(ids, close=textual, chunk_sizes=[1] * len(ids))
    checker.check("textual-close-not-leaked",
                  parser.reasoning == "think text" and parser.answer == "answer",
                  repr(parser.result()))

    ids = tokenize("reason") + [CLOSE] + tokenize("answer")
    parser, deltas = run(ids, chunk_sizes=[6, 1, 6])
    checker.check("boundary-exactly-on-chunk-edge",
                  parser.reasoning == "reason" and parser.answer == "answer",
                  repr(parser.result()))

    ids = tokenize("abcdef") + [CLOSE] + tokenize("ghij")
    first = reasoning.GenerationRegionParser(ByteDecoder(), (CLOSE,))
    second = reasoning.GenerationRegionParser(ByteDecoder(), (CLOSE,))
    first.feed(ids[:2])
    second.feed(ids)
    first.feed(ids[2:])
    first.finalize()
    second.finalize()
    checker.check("independent-parsers",
                  (first.reasoning, first.answer) == ("abcdef", "ghij")
                  and (second.reasoning, second.answer) == ("abcdef", "ghij"),
                  f"first={(first.reasoning, first.answer)} "
                  f"second={(second.reasoning, second.answer)}")

    ids = tokenize("plain answer")
    parser, deltas = run(ids, thinking=False)
    checker.check("non-thinking-is-answer",
                  parser.reasoning == "" and parser.answer == "plain answer"
                  and parser.closed, repr(parser.result()))

    ids = [CLOSE] + tokenize("plain")
    parser, _ = run(ids, thinking=False)
    checker.check("non-thinking-ignores-close-token",
                  parser.answer == "plain" and parser.reasoning == "",
                  repr(parser.result()))

    parser = reasoning.GenerationRegionParser(ByteDecoder(), (CLOSE,))
    parser.feed(tokenize("reason") + [CLOSE] + tokenize("ans"))
    parser.finalize()
    parser.resync(reasoning.ANSWER, "answer")
    checker.check("resync-keeps-regions",
                  parser.answer == "answer" and parser.reasoning == "reason",
                  repr(parser.result()))

    result = reasoning.split_regions(
        ByteDecoder(), tokenize("a") + [CLOSE] + tokenize(" b"), (CLOSE,))
    checker.check("split-regions-helper",
                  result.reasoning == "a" and result.answer == "b"
                  and result.reasoning_tokens == 1,
                  repr(result))

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
