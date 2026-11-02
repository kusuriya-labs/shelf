#!/usr/bin/env python3
"""Reasoning (thinking) acceptance: non-streaming Chat Completions.

Qwen3.8-27B-FP8 defaults to thinking on. A response must carry the thinking
region in ``message.reasoning_content`` and the answer in ``message.content``,
with no chat-template control tags in either.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    ensure_chat_import,
    http_get_json,
    http_json,
    load_processor,
    model_dir,
)

codec = ensure_chat_import()

OPEN_TAG = "<" + "think" + ">"
CLOSE_TAG = "</" + "think" + ">"

QUESTION = "2+2は何ですか。数字だけで答えてください。"
HISTORY_QUESTION = "その数字に1を足すと？"


def read_token_log(path: Path) -> list[list[int]]:
    if not path.is_file():
        return []
    return [[int(x) for x in line.split(",") if x]
            for line in path.read_text().splitlines() if line.strip()]


def message_of(result: dict) -> dict:
    return result["choices"][0]["message"]


def check_default_thinking(checker, server, processor, token_log):
    result = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [{"role": "user", "content": QUESTION}],
         "temperature": 0, "max_completion_tokens": 768})
    choice = result["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""

    checker.check("e01-reasoning-content-present", bool(reasoning.strip()),
                  repr(message)[:300])
    checker.check("e01-answer-present", bool(content.strip()), repr(content)[:200])
    checker.check("e01-finish-reason", choice.get("finish_reason") in ("stop", "length"),
                  repr(choice.get("finish_reason")))
    checker.check("e01-no-close-tag-in-reasoning", CLOSE_TAG not in reasoning,
                  repr(reasoning[-80:]))
    checker.check("e01-no-open-tag-in-content", OPEN_TAG not in content,
                  repr(content[:80]))
    checker.check("e01-usage-completion-tokens",
                  result["usage"]["completion_tokens"] > 0, repr(result["usage"]))

    runs = read_token_log(token_log)
    checker.check("e01-token-log-present", bool(runs), repr(runs))
    if runs:
        split = codec.split_generation(processor, runs[-1], thinking=True)
        checker.check("e01-reasoning-matches-ids",
                      reasoning == split.reasoning,
                      f"wire={reasoning[-60:]!r} ids={split.reasoning[-60:]!r}")
        checker.check("e01-content-matches-ids",
                      content == split.answer,
                      f"wire={content[:60]!r} ids={split.answer[:60]!r}")
        checker.check("e01-close-token-in-ids", split.closed, "")
        checker.check("e01-reasoning-tokens-counted",
                      split.reasoning_tokens > 0
                      and result["usage"]["completion_tokens"] == len(runs[-1]),
                      f"split={split.reasoning_tokens} "
                      f"completion={result['usage']['completion_tokens']}")

    return result


def check_disable_thinking(checker, server):
    result = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [{"role": "user", "content": QUESTION}],
         "temperature": 0, "max_completion_tokens": 256,
         "chat_template_kwargs": {"enable_thinking": False}})
    message = message_of(result)
    content = message.get("content") or ""
    checker.check("e03-no-reasoning-field", "reasoning_content" not in message,
                  repr(message)[:300])
    checker.check("e03-answer-present", bool(content.strip()), repr(content)[:200])
    checker.check("e03-no-tags", OPEN_TAG not in content and CLOSE_TAG not in content,
                  repr(content)[:200])


def check_budget_exhausted_in_thinking(checker, server):
    result = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [{"role": "user", "content": QUESTION}],
         "temperature": 0, "max_completion_tokens": 16})
    choice = result["choices"][0]
    message = choice["message"]
    checker.check("e05-finish-length", choice.get("finish_reason") == "length",
                  repr(choice.get("finish_reason")))
    checker.check("e05-answer-not-fabricated",
                  (message.get("content") or "") == "", repr(message.get("content")))
    checker.check("e05-reasoning-present",
                  bool((message.get("reasoning_content") or "").strip()),
                  repr(message)[:300])


def check_reasoning_history(checker, server):
    first = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [{"role": "user", "content": QUESTION}],
         "temperature": 0, "max_completion_tokens": 768})
    first_message = message_of(first)
    reasoning = first_message.get("reasoning_content") or ""
    content = first_message.get("content") or ""

    second = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [
             {"role": "user", "content": QUESTION},
             {"role": "assistant", "reasoning_content": reasoning,
              "content": content},
             {"role": "user", "content": HISTORY_QUESTION},
         ],
         "temperature": 0, "max_completion_tokens": 768})
    message = message_of(second)
    checker.check("e07-history-accepted",
                  second["choices"][0].get("finish_reason") in ("stop", "length"),
                  repr(second["choices"][0].get("finish_reason")))
    checker.check("e07-history-answer-present",
                  bool((message.get("content") or "").strip()),
                  repr(message)[:300])
    checker.check("e07-history-reasoning-separate",
                  CLOSE_TAG not in (message.get("content") or ""),
                  repr((message.get("content") or "")[:200]))

    dropped = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "phaseshift",
         "messages": [
             {"role": "user", "content": QUESTION},
             {"role": "assistant", "reasoning_content": reasoning,
              "content": content},
             {"role": "user", "content": HISTORY_QUESTION},
         ],
         "temperature": 0, "max_completion_tokens": 768,
         "chat_template_kwargs": {"preserve_thinking": False}})
    checker.check("e08-preserve-off-accepted",
                  dropped["choices"][0].get("finish_reason") in ("stop", "length"),
                  repr(dropped["choices"][0].get("finish_reason")))


def check_openai_client(checker, server):
    try:
        from openai import OpenAI

        client = OpenAI(base_url=server.base_url, api_key="unused")
        completion = client.chat.completions.create(
            model="phaseshift",
            messages=[{"role": "user", "content": QUESTION}],
            temperature=0,
            max_completion_tokens=768,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        checker.check("e17-sdk-reasoning-content",
                      bool((reasoning or "").strip()), repr(message)[:300])
        checker.check("e17-sdk-content",
                      bool((message.content or "").strip()),
                      repr(message.content)[:200])
        checker.check("e17-sdk-separated",
                      OPEN_TAG not in (message.content or "")
                      and CLOSE_TAG not in (message.content or ""), "")
    except ImportError as exc:
        checker.check("e17-sdk-reasoning-content", False, repr(exc))


def check_meta(checker, server):
    meta = http_get_json(f"{server.base_url}/meta")
    reasoning = meta.get("reasoning") or {}
    checker.check("meta-reasoning-enabled", reasoning.get("enabled") is True,
                  repr(reasoning))
    checker.check("meta-default-thinking-on",
                  reasoning.get("default_enable_thinking") is True, repr(reasoning))
    checker.check("meta-default-effort",
                  reasoning.get("default_reasoning_effort") == "xhigh",
                  repr(reasoning))


def main() -> int:
    checker = Checker("server-reasoning")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    token_log = Path(tempfile.mkdtemp(prefix="ps-reasoning-")) / "tokens.txt"
    try:
        processor = load_processor()
        with ServerHarness(max_seq_len=1024,
                           env={"PHASESHIFT_SERVER_TOKEN_LOG": str(token_log)}) as server:
            check_meta(checker, server)
            check_default_thinking(checker, server, processor, token_log)
            check_disable_thinking(checker, server)
            check_budget_exhausted_in_thinking(checker, server)
            check_reasoning_history(checker, server)
            check_openai_client(checker, server)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
