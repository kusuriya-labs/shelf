#!/usr/bin/env python3
"""Reasoning (thinking) acceptance: streaming Chat Completions.

Streaming and non-streaming must agree on both regions, every chunk must carry
only one region, and the thinking to answer transition must happen exactly once.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    ensure_chat_import,
    http_json,
    http_sse,
    load_processor,
    model_dir,
)

codec = ensure_chat_import()

OPEN_TAG = "<" + "think" + ">"
CLOSE_TAG = "</" + "think" + ">"

QUESTION = "12と19を足すと何ですか。数字だけで答えてください。"
PLAIN_QUESTION = "Reply with the single word: hello"


def read_token_log(path: Path) -> list[list[int]]:
    if not path.is_file():
        return []
    return [[int(x) for x in line.split(",") if x]
            for line in path.read_text().splitlines() if line.strip()]


def collect(chunks):
    reasoning, content = [], []
    mixed = 0
    order_violation = 0
    finish = None
    usage = None
    for chunk in chunks:
        if chunk.get("usage") is not None and not chunk.get("choices"):
            usage = chunk["usage"]
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content") and delta.get("content"):
                mixed += 1
            if delta.get("reasoning_content"):
                if content:
                    order_violation += 1
                reasoning.append(delta["reasoning_content"])
            if delta.get("content") is not None and delta.get("content") != "":
                content.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "mixed": mixed,
        "order_violation": order_violation,
        "finish": finish,
        "usage": usage,
        "chunks": len(chunks),
    }


def stream(server, payload):
    return collect(list(http_sse(f"{server.base_url}/chat/completions", payload)))


def check_stream_parity(checker, server, processor, token_log):
    non = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "shelf",
         "messages": [{"role": "user", "content": QUESTION}],
         "temperature": 0, "max_completion_tokens": 768})

    result = stream(server, {
        "model": "shelf",
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0, "max_completion_tokens": 768, "stream": True,
        "stream_options": {"include_usage": True}})

    message = non["choices"][0]["message"]
    checker.check("e02-chunks-multiple", result["chunks"] > 1, repr(result["chunks"]))
    checker.check("e02-reasoning-deltas", bool(result["reasoning"].strip()),
                  repr(result["reasoning"])[:200])
    checker.check("e02-answer-deltas", bool(result["content"].strip()),
                  repr(result["content"])[:200])
    checker.check("e02-no-mixed-delta", result["mixed"] == 0, repr(result["mixed"]))
    checker.check("e02-single-transition", result["order_violation"] == 0,
                  repr(result["order_violation"]))
    checker.check("e02-reasoning-parity",
                  result["reasoning"] == (message.get("reasoning_content") or ""),
                  f"stream={result['reasoning'][-60:]!r} "
                  f"non={(message.get('reasoning_content') or '')[-60:]!r}")
    checker.check("e02-content-parity",
                  result["content"] == (message.get("content") or ""),
                  f"stream={result['content'][:60]!r} "
                  f"non={(message.get('content') or '')[:60]!r}")
    checker.check("e02-no-tags",
                  OPEN_TAG not in result["reasoning"] + result["content"]
                  and CLOSE_TAG not in result["reasoning"] + result["content"], "")
    checker.check("e02-finish-reason", result["finish"] in ("stop", "length"),
                  repr(result["finish"]))
    checker.check("e02-usage-chunk", result["usage"] is not None, repr(result["usage"]))

    runs = read_token_log(token_log)
    if result["usage"] is not None and runs:
        checker.check("e02-usage-completion-tokens",
                      result["usage"]["completion_tokens"] == len(runs[-1]),
                      f"usage={result['usage']['completion_tokens']} "
                      f"ids={len(runs[-1])}")
        split = codec.split_generation(processor, runs[-1], thinking=True)
        checker.check("e02-reasoning-matches-ids",
                      result["reasoning"] == split.reasoning, "")
        checker.check("e02-content-matches-ids",
                      result["content"] == split.answer, "")

    repeat = stream(server, {
        "model": "shelf",
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0, "max_completion_tokens": 768, "stream": True})
    checker.check("e02-repeat-deterministic",
                  repeat["reasoning"] == result["reasoning"]
                  and repeat["content"] == result["content"], "")


def check_stream_without_thinking(checker, server):
    non = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "shelf",
         "messages": [{"role": "user", "content": PLAIN_QUESTION}],
         "temperature": 0, "max_completion_tokens": 64,
         "chat_template_kwargs": {"enable_thinking": False}})
    result = stream(server, {
        "model": "shelf",
        "messages": [{"role": "user", "content": PLAIN_QUESTION}],
        "temperature": 0, "max_completion_tokens": 64, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}})
    checker.check("e03-stream-no-reasoning", result["reasoning"] == "",
                  repr(result["reasoning"])[:120])
    checker.check("e03-stream-content-parity",
                  result["content"] == (non["choices"][0]["message"].get("content") or ""),
                  f"stream={result['content']!r}")


def check_concurrent_mixed(checker, server):
    results = {}

    def run(name, payload):
        results[name] = stream(server, payload)

    thinking = {
        "model": "shelf",
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0, "max_completion_tokens": 384, "stream": True}
    plain = {
        "model": "shelf",
        "messages": [{"role": "user", "content": PLAIN_QUESTION}],
        "temperature": 0, "max_completion_tokens": 64, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}

    threads = [threading.Thread(target=run, args=("thinking", thinking)),
               threading.Thread(target=run, args=("plain", plain))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    checker.check("e12-both-finished", set(results) == {"thinking", "plain"},
                  repr(results.keys()))
    if set(results) == {"thinking", "plain"}:
        checker.check("e12-thinking-has-reasoning",
                      bool(results["thinking"]["reasoning"].strip()), "")
        checker.check("e12-plain-has-no-reasoning",
                      results["plain"]["reasoning"] == "",
                      repr(results["plain"]["reasoning"])[:120])
        checker.check("e12-plain-content-not-contaminated",
                      results["plain"]["content"].strip()
                      and results["plain"]["content"] not in results["thinking"]["reasoning"],
                      repr(results["plain"]["content"])[:120])
        checker.check("e12-no-tag-leak",
                      CLOSE_TAG not in results["thinking"]["reasoning"]
                      + results["thinking"]["content"]
                      + results["plain"]["content"], "")


def main() -> int:
    checker = Checker("server-reasoning-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    token_log = Path(tempfile.mkdtemp(prefix="shelf-reasoning-stream-")) / "tokens.txt"
    try:
        processor = load_processor()
        with ServerHarness(max_seq_len=1024, max_concurrent_requests=2,
                           env={"SHELF_SERVER_TOKEN_LOG": str(token_log)}) as server:
            check_stream_parity(checker, server, processor, token_log)
            check_stream_without_thinking(checker, server)
            check_concurrent_mixed(checker, server)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
