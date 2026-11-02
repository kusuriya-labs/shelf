#!/usr/bin/env python3
"""Gate 2 acceptance: streaming chat completions through the server."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    compute_pids,
    http_json,
    http_sse,
    model_dir,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def read_token_log(path: Path) -> list[list[int]]:
    if not path.is_file():
        return []
    return [[int(x) for x in line.split(",") if x]
            for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    checker = Checker("server-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    token_log = Path(tempfile.mkdtemp(prefix="shelf-tokenlog-")) / "tokens.txt"

    try:
        with ServerHarness(env={"SHELF_SERVER_TOKEN_LOG": str(token_log)}) as server:
            non_stream = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
            non_text = non_stream["choices"][0]["message"]["content"]

            pids_before = compute_pids()
            chunks = list(http_sse(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES, "stream": True,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
            content_parts = []
            for chunk in chunks:
                delta = chunk.get("choices", [{}])[0].get("delta", {}) or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
            stream_text = "".join(content_parts)

            checker.check("stream-multiple-chunks", len(chunks) > 1, f"chunks={len(chunks)}")
            checker.check("stream-parity", stream_text == non_text,
                          f"stream={stream_text!r} non={non_text!r}")

            pids_after = compute_pids()
            checker.check("single-compute-process", len(pids_after) == 1, repr(pids_after))
            checker.check("compute-pid-stable", pids_before == pids_after,
                          f"{pids_before} -> {pids_after}")

            runs = read_token_log(token_log)
            checker.check("stream-and-nonstream-token-parity",
                          len(runs) >= 2 and runs[0] == runs[1],
                          f"runs={runs}")

            repeat = list(http_sse(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES, "stream": True,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
            repeat_text = "".join(
                (chunk.get("choices", [{}])[0].get("delta", {}) or {}).get("content") or ""
                for chunk in repeat)
            checker.check("repeated-stream-parity", repeat_text == non_text,
                          f"repeat={repeat_text!r} non={non_text!r}")
            checker.check("compute-pid-stable-repeat", compute_pids() == pids_before,
                          repr(compute_pids()))

            checker.check("no-usage-chunk-unless-requested",
                          all("usage" not in chunk for chunk in repeat),
                          repr([chunk for chunk in repeat if "usage" in chunk]))

            usage_chunks = list(http_sse(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES, "stream": True,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
                 "stream_options": {"include_usage": True}}))
            checker.check("usage-chunk-count",
                          sum(1 for chunk in usage_chunks if "usage" in chunk) == 1,
                          repr([chunk for chunk in usage_chunks if "usage" in chunk]))
            usage_chunk = next((chunk for chunk in usage_chunks if "usage" in chunk), None)
            if usage_chunk is None:
                checker.check("usage-chunk-present", False, "no usage chunk")
            else:
                checker.check("usage-chunk-empty-choices",
                              usage_chunk.get("choices") == [],
                              repr(usage_chunk))
                usage = usage_chunk["usage"]
                checker.check("usage-prompt-tokens",
                              isinstance(usage.get("prompt_tokens"), int)
                              and usage["prompt_tokens"] > 0, repr(usage))
                checker.check("usage-completion-tokens",
                              isinstance(usage.get("completion_tokens"), int)
                              and usage["completion_tokens"] > 0, repr(usage))
                checker.check("usage-total-tokens",
                              usage.get("total_tokens")
                              == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                              repr(usage))
                checker.check("usage-prompt-parity",
                              usage.get("prompt_tokens")
                              == non_stream["usage"]["prompt_tokens"],
                              f"stream={usage.get('prompt_tokens')} "
                              f"non={non_stream['usage']['prompt_tokens']}")
                checker.check("usage-completion-parity",
                              usage.get("completion_tokens")
                              == non_stream["usage"]["completion_tokens"],
                              f"stream={usage.get('completion_tokens')} "
                              f"non={non_stream['usage']['completion_tokens']}")
                checker.check("usage-details-present",
                              isinstance(usage.get("prompt_tokens_details"), dict)
                              and isinstance(
                                  usage["prompt_tokens_details"].get("cached_tokens"), int),
                              repr(usage))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
