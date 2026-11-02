#!/usr/bin/env python3
"""削除した OpenAI パラメータが silent ignore されず 400 で拒否される."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_json,
    http_post_status,
    model_dir,
)

MESSAGES = [{"role": "user", "content": "hello"}]

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

STRICT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": CITY_SCHEMA,
        "strict": True,
    },
}

PLAIN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": CITY_SCHEMA,
    },
}

REJECTED = [
    ("response_format", {"response_format": {"type": "json_object"}},
     "response_format"),
    ("response_format-json-schema",
     {"response_format": {"type": "json_schema",
                          "json_schema": {"name": "probe", "strict": True,
                                          "schema": CITY_SCHEMA}}},
     "response_format"),
    ("reasoning_effort", {"reasoning_effort": "high"}, "reasoning_effort"),
    ("reasoning", {"reasoning": {"effort": "high"}}, "reasoning"),
    ("tool-choice-required", {"tool_choice": "required", "tools": [PLAIN_TOOL]},
     "tool_choice"),
    ("tool-choice-named",
     {"tool_choice": {"type": "function",
                      "function": {"name": "get_weather"}},
      "tools": [PLAIN_TOOL]},
     "tool_choice"),
    ("strict-tool", {"tools": [STRICT_TOOL]}, "tools[0].function.strict"),
    ("parallel-tool-calls-false", {"parallel_tool_calls": False},
     "parallel_tool_calls"),
    ("stream-options-unknown", {"stream_options": {"extra": True}},
     "stream_options.extra"),
]


def main() -> int:
    checker = Checker("server-unsupported")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=512) as server:
            url = f"{server.base_url}/chat/completions"

            for label, extra, param in REJECTED:
                payload = {"model": "phaseshift", "messages": MESSAGES,
                           "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
                payload.update(extra)
                status, body = http_post_status(url, payload)
                checker.check(f"{label}-400", status == 400,
                              f"status={status} body={body[:300]}")
                if status == 400:
                    error = json.loads(body).get("error", {})
                    checker.check(f"{label}-code",
                                  error.get("code") == "unsupported_parameter",
                                  repr(error))
                    checker.check(f"{label}-param", error.get("param") == param,
                                  repr(error))

            status, body = http_post_status(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 16})
            checker.check("conflicting-max-tokens-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 0, "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("zero-max-tokens-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}, "n": 2})
            checker.check("n-gt-one-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": "yes"})
            checker.check("stream-options-not-object-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": "yes"}})
            checker.check("stream-options-bad-flag-400", status == 400,
                          f"status={status} body={body[:300]}")

            result = http_json(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "tools": [PLAIN_TOOL], "tool_choice": "auto"})
            checker.check("accepted-baseline", result["choices"][0]["finish_reason"]
                          in ("stop", "length", "tool_calls"), repr(result))

            request = urllib.request.Request(
                url, data=b"not json",
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw_status, raw_body = response.status, response.read().decode()
            except urllib.error.HTTPError as exc:
                raw_status, raw_body = exc.code, exc.read().decode()
            checker.check("malformed-json-400", raw_status == 400,
                          f"status={raw_status} body={raw_body[:300]}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
