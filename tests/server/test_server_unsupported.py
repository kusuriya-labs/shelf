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
    ensure_chat_import,
)

ensure_chat_import()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "apps" / "server"))
from serverlib.openai_protocol import OpenAIProtocolError, parse_chat_request  # noqa: E402

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
    ("store-true", {"store": True}, "store"),
    ("frequency-penalty", {"frequency_penalty": 1}, "frequency_penalty"),
    ("presence-penalty", {"presence_penalty": -1}, "presence_penalty"),
    *[(key, {key: value}, key) for key, value in [
        ("logit_bias", {}), ("logprobs", True), ("top_logprobs", 2),
        ("prediction", {}), ("modalities", ["audio"]), ("audio", {}),
        ("web_search_options", {}), ("unknown_parameter", 1)]],
    ("response-format-thinking", {"response_format": {"type": "json_object"}},
     "response_format"),
    ("response-format-json-schema-thinking",
     {"response_format": {"type": "json_schema",
                          "json_schema": {"name": "probe", "strict": True,
                                          "schema": CITY_SCHEMA}}},
     "response_format"),
    ("response-format-unsupported-schema",
     {"chat_template_kwargs": {"enable_thinking": False},
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "probe", "schema": {"type": "string", "pattern": "a+"}}}},
     "response_format.json_schema.schema"),
    ("response-format-conflicting-enum",
     {"chat_template_kwargs": {"enable_thinking": False},
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "probe", "schema": {"type": "integer", "enum": [1, "bad"]}}}},
     "response_format.json_schema.schema"),
    ("reasoning_effort", {"reasoning_effort": "high"}, "reasoning_effort"),
    ("reasoning", {"reasoning": {"effort": "high"}}, "reasoning"),
    ("tool-choice-required-without-tools", {"tool_choice": "required"}, "tool_choice"),
    ("tool-choice-unknown-name",
     {"tool_choice": {"type": "function", "function": {"name": "missing"}},
      "tools": [PLAIN_TOOL]}, "tool_choice"),
    ("strict-tool", {"tools": [STRICT_TOOL]}, "tools[0].function.strict"),
    ("invalid-strict", {"tools": [{"type": "function", "function": {"name": "f", "strict": 0}}]}, "tools[0].function.strict"),
    ("parallel-tool-calls-false", {"parallel_tool_calls": False},
     "parallel_tool_calls"),
    ("stream-options-unknown", {"stream_options": {"extra": True}},
     "stream_options.extra"),
]


def unit_main() -> int:
    checker = Checker("server-unsupported-unit")
    base = {"model": "shelf", "messages": MESSAGES}
    for label, extra, param in REJECTED:
        try:
            parse_chat_request({**base, **extra})
        except OpenAIProtocolError as exc:
            checker.check(label, exc.param == param and exc.code == "unsupported_parameter",
                          repr((exc.param, exc.code)))
        else:
            checker.check(label, False, "accepted")
    for key, value in [("user", "u"), ("store", False), ("metadata", {"x": "y"}),
                       ("service_tier", "auto"), ("parallel_tool_calls", True),
                       ("frequency_penalty", 0), ("presence_penalty", 0)]:
        try:
            parse_chat_request({**base, key: value})
            checker.check(f"accept-{key}", True)
        except OpenAIProtocolError as exc:
            checker.check(f"accept-{key}", False, str(exc))
    return checker.done()


def main() -> int:
    checker = Checker("server-unsupported")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=512) as server:
            url = f"{server.base_url}/chat/completions"

            for label, extra, param in REJECTED:
                payload = {"model": "shelf", "messages": MESSAGES,
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
                "model": "shelf", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 16})
            checker.check("conflicting-max-tokens-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "shelf", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 0, "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("zero-max-tokens-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "shelf", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}, "n": 2})
            checker.check("n-gt-one-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "shelf", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": "yes"})
            checker.check("stream-options-not-object-400", status == 400,
                          f"status={status} body={body[:300]}")

            status, body = http_post_status(url, {
                "model": "shelf", "messages": MESSAGES,
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": "yes"}})
            checker.check("stream-options-bad-flag-400", status == 400,
                          f"status={status} body={body[:300]}")

            result = http_json(url, {
                "model": "shelf", "messages": MESSAGES,
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
    sys.exit(unit_main() if "--unit" in sys.argv else main())
