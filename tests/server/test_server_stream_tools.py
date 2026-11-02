#!/usr/bin/env python3
"""Gate 4 acceptance: streaming tool calls through Chat Completions."""

from __future__ import annotations

import json
import sys
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

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
USER_MESSAGE = "大阪の天気をget_weatherで確認して"
MESSAGES = [{"role": "user", "content": USER_MESSAGE}]


def reconstruct_tool_calls(chunks):
    calls: dict[int, dict] = {}
    finish = None
    raw_text = ""
    for chunk in chunks:
        choice = chunk.get("choices", [{}])[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        for tc in delta.get("tool_calls") or []:
            index = tc.get("index", 0)
            entry = calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
            if tc.get("id"):
                entry["id"] = tc["id"]
            function = tc.get("function") or {}
            if function.get("name"):
                entry["name"] = function["name"]
            if function.get("arguments"):
                entry["arguments"] += function["arguments"]
        if delta.get("content"):
            raw_text += delta["content"]
        raw_text += json.dumps(delta)
    ordered = [calls[i] for i in sorted(calls)]
    return ordered, finish, raw_text


def main() -> int:
    checker = Checker("server-stream-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness() as server:
            url = f"{server.base_url}/chat/completions"
            chunks = list(http_sse(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "tools": [WEATHER_TOOL], "tool_choice": "auto",
                "stream": True, "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}))
            calls, finish, raw = reconstruct_tool_calls(chunks)
            pids_before = compute_pids()

            checker.check("case1-delta-tool-calls", len(calls) == 1, repr(calls))
            if calls:
                checker.check("case2-name", calls[0]["name"] == "get_weather", repr(calls[0]))
                try:
                    args = json.loads(calls[0]["arguments"])
                    checker.check("case3-arguments-json",
                                  isinstance(args, dict) and "city" in args,
                                  repr(calls[0]["arguments"]))
                except json.JSONDecodeError as exc:
                    checker.check("case3-arguments-json", False, repr(exc))
            checker.check("case5-finish-reason", finish == "tool_calls", repr(finish))
            checker.check("case6-no-tool-markup",
                          "<tool_call>" not in raw and "<function=" not in raw
                          and "<parameter=" not in raw,
                          raw[:300])

            non = http_json(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "tools": [WEATHER_TOOL], "tool_choice": "auto",
                "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}})
            non_calls = non["choices"][0]["message"].get("tool_calls") or []
            checker.check("case4-nonstream-count", len(non_calls) == len(calls),
                          f"stream={len(calls)} non={len(non_calls)}")
            if non_calls and calls:
                non_fn = non_calls[0]["function"]
                stream_args = json.loads(calls[0]["arguments"] or "{}")
                non_args = json.loads(non_fn["arguments"] or "{}")
                checker.check("case4-parity-name", non_fn["name"] == calls[0]["name"],
                              f"{non_fn['name']} vs {calls[0]['name']}")
                checker.check("case4-parity-args", non_args == stream_args,
                              f"{non_args} vs {stream_args}")

            pids_after = compute_pids()
            checker.check("case8-compute-pid-stable", pids_before == pids_after,
                          f"{pids_before} -> {pids_after}")

            repeat = list(http_sse(url, {
                "model": "phaseshift", "messages": MESSAGES,
                "tools": [WEATHER_TOOL], "tool_choice": "auto",
                "stream": True, "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}))
            repeat_calls, _, _ = reconstruct_tool_calls(repeat)
            checker.check("case8-repeat-tool-call",
                          len(repeat_calls) == 1
                          and repeat_calls[0]["name"] == "get_weather",
                          repr(repeat_calls))
            checker.check("case8-compute-pid-stable-repeat",
                          compute_pids() == pids_before, repr(compute_pids()))

            # Case 9: the Gate 2 no-tools streaming path is unchanged.
            plain_chunks = list(http_sse(url, {
                "model": "phaseshift", "messages": [{"role": "user", "content": "Hello"}],
                "stream": True, "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}}))
            plain_text = "".join(
                (c.get("choices", [{}])[0].get("delta", {}) or {}).get("content") or ""
                for c in plain_chunks)
            plain = http_json(url, {
                "model": "phaseshift", "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("case9-plain-stream-parity",
                          plain_text == plain["choices"][0]["message"]["content"],
                          f"stream={plain_text!r}")

            # Case 7: OpenAI Python client streaming.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="unused")
                stream = client.chat.completions.create(
                    model="phaseshift", messages=MESSAGES, tools=[WEATHER_TOOL],
                    tool_choice="auto", temperature=0, max_tokens=128, stream=True,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                acc: dict[int, dict] = {}
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    for tc in delta.tool_calls or []:
                        entry = acc.setdefault(tc.index, {"name": "", "arguments": ""})
                        if tc.function and tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            entry["arguments"] += tc.function.arguments
                checker.check("case7-openai-client-stream-tool-call",
                              len(acc) == 1 and acc[0]["name"] == "get_weather",
                              repr(acc))
                if acc:
                    parsed = json.loads(acc[0]["arguments"])
                    checker.check("case7-openai-client-arguments",
                                  isinstance(parsed, dict) and "city" in parsed,
                                  repr(acc[0]["arguments"]))
            except ImportError as exc:
                checker.check("case7-openai-client-stream-tool-call", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
