#!/usr/bin/env python3
"""Reasoning (thinking) acceptance: tool calling with thinking enabled.

The thinking region is never fed to the tool parser, so a tool call is only
recognised in the answer region, and neither region leaks chat markup.
"""

from __future__ import annotations

import json
import sys
import tempfile
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
TOOL_CALL_TAG = "<tool_" + "call>"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

USER_MESSAGE = "大阪の天気をget_weatherで確認して"
MESSAGES = [{"role": "user", "content": USER_MESSAGE}]


def read_token_log(path: Path) -> list[list[int]]:
    if not path.is_file():
        return []
    return [[int(x) for x in line.split(",") if x]
            for line in path.read_text().splitlines() if line.strip()]


def reconstruct_tool_calls(chunks):
    calls: dict[int, dict] = {}
    finish = None
    reasoning = ""
    content = ""
    mixed = 0
    for chunk in chunks:
        choice = (chunk.get("choices") or [{}])[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content") and delta.get("content"):
            mixed += 1
        reasoning += delta.get("reasoning_content") or ""
        content += delta.get("content") or ""
        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            entry = calls.setdefault(
                index, {"id": None, "name": "", "arguments": ""})
            if call.get("id"):
                entry["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                entry["name"] = function["name"]
            if function.get("arguments"):
                entry["arguments"] += function["arguments"]
    return {
        "calls": [calls[index] for index in sorted(calls)],
        "finish": finish,
        "reasoning": reasoning,
        "content": content,
        "mixed": mixed,
    }


def check_thinking_tool_call(checker, server, processor, token_log):
    result = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "shelf", "messages": MESSAGES,
         "tools": [WEATHER_TOOL], "tool_choice": "auto",
         "temperature": 0, "max_completion_tokens": 768})
    choice = result["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content")

    checker.check("e06-tool-call-present", len(calls) == 1, repr(message)[:400])
    checker.check("e06-finish-tool-calls", choice.get("finish_reason") == "tool_calls",
                  repr(choice.get("finish_reason")))
    checker.check("e06-reasoning-present", bool(reasoning.strip()),
                  repr(reasoning)[:200])
    checker.check("e06-content-null", content is None, repr(content))
    checker.check("e06-no-markup-in-reasoning", TOOL_CALL_TAG not in reasoning,
                  repr(reasoning[-120:]))
    if calls:
        function = calls[0]["function"]
        checker.check("e06-function-name", function["name"] == "get_weather",
                      repr(function))
        try:
            arguments = json.loads(function["arguments"])
            checker.check("e06-arguments-json",
                          isinstance(arguments, dict) and "city" in arguments,
                          repr(function["arguments"]))
        except json.JSONDecodeError as exc:
            checker.check("e06-arguments-json", False, repr(exc))

    runs = read_token_log(token_log)
    if runs:
        split = codec.split_generation(processor, runs[-1], thinking=True)
        checker.check("e06-reasoning-matches-ids", reasoning == split.reasoning,
                      f"wire={reasoning[-40:]!r} ids={split.reasoning[-40:]!r}")
        checker.check("e06-answer-had-tool-markup",
                      TOOL_CALL_TAG in split.answer, repr(split.answer[:80]))
        checker.check("e06-tool-markup-not-in-reasoning",
                      TOOL_CALL_TAG not in split.reasoning, "")

    round_trip = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "shelf",
         "messages": MESSAGES + [
             {"role": "assistant", "content": None, "reasoning_content": reasoning,
              "tool_calls": [
                  {"id": "call_0", "type": "function",
                   "function": {"name": "get_weather",
                                "arguments": "{\"city\":\"大阪\"}"}}]},
             {"role": "tool", "tool_call_id": "call_0", "name": "get_weather",
              "content": json.dumps({"temperature": 22, "unit": "celsius"})},
         ],
         "tools": [WEATHER_TOOL], "tool_choice": "auto",
         "temperature": 0, "max_completion_tokens": 768})
    final = round_trip["choices"][0]
    final_message = final["message"]
    checker.check("e07-roundtrip-finish",
                  final.get("finish_reason") in ("stop", "length"),
                  repr(final.get("finish_reason")))
    checker.check("e07-roundtrip-answer",
                  bool((final_message.get("content") or "").strip()),
                  repr(final_message)[:300])
    checker.check("e07-roundtrip-markup-free",
                  TOOL_CALL_TAG not in (final_message.get("content") or ""), "")
    return reasoning


def check_streaming_tool_call(checker, server, expected_reasoning):
    chunks = list(http_sse(f"{server.base_url}/chat/completions", {
        "model": "shelf", "messages": MESSAGES,
        "tools": [WEATHER_TOOL], "tool_choice": "auto",
        "stream": True, "temperature": 0, "max_completion_tokens": 768}))
    result = reconstruct_tool_calls(chunks)
    checker.check("e06-stream-tool-call", len(result["calls"]) == 1,
                  repr(result["calls"]))
    checker.check("e06-stream-finish", result["finish"] == "tool_calls",
                  repr(result["finish"]))
    checker.check("e06-stream-no-mixed-delta", result["mixed"] == 0,
                  repr(result["mixed"]))
    checker.check("e06-stream-reasoning", bool(result["reasoning"].strip()), "")
    checker.check("e06-stream-no-markup",
                  TOOL_CALL_TAG not in result["reasoning"] + result["content"], "")
    if result["calls"]:
        call = result["calls"][0]
        checker.check("e06-stream-function-name", call["name"] == "get_weather",
                      repr(call["name"]))
        try:
            arguments = json.loads(call["arguments"] or "{}")
            checker.check("e06-stream-arguments",
                          isinstance(arguments, dict) and "city" in arguments,
                          repr(call["arguments"]))
        except json.JSONDecodeError as exc:
            checker.check("e06-stream-arguments", False, repr(exc))
    if expected_reasoning:
        checker.check("e06-stream-reasoning-parity",
                      result["reasoning"] == expected_reasoning,
                      f"stream={result['reasoning'][-40:]!r} "
                      f"non={expected_reasoning[-40:]!r}")


def check_tools_without_thinking(checker, server):
    result = http_json(
        f"{server.base_url}/chat/completions",
        {"model": "shelf", "messages": MESSAGES,
         "tools": [WEATHER_TOOL], "tool_choice": "auto",
         "temperature": 0, "max_completion_tokens": 256,
         "chat_template_kwargs": {"enable_thinking": False}})
    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    checker.check("e03-nothink-tool-call", len(calls) == 1, repr(message)[:300])
    checker.check("e03-nothink-no-reasoning", "reasoning_content" not in message,
                  repr(message)[:200])


def main() -> int:
    checker = Checker("server-reasoning-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    token_log = Path(tempfile.mkdtemp(prefix="ps-reasoning-tools-")) / "tokens.txt"
    try:
        processor = load_processor()
        with ServerHarness(max_seq_len=1024,
                           env={"SHELF_SERVER_TOKEN_LOG": str(token_log)}) as server:
            reasoning = check_thinking_tool_call(
                checker, server, processor, token_log)
            check_streaming_tool_call(checker, server, reasoning)
            check_tools_without_thinking(checker, server)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
