#!/usr/bin/env python3
"""Gate 3 acceptance: tool calling and tool-result round-trip."""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json, model_dir  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    checker = Checker("server-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness() as server:
            # Case 1: no tools.
            plain = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "phaseshift",
                 "messages": [{"role": "user", "content": "Hello"}],
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
            plain_content = plain["choices"][0]["message"]["content"]
            checker.check("case1-no-tools-content", bool(plain_content.strip()),
                          repr(plain_content))
            checker.check("case1-no-tool-calls",
                          not plain["choices"][0]["message"].get("tool_calls"))
            checker.check("case7-no-markup-leak", "<tool_call>" not in plain_content,
                          repr(plain_content))

            # Case 2: tools present, no tool needed.
            no_call = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "phaseshift",
                 "messages": [{"role": "user", "content": "Reply with the single word: hello"},
                              ],
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
                 "tools": [WEATHER_TOOL], "tool_choice": "auto"})
            no_call_choice = no_call["choices"][0]
            no_call_content = no_call_choice["message"]["content"]
            checker.check("case2-content", bool(no_call_content.strip()), repr(no_call_content))
            checker.check("case7-no-markup-leak-2", "<tool_call>" not in no_call_content,
                          repr(no_call_content))

            # Case 3/4: single tool call.
            tool_result = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "phaseshift",
                 "messages": [{"role": "user",
                               "content": "大阪の現在の気温をget_weatherを使って確認して"}],
                 "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False},
                 "tools": [WEATHER_TOOL], "tool_choice": "auto"})
            tool_choice = tool_result["choices"][0]
            message = tool_choice["message"]
            calls = message.get("tool_calls") or []
            checker.check("case3-tool-call-present", len(calls) == 1, repr(message))
            checker.check("case3-finish-reason",
                          tool_choice.get("finish_reason") == "tool_calls",
                          repr(tool_choice.get("finish_reason")))
            checker.check("case7-no-markup-in-tool-content",
                          "<tool_call>" not in (message.get("content") or ""),
                          repr(message.get("content")))
            if calls:
                call = calls[0]
                checker.check("case3-function-name",
                              call["function"]["name"] == "get_weather", repr(call))
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    checker.check("case4-arguments-json-object",
                                  isinstance(arguments, dict) and "city" in arguments,
                                  repr(call["function"]["arguments"]))
                except json.JSONDecodeError as exc:
                    checker.check("case4-arguments-json-object", False, repr(exc))
                call_id = call.get("id", "call_0")

                # Case 5: tool result round-trip.
                round_trip = http_json(
                    f"{server.base_url}/chat/completions",
                    {"model": "phaseshift",
                     "messages": [
                         {"role": "user",
                          "content": "大阪の現在の気温をget_weatherを使って確認して"},
                         {"role": "assistant", "content": "",
                          "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": call_id,
                          "name": "get_weather",
                          "content": json.dumps({"temperature": 22, "unit": "celsius"})},
                     ],
                     "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
                     "tools": [WEATHER_TOOL]})
                final_choice = round_trip["choices"][0]
                final_content = final_choice["message"]["content"]
                checker.check("case5-final-content", bool(final_content.strip()),
                              repr(final_content))
                checker.check("case5-final-finish-stop",
                              final_choice.get("finish_reason") == "stop",
                              repr(final_choice.get("finish_reason")))
                checker.check("case7-no-markup-leak-final", "<tool_call>" not in final_content,
                              repr(final_content))

            # Case 6: malformed tool history must not crash the server.
            bad_status = None
            bad_message = ""
            try:
                http_json(
                    f"{server.base_url}/chat/completions",
                    {"model": "phaseshift",
                     "messages": [
                         {"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "",
                          "tool_calls": [{"id": "call_x", "type": "function",
                                          "function": {"name": "get_weather",
                                                       "arguments": "not json"}}]},
                     ],
                     "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False},
                     "tools": [WEATHER_TOOL]})
            except urllib.error.HTTPError as exc:
                bad_status = exc.code
                bad_message = exc.read().decode("utf-8", errors="replace")
            checker.check("case6-malformed-history-error",
                          bad_status in (400, 422, 500),
                          f"status={bad_status} body={bad_message[:200]}")
            checker.check("case6-malformed-history-message",
                          "arguments" in bad_message or "JSON" in bad_message,
                          bad_message[:200])

            recovered = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "phaseshift",
                 "messages": [{"role": "user", "content": "Hello"}],
                 "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("case6-server-recovers",
                          bool(recovered["choices"][0]["message"]["content"].strip()),
                          repr(recovered))

            # OpenAI Python client E2E.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="unused")
                completion = client.chat.completions.create(
                    model="phaseshift",
                    messages=[{"role": "user",
                               "content": "大阪の現在の気温をget_weatherを使って確認して"}],
                    tools=[WEATHER_TOOL],
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=128,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                client_calls = completion.choices[0].message.tool_calls or []
                checker.check("openai-client-tool-calls", len(client_calls) == 1,
                              repr(completion.choices[0].message))
                if client_calls:
                    checker.check("openai-client-function-name",
                                  client_calls[0].function.name == "get_weather",
                                  repr(client_calls[0]))
                    parsed = json.loads(client_calls[0].function.arguments)
                    checker.check("openai-client-arguments",
                                  isinstance(parsed, dict) and "city" in parsed,
                                  repr(client_calls[0].function.arguments))
            except ImportError as exc:
                checker.check("openai-client-tool-calls", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
