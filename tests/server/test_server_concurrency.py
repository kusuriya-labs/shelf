#!/usr/bin/env python3
"""Gate 7B: end-to-end concurrent serving acceptance."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time
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
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}


def chat_payload(content, max_tokens=16, tools=None, stream=False):
    payload = {"model": "shelf",
               "messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        payload["tools"] = tools
    if stream:
        payload["stream"] = True
    return payload


def collect_chat_stream(server, content, max_tokens=16, tools=None):
    text = ""
    chunks = 0
    specs = []
    for event in http_sse(f"{server.base_url}/chat/completions",
                          chat_payload(content, max_tokens, tools, stream=True)):
        chunks += 1
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                text += delta["content"]
            for call in delta.get("tool_calls") or []:
                specs.append(call)
    return chunks, text, specs


def run_threads(fns):
    barrier = threading.Barrier(len(fns))
    results = [None] * len(fns)
    errors = [None] * len(fns)

    def wrap(index, fn):
        try:
            barrier.wait(timeout=60)
            results[index] = fn()
        except Exception as exc:  # noqa: BLE001
            errors[index] = exc

    threads = [threading.Thread(target=wrap, args=(i, fn))
               for i, fn in enumerate(fns)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    return results, errors


def main() -> int:
    checker = Checker("server-concurrency")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_concurrent_requests=4, max_seq_len=1024) as server:
            warm = http_json(f"{server.base_url}/chat/completions",
                             chat_payload("Hello", max_tokens=8))
            checker.check("warmup", bool(warm["choices"][0]["message"]["content"].strip()))
            pids_before = compute_pids()

            # 1. Concurrent non-stream Chat Completions.
            prompts = ["Reply with the single word: alpha",
                       "Reply with the single word: beta",
                       "Reply with the single word: gamma",
                       "Reply with the single word: delta"]

            def nonstream(content):
                response = http_json(f"{server.base_url}/chat/completions",
                                     chat_payload(content, max_tokens=16))
                message = response["choices"][0]["message"]
                return message.get("content", "")

            results, errors = run_threads(
                [lambda c=c: nonstream(c) for c in prompts])
            checker.check("chat-concurrent-no-errors",
                          all(e is None for e in errors), repr(errors))
            checker.check("chat-concurrent-content",
                          all(r and r.strip() for r in results), repr(results))

            # 2. Concurrent streaming Chat Completions.
            stream_prompts = prompts[:3]
            results, errors = run_threads(
                [lambda c=c: collect_chat_stream(server, c, max_tokens=16)
                 for c in stream_prompts])
            checker.check("chat-stream-concurrent-no-errors",
                          all(e is None for e in errors), repr(errors))
            checker.check("chat-stream-concurrent-content",
                          all(r[0] >= 1 and r[1].strip() for r in results if r),
                          repr(results))

            # 3. Mixed stream / non-stream Chat Completions.
            def mixed_chat_a():
                return nonstream("Reply with the single word: one")

            def mixed_chat_b():
                return collect_chat_stream(server, "Reply with the single word: two")

            def mixed_chat_c():
                return nonstream("Reply with the single word: three")

            def mixed_chat_d():
                return collect_chat_stream(server, "Reply with the single word: four")

            results, errors = run_threads(
                [mixed_chat_a, mixed_chat_b, mixed_chat_c, mixed_chat_d])
            checker.check("mixed-no-errors", all(e is None for e in errors), repr(errors))
            checker.check("mixed-chat-nonstream",
                          bool(results[0] and results[0].strip()),
                          repr(results[0]))
            checker.check("mixed-chat-stream",
                          bool(results[1] and results[1][0] >= 1 and results[1][1].strip()),
                          repr(results[1]))
            checker.check("mixed-chat-nonstream-b",
                          bool(results[2] and results[2].strip()),
                          repr(results[2]))
            checker.check("mixed-chat-stream-b",
                          bool(results[3] and results[3][0] >= 1
                               and results[3][1].strip()),
                          repr(results[3]))

            # 4. Concurrent tool call isolation.
            def plain_with_tools():
                response = http_json(f"{server.base_url}/chat/completions",
                                     chat_payload("Reply with the single word: plain",
                                                  max_tokens=32, tools=[WEATHER_TOOL]))
                return response["choices"][0]["message"]

            def tool_call():
                response = http_json(f"{server.base_url}/chat/completions",
                                     chat_payload(
                                         "大阪の現在の気温をget_weatherを使って確認して",
                                         max_tokens=128, tools=[WEATHER_TOOL]))
                return response["choices"][0]["message"]

            results, errors = run_threads([plain_with_tools, tool_call])
            checker.check("tool-concurrent-no-errors",
                          all(e is None for e in errors), repr(errors))
            plain_message, tool_message = results
            plain_content = (plain_message or {}).get("content", "")
            checker.check("tool-concurrent-plain-no-calls",
                          bool(plain_content.strip())
                          and not (plain_message or {}).get("tool_calls"),
                          repr(plain_message))
            checker.check("tool-concurrent-plain-no-markup",
                          "<tool_call>" not in plain_content, repr(plain_content))
            calls = (tool_message or {}).get("tool_calls") or []
            checker.check("tool-concurrent-call-present", len(calls) == 1,
                          repr(tool_message))
            if calls:
                arguments = calls[0].get("function", {}).get("arguments", "")
                checker.check("tool-concurrent-arguments",
                              "city" in arguments, repr(arguments))
                checker.check("tool-concurrent-id",
                              bool(calls[0].get("id")), repr(calls[0]))
            checker.check("tool-concurrent-no-markup-in-arguments",
                          "<tool_call>" not in str(tool_message), repr(tool_message))

            # 5. Streaming disconnect isolation: cancel A, B continues.
            combined = {}

            def stream_b():
                combined["b"] = collect_chat_stream(
                    server, "Count from one to fifty slowly.", max_tokens=400)

            thread_b = threading.Thread(target=stream_b)
            thread_b.start()

            connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=120)
            connection.request(
                "POST", "/v1/chat/completions",
                body=json.dumps(chat_payload("Count from one to fifty slowly.",
                                             max_tokens=400, stream=True)),
                headers={"Content-Type": "application/json",
                         "Accept": "text/event-stream"})
            response = connection.getresponse()
            first = response.readline()
            time.sleep(0.2)
            connection.close()
            checker.check("disconnect-a-started", b"data:" in first, repr(first))

            thread_b.join(timeout=300)
            checker.check("disconnect-b-survived",
                          "b" in combined and combined["b"][0] >= 1
                          and combined["b"][1].strip(),
                          repr(combined.get("b")))

            checker.check("compute-process-single", len(compute_pids()) == 1,
                          f"pids={compute_pids()}")
            checker.check("compute-pid-stable", compute_pids() == pids_before,
                          f"{pids_before} -> {compute_pids()}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
