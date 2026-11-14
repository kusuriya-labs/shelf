#!/usr/bin/env python3
"""DFlash2 構成に対する shelf-server の E2E 検証.

通常の DFlash2 生成（chat / stream / tools / stochastic sampling）が成立し、
structured output が通常 decode に切り替わって動作することを確認する.
"""

from __future__ import annotations

import json
import importlib.util
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (Checker, ServerHarness, http_get_json,  # noqa: E402
                     http_json, http_post_status, model_dir)

DFLASH2_MODEL_DIR = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
KV_DTYPE = os.environ.get("SHELF_TEST_KV_DTYPE", "fp8")


def arena_used_values(log_text: str) -> list[int]:
    out = []
    for line in log_text.splitlines():
        marker = "arena_used="
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].split()[0]
        if tail.isdigit():
            out.append(int(tail))
    return out

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def main() -> int:
    checker = Checker("server-dflash2")
    harness = ServerHarness(max_seq_len=512, arena_gib=31, device=1,
                            max_concurrent_requests=1,
                            kv_cache_dtype=KV_DTYPE,
                            dflash2_model_dir=DFLASH2_MODEL_DIR,
                            env={"SHELF_BATCH_TRACE": "1"},
                            startup_timeout=420.0)
    try:
        harness.start()

        log_text = ""
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            log_text = harness.log_path.read_text(errors="replace")
            if "Ready." in log_text or "failed" in log_text:
                break
            time.sleep(0.5)
        checker.check("server reached Ready", "Ready." in log_text, log_text[-500:])
        checker.check("log reports DFlash2",
                      "Speculative:        DFlash2" in log_text, log_text[-800:])
        checker.check("log reports kv dtype",
                      f"KV dtype:           {KV_DTYPE}" in log_text, log_text[-800:])
        checker.check("log reports prefix cache capability",
                      "capabilities.prefix_cache=" in log_text, log_text[-800:])
        checker.check("log concurrency is 1",
                      "Concurrency:        1" in log_text, log_text[-800:])
        checker.check("compute reports dflash2 enabled",
                      "DFLASH2_ENABLED=1" in log_text, log_text[-800:])

        models = http_get_json(f"{harness.base_url}/models")
        checker.check("model listed",
                      any(m.get("id") == "shelf" for m in models.get("data", [])))

        chat = http_json(f"{harness.base_url}/chat/completions", {
            "model": "shelf",
            "messages": [{"role": "user", "content": "Say exactly: hello"}],
            "temperature": 0,
            "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
        })
        content = chat["choices"][0]["message"]["content"]
        checker.check("chat returns content", "hello" in content, content)
        checker.check("chat finish_reason",
                      chat["choices"][0]["finish_reason"] in ("stop", "length"),
                      str(chat["choices"][0]["finish_reason"]))

        stochastic = http_json(f"{harness.base_url}/chat/completions", {
            "model": "shelf",
            "messages": [{"role": "user", "content": "Say exactly: hello"}],
            "temperature": 0.7, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
        })
        checker.check("temperature > 0 accepted",
                      bool(stochastic["choices"][0]["message"]["content"]),
                      str(stochastic["choices"][0]["message"])[:200])

        status, body = http_post_status(
            f"{harness.base_url}/chat/completions",
            {"model": "shelf",
             "messages": [{"role": "user", "content": "Produce the JSON value."}],
             "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
             "response_format": {"type": "json_object"}})
        if importlib.util.find_spec("xgrammar") is None:
            checker.check("structured requires xgrammar", status == 400 and
                          json.loads(body)["error"]["code"] == "unsupported_parameter", body[:300])
        else:
            checker.check("structured accepted", status == 200, f"{status} {body[:300]}")
        if status == 200:
            content = json.loads(body)["choices"][0]["message"]["content"]
            checker.check("structured object", isinstance(json.loads(content), dict), content)

        tools = http_json(f"{harness.base_url}/chat/completions", {
            "model": "shelf",
            "messages": [{"role": "user",
                          "content": "大阪の現在の気温をget_weatherで確認して"}],
            "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}, "tools": TOOLS,
        })
        tool_calls = tools["choices"][0]["message"].get("tool_calls") or []
        checker.check("tools accepted", bool(tool_calls),
                      str(tools["choices"][0]["message"])[:300])
        if tool_calls:
            checker.check("tool call name",
                          tool_calls[0]["function"]["name"] == "get_weather",
                          str(tool_calls[0])[:300])
            checker.check("tool arguments are a json string",
                          isinstance(tool_calls[0]["function"]["arguments"], str),
                          str(tool_calls[0])[:300])
            tool_content = tools["choices"][0]["message"].get("content") or ""
            checker.check("tools content has no tool markup",
                          "<tool_call>" not in tool_content, repr(tool_content[:300]))

        status, body = http_post_status(
            f"{harness.base_url}/responses",
            {"model": "shelf", "input": "Say exactly: ok",
             "text": {"format": {"type": "json_schema"}}})
        checker.check("responses structured rejected", status == 400, f"{status} {body[:300]}")

        request = urllib.request.Request(
            f"{harness.base_url}/chat/completions",
            data=json.dumps({
                "model": "shelf",
                "messages": [{"role": "user", "content": "stream test"}],
                "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        chunks = []
        finish = None
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                payload = decoded[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                delta = event["choices"][0]["delta"].get("content")
                if delta:
                    chunks.append(delta)
                if event["choices"][0].get("finish_reason"):
                    finish = event["choices"][0]["finish_reason"]
        checker.check("stream returns content", len(chunks) > 0, str(chunks))
        checker.check("stream finish reason", finish == "stop", str(finish))

        used = arena_used_values(harness.log_path.read_text(errors="replace"))
        checker.check("arena trace captured", len(used) > 0, f"{len(used)} steps")
        if used:
            checker.check("arena does not grow per request", max(used) == used[0],
                          f"first={used[0]} max={max(used)}")
    finally:
        harness.close()

    return checker.done()


if __name__ == "__main__":
    if not DFLASH2_MODEL_DIR:
        print("SKIP: SHELF_DFLASH2_MODEL_DIR is not set")
        raise SystemExit(77)
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        raise SystemExit(77)
    raise SystemExit(main())
