#!/usr/bin/env python3
"""Gate 7A: HTTP streaming disconnect cancels the in-flight generation."""

from __future__ import annotations

import http.client
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    compute_pids,
    http_json,
    model_dir,
)

LONG_MESSAGE = {"role": "user", "content": "Count from 1 to 500 slowly, one number per line."}
STREAM_BUDGET = 256


def open_stream_and_disconnect(port: int, path: str, payload: dict) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    connection.request(
        "POST", path, body=json.dumps(payload),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    response = connection.getresponse()
    first = response.readline().decode("utf-8", errors="replace")
    connection.close()
    return first


def main() -> int:
    checker = Checker("server-stream-cancel")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness() as server:
            # Warm up (loads the model once).
            warm = http_json(f"{server.base_url}/chat/completions", {
                "model": "shelf",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0, "max_tokens": 8,
                 "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("warmup", bool(warm["choices"][0]["message"]["content"].strip()))
            pids_before = compute_pids()

            # Chat Completions streaming disconnect.
            first = open_stream_and_disconnect(
                server.port, "/v1/chat/completions", {
                    "model": "shelf", "messages": [LONG_MESSAGE],
                    "stream": True, "temperature": 0, "max_tokens": STREAM_BUDGET,
                    "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("chat-disconnect-started", "data:" in first, repr(first[:120]))

            started = time.monotonic()
            recovered = http_json(f"{server.base_url}/chat/completions", {
                "model": "shelf",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0, "max_tokens": 16,
                 "chat_template_kwargs": {"enable_thinking": False}})
            chat_latency = time.monotonic() - started
            checker.check("chat-after-disconnect",
                          bool(recovered["choices"][0]["message"]["content"].strip()),
                          repr(recovered))
            checker.check("chat-after-disconnect-latency",
                          chat_latency < 120.0,
                          f"latency={chat_latency:.1f}s (cancelled if well below full generation)")

            first = open_stream_and_disconnect(
                server.port, "/v1/responses", {
                    "model": "shelf", "input": LONG_MESSAGE["content"],
                    "stream": True, "temperature": 0, "max_output_tokens": STREAM_BUDGET})
            checker.check("responses-disconnect-started", "event: response.created" in first,
                          repr(first[:120]))

            recovered = http_json(f"{server.base_url}/chat/completions", {
                "model": "shelf", "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0, "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False}})
            checker.check("responses-after-disconnect",
                          bool(recovered["choices"][0]["message"]["content"].strip()),
                          repr(recovered))

            checker.check("compute-pid-stable", compute_pids() == pids_before,
                          f"{pids_before} -> {compute_pids()}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
