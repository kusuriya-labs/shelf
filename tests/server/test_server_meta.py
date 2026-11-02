#!/usr/bin/env python3
"""GET /v1/meta の server メタ情報と in-flight 状態の契約."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_get_json,
    http_sse,
    model_dir,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def poll_meta(base_url: str, predicate, timeout: float = 10.0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        meta = http_get_json(f"{base_url}/meta", timeout=5.0)
        if predicate(meta):
            return meta
        time.sleep(0.01)
    return None


def main() -> int:
    checker = Checker("server-meta")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_concurrent_requests=1) as server:
            meta = http_get_json(f"{server.base_url}/meta")
            checker.check("model", meta.get("model") == "shelf", repr(meta))
            checker.check("max-seq-len",
                          meta.get("max_seq_len") == server.max_seq_len, repr(meta))
            checker.check("max-concurrent-requests",
                          meta.get("max_concurrent_requests") == 1, repr(meta))
            checker.check("kv-cache-dtype",
                          meta.get("kv_cache_dtype") == "bf16", repr(meta))
            checker.check("arena-gib",
                          meta.get("arena_gib") == server.arena_gib, repr(meta))
            checker.check("prefix-cache-is-bool",
                          isinstance(meta.get("prefix_cache"), bool), repr(meta))
            checker.check("speculative-none",
                          meta.get("speculative") is None, repr(meta))
            checker.check("idle-inflight",
                          meta.get("inflight") == 0, repr(meta))
            checker.check("idle-queued", meta.get("queued") == 0, repr(meta))

            idle_kv = meta.get("kv_cache")
            checker.check("kv-cache-present", isinstance(idle_kv, dict), repr(meta))
            if isinstance(idle_kv, dict):
                checker.check("kv-page-geometry",
                              idle_kv.get("page_tokens", 0) > 0
                              and idle_kv.get("total_pages", 0) > 0
                              and idle_kv.get("capacity_tokens")
                              == idle_kv.get("total_pages", 0)
                              * idle_kv.get("page_tokens", 0),
                              repr(idle_kv))
                checker.check("kv-idle-free",
                              idle_kv.get("used_pages") == 0
                              and idle_kv.get("free_pages") == idle_kv.get("total_pages")
                              and idle_kv.get("free_tokens")
                              == idle_kv.get("free_pages", 0)
                              * idle_kv.get("page_tokens", 0),
                              repr(idle_kv))
                checker.check("kv-idle-requests",
                              idle_kv.get("active_requests") == 0
                              and idle_kv.get("queued_requests") == 0
                              and idle_kv.get("banker_safe") is True,
                              repr(idle_kv))
            else:
                checker.check("kv-page-geometry", False, repr(meta))
                checker.check("kv-idle-free", False, repr(meta))
                checker.check("kv-idle-requests", False, repr(meta))

            long_run = threading.Thread(
                target=lambda: list(http_sse(
                    f"{server.base_url}/chat/completions",
                    {"model": "shelf", "messages": MESSAGES, "stream": True,
                     "temperature": 0, "max_tokens": 384, "chat_template_kwargs": {"enable_thinking": False}})),
                daemon=True)
            long_run.start()

            busy = poll_meta(
                server.base_url,
                lambda m: m.get("inflight") == 1 and m.get("queued") == 0,
                timeout=30.0)
            checker.check("busy-inflight", busy is not None,
                          f"inflight did not reach 1 within timeout")

            waiter = threading.Thread(
                target=lambda: list(http_sse(
                    f"{server.base_url}/chat/completions",
                    {"model": "shelf", "messages": MESSAGES, "stream": True,
                     "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}})),
                daemon=True)
            waiter.start()

            queued_meta = poll_meta(
                server.base_url,
                lambda m: m.get("inflight") == 1 and m.get("queued") == 1,
                timeout=30.0)
            checker.check("queued-while-full", queued_meta is not None,
                          "second request did not queue within timeout")

            long_run.join(timeout=300)
            waiter.join(timeout=300)
            checker.check("streams-finished",
                          not long_run.is_alive() and not waiter.is_alive(),
                          f"long={long_run.is_alive()} waiter={waiter.is_alive()}")

            final = http_get_json(f"{server.base_url}/meta")
            checker.check("drained-inflight", final.get("inflight") == 0, repr(final))
            checker.check("drained-queued", final.get("queued") == 0, repr(final))

            status_body = None
            try:
                status_body = http_get_json(f"{server.base_url}/meta/missing")
            except Exception as exc:  # noqa: BLE001
                status_body = repr(exc)
            checker.check("unknown-path-not-meta", status_body != meta, repr(status_body))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
