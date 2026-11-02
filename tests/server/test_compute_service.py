#!/usr/bin/env python3
"""shelf-compute --serve-stdio protocol acceptance.

Covers the Gate 1 JSONL contract and the Gate 7A asynchronous control plane:
generate-time ping / cancel / shutdown, request-specific cancellation, and
reuse after cancel.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness  # noqa: E402

PROMPT = [248041, 77091]


def collect(harness, request_id, *, ping_after=None, cancel_after=None,
            stats_after=None, stale_ids=(), expect_error=False):
    tokens = []
    pong = False
    done = None
    errors = []
    stale = []
    stats = []
    while True:
        event = harness.recv()
        kind = event.get("event")
        event_id = event.get("request_id")
        if event_id in stale_ids:
            stale.append(event)
        if kind == "token":
            tokens.append(event["token_id"])
        elif kind == "pong":
            pong = True
        elif kind == "stats":
            stats.append(event)
        elif kind == "error":
            errors.append(event)
            return {"tokens": tokens, "pong": pong, "done": done,
                    "errors": errors, "stale": stale, "stats": stats}
        elif kind == "done":
            done = event
            return {"tokens": tokens, "pong": pong, "done": done,
                    "errors": errors, "stale": stale, "stats": stats}
        if ping_after is not None and len(tokens) == ping_after:
            harness.send({"op": "ping"})
            ping_after = None
        if stats_after is not None and len(tokens) == stats_after:
            harness.send({"op": "stats"})
            stats_after = None
        if cancel_after is not None and len(tokens) == cancel_after:
            harness.send({"op": "cancel", "request_id": request_id})
            cancel_after = None


def read_stats(harness) -> dict:
    harness.send({"op": "stats"})
    return harness.recv()


def main() -> int:
    checker = Checker("compute-service")
    harness = ComputeHarness()
    try:
        harness.start()
        harness.wait_ready()
        pid = harness.proc.pid

        harness.send({"op": "ping"})
        event = harness.recv()
        checker.check("ping-pong", event.get("event") == "pong", repr(event))

        # Malformed JSON must not corrupt subsequent service (Gate 7A case 41).
        harness.proc.stdin.write("this is not json\n")
        harness.proc.stdin.flush()
        error_event = harness.recv()
        checker.check("malformed-json-error", error_event.get("event") == "error",
                      repr(error_event))
        harness.send({"op": "ping"})
        checker.check("survives-malformed-json",
                      harness.recv().get("event") == "pong")

        idle = read_stats(harness)
        idle_kv = idle.get("kv") or {}
        checker.check("stats-event", idle.get("event") == "stats", repr(idle))
        checker.check("stats-idle-free-all",
                      idle_kv.get("total_pages", 0) > 0
                      and idle_kv.get("used_pages") == 0
                      and idle_kv.get("free_pages") == idle_kv.get("total_pages"),
                      repr(idle_kv))
        checker.check("stats-idle-capacity",
                      idle_kv.get("capacity_tokens")
                      == idle_kv.get("total_pages", 0) * idle_kv.get("page_tokens", 0)
                      and idle_kv.get("free_tokens")
                      == idle_kv.get("free_pages", 0) * idle_kv.get("page_tokens", 0),
                      repr(idle_kv))
        checker.check("stats-idle-requests",
                      idle_kv.get("active_requests") == 0
                      and idle_kv.get("queued_requests") == 0
                      and idle_kv.get("banker_safe") is True,
                      repr(idle_kv))

        # Plain generation.
        harness.send({
            "op": "generate", "request_id": 1, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "seed": 0})
        result = collect(harness, 1)
        done = result["done"]
        checker.check("generate-done", done is not None and done.get("request_id") == 1)
        checker.check("token-events-match-done",
                      result["tokens"] == done.get("generated_ids"),
                      f"tokens={result['tokens']} done={done.get('generated_ids')}")

        # Case A: ping is processed while a generation is in flight, and the
        # same generation can then be cancelled.
        harness.send({
            "op": "generate", "request_id": 50, "input_ids": PROMPT,
            "max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "seed": 0})
        result = collect(harness, 50, ping_after=3, stats_after=4, cancel_after=6)
        checker.check("ping-during-generation", result["pong"],
                      repr(result))
        busy_kv = (result["stats"][0].get("kv") or {}) if result["stats"] else {}
        checker.check("stats-during-generation",
                      bool(result["stats"])
                      and busy_kv.get("active_requests") == 1
                      and busy_kv.get("used_pages", 0) > 0
                      and busy_kv.get("queued_requests") == 0
                      and busy_kv.get("banker_safe") is True,
                      repr(busy_kv))
        checker.check("ping-then-cancel-done",
                      result["done"] is not None
                      and result["done"].get("request_id") == 50
                      and result["done"].get("finish_reason") == "cancelled",
                      repr(result["done"]))
        checker.check("cancel-token-parity",
                      result["tokens"] == result["done"].get("generated_ids"),
                      f"tokens={result['tokens']} done={result['done'].get('generated_ids')}")

        # Case 37: deterministic cancel (generate + cancel back to back).
        harness.send({
            "op": "generate", "request_id": 100, "input_ids": PROMPT,
            "max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "seed": 0})
        harness.send({"op": "cancel", "request_id": 100})
        result = collect(harness, 100)
        done100 = result["done"]
        checker.check("cancel-finish-reason",
                      done100 is not None and done100.get("finish_reason") == "cancelled",
                      repr(done100))
        checker.check("cancel-token-count-bounded",
                      len(result["tokens"]) <= 200
                      and len(result["tokens"]) >= 0,
                      f"tokens={len(result['tokens'])}")
        checker.check("cancel-token-parity-2",
                      result["tokens"] == done100.get("generated_ids"),
                      f"tokens={result['tokens']} done={done100.get('generated_ids')}")

        # Case 38/39: reuse after cancel, no stale events from the cancelled request.
        harness.send({
            "op": "generate", "request_id": 101, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "seed": 0})
        result = collect(harness, 101, stale_ids=(100,))
        checker.check("reuse-after-cancel",
                      result["done"] is not None
                      and result["done"].get("request_id") == 101
                      and result["done"].get("finish_reason") == "length",
                      repr(result["done"]))
        checker.check("no-stale-events", not result["stale"], repr(result["stale"]))
        drained_kv = (read_stats(harness).get("kv") or {})
        checker.check("stats-drained",
                      drained_kv.get("used_pages") == 0
                      and drained_kv.get("free_pages") == drained_kv.get("total_pages")
                      and drained_kv.get("active_requests") == 0,
                      repr(drained_kv))
        checker.check("compute-pid-stable", harness.proc.pid == pid,
                      f"{pid} -> {harness.proc.pid}")
        checker.check("model-load-once", harness.count_load_events() == 1,
                      f"load_events={harness.count_load_events()}")

        # Late cancel of an already-terminal request is an idempotent no-op.
        harness.send({"op": "cancel", "request_id": 101})
        harness.send({"op": "ping"})
        checker.check("late-cancel-no-op",
                      harness.recv().get("event") == "pong")

        # Unknown cancel target is an error, not a crash.
        harness.send({"op": "cancel", "request_id": 999999})
        checker.check("unknown-cancel-error",
                      harness.recv().get("event") == "error")

        # Case 40: shutdown while a generation is in flight.
        harness.send({
            "op": "generate", "request_id": 200, "input_ids": PROMPT,
            "max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "seed": 0})
        harness.send({"op": "shutdown"})
        result = collect(harness, 200)
        checker.check("shutdown-cancels-active",
                      result["done"] is not None
                      and result["done"].get("request_id") == 200
                      and result["done"].get("finish_reason") == "cancelled",
                      repr(result["done"]))
        shutdown_event = harness.recv()
        checker.check("shutdown-event", shutdown_event.get("event") == "shutdown",
                      repr(shutdown_event))
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0,
                      f"rc={harness.proc.returncode}")
    except Exception as exc:  # noqa: BLE001 - report as test failure
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
