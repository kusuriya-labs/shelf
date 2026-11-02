#!/usr/bin/env python3
"""Gate 7B: compute-level concurrent continuous batching acceptance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness  # noqa: E402

PROMPTS = [
    [248041, 77091],
    [248041, 77092],
    [248041, 77091, 198, 220],
    [248041, 77093, 198],
]


def collect(harness, target_ids, timeout_each=300.0):
    tokens = {i: [] for i in target_ids}
    done = {}
    other = []
    while len(done) < len(target_ids):
        event = harness.recv(timeout=timeout_each)
        kind = event.get("event")
        rid = event.get("request_id")
        if rid in target_ids:
            if kind == "token":
                tokens[rid].append(event["token_id"])
            elif kind == "done":
                done[rid] = event
            elif kind == "error":
                raise AssertionError(f"error for {rid}: {event}")
        else:
            other.append(event)
    return tokens, done, other


def main() -> int:
    checker = Checker("compute-concurrency")
    device = int(os.environ.get("PHASESHIFT_TEST_DEVICE", "0"))
    harness = ComputeHarness(max_concurrent_requests=4, batch_trace=True, device=device)
    try:
        harness.start()
        harness.wait_ready()
        pid = harness.proc.pid

        # Serial baseline.
        serial = {}
        for i, prompt in enumerate(PROMPTS):
            rid = 100 + i
            harness.send({"op": "generate", "request_id": rid, "input_ids": prompt,
                          "max_new_tokens": 12, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
            tokens, done, _ = collect(harness, [rid])
            checker.check(f"serial-{rid}-parity",
                          tokens[rid] == done[rid].get("generated_ids"),
                          f"tokens={tokens[rid]} done={done[rid].get('generated_ids')}")
            serial[rid] = done[rid]["generated_ids"]

        # Concurrent batch: send all four without waiting.
        ids = [200, 201, 202, 203]
        for rid, prompt in zip(ids, PROMPTS):
            harness.send({"op": "generate", "request_id": rid, "input_ids": prompt,
                          "max_new_tokens": 12, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
        tokens, done, other = collect(harness, ids)
        checker.check("no-stray-events", not other, repr(other))
        concurrent_first = {}
        for rid in ids:
            checker.check(f"concurrent-{rid}-token-parity",
                          tokens[rid] == done[rid].get("generated_ids"),
                          f"tokens={tokens[rid]} done={done[rid].get('generated_ids')}")
            checker.check(f"concurrent-{rid}-valid",
                          done[rid].get("finish_reason") in ("length", "stop")
                          and len(done[rid]["generated_ids"]) > 0,
                          repr(done[rid]))
            concurrent_first[rid] = done[rid]["generated_ids"]

        # Determinism: an identical second batch reproduces the first exactly.
        ids2 = [210, 211, 212, 213]
        for rid, prompt in zip(ids2, PROMPTS):
            harness.send({"op": "generate", "request_id": rid, "input_ids": prompt,
                          "max_new_tokens": 12, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
        _, done2, _ = collect(harness, ids2)
        checker.check("concurrent-deterministic",
                      all(done2[rid2]["generated_ids"] == concurrent_first[rid]
                          for rid, rid2 in zip(ids, ids2)),
                      f"{[done2[rid2]['generated_ids'] for rid2 in ids2]}")

        steps = harness.batch_steps()
        max_requests = max((r for r, _ in steps), default=0)
        checker.check("batch-trace-recorded", len(steps) > 0, f"steps={len(steps)}")
        checker.check("step-requests-ge-2", max_requests >= 2,
                      f"max requests/step={max_requests} steps={steps[-8:]}")

        # Cancel isolation: A and B concurrent; cancel only A. B uses a prompt
        # whose greedy output is batch-invariant so it can be compared exactly.
        a_id, b_id = 300, 301
        for rid, prompt in ((a_id, PROMPTS[2]), (b_id, PROMPTS[0])):
            harness.send({"op": "generate", "request_id": rid, "input_ids": prompt,
                          "max_new_tokens": 200 if rid == a_id else 12,
                          "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
        a_tokens = []
        a_done = None
        b_done = None
        cancel_sent = False
        while a_done is None or b_done is None:
            event = harness.recv()
            kind = event.get("event")
            rid = event.get("request_id")
            if rid == a_id:
                if kind == "token":
                    a_tokens.append(event["token_id"])
                    if len(a_tokens) >= 3 and not cancel_sent:
                        harness.send({"op": "cancel", "request_id": a_id})
                        cancel_sent = True
                elif kind == "done":
                    a_done = event
            elif rid == b_id:
                if kind == "done":
                    b_done = event
        checker.check("cancel-isolation-a-cancelled",
                      a_done.get("finish_reason") == "cancelled", repr(a_done))
        checker.check("cancel-isolation-a-parity",
                      a_tokens == a_done.get("generated_ids"))
        checker.check("cancel-isolation-b-normal",
                      b_done.get("finish_reason") in ("length", "stop"), repr(b_done))
        checker.check("cancel-isolation-b-parity",
                      b_done["generated_ids"] == serial[100],
                      f"{b_done['generated_ids']} vs {serial[100]}")

        # Queued cancel: two generates plus cancel issued before any step.
        qa, qb = 400, 401
        harness.send({"op": "generate", "request_id": qa, "input_ids": PROMPTS[0],
                      "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0,
                      "top_k": 0, "seed": 0})
        harness.send({"op": "generate", "request_id": qb, "input_ids": PROMPTS[1],
                      "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0,
                      "top_k": 0, "seed": 0})
        harness.send({"op": "cancel", "request_id": qb})
        tokens, done, _ = collect(harness, [qa, qb])
        checker.check("queued-cancel-finish-reason",
                      done[qb].get("finish_reason") == "cancelled", repr(done[qb]))
        checker.check("queued-cancel-sibling-ok",
                      done[qa].get("finish_reason") == "length"
                      and len(done[qa]["generated_ids"]) == 8
                      and done[qa]["generated_ids"] == serial[100][:8],
                      repr(done[qa]))

        # KV Banker queuing: physical KV capacity fits roughly one large
        # request, so the rest must wait and be admitted after earlier ones
        # finish. All must still complete without invariant violations.
        big_ids = [600, 601, 602]
        for rid in big_ids:
            harness.send({"op": "generate", "request_id": rid, "input_ids": PROMPTS[0],
                          "max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
        tokens, done, _ = collect(harness, big_ids)
        checker.check("banker-queue-all-complete",
                      all(done[i].get("finish_reason") in ("length", "stop")
                          for i in big_ids), repr({i: done[i] for i in big_ids}))
        checker.check("banker-queue-token-parity",
                      all(tokens[i] == done[i]["generated_ids"] for i in big_ids),
                      repr({i: (tokens[i], done[i].get("generated_ids")) for i in big_ids}))

        # Late cancel idempotent no-op.
        harness.send({"op": "cancel", "request_id": qa})
        harness.send({"op": "ping"})
        checker.check("late-cancel-no-op", harness.recv().get("event") == "pong")

        # Enterprise retirement: many short requests on one process.
        for n in range(300):
            rid = 1000 + n
            harness.send({"op": "generate", "request_id": rid, "input_ids": PROMPTS[0],
                          "max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
            collect(harness, [rid])
        checker.check("retirement-pid-stable", harness.proc.pid == pid)
        checker.check("retirement-model-load-once", harness.count_load_events() == 1,
                      f"loads={harness.count_load_events()}")

        # Shutdown with three in flight.
        sh_ids = [500, 501, 502]
        for rid, prompt in zip(sh_ids, PROMPTS[:3]):
            harness.send({"op": "generate", "request_id": rid, "input_ids": prompt,
                          "max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0,
                          "top_k": 0, "seed": 0})
        harness.send({"op": "shutdown"})
        terminals = {}
        saw_shutdown = False
        while not saw_shutdown:
            event = harness.recv()
            if event.get("event") == "done":
                terminals[event["request_id"]] = event.get("finish_reason")
            elif event.get("event") == "shutdown":
                saw_shutdown = True
        checker.check("shutdown-all-cancelled",
                      all(terminals.get(i) == "cancelled" for i in sh_ids),
                      repr(terminals))
        checker.check("shutdown-single-terminal", len(terminals) == len(sh_ids),
                      repr(terminals))
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0,
                      f"rc={harness.proc.returncode}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
