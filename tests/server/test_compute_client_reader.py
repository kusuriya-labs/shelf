#!/usr/bin/env python3
"""AsyncComputeClient の stdout reader 契約.

64 KiB を超える done 行が reader を殺さないこと、reader が死んだ場合は
保留中の stream をハングさせずエラーで終端することを、偽 compute で確認する。
GPU と model は不要。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker  # noqa: E402

SERVER_SRC = Path(__file__).resolve().parents[2] / "src" / "apps" / "server"
sys.path.insert(0, str(SERVER_SRC))

from serverlib import compute_client  # noqa: E402

PROMPT = [1, 2, 3]

FAKE_COMPUTE = r'''
import json
import sys

mode = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    op = request.get("op")
    if op == "ping":
        event = {"event": "pong", "capabilities": {"prefix_cache": True}}
        sys.stdout.write(json.dumps(event) + "\n")
    elif op == "generate":
        rid = request["request_id"]
        sys.stdout.write(json.dumps(
            {"event": "token", "request_id": rid, "token_id": 7}) + "\n")
        if mode == "huge":
            payload = {"event": "done", "request_id": rid,
                       "generated_ids": list(range(30000)),
                       "finish_reason": "length"}
        else:
            payload = {"event": "token", "request_id": [], "token_id": 7}
        sys.stdout.write(json.dumps(payload) + "\n")
    elif op == "cancel":
        rid = request["request_id"]
        sys.stdout.write(json.dumps(
            {"event": "done", "request_id": rid,
             "finish_reason": "cancelled"}) + "\n")
    elif op == "shutdown":
        sys.stdout.write(json.dumps({"event": "shutdown"}) + "\n")
        sys.stdout.flush()
        break
    sys.stdout.flush()
'''


async def collect(client, timeout: float):
    stream = client.generate_stream(PROMPT, 10)
    events = []
    try:
        async def drive():
            async for event in stream:
                events.append(event)
                if "done" in event:
                    return

        await asyncio.wait_for(drive(), timeout=timeout)
    finally:
        await stream.aclose()
    return events


async def delayed_grammar_test(checker: Checker) -> None:
    class Grammar:
        after_reasoning = True
        def __init__(self):
            self.tokens = []
        def next_mask(self):
            return "initial" if not self.tokens else "next"
        def accept(self, token):
            self.tokens.append(token)
            return True

    client = compute_client.AsyncComputeClient([sys.executable])
    sent = []
    mailbox = asyncio.Queue()
    for event in ({"event": "token", "token_id": 10},
                  {"event": "token", "token_id": 99},
                  {"event": "token", "token_id": 20},
                  {"event": "done", "finish_reason": "length"}):
        mailbox.put_nowait(event)
    async def begin(): return 1, mailbox
    async def end(_): pass
    async def send(payload): sent.append(payload)
    client._begin_request = begin
    client._end_request = end
    client._send = send
    grammar = Grammar()
    events = [event async for event in client.generate_stream(
        PROMPT, 10, reasoning_close_token=99, grammar=grammar)]
    checker.check("delayed-grammar-transitions",
                  [event.get("token_id") for event in events[:-1]] == [10, 99, 20]
                  and grammar.tokens == [20]
                  and sent[0].get("grammar_after_reasoning") is True
                  and sent[0].get("reasoning_close_token") == 99
                  and [item["bitmask"] for item in sent[1:]] == ["initial", "next"], repr(sent))


async def run(checker: Checker) -> None:
    await delayed_grammar_test(checker)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "fake_compute.py"
        script.write_text(FAKE_COMPUTE, encoding="utf-8")

        huge = compute_client.AsyncComputeClient(
            [sys.executable, "-I", str(script), "huge"], max_inflight=1)
        await huge.start()
        try:
            events = await collect(huge, 10.0)
            done = events[-1].get("done") or {}
            checker.check("huge-done-delivered",
                          len(done.get("generated_ids", [])) == 30000,
                          repr(done)[:200])
            checker.check("huge-inflight-released", huge.inflight == 0,
                          f"inflight={huge.inflight}")
        except Exception as exc:  # noqa: BLE001
            checker.check("huge-done-delivered", False, repr(exc))
        finally:
            await huge.shutdown()

        fault = compute_client.AsyncComputeClient(
            [sys.executable, "-I", str(script), "fault"], max_inflight=1)
        await fault.start()
        try:
            await collect(fault, 10.0)
            checker.check("reader-fault-raises", False, "stream ended without error")
        except asyncio.TimeoutError:
            checker.check("reader-fault-raises", False, "stream hung")
        except compute_client.ComputeError as exc:
            checker.check("reader-fault-raises", True, repr(exc))
        finally:
            await fault.shutdown()


def main() -> int:
    checker = Checker("compute-client-reader")
    asyncio.run(run(checker))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
