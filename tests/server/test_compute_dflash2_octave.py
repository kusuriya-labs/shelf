#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, model_dir  # noqa: E402


PROMPTS = [
    [248041, 77091],
    [248041, 77092],
    [248041, 77091, 198, 220],
    [248041, 77093, 198],
]


def collect(harness, request_ids):
    tokens = {request_id: [] for request_id in request_ids}
    done = {}
    while len(done) != len(request_ids):
        event = harness.recv(timeout=300.0)
        request_id = event.get("request_id")
        if request_id not in tokens:
            continue
        if event.get("event") == "token":
            tokens[request_id].append(event["token_id"])
        elif event.get("event") == "done":
            done[request_id] = event
        elif event.get("event") == "error":
            raise RuntimeError(str(event))
    return tokens, done


def main() -> int:
    checker = Checker("compute-dflash2-octave")
    draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
    if not draft:
        print("SKIP: SHELF_DFLASH2_MODEL_DIR is not set")
        return 77
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        return 77
    device = int(os.environ.get("SHELF_TEST_DEVICE", "0"))
    harness = ComputeHarness(
        max_seq_len=512,
        max_concurrent_requests=4,
        batch_trace=True,
        device=device,
        kv_cache_dtype="octave:c4v4",
        dflash2_model_dir=draft,
        dflash2_drafts=7,
        auto_arena=True,
        prefix_cache_enabled=False,
    )
    try:
        harness.start()
        harness.wait_ready()
        serial = []
        for index, prompt in enumerate(PROMPTS):
            request_id = 100 + index
            harness.send({
                "op": "generate", "request_id": request_id,
                "input_ids": prompt, "max_new_tokens": 16,
                "temperature": 0.0,
            })
            tokens, done = collect(harness, [request_id])
            checker.check(f"serial-{index}-stream",
                          tokens[request_id] == done[request_id]["generated_ids"])
            serial.append(done[request_id]["generated_ids"])
        for count in (2, 3, 4):
            request_ids = [count * 1000 + index for index in range(count)]
            for request_id, prompt in zip(request_ids, PROMPTS[:count]):
                harness.send({
                    "op": "generate", "request_id": request_id,
                    "input_ids": prompt, "max_new_tokens": 16,
                    "temperature": 0.0,
                })
            tokens, done = collect(harness, request_ids)
            first = []
            for index, request_id in enumerate(request_ids):
                checker.check(
                    f"concurrent-{count}-{index}-stream",
                    tokens[request_id] == done[request_id]["generated_ids"],
                )
                checker.check(
                    f"concurrent-{count}-{index}-valid",
                    bool(done[request_id]["generated_ids"]),
                )
                first.append(done[request_id]["generated_ids"])
            repeat_ids = [count * 10000 + index for index in range(count)]
            for request_id, prompt in zip(repeat_ids, PROMPTS[:count]):
                harness.send({
                    "op": "generate", "request_id": request_id,
                    "input_ids": prompt, "max_new_tokens": 16,
                    "temperature": 0.0,
                })
            _, repeated = collect(harness, repeat_ids)
            checker.check(
                f"concurrent-{count}-deterministic",
                all(repeated[request_id]["generated_ids"] == first[index]
                    for index, request_id in enumerate(repeat_ids)),
                str([repeated[request_id]["generated_ids"]
                     for request_id in repeat_ids]),
            )
        steps = harness.batch_steps()
        checker.check("mixed-batch-observed",
                      max((requests for requests, _ in steps), default=0) >= 4,
                      str(steps[-16:]))
    except Exception as exc:
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    raise SystemExit(main())
