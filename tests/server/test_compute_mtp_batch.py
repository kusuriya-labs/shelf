#!/usr/bin/env python3
"""MTP のマルチリクエスト融合 verify を shelf-compute --serve-stdio で確認する。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness  # noqa: E402

PROMPTS = [
    [248041, 77091],
    [248041, 77092],
    [248041, 77091, 198, 220],
    [248041, 77093, 198],
    [248041, 77091, 279, 2614],
    [248041, 77092, 279, 2614],
    [248041, 77093, 279, 2614],
    [248041, 77094, 279, 2614],
]
MAX_NEW_TOKENS = 8


def model_dir() -> Path:
    env = os.environ.get("SHELF_MODEL_DIR_MTP") or os.environ.get("SHELF_MODEL_DIR")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parents[2]
            / "models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw")


def collect(harness: ComputeHarness, request_ids: list[int]) -> dict[int, dict]:
    tokens = {request_id: [] for request_id in request_ids}
    done: dict[int, dict] = {}
    while len(done) < len(request_ids):
        event = harness.recv(timeout=900.0)
        kind = event.get("event")
        request_id = event.get("request_id")
        if request_id in tokens:
            if kind == "token":
                tokens[request_id].append(event["token_id"])
            elif kind == "done":
                if tokens[request_id] != event.get("generated_ids"):
                    raise AssertionError(
                        f"token stream mismatch for {request_id}: "
                        f"{tokens[request_id]} vs {event.get('generated_ids')}")
                done[request_id] = event
            elif kind == "error":
                raise AssertionError(f"error for {request_id}: {event}")
    return done


def run_batch(harness: ComputeHarness, base_id: int) -> dict[int, dict]:
    ids = [base_id + i for i in range(len(PROMPTS))]
    for request_id, prompt in zip(ids, PROMPTS):
        harness.send({
            "op": "generate",
            "request_id": request_id,
            "input_ids": prompt,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
        })
    return collect(harness, ids)


def target_baseline() -> dict[int, list[int]]:
    harness = ComputeHarness(
        model_path=model_dir(),
        auto_arena=True,
        max_seq_len=512,
        max_concurrent_requests=len(PROMPTS),
        max_num_batched_tokens=512,
        prefix_cache_enabled=False,
        devices=os.environ.get("SHELF_MTP_DEVICES", "0,1,2,3"),
        ep=4,
    )
    try:
        harness.start()
        harness.wait_ready()
        baseline = {}
        for request_id, prompt in enumerate(PROMPTS):
            harness.send({
                "op": "generate",
                "request_id": request_id,
                "input_ids": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            })
            done = collect(harness, [request_id])
            baseline[request_id] = done[request_id]["generated_ids"]
        return baseline
    finally:
        harness.close()


def main() -> int:
    checker = Checker("compute-mtp-batch")
    devices = os.environ.get("SHELF_MTP_DEVICES", "0,1,2,3")
    baseline = target_baseline()
    harness = ComputeHarness(
        model_path=model_dir(),
        auto_arena=True,
        max_seq_len=512,
        max_concurrent_requests=len(PROMPTS),
        max_num_batched_tokens=512,
        prefix_cache_enabled=False,
        devices=devices,
        ep=4,
        mtp=True,
    )
    try:
        harness.start()
        harness.wait_ready()

        first = run_batch(harness, 100)
        second = run_batch(harness, 200)
        checker.check("completed", all(
            item.get("finish_reason") in ("length", "stop") for item in first.values()),
            repr(first))
        checker.check("matches-target", all(
            first[100 + i]["generated_ids"] == baseline[i]
            for i in range(len(PROMPTS))),
            repr([(first[100 + i]["generated_ids"], baseline[i])
                  for i in range(len(PROMPTS))]))
        checker.check("deterministic", all(
            first[100 + i]["generated_ids"] == second[200 + i]["generated_ids"]
            for i in range(len(PROMPTS))),
            repr([first[100 + i]["generated_ids"] for i in range(len(PROMPTS))]))

        stderr = "\n".join(harness.stderr_lines)
        checker.check("mtp-batch-enabled", "MTP_SERVE=batch" in stderr, stderr[-500:])
        match = re.search(
            r"MTP_BATCH rounds=(\d+) accepted=(\d+) emitted=(\d+) chunks=(\d+) "
            r"max_active=(\d+)",
            stderr)
        checker.check("mtp-batch-stats", match is not None, stderr[-500:])
        if match is not None:
            rounds = int(match.group(1))
            chunks = int(match.group(4))
            max_active = int(match.group(5))
            checker.check("multi-request-round", max_active >= 2, match.group(0))
            checker.check("multi-batch-verify", chunks >= 2 and rounds >= 2,
                          match.group(0))
    except Exception as exc:  # noqa: BLE001 - the harness reports failures as text
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    raise SystemExit(main())
