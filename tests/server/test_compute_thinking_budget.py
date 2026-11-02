#!/usr/bin/env python3
"""compute stdio: reasoning_budget_tokens closes the thinking region.

One prompt that keeps thinking, with a deliberately small budget so the
injection point falls inside the thinking region. The decode path always runs.
The DFlash2 speculative path runs when ``SHELF_DFLASH2_MODEL_DIR`` is set; the
27B target plus its drafter needs two GPUs, so ``SHELF_TEST_TP_DEVICES`` (for
example ``0,1``) is required as well. The two cases are not compared against
each other: TP degree changes the reduction, so token sequences may diverge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    ensure_chat_import,
    load_processor,
)

codec = ensure_chat_import()

BUDGET = 128
MAX_NEW_TOKENS = 320
KV_DTYPE = "fp8"
DFLASH2_MODEL_DIR = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
TP_DEVICES = os.environ.get("SHELF_TEST_TP_DEVICES")


def generate(harness, prompt_ids, request_id, **fields) -> dict:
    harness.send({
        "op": "generate",
        "request_id": request_id,
        "input_ids": list(prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        **fields,
    })
    while True:
        event = harness.recv()
        kind = event.get("event")
        if kind == "done":
            return event
        if kind == "error":
            raise RuntimeError(f"compute error: {event}")


def generated_of(done) -> list[int]:
    return [int(token) for token in done.get("generated_ids", [])]


def run_case(checked, label, harness, prompt_ids, close_token) -> list[int]:
    baseline = generated_of(generate(harness, prompt_ids, 1))
    natural = next(
        (index for index, token in enumerate(baseline) if token == close_token), None)
    checked.check(f"{label}-baseline-thinking-exceeds-the-budget",
                  natural is None or natural >= BUDGET,
                  f"baseline close token at {natural}")

    generated = generated_of(generate(
        harness, prompt_ids, 2,
        reasoning_budget_tokens=BUDGET,
        reasoning_close_token=close_token))

    checked.check(f"{label}-generation-reaches-the-budget",
                  len(generated) >= BUDGET + 8, f"generated={len(generated)}")
    checked.check(f"{label}-close-token-at-the-budget",
                  generated[BUDGET] == close_token,
                  f"token {BUDGET} is {generated[BUDGET]}, want {close_token}")
    checked.check(f"{label}-thinking-stayed-open-before-the-budget",
                  close_token not in generated[:BUDGET],
                  f"first close token at "
                  f"{next((i for i, t in enumerate(generated) if t == close_token), None)}")
    checked.check(f"{label}-generation-continues-after-the-close-token",
                  len(generated) > BUDGET + 1, f"generated={len(generated)}")
    checked.check(f"{label}-close-token-appears-only-once",
                  generated.count(close_token) == 1,
                  f"count={generated.count(close_token)}")
    return generated


def main() -> int:
    checked = Checker("compute-thinking-budget")
    processor = load_processor()
    close_ids = codec.reasoning_close_ids(processor)
    if len(close_ids) != 1:
        print(f"[SKIP] the reasoning close tag is {len(close_ids)} tokens")
        return 77
    close_token = int(close_ids[0])

    messages = [
        {"role": "user",
         "content": "991 掛ける 997 を、手順を詳しく書いて計算してください。"},
    ]
    prompt_ids = codec.prompt_ids(processor, messages, enable_thinking=True)

    with ComputeHarness(max_seq_len=4096, kv_cache_dtype=KV_DTYPE) as harness:
        harness.wait_ready()
        run_case(checked, "decode", harness, prompt_ids, close_token)

    if not DFLASH2_MODEL_DIR or not TP_DEVICES:
        print("SKIP dflash2: set SHELF_DFLASH2_MODEL_DIR and SHELF_TEST_TP_DEVICES")
        return checked.done()

    with ComputeHarness(max_seq_len=4096, kv_cache_dtype=KV_DTYPE,
                        dflash2_model_dir=DFLASH2_MODEL_DIR,
                        auto_arena=True,
                        tp=len(TP_DEVICES.split(",")),
                        devices=TP_DEVICES) as harness:
        harness.wait_ready()
        run_case(checked, "dflash2", harness, prompt_ids, close_token)

    return checked.done()


if __name__ == "__main__":
    sys.exit(main())
