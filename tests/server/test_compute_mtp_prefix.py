#!/usr/bin/env python3
"""MTP serve の prefix cache 復元を確認する。

環境変数:
    SHELF_MODEL_DIR_MTP      対象 checkpoint
    SHELF_MTP_PREFIX_DEVICES device list（default 0,1）
    SHELF_MTP_PREFIX_TP      tensor parallel（default 1）
    SHELF_MTP_PREFIX_EP      expert parallel（default 2）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, prompt_ids  # noqa: E402


def model_dir() -> Path:
    env = os.environ.get("SHELF_MODEL_DIR_MTP") or os.environ.get("SHELF_MODEL_DIR")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parents[2]
            / "models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw")


def main() -> int:
    checker = Checker("compute-mtp-prefix")
    model = model_dir()
    os.environ["SHELF_MODEL_DIR"] = str(model)
    prefix_text = "This is a stable shared system document used for prefix caching. " * 260
    prefix = prompt_ids(prefix_text)
    first_prompt = prefix + prompt_ids("Count from 1 to 20: 1, 2, 3,")
    second_prompt = prefix + prompt_ids("Count from 20 to 1: 20, 19, 18,")
    short_prompt = prompt_ids("Shelf keeps the tail checkpoint for an identical short request. " * 40)

    harness = ComputeHarness(
        model_path=model,
        auto_arena=True,
        max_seq_len=8192,
        max_concurrent_requests=1,
        max_num_batched_tokens=2048,
        prefix_cache_enabled=True,
        prefix_cache_strategy="radix",
        prefix_cache_save_interval=16384,
        devices=os.environ.get("SHELF_MTP_PREFIX_DEVICES", "0,1"),
        tp=int(os.environ.get("SHELF_MTP_PREFIX_TP", "1")),
        ep=int(os.environ.get("SHELF_MTP_PREFIX_EP", "2")),
        mtp=True,
        mtp_drafts=3,
        mtp_window=4096,
        mtp_sink=64,
        kv_cache_dtype="octave:c5v5",
    )
    try:
        harness.start()
        harness.wait_ready(timeout=1200.0)
        done = {}
        for request_id, prompt in ((1, first_prompt), (2, second_prompt),
                                   (3, first_prompt), (4, short_prompt),
                                   (5, short_prompt)):
            harness.send({
                "op": "generate",
                "request_id": request_id,
                "input_ids": prompt,
                "max_new_tokens": 16,
                "temperature": 0.0,
            })
            while True:
                event = harness.recv(timeout=900.0)
                if event.get("event") == "error":
                    checker.check("completed", False, repr(event))
                    return 1
                if event.get("event") == "done":
                    done[request_id] = event
                    break
        checker.check("first-completed", 1 in done, repr(done))
        checker.check("second-completed", 2 in done, repr(done))
        checker.check("third-completed", 3 in done, repr(done))
        checker.check("first-no-restore", done.get(1, {}).get("restored_tokens") == 0,
                      repr(done.get(1)))
        restored = int(done.get(2, {}).get("restored_tokens", 0))
        checker.check("second-restored", restored > 0, repr(done.get(2)))
        checker.check("second-prefill-shorter",
                      done.get(2, {}).get("prefill_tokens", 1 << 30) < len(second_prompt),
                      repr(done.get(2)))
        repeat_restored = int(done.get(3, {}).get("restored_tokens", 0))
        checker.check("repeat-restored", repeat_restored > 0, repr(done.get(3)))
        checker.check("repeat-prefill-shorter",
                      done.get(3, {}).get("prefill_tokens", 1 << 30) < len(first_prompt),
                      repr(done.get(3)))
        checker.check("short-first-no-restore",
                      done.get(4, {}).get("restored_tokens") == 0,
                      repr(done.get(4)))
        short_restored = int(done.get(5, {}).get("restored_tokens", 0))
        checker.check("short-repeat-restored", short_restored > 0, repr(done.get(5)))
        checker.check("short-repeat-prefill-shorter",
                      done.get(5, {}).get("prefill_tokens", 1 << 30) < len(short_prompt),
                      repr(done.get(5)))
        continuation = first_prompt + done[1]["generated_ids"]
        harness.send({
            "op": "generate",
            "request_id": 6,
            "input_ids": continuation,
            "max_new_tokens": 16,
            "temperature": 0.0,
        })
        while True:
            event = harness.recv(timeout=900.0)
            if event.get("event") == "error":
                checker.check("continuation-completed", False, repr(event))
                return 1
            if event.get("event") == "done":
                done[6] = event
                break
        checker.check("continuation-completed", 6 in done, repr(done))
        checker.check("continuation-restored",
                      int(done.get(6, {}).get("restored_tokens", 0)) > len(first_prompt),
                      repr(done.get(6)))
        return checker.done()
    finally:
        harness.close()


if __name__ == "__main__":
    raise SystemExit(main())
