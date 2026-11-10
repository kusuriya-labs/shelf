#!/usr/bin/env python3
"""Exercise per-token masks and waiting-request isolation through JSONL."""
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ComputeHarness


def mask(token):
    bits = bytearray(248320 // 8)
    bits[token // 8] |= 1 << (token % 8)
    return base64.b64encode(bits).decode("ascii")


def main():
    mode = os.environ.get("SHELF_TEST_MASK_SPEC", "")
    h = ComputeHarness(max_concurrent_requests=4, device=int(os.environ.get("SHELF_TEST_DEVICE", "0")),
                       mtp=mode == "mtp",
                       dflash2_model_dir=os.environ.get("SHELF_DFLASH2_MODEL_DIR") if mode == "dflash2" else None)
    try:
        h.start(); h.wait_ready()
        for rid, temperature in ((10, 0.0), (11, 0.8)):
            h.send({"op":"generate", "request_id":rid, "input_ids":[248041,77091],
                    "max_new_tokens":4, "temperature":temperature, "grammar":True,
                    "bitmask":mask(100)})
            expected = [100, 101, 102, 103]
            seen = []
            while True:
                event = h.recv()
                assert event.get("event") != "error", event
                if event.get("event") == "token":
                    seen.append(event["token_id"])
                    assert seen[-1] == expected[len(seen)-1], event
                    if len(seen) < len(expected):
                        h.send({"op":"mask", "request_id":rid, "bitmask":mask(expected[len(seen)])})
                if event.get("event") == "done":
                    assert seen == expected and event["generated_ids"] == seen, event
                    break
        # A forced reasoning close activates the initial mask, and the close
        # itself is not sampled through it.
        h.send({"op":"generate", "request_id":12, "input_ids":[248041,77091],
                "max_new_tokens":4, "grammar":True, "grammar_after_reasoning":True,
                "reasoning_budget_tokens":1, "reasoning_close_token":104,
                "bitmask":mask(100)})
        seen = []
        while True:
            event = h.recv()
            assert event.get("event") != "error", event
            if event.get("event") == "token":
                seen.append(event["token_id"])
                if len(seen) == 1:
                    assert seen[0] != 104, event
                elif len(seen) == 2:
                    assert seen[-1] == 104, event
                    h.send({"op":"mask", "request_id":12, "bitmask":mask(101)})
                elif len(seen) == 3:
                    assert seen[-1] == 101, event
                    h.send({"op":"mask", "request_id":12, "bitmask":mask(102)})
                else:
                    assert seen[-1] == 102, event
            elif event.get("event") == "done":
                assert len(seen) == 4 and event["generated_ids"] == seen, event
                break
        # A waiting grammar request must not block its unconstrained peer.
        h.send({"op":"generate", "request_id":20, "input_ids":[248041,77091],
                "max_new_tokens":10, "grammar":True, "bitmask":mask(100)})
        assert h.recv()["token_id"] == 100
        h.send({"op":"generate", "request_id":21, "input_ids":[248041,77091], "max_new_tokens":3})
        while True:
            e = h.recv()
            assert e.get("event") != "error" and e.get("request_id") != 20, e
            if e.get("request_id") == 21 and e.get("event") == "done": break
        h.send({"op":"cancel", "request_id":20})
        assert h.recv()["finish_reason"] == "cancelled"
        print("compute token mask passed")
        return 0
    finally:
        h.close()

if __name__ == "__main__":
    raise SystemExit(main())
