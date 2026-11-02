#!/usr/bin/env python3
"""prefix cache の JSONL protocol contract.

  * ping の capabilities.prefix_cache（PrefixCache 作成成功時 true）
  * generate の prefix_cache_checkpoint_position（0 のみ受理）
  * done の prefill_tokens / restored_tokens / cache_checkpoint_tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    ensure_chat_import,
    model_dir,
    prompt_ids,
)

LONG_PROMPT = prompt_ids(
    "The quick brown fox jumps over the lazy dog. " * 40
    + " Answer with one short sentence.")


def generate(harness, request_id, **extra):
    payload = {
        "op": "generate",
        "request_id": request_id,
        "input_ids": extra.pop("input_ids", LONG_PROMPT),
        "max_new_tokens": extra.pop("max_new_tokens", 8),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 0,
    }
    payload.update(extra)
    harness.send(payload)
    tokens = []
    done = None
    errors = []
    while True:
        event = harness.recv()
        kind = event.get("event")
        if kind == "token":
            tokens.append(event["token_id"])
        elif kind == "error":
            errors.append(event)
            return tokens, done, errors
        elif kind == "done":
            done = event
            return tokens, done, errors


def main() -> int:
    checker = Checker("compute-prefix-contract")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    if len(LONG_PROMPT) < 160:
        checker.check("prompt-length", False, f"len={len(LONG_PROMPT)}")
        return checker.done()

    ensure_chat_import()

    harness = ComputeHarness(max_seq_len=512, arena_gib=24, device=0)
    try:
        harness.start()
        harness.wait_ready()

        harness.send({"op": "ping"})
        pong = None
        while True:
            event = harness.recv()
            if event.get("event") == "pong":
                pong = event
                break
        caps = (pong or {}).get("capabilities") or {}
        checker.check("protocol-version", pong is not None and
                      pong.get("protocol_version") == 1, repr(pong))
        checker.check("capability-prefix-cache-true",
                      caps.get("prefix_cache") is True, repr(pong))
        checker.check("capabilities-is-object", isinstance(caps, dict), repr(pong))

        _, done, errors = generate(
            harness, 1, prefix_cache_checkpoint_position=0)
        checker.check("zero-checkpoint-ok", not errors and done is not None,
                      str(errors))
        if done is not None:
            for key in ("prefill_tokens", "restored_tokens",
                        "cache_checkpoint_tokens", "prompt_tokens",
                        "finish_reason", "generated_ids"):
                checker.check(f"done-has-{key}", key in done, repr(sorted(done)))
            checker.check("restored-tokens-zero",
                          done.get("restored_tokens") == 0,
                          repr(done.get("restored_tokens")))
            checker.check("cache-checkpoint-tokens-zero",
                          done.get("cache_checkpoint_tokens") == 0,
                          repr(done.get("cache_checkpoint_tokens")))
            checker.check("prefill-equals-prompt",
                          done.get("prefill_tokens") == done.get("prompt_tokens"),
                          repr({k: done.get(k) for k in
                                ("prefill_tokens", "prompt_tokens")}))
            checker.check("generated-ids-present",
                          bool(done.get("generated_ids")), repr(done)[:300])

        _, hit_done, hit_errors = generate(
            harness, 2, prefix_cache_checkpoint_position=128)
        checker.check("nonzero-checkpoint-rejected",
                      hit_done is None and len(hit_errors) == 1,
                      str(hit_errors))
        if hit_errors:
            checker.check("nonzero-checkpoint-invalid-argument",
                          hit_errors[0].get("code") == "invalid_argument",
                          str(hit_errors[0]))
            checker.check("nonzero-checkpoint-message",
                          "prefix cache" in str(hit_errors[0].get("message", "")),
                          str(hit_errors[0]))
            checker.check("nonzero-checkpoint-request-id",
                          hit_errors[0].get("request_id") == 2, str(hit_errors[0]))

        _, after_done, after_errors = generate(harness, 3, max_new_tokens=8)
        checker.check("service-alive-after-rejection",
                      not after_errors and after_done is not None,
                      str(after_errors))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
