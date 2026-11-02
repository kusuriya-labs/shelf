#!/usr/bin/env python3
"""Gate 8R: generation stop token (EOS) contract.

Confirms that Shelf uses every token listed by the model
``generation_config.eos_token_id`` (for Qwen3.5/3.8: ``<|im_end|>`` and
``<|endoftext|>``), so an assistant turn terminates on ``<|im_end|>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    generation_stop_probe,
    prompt_ids,
)


def main() -> int:
    checker = Checker("eos-contract")
    probe = generation_stop_probe()
    print("generation stop probe:", probe)

    model_eos = probe["model_config_eos"]
    tokenizer_eos = probe["tokenizer_eos_metadata"]
    stop_tokens = probe["generation_stop_tokens"]
    stop_names = probe["generation_stop_token_names"]

    checker.check("generation-stop-tokens-present", len(stop_tokens) >= 2, repr(stop_tokens))
    checker.check("generation-stop-includes-imend", tokenizer_eos in stop_tokens,
                  f"stop={stop_tokens} tokenizer={tokenizer_eos}")
    checker.check("generation-stop-includes-endoftext", model_eos in stop_tokens,
                  f"stop={stop_tokens} model={model_eos}")
    checker.check("generation-stop-names",
                  "<|im_end|>" in stop_names and "<|endoftext|>" in stop_names,
                  repr(stop_names))

    # The runtime must stop at the first stop token (the assistant turn end),
    # not run through to the secondary <|endoftext|> stop.
    checker.check("turn-end-is-first-stop",
                  bool(stop_tokens) and stop_tokens[0] == tokenizer_eos,
                  f"stop={stop_tokens} tokenizer={tokenizer_eos}")

    input_ids = prompt_ids("Say OK.")
    harness = ComputeHarness(max_concurrent_requests=1, max_seq_len=256, device=1)
    try:
        harness.start()
        harness.wait_ready()
        harness.send({
            "op": "generate",
            "request_id": 1,
            "input_ids": input_ids,
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 0,
        })
        done = None
        while done is None:
            event = harness.recv()
            if event.get("request_id") != 1:
                continue
            if event.get("event") in ("done", "error"):
                done = event
        generated = list(done.get("generated_ids", []))
        checker.check("runtime-non-empty", bool(generated), repr(generated))
        checker.check("runtime-terminates-with-stop",
                      done.get("finish_reason") == "stop", repr(done))
        checker.check("terminal-token-is-turn-end",
                      bool(generated) and generated[-1] == tokenizer_eos,
                      f"got={generated} tokenizer_eos={tokenizer_eos}")
        checker.check("terminal-is-first-stop",
                      bool(generated) and generated[-1] == stop_tokens[0],
                      f"got={generated} stop={stop_tokens}")

        harness.send({"op": "shutdown"})
        harness.recv()
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
