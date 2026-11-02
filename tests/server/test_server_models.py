#!/usr/bin/env python3
"""GET /v1/models と model 名の契約."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_get_json,
    http_post_status,
    model_dir,
)


def main() -> int:
    checker = Checker("server-models")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness() as server:
            models = http_get_json(f"{server.base_url}/models")
            checker.check("object-list", models.get("object") == "list", repr(models))
            checker.check("model-listed",
                          any(m.get("id") == "phaseshift" for m in models.get("data", [])),
                          repr(models))
            checker.check("model-owned-by",
                          all(m.get("owned_by") == "phaseshift"
                              for m in models.get("data", [])), repr(models))
            checker.check("model-context-length",
                          all(m.get("context_length") == server.max_seq_len
                              for m in models.get("data", [])), repr(models))

            status, body = http_post_status(
                f"{server.base_url}/chat/completions",
                {"model": "does-not-exist",
                 "messages": [{"role": "user", "content": "hello"}],
                 "temperature": 0, "max_tokens": 8})
            checker.check("unknown-model-404", status == 404, f"{status} {body[:300]}")

            status, body = http_post_status(
                f"{server.base_url}/responses",
                {"model": "phaseshift", "input": "hello"})
            checker.check("responses-404", status == 404, f"{status} {body[:300]}")

            status, body = http_post_status(
                f"{server.base_url}/completions",
                {"model": "phaseshift", "prompt": "hello", "max_tokens": 8})
            checker.check("completions-404", status == 404, f"{status} {body[:300]}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
