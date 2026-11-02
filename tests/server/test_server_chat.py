#!/usr/bin/env python3
"""Gate 1 acceptance: OpenAI chat/completions (non-stream) through the server."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    cli_generated_ids,
    compute_pids,
    ensure_chat_import,
    http_get_json,
    http_json,
    model_dir,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def read_token_log(path: Path) -> list[list[int]]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [[int(x) for x in line.split(",") if x] for line in lines]


def main() -> int:
    checker = Checker("server-chat")
    token_log = Path(tempfile.mkdtemp(prefix="ps-tokenlog-")) / "tokens.txt"
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(env={"SHELF_SERVER_TOKEN_LOG": str(token_log)}) as server:
            models = http_get_json(f"{server.base_url}/models")
            checker.check(
                "models-listed",
                any(m.get("id") == "shelf" for m in models.get("data", [])),
                repr(models))

            result = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
            choice = result["choices"][0]
            content = choice["message"]["content"]
            checker.check("response-id", result.get("model") == "shelf", repr(result))
            checker.check("finish-reason", choice.get("finish_reason") == "stop",
                          repr(choice))
            checker.check("content-nonempty", bool(content.strip()), repr(content))

            pids_before = compute_pids()
            result2 = http_json(
                f"{server.base_url}/chat/completions",
                {"model": "shelf", "messages": MESSAGES,
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
            content2 = result2["choices"][0]["message"]["content"]
            checker.check("second-request-consistent", content == content2,
                          f"{content!r} vs {content2!r}")
            pids_after = compute_pids()
            checker.check("single-compute-process", len(pids_after) == 1, repr(pids_after))
            checker.check("compute-pid-stable", pids_before == pids_after,
                          f"{pids_before} -> {pids_after}")

            token_runs = read_token_log(token_log)
            checker.check("model-load-once-two-requests", len(token_runs) == 2,
                          f"token_runs={len(token_runs)}")

        cli_ids = cli_generated_ids(MESSAGES, 32, enable_thinking=False)
        codec = ensure_chat_import()
        processor = codec.load_processor(str(model_dir()))
        cli_text = codec.decode_generated(processor, cli_ids)
        checker.check("cli-token-parity",
                      bool(token_runs) and token_runs[-1] == cli_ids,
                      f"server={token_runs[-1] if token_runs else None} cli={cli_ids}")
        checker.check("cli-text-parity", content == cli_text,
                      f"server={content!r} cli={cli_text!r}")
    except Exception as exc:  # noqa: BLE001 - report as test failure
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
