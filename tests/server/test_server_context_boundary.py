#!/usr/bin/env python3
"""Gate 11C: 32K context boundary and physical concurrency closure tests.

Boundary:
  * a near-limit prompt (actual rendered tokens) completes with HTTP 200
  * an over-limit max_tokens is clamped to the remaining window (HTTP 200)
  * the server stays healthy for a plain request afterwards

Concurrency:
  * four ~6.5K-token requests run concurrently within the single-context KV
    capacity and all complete
  * the compute batch trace shows multi-request scheduler steps
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    ensure_chat_import,
    load_processor,
    model_dir,
    test_arena_gib,
)

codec = ensure_chat_import()

MAX_SEQ_LEN = 32768
FILLER = "The quick brown fox jumps over the lazy dog. "
TAIL = "\nReply with the single word: OK"
CONTINUE = "\nNow write a long detailed paragraph about the text above."


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))
    except (urllib.error.URLError, OSError):
        return 0, {}


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def rendered_tokens(processor, reps, enable_thinking):
    messages = [{"role": "user", "content": FILLER * reps + TAIL}]
    return len(codec.prompt_ids(processor, messages, None, enable_thinking))


def fit_reps(processor, target, enable_thinking):
    lo, hi = 1, 40000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rendered_tokens(processor, mid, enable_thinking) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def boundary_cases(checker, processor, url):
    reps = fit_reps(processor, 32500, False)
    rendered = rendered_tokens(processor, reps, False)
    checker.check("boundary-reps", rendered <= 32500, f"rendered={rendered} reps={reps}")
    max_tokens = MAX_SEQ_LEN - rendered - 2
    checker.check("boundary-budget-fits", max_tokens >= 32,
                  f"rendered={rendered} max_tokens={max_tokens}")

    status, body = post(url, {
        "model": "shelf", "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": FILLER * reps + TAIL}]})
    checker.check("near-limit-200", status == 200, f"{status} {repr(body)[:200]}")
    checker.check("near-limit-content", bool((message(body).get("content") or "").strip()),
                  repr(message(body))[:200])

    over = max_tokens + 256
    status, body = post(url, {
        "model": "shelf", "temperature": 0, "max_tokens": over,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": FILLER * reps + TAIL}]})
    checker.check("over-limit-clamped", status == 200, f"{status} {repr(body)[:200]}")
    remaining = MAX_SEQ_LEN - rendered
    completion = int((body.get("usage") or {}).get("completion_tokens", 0))
    checker.check("over-limit-budget", 0 < completion <= remaining,
                  f"completion={completion} remaining={remaining}")

    status, body = post(url, {
        "model": "shelf", "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "What is 2+2? Answer with only the number."}]})
    checker.check("post-over-healthy", status == 200, f"{status} {repr(body)[:200]}")
    checker.check("post-over-answer",
                  (message(body).get("content") or "").strip() == "4",
                  repr(message(body).get("content")))


def concurrency_batch(checker, processor, harness):
    url = f"{harness.base_url}/chat/completions"
    per_request = fit_reps(processor, 6500, False)
    rendered = rendered_tokens(processor, per_request, False)
    checker.check("concurrency-chunk", 6000 <= rendered <= 7000,
                  f"rendered={rendered} reps={per_request}")

    # A long continuation keeps every request resident long enough that the four
    # staggered HTTP arrivals overlap in the scheduler; a short answer would let
    # the first request finish before the others are admitted.
    content = FILLER * per_request + CONTINUE
    payload = {"model": "shelf", "temperature": 0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False},
                      "messages": [{"role": "user", "content": content}]}

    results = {}

    def one(index):
        results[index] = post(url, dict(payload))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    statuses = [results[i][0] for i in sorted(results)]
    checker.check("four-complete", statuses == [200, 200, 200, 200], repr(statuses))
    checker.check("four-content",
                  all((message(results[i][1]).get("content") or "").strip()
                      for i in range(4)),
                  repr({i: message(results[i][1]).get("content") for i in range(4)}))

    status, body = post(url, {"model": "shelf", "temperature": 0,
                              "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False},
                              "messages": [{"role": "user",
                                            "content": "What is 2+2? Answer with only the number."}]})
    checker.check("concurrency-health", status == 200
                  and (message(body).get("content") or "").strip() == "4",
                  f"{status} {repr(message(body).get('content'))}")


def main() -> int:
    checker = Checker("server-context-boundary")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    boundary_log = Path(tempfile.gettempdir()) / "shelf-g11c-boundary-compute.log"
    concurrency_log = Path(tempfile.gettempdir()) / "shelf-g11c-concurrency-compute.log"
    for path in (boundary_log, concurrency_log):
        path.unlink(missing_ok=True)
    try:
        processor = load_processor()
        # Boundary uses the production default profile.
        with ServerHarness(use_defaults=True, startup_timeout=360.0,
                           arena_override=test_arena_gib(),
                           env={"SHELF_COMPUTE_LOG": str(boundary_log)}) as h1:
            boundary_cases(checker, processor, f"{h1.base_url}/chat/completions")

        # Concurrency uses a fresh default server: the near-limit request above
        # would otherwise prime the prefix cache and collapse the four prompts
        # to a generation-prompt tail, hiding the scheduler batching.
        with ServerHarness(use_defaults=True, startup_timeout=360.0,
                           arena_override=test_arena_gib(),
                           env={"SHELF_BATCH_TRACE": "1",
                                "SHELF_COMPUTE_LOG": str(concurrency_log)}) as h2:
            concurrency_batch(checker, processor, h2)

        text = concurrency_log.read_text(errors="replace") \
            if concurrency_log.exists() else ""
        steps = [int(m) for m in re.findall(r"SHELF_BATCH_STEP requests=(\d+)", text)]
        checker.check("batch-trace-present", bool(steps), f"steps={len(steps)}")
        checker.check("multi-request-step", bool(steps) and max(steps) >= 2,
                      f"max_requests={max(steps) if steps else 0} sample={steps[:10]}")
        print(f"batch steps={len(steps)} max_requests={max(steps) if steps else 0}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
