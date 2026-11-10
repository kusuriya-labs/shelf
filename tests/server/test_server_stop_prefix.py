#!/usr/bin/env python3
"""実モデルで stop と prefix cache の cancel 後の状態を検証する。"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import Checker, ServerHarness, compute_binary, http_json, http_sse, model_dir, load_processor, ensure_chat_import
codec = ensure_chat_import()

PREFIX = "The quick brown fox jumps over the lazy dog. " * 75


def aligned_prefix():
    processor = load_processor()
    base = PREFIX + "\nReply with the word OK."
    for count in range(64):
        text = base + " padding" * count
        messages = [{"role": "user", "content": text}]
        if len(codec.prompt_ids(processor, messages, enable_thinking=False)) % 16 == 0:
            return text
    raise RuntimeError("page-aligned prompt not found")


def body(prefix, suffix, maximum=96, **extra):
    return {"model": "shelf", "messages": [{"role": "user", "content": prefix},
            {"role": "assistant", "content": "OK"}, {"role": "user", "content": suffix}],
            "temperature": 0, "max_tokens": maximum,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}


def tokens(path):
    return [line for line in path.read_text().splitlines() if line]


def run_case(checker, root, draft):
    label = "dflash2" if draft else "plain"
    log = root / (label + "-server.log")
    token_log = root / (label + "-tokens.log")
    prefix = aligned_prefix()
    with ServerHarness(log_path=log, max_seq_len=2048, arena_gib=31,
                       max_concurrent_requests=2, kv_cache_dtype="fp8",
                       dflash2_model_dir=draft,
                       env={"SHELF_SERVER_TOKEN_LOG": str(token_log)},
                       startup_timeout=420) as server:
        url = server.base_url + "/chat/completions"
        b = body(prefix, "Give a short description of the fox.", 64)
        c = body(prefix, "Count from 1 to 500, one number per line.", 256)
        a = body(prefix, "Count from 1 to 400, one number per line.", 256)
        first = http_json(url, b)
        first_ids = tokens(token_log)[-1]
        checker.check(label + "-baseline-cache-miss", first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, repr(first["usage"]))
        baseline_c = http_json(url, c)
        baseline_c_ids = tokens(token_log)[-1]
        prime = {"model": "shelf", "messages": [{"role": "user", "content": prefix}],
                 "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
        primed = http_json(url, prime)
        checker.check(label + "-priming-completed", bool(primed["choices"]), repr(primed))
        baseline_a = http_json(url, a)
        content_a = baseline_a["choices"][0]["message"]["content"]
        checker.check(label + "-prefix-long", first["usage"]["prompt_tokens"] >= 224, repr(first["usage"]))
        checker.check(label + "-warm-hit", baseline_a["usage"]["prompt_tokens_details"]["cached_tokens"] >= 224, repr(baseline_a["usage"]))
        if not content_a:
            checker.check(label + "-stop-text", False, repr(baseline_a))
            return
        marker = content_a[:1]
        stopped = http_json(url, {**a, "stop": marker})
        checker.check(label + "-stop-cancel", stopped["choices"][0]["finish_reason"] == "stop" and stopped["choices"][0]["message"]["content"] == "", repr(stopped))
        after = http_json(url, b)
        after_ids = tokens(token_log)[-1]
        checker.check(label + "-after-hit", after["usage"]["prompt_tokens_details"]["cached_tokens"] >= 224, repr(after["usage"]))
        checker.check(label + "-after-same-text", after["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"], "")
        checker.check(label + "-after-same-tokens", first_ids == after_ids, f"{first_ids} != {after_ids}")
        started = threading.Event()
        def concurrent():
            text = ""
            for event in http_sse(url, {**c, "stream": True}):
                for choice in event.get("choices", []):
                    chunk = (choice.get("delta") or {}).get("content", "")
                    if chunk:
                        text += chunk
                        started.set()
            return text
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(concurrent)
            if not started.wait(120):
                checker.check(label + "-concurrent-start", False, "no output")
                return
            running_before = not future.done()
            stopped2 = http_json(url, {**a, "stop": marker})
            running_after = not future.done()
            concurrent_text = future.result(timeout=300)
        checker.check(label + "-concurrent-overlap", running_before and running_after, f"before={running_before} after={running_after}")
        checker.check(label + "-concurrent-stop", stopped2["choices"][0]["finish_reason"] == "stop", repr(stopped2))
        checker.check(label + "-concurrent-same-text", concurrent_text == baseline_c["choices"][0]["message"]["content"], "")
        checker.check(label + "-concurrent-same-tokens", tokens(token_log)[-1] == baseline_c_ids, "")
        entries = {int(req): (int(restored), int(generated)) for req, restored, generated in
                   re.findall(r"\[req (\d+)\] prompt=\d+ restored=(\d+) prefill=\d+ gen=(\d+)", log.read_text(errors="replace"))}
        for req in (5, 8):
            restored, generated = entries.get(req, (0, 256))
            checker.check(f"{label}-cancel-{req}-hit", restored >= 224, repr(entries))
            checker.check(f"{label}-cancel-{req}-early", generated < 256, repr(entries))


def main():
    if not model_dir().is_dir() or not compute_binary().is_file():
        print("SKIP: モデルまたは compute binary がありません")
        return 77
    checker = Checker("server-stop-prefix")
    try:
        with tempfile.TemporaryDirectory(prefix="shelf-stop-prefix-") as tmp:
            root = Path(tmp)
            run_case(checker, root, None)
            draft = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
            if draft and Path(draft).is_dir():
                run_case(checker, root, draft)
            else:
                print("SKIP: DFlash2 モデルがありません")
    except Exception as exc:
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
