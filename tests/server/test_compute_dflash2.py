#!/usr/bin/env python3
"""shelf-compute --serve-stdio と DFlash2 speculative decoding の結合検証.

serve 経路で DFlash2 が動作すること、target-only greedy と完全一致すること、
通常の stochastic sampling が成立すること、および constraint 相当の
request field が fail-closed に拒否されることを確認する。
bf16 と fp8 の両方の KV dtype で回す。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (Checker, ComputeHarness, compute_binary,  # noqa: E402
                     model_dir, prompt_ids)

PROMPT = [248041, 77091]
DFLASH2_MODEL_DIR = os.environ.get("SHELF_DFLASH2_MODEL_DIR")
MAX_NEW_TOKENS = 64
PREFIX_PROMPT_TOKENS = 64
PREFIX_SEED_TOKENS = 16
PREFIX_EXTENSION_TOKENS = 8


def paged_prompt(base, length):
    out = []
    while len(out) < length:
        out.extend(base)
    return out[:length]


def collect(harness, request_id):
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


def generated_ids(done) -> list[int]:
    return list(done.get("generated_ids", []))


def arena_used_values(stderr_lines) -> list[int]:
    out = []
    for line in stderr_lines:
        marker = "arena_used="
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].split()[0]
        if tail.isdigit():
            out.append(int(tail))
    return out


def run_single_shot(input_ids, max_new_tokens, *, device, kv_cache_dtype,
                    dflash2=False):
    argv = [
        str(compute_binary()),
        "--model-dir", str(model_dir()),
        "--input-ids-file", "/dev/stdin",
        "--max-new-tokens", str(max_new_tokens),
        "--max-seq-len", "512",
        "--arena-gib", "auto",
        "--device", str(device),
        "--temperature", "0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--seed", "0",
        "--kv-cache-dtype", kv_cache_dtype,
    ]
    if dflash2:
        argv += ["--dflash2-model-dir", DFLASH2_MODEL_DIR, "--dflash2-drafts", "7"]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as proc:
        stdout, stderr = proc.communicate(
            "\n".join(str(t) for t in input_ids) + "\n", timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"single-shot failed: {stderr[-2000:]}")
        for line in stdout.splitlines():
            if line.startswith("GENERATED_IDS="):
                return [int(x) for x in
                        line[len("GENERATED_IDS="):].split(",") if x.strip()]
    raise RuntimeError("GENERATED_IDS not found in single-shot output")


def run_suite(kv_cache_dtype: str, checker: Checker) -> None:
    label = f"kv={kv_cache_dtype}"
    serve_ids = None
    ext_ids = None
    extension = None
    harness = ComputeHarness(max_seq_len=512, arena_gib=31, device=1,
                             kv_cache_dtype=kv_cache_dtype, batch_trace=True,
                             dflash2_model_dir=DFLASH2_MODEL_DIR)
    try:
        harness.start()
        harness.wait_ready()
        checker.check(f"{label} dflash2 enabled",
                      any("DFLASH2_ENABLED=1" in line for line in harness.stderr_lines),
                      str(harness.stderr_lines[-10:]))

        harness.send({
            "op": "generate", "request_id": 1, "input_ids": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
            "prefix_cache_checkpoint_position": 0,
        })
        tokens, done, errors = collect(harness, 1)
        checker.check(f"{label} no error", not errors, str(errors))
        checker.check(f"{label} done received", done is not None)
        if done is None:
            return

        ids = generated_ids(done)
        checker.check(f"{label} tokens emitted", len(tokens) == len(ids),
                      f"{len(tokens)} tokens vs {len(ids)} ids")
        checker.check(f"{label} stream matches generated_ids", tokens == ids)
        checker.check(f"{label} finish_reason", done.get("finish_reason") in
                      ("stop", "length"), str(done.get("finish_reason")))

        harness.send({
            "op": "generate", "request_id": 2, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.7, "top_p": 0.95, "top_k": 0,
            "seed": 7,
        })
        _, stochastic_done, stochastic_errors = collect(harness, 2)
        checker.check(f"{label} stochastic generation ok",
                      not stochastic_errors and stochastic_done is not None,
                      str(stochastic_errors))
        if stochastic_done is not None:
            checker.check(f"{label} stochastic tokens emitted",
                          len(generated_ids(stochastic_done)) > 0,
                          str(generated_ids(stochastic_done)))

        harness.send({
            "op": "generate", "request_id": 3, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
            "grammar": 'root ::= "YES" | "NO"',
        })
        _, grammar_done, grammar_errors = collect(harness, 3)
        checker.check(f"{label} grammar rejected",
                      grammar_done is None and len(grammar_errors) == 1,
                      str(grammar_errors))
        if grammar_errors:
            checker.check(f"{label} grammar rejection code",
                          grammar_errors[0].get("code") == "invalid_argument",
                          str(grammar_errors[0]))
            checker.check(f"{label} grammar rejection names the field",
                          "grammar" in str(grammar_errors[0].get("message", "")),
                          str(grammar_errors[0]))

        harness.send({
            "op": "generate", "request_id": 4, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
            "structural_tag": "{}",
        })
        _, tag_done, tag_errors = collect(harness, 4)
        checker.check(f"{label} structural_tag rejected",
                      tag_done is None and len(tag_errors) == 1,
                      str(tag_errors))

        harness.send({"op": "ping"})
        while True:
            if harness.recv().get("event") == "pong":
                break

        harness.send({
            "op": "generate", "request_id": 5, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
        })
        _, second_done, second_errors = collect(harness, 5)
        checker.check(f"{label} second generation ok",
                      not second_errors and second_done, str(second_errors))
        if second_done is not None:
            checker.check(f"{label} repeat deterministic",
                          generated_ids(second_done) == ids[:8],
                          f"{generated_ids(second_done)} vs {ids[:8]}")

        serve_ids = ids

        prefix_prompt = paged_prompt(PROMPT, PREFIX_PROMPT_TOKENS)
        harness.send({
            "op": "generate", "request_id": 6, "input_ids": prefix_prompt,
            "max_new_tokens": PREFIX_SEED_TOKENS, "temperature": 0.0,
        })
        _, seed_done, seed_errors = collect(harness, 6)
        checker.check(f"{label} prefix seed ok",
                      not seed_errors and seed_done is not None, str(seed_errors))
        if seed_done is None:
            return
        seed_ids = generated_ids(seed_done)
        checker.check(f"{label} prefix seed tokens", len(seed_ids) > 0, str(seed_ids))
        if not seed_ids:
            return
        extension = prefix_prompt + seed_ids[:-1] + [prefix_prompt[-1]]
        harness.send({
            "op": "generate", "request_id": 7, "input_ids": extension,
            "max_new_tokens": PREFIX_EXTENSION_TOKENS, "temperature": 0.0,
        })
        _, ext_done, ext_errors = collect(harness, 7)
        checker.check(f"{label} extension generation ok",
                      not ext_errors and ext_done is not None, str(ext_errors))
        if ext_done is None:
            return
        restored = ext_done.get("restored_tokens")
        checker.check(f"{label} prefix restored_tokens",
                      restored == PREFIX_PROMPT_TOKENS,
                      f"restored={restored} expected={PREFIX_PROMPT_TOKENS}")
        checker.check(f"{label} prefix prefill_tokens",
                      ext_done.get("prefill_tokens") ==
                      len(extension) - PREFIX_PROMPT_TOKENS,
                      f"prefill={ext_done.get('prefill_tokens')} "
                      f"input={len(extension)} restored={restored}")
        ext_ids = generated_ids(ext_done)
        checker.check(f"{label} prefix extension tokens", len(ext_ids) > 0, str(ext_ids))

        harness.send({
            "op": "generate", "request_id": 8, "input_ids": extension,
            "max_new_tokens": PREFIX_EXTENSION_TOKENS, "temperature": 0.0,
        })
        _, ext2_done, ext2_errors = collect(harness, 8)
        checker.check(f"{label} prefix replay ok",
                      not ext2_errors and ext2_done is not None, str(ext2_errors))
        if ext2_done is not None:
            checker.check(f"{label} same prompt does not restore",
                          ext2_done.get("restored_tokens") == 0,
                          f"restored={ext2_done.get('restored_tokens')}")
            checker.check(f"{label} prefix replay deterministic",
                          generated_ids(ext2_done) == ext_ids,
                          f"{generated_ids(ext2_done)} vs {ext_ids}")
    finally:
        harness.close()

    used = arena_used_values(harness.stderr_lines)
    checker.check(f"{label} arena trace captured", len(used) > 0, f"{len(used)} steps")
    if used:
        checker.check(f"{label} arena does not grow per request",
                      max(used) == used[0],
                      f"first={used[0]} max={max(used)} delta={max(used) - used[0]}")

    if serve_ids:
        target_only = run_single_shot(PROMPT, MAX_NEW_TOKENS, device=1,
                                      kv_cache_dtype=kv_cache_dtype,
                                      dflash2=False)
        shot_dflash = run_single_shot(PROMPT, MAX_NEW_TOKENS, device=1,
                                      kv_cache_dtype=kv_cache_dtype,
                                      dflash2=True)
        checker.check(f"{label} serve == target-only greedy",
                      serve_ids == target_only,
                      f"serve={serve_ids} target={target_only}")
        checker.check(f"{label} serve == single-shot dflash2",
                      serve_ids == shot_dflash,
                      f"serve={serve_ids} shot={shot_dflash}")


def main() -> int:
    checker = Checker("compute-dflash2")
    for kv_cache_dtype in ("bf16", "fp8"):
        run_suite(kv_cache_dtype, checker)
    return checker.done()


if __name__ == "__main__":
    if not DFLASH2_MODEL_DIR:
        print("SKIP: SHELF_DFLASH2_MODEL_DIR is not set")
        raise SystemExit(77)
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        raise SystemExit(77)
    raise SystemExit(main())
