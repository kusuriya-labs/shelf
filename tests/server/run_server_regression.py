#!/usr/bin/env python3
"""Canonical Shelf server regression runner.

The manifest is explicit, not a glob. `support.py` must not be pulled into the
default regression run by accident, so every participating test is named here.

Usage:
    python3 tests/server/run_server_regression.py --all
    python3 tests/server/run_server_regression.py --group tools --group agent
    python3 tests/server/run_server_regression.py --list

Each selected test runs as a subprocess and must exit 0. A non-zero exit is a
failure, including a missing prerequisite; nothing is silently treated as a
skip (a test that exits 77 reports itself as skipped and is not a failure).

Environment (inherited by every test):
    SHELF_BUILD_DIR       default: <repo>/build
    SHELF_MODEL_DIR       model directory used by every server test
    SHELF_TEST_DEVICE     GPU device index override
    SHELF_TEST_ARENA_GIB  arena capacity override for harness tests
    SHELF_TEST_WORKING_SET_RESERVE_GIB
                               working-set reserve override for harness tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = Path(__file__).resolve().parent

# Group order is the canonical `--all` order. A test appears in exactly one
# group. Intentionally excluded from the manifest:
#   support.py                  shared helpers, not a test
GROUPS = OrderedDict([
    ("core", [
        "test_compute_client_reader.py",
        "test_compute_pids_scope.py",
        "test_compute_service.py",
        "test_compute_dflash2.py",
        "test_compute_dflash2_octave.py",
        "test_compute_dflash2_unaligned.py",
        "test_compute_dflash2_prefix_long.py",
        "test_server_dflash2.py",
        "test_server_models.py",
        "test_server_meta.py",
        "test_server_chat.py",
        "test_server_responses.py",
        "test_server_stream.py",
        "test_server_eos_contract.py",
        "test_plamo3_template.py",
        "test_inference_budget.py",
    ]),
    ("tools", [
        "test_server_tools.py",
        "test_server_stream_tools.py",
        "test_server_tool_regression.py",
    ]),
    ("reasoning", [
        "test_reasoning_parser.py",
        "test_reasoning_request.py",
        "test_compute_thinking_budget.py",
        "test_server_reasoning.py",
        "test_server_reasoning_stream.py",
        "test_server_reasoning_tools.py",
    ]),
    ("unsupported", [
        "test_server_unsupported.py",
        "test_server_stop.py",
        "test_server_stop_prefix.py",
        "test_compute_prefix_contract.py",
    ]),
    ("concurrency-cancel", [
        "test_compute_cancel.py",
        "test_compute_concurrency.py",
        "test_server_concurrency.py",
        "test_server_stream_cancel.py",
        "test_server_context_boundary.py",
    ]),
    ("agent", [
        "test_agent_opencode.py",
    ]),
])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[],
                        help="group to run (repeatable)")
    parser.add_argument("--all", action="store_true", help="run every group")
    parser.add_argument("--list", action="store_true", help="list groups and exit")
    parser.add_argument("--tail", type=int, default=40,
                        help="output lines to show for a failing test")
    return parser.parse_args(argv)


def selected(args):
    if args.all:
        return list(GROUPS)
    if args.group:
        unknown = [g for g in args.group if g not in GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s): {', '.join(unknown)}")
        return args.group
    raise SystemExit("select at least one --group or --all")


def test_env(extra=None):
    env = dict(os.environ)
    env.setdefault("SHELF_BUILD_DIR", str(REPO_ROOT / "build"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def run_test(name, env, tail):
    path = SERVER_DIR / name
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-I", "-u", str(path)],
        cwd=str(SERVER_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.monotonic() - started
    output = proc.stdout or ""
    if proc.returncode == 77:
        print(f"[SKIP] {name} ({duration:.1f}s)")
        return "skip"
    if proc.returncode != 0:
        lines = output.rstrip().splitlines()
        shown = "\n".join(lines[-tail:]) if tail > 0 else ""
        print(shown)
        print(f"[FAIL] {name} (rc={proc.returncode}, {duration:.1f}s)")
        return "fail"
    for line in output.splitlines():
        if "skip" in line.lower():
            print(f"  {name}: {line.strip()}")
    summary = ""
    for line in reversed(output.splitlines()):
        if "passed=" in line:
            summary = line.strip()
            break
    print(f"[PASS] {name} ({duration:.1f}s) {summary}")
    return "pass"


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        for group, tests in GROUPS.items():
            print(f"{group}: {len(tests)}")
            for test in tests:
                print(f"  {test}")
        return 0

    groups = selected(args)
    env = test_env()

    failures = []
    passed = []
    skipped = []
    print(f"server regression: groups={','.join(groups)}")
    for group in groups:
        print(f"===== group {group} =====")
        for test in GROUPS[group]:
            result = run_test(test, env, args.tail)
            if result == "pass":
                passed.append(test)
            elif result == "skip":
                skipped.append(test)
            else:
                failures.append(test)

    print(f"===== SUMMARY passed={len(passed)} skipped={len(skipped)} failed={len(failures)}")
    for test in skipped:
        print(f"  SKIP {test}")
    if failures:
        for test in failures:
            print(f"  FAIL {test}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
