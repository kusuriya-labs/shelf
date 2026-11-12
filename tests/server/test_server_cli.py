#!/usr/bin/env python3
"""shelf-server の CLI が prefix cache の option を shelf-compute へ転送することを固定する。"""

from __future__ import annotations

import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2] / "src" / "apps" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shelf_server  # noqa: E402
from support import Checker  # noqa: E402


def value_of(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def main() -> int:
    checker = Checker("server-cli")
    opts = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
        "--prefix-cache", "0",
        "--prefix-cache-concurrency", "6",
        "--prefix-cache-strategy", "slot",
    ])
    argv = shelf_server.build_compute_argv(opts)
    checker.check("enabled-forwarded", value_of(argv, "--prefix-cache") == "0",
                  repr(value_of(argv, "--prefix-cache")))
    checker.check("concurrency-forwarded",
                  value_of(argv, "--prefix-cache-concurrency") == "6",
                  repr(value_of(argv, "--prefix-cache-concurrency")))
    checker.check("strategy-forwarded",
                  value_of(argv, "--prefix-cache-strategy") == "slot",
                  repr(value_of(argv, "--prefix-cache-strategy")))

    defaults = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
    ])
    default_argv = shelf_server.build_compute_argv(defaults)
    checker.check("default-enabled",
                  value_of(default_argv, "--prefix-cache") == "1",
                  repr(value_of(default_argv, "--prefix-cache")))
    checker.check("default-strategy-radix",
                  value_of(default_argv, "--prefix-cache-strategy") == "radix",
                  repr(value_of(default_argv, "--prefix-cache-strategy")))
    checker.check("default-concurrency",
                  value_of(default_argv, "--prefix-cache-concurrency") == "4",
                  repr(value_of(default_argv, "--prefix-cache-concurrency")))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
