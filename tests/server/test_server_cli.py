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
        "--prefix-cache-checkpoints", "6",
        "--prefix-cache-save-interval", "4096",
        "--prefix-cache-strategy", "slot",
    ])
    argv = shelf_server.build_compute_argv(opts)
    checker.check("enabled-forwarded", value_of(argv, "--prefix-cache") == "0",
                  repr(value_of(argv, "--prefix-cache")))
    checker.check("checkpoints-forwarded",
                  value_of(argv, "--prefix-cache-checkpoints") == "6",
                  repr(value_of(argv, "--prefix-cache-checkpoints")))
    checker.check("save-interval-forwarded",
                  value_of(argv, "--prefix-cache-save-interval") == "4096",
                  repr(value_of(argv, "--prefix-cache-save-interval")))
    checker.check("strategy-forwarded",
                  value_of(argv, "--prefix-cache-strategy") == "slot",
                  repr(value_of(argv, "--prefix-cache-strategy")))

    chunk = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
        "--prefill-chunk", "1024",
    ])
    chunk_argv = shelf_server.build_compute_argv(chunk)
    checker.check("prefill-chunk-forwarded",
                  value_of(chunk_argv, "--prefill-chunk") == "1024",
                  repr(value_of(chunk_argv, "--prefill-chunk")))

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
    checker.check("default-checkpoints",
                  value_of(default_argv, "--prefix-cache-checkpoints") == "16",
                  repr(value_of(default_argv, "--prefix-cache-checkpoints")))
    checker.check("default-save-interval",
                  value_of(default_argv, "--prefix-cache-save-interval") == "16384",
                  repr(value_of(default_argv, "--prefix-cache-save-interval")))
    checker.check("既定の prefill chunk を転送する",
                  value_of(default_argv, "--prefill-chunk") == "8192",
                  repr(value_of(default_argv, "--prefill-chunk")))

    ngram_opts = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
        "--ngram",
        "--ngram-drafts", "15",
        "--ngram-min-match", "6",
        "--ngram-max-match", "32",
        "--ngram-stats", "1",
    ])
    ngram_argv = shelf_server.build_compute_argv(ngram_opts)
    checker.check("ngram-flag-forwarded", "--ngram" in ngram_argv, repr(ngram_argv))
    checker.check("ngram-drafts-forwarded",
                  value_of(ngram_argv, "--ngram-drafts") == "15",
                  repr(value_of(ngram_argv, "--ngram-drafts")))
    checker.check("ngram-min-match-forwarded",
                  value_of(ngram_argv, "--ngram-min-match") == "6",
                  repr(value_of(ngram_argv, "--ngram-min-match")))
    checker.check("ngram-max-match-forwarded",
                  value_of(ngram_argv, "--ngram-max-match") == "32",
                  repr(value_of(ngram_argv, "--ngram-max-match")))
    checker.check("ngram-stats-forwarded",
                  value_of(ngram_argv, "--ngram-stats") == "1",
                  repr(value_of(ngram_argv, "--ngram-stats")))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
