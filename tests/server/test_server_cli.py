#!/usr/bin/env python3
"""shelf-server の CLI が prefix cache の option を shelf-compute へ転送することを固定する。"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

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

    interval = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
        "--prefill-decode-interval", "4",
    ])
    checker.check("prefill-decode-interval-forwarded",
                  value_of(shelf_server.build_compute_argv(interval),
                           "--prefill-decode-interval") == "4",
                  repr(value_of(shelf_server.build_compute_argv(interval),
                                "--prefill-decode-interval")))
    disabled = shelf_server.parse_args([
        "--model-dir", "models/unused",
        "--device", "0",
        "--prefill-decode-interval", "0",
    ])
    checker.check("prefill-decode-interval-zero-forwarded",
                  value_of(shelf_server.build_compute_argv(disabled),
                           "--prefill-decode-interval") == "0",
                  "zero must be forwarded explicitly")
    for bad in ("-1", "1025"):
        try:
            shelf_server.parse_args([
                "--model-dir", "models/unused",
                "--device", "0",
                "--prefill-decode-interval", bad,
            ])
            checker.check(f"prefill-decode-interval-reject-{bad}", False, "accepted")
        except SystemExit:
            checker.check(f"prefill-decode-interval-reject-{bad}", True)

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
    checker.check("既定では decode 割り込みを転送しない",
                  "--prefill-decode-interval" not in default_argv,
                  repr(default_argv))
    hy4 = shelf_server.parse_args([
        "--model-dir", "body", "--placement", "owners.json", "--engram-dir", "fp8",
        "--dfly-dir", "draft", "--max-seq-len", "2304", "--compute", "shelf-compute",
    ])
    checker.check("hy4-ep3", shelf_server.build_compute_argv(hy4) == [
        "shelf-compute", "hy4", "serve", "--model-dir", "body", "--engram-dir", "fp8",
        "--placement", "owners.json", "--max-seq-len", "2304", "--chunk", "1024",
        "--max-concurrent-requests", "4", "--dfly-dir", "draft", "--dfly-drafts", "4",
    ])
    checker.check("hy4-meta-default", shelf_server.speculative_meta(hy4) == {"method": "dfly2", "drafts": 4})
    for k in (2, 3, 4, 5):
        opts = shelf_server.parse_args([
            "--model-dir", "body", "--placement", "owners.json", "--dfly-dir", "draft",
            "--dfly-drafts", str(k),
        ])
        checker.check(f"hy4-drafts-argv-{k}", value_of(shelf_server.build_compute_argv(opts), "--dfly-drafts") == str(k))
        checker.check(f"hy4-meta-{k}", shelf_server.speculative_meta(opts) == {"method": "dfly2", "drafts": k})
    old_env = os.environ.get("SHELF_HY4_DFLY_DRAFTS")
    try:
        os.environ["SHELF_HY4_DFLY_DRAFTS"] = "3"
        env_opts = shelf_server.parse_args(["--model-dir", "body", "--placement", "owners.json", "--dfly-dir", "draft"])
        checker.check("hy4-env-drafts", value_of(shelf_server.build_compute_argv(env_opts), "--dfly-drafts") == "3")
        checker.check("hy4-env-meta", shelf_server.speculative_meta(env_opts) == {"method": "dfly2", "drafts": 3})
        os.environ["SHELF_HY4_DFLY_DRAFTS"] = "6"
        try:
            shelf_server.parse_args(["--model-dir", "body", "--placement", "owners.json"])
            checker.check("hy4-env-reject-6", False)
        except SystemExit:
            checker.check("hy4-env-reject-6", True)
    finally:
        if old_env is None:
            os.environ.pop("SHELF_HY4_DFLY_DRAFTS", None)
        else:
            os.environ["SHELF_HY4_DFLY_DRAFTS"] = old_env
    for bad in ("0", "1", "6", "-1", "abc"):
        try:
            shelf_server.parse_args(["--model-dir", "body", "--placement", "owners.json", "--dfly-drafts", bad])
            checker.check(f"hy4-reject-{bad}", False)
        except SystemExit:
            checker.check(f"hy4-reject-{bad}", True)
    hy4_pc_off = shelf_server.parse_args([
        "--model-dir", "body", "--placement", "owners.json", "--engram-dir", "fp8",
        "--dfly-dir", "draft", "--max-seq-len", "2304", "--compute", "shelf-compute",
        "--prefix-cache", "0",
    ])
    checker.check("hy4-prefix-cache-off",
                  value_of(shelf_server.build_compute_argv(hy4_pc_off), "--prefix-cache") == "0")
    hy4_strat_off = shelf_server.parse_args([
        "--model-dir", "body", "--placement", "owners.json", "--prefix-cache-strategy", "off",
    ])
    checker.check("hy4-strategy-off",
                  value_of(shelf_server.build_compute_argv(hy4_strat_off), "--prefix-cache") == "0")
    qwen_strat_off = shelf_server.parse_args([
        "--model-dir", "models/unused", "--device", "0", "--prefix-cache-strategy", "off",
    ])
    qwen_strat_off_argv = shelf_server.build_compute_argv(qwen_strat_off)
    checker.check("qwen-strategy-off-disables-cache",
                  value_of(qwen_strat_off_argv, "--prefix-cache") == "0")
    checker.check("qwen-strategy-off-compute-strategy",
                  value_of(qwen_strat_off_argv, "--prefix-cache-strategy") == "radix")
    hy4_opts = shelf_server.parse_args([
        "--model-dir", "body", "--placement", "owners.json", "--kv-pool-tokens", "65536",
        "--prefill-decode-interval", "2",
    ])
    hy4_opts_argv = shelf_server.build_compute_argv(hy4_opts)
    checker.check("hy4-kv-pool-tokens", value_of(hy4_opts_argv, "--kv-pool-tokens") == "65536")
    checker.check("hy4-prefill-decode-interval", value_of(hy4_opts_argv, "--prefill-decode-interval") == "2")
    for chunk in (2048, 4096):
        opts = shelf_server.parse_args(["--model-dir", "body", "--placement", "owners.json", "--prefill-chunk", str(chunk)])
        checker.check("hy4-chunk", value_of(shelf_server.build_compute_argv(opts), "--chunk") == str(chunk))
    for ep, tp, slots in ((3, 1, 16), (4, 1, 4), (2, 2, 1)):
        opts = shelf_server.parse_args([
            "--model-dir", "body", "--placement", "owners.json", "--expert-parallel", str(ep),
            "--tensor-parallel", str(tp), "--device", ",".join(map(str, range(ep * tp))),
            "--max-concurrent-requests", str(slots),
        ])
        argv = shelf_server.build_compute_argv(opts)
        checker.check("hy4-parallel", value_of(argv, "--expert-parallel") == str(ep))
        checker.check("hy4-tp", value_of(argv, "--tensor-parallel") == (str(tp) if tp > 1 else None))
        checker.check("hy4-slots", value_of(argv, "--max-concurrent-requests") == str(slots))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
