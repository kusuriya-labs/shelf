#!/usr/bin/env python3
"""Prefix Cache（T16）の E2E 検査。

同じプロンプトを 2 回流し、2 回目が
  * キャッシュヒットしている（engine の stderr に `[prefix] hit`）
  * 生成列が 1 回目と一致する
  * prefill が短くなり、時間が短い
を確認する。

使い方:
  python3 tests/scripts/check_prefix_cache.py \
      --model models/Qwen3.5-9B --blocks 512
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prefix Cache の E2E 検査（T16）")
    parser.add_argument("--engine", default="build/hip_rel/apps/shelf-compute")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--blocks", type=int, default=512, help="prefix cache の追加ブロック数")
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--kv-in-host", action="store_true",
                        help="T14: KV を pinned host に置く")
    parser.add_argument("--online-kv-pages", type=int, default=0,
                        help="T14: 直近 N ページを VRAM にミラーする")
    args = parser.parse_args(argv)

    engine = os.path.abspath(os.path.join(ROOT, args.engine))
    model = os.path.abspath(os.path.join(ROOT, args.model))
    prompt = [9707] * args.prompt_tokens

    proc = subprocess.Popen(
        [engine, "--model", model, "--max-batch", "4",
         "--prefix-cache-blocks", str(args.blocks)]
        + (["--kv-in-host"] if args.kv_in_host else [])
        + (["--online-kv-pages", str(args.online_kv_pages)] if args.online_kv_pages else []),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1)

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def read_done():
        while True:
            ev = json.loads(proc.stdout.readline())
            if ev.get("event") == "done":
                return ev
            if ev.get("event") == "error":
                raise RuntimeError(ev)

    send({"op": "ping"})
    while True:
        ev = json.loads(proc.stdout.readline())
        if ev.get("event") == "pong":
            break

    def generate(rid):
        t0 = time.perf_counter()
        send({"op": "generate", "request_id": rid, "input_ids": prompt,
              "max_new_tokens": args.max_tokens})
        done = read_done()
        return time.perf_counter() - t0, done["generated_ids"]

    cold_ms, cold_ids = generate("cold")
    warm_ms, warm_ids = generate("warm")
    proc.stdin.close()
    err = proc.stderr.read()
    proc.wait(timeout=30)

    hits = [line for line in err.splitlines() if "[prefix] hit" in line]
    ok = True
    print(f"cold: {cold_ms * 1000:.0f} ms {cold_ids}")
    print(f"warm: {warm_ms * 1000:.0f} ms {warm_ids}")
    if cold_ids != warm_ids:
        print("FAIL: 生成列が一致しない")
        ok = False
    if not hits:
        print("FAIL: prefix hit のログが無い")
        ok = False
    else:
        print("hit:", hits[-1])
    if not (warm_ms < cold_ms * 0.8):
        print(f"FAIL: 2 回目が速くならない（{warm_ms / cold_ms:.2f}x）")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
