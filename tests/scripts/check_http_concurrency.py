#!/usr/bin/env python3
"""HTTP の同時要求（T15）を検証する: 4 並列と直列で応答が一致するか。

サーバを子プロセスで起動し、同じ 4 つのプロンプトを
  * 同時（HTTP ハンドラを並行）→ エンジンは 4 スロットでバッチ
  * 直列（1 つずつ完了を待つ）
で送り、content / finish_reason / usage を比較する。

使い方:
  python3 tests/scripts/check_http_concurrency.py \
      --model models/Qwen3.5-9B --engine build/hip_rel/apps/shelf-compute
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

PROMPTS = [
    "1+1は? 短く",
    "日本の首都は? 一語で",
    "2+2は? 短く",
    "空は何色? 一語で",
]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> bool:
    import aiohttp
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(base + "/healthz") as resp:
                    if resp.status == 200:
                        return True
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)
    return False


async def ask(session, base, prompt: str, max_tokens: int = 3):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "enable_thinking": False}
    async with session.post(base + "/v1/chat/completions", json=body) as resp:
        data = await resp.json()
    if resp.status != 200:
        return {"status": resp.status, "body": data}
    return {
        "status": resp.status,
        "text": data["choices"][0]["message"]["content"],
        "finish": data["choices"][0]["finish_reason"],
        "usage": data["usage"],
    }


async def run(base: str) -> int:
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        concurrent = await asyncio.gather(
            *[ask(session, base, p) for p in PROMPTS])
        sequential = [await ask(session, base, p) for p in PROMPTS]

    ok = True
    for i, (c, s) in enumerate(zip(concurrent, sequential)):
        same = (c.get("text") == s.get("text") and c.get("finish") == s.get("finish")
                and c.get("usage") == s.get("usage"))
        print(f"{'PASS' if same else 'FAIL'}  prompt[{i}] "
              f"concurrent={c.get('text')!r} sequential={s.get('text')!r} "
              f"usage {c.get('usage')} / {s.get('usage')}")
        ok = ok and same
    return 0 if ok else 1


async def main_async(args) -> int:
    base = f"http://127.0.0.1:{args.port}"
    log = tempfile.NamedTemporaryFile(prefix="shelf_conc_", suffix=".log", delete=False)
    cmd = [sys.executable, "server.py", "--engine", args.engine, "--model", args.model,
           "--model-name", args.model_name, "--port", str(args.port),
           "--gpu", str(args.gpu), "--pipeline-parallel", str(args.pp),
           "--prefill-rows", str(args.prefill_rows),
           "--max-inflight", str(args.n)]
    if args.kv_in_host:
        cmd.append("--kv-in-host")
    if args.online_kv_pages:
        cmd += ["--online-kv-pages", str(args.online_kv_pages)]
    print("起動:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=os.path.join(ROOT, "apps", "server"),
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        if not await wait_ready(base, proc, args.max_wait):
            print("サーバが起動しない:", log.name)
            print(open(log.name, encoding="utf-8", errors="replace").read()[-3000:])
            return 1
        rc = await run(base)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print(f"ログ: {log.name}")
    return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HTTP 同時要求（T15）の検証")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--engine", default="build/hip_rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--kv-in-host", action="store_true",
                        help="T14: KV を pinned host に置く")
    parser.add_argument("--online-kv-pages", type=int, default=0,
                        help="T14: 直近 N ページを VRAM にミラーする")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-wait", type=float, default=900.0)
    args = parser.parse_args(argv)
    if not args.port:
        args.port = free_port()
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(args.engine)
    if not args.model_name:
        args.model_name = os.path.basename(args.model.rstrip("/"))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
