#!/usr/bin/env python3
"""バッチ decode の数値一貫性（N01）を検証する。

同じプロンプトを N 並列で同時に投げ、直列（1 並列）の結果と **生成列の完全一致**
を比較する。対応する行がスロットごとに違っても、同じ入力なら同じ出力になること
（= バッチ GEMM と GEMV の数値経路が揃っていること）を確かめる。

あわせて実効性能（直列の合計時間 vs 並列の時間、Aggregate tok/s）を出す。
数値一貫性と性能のトレードオフを同じ条件で測るための道具。

使い方:
  python3 tests/scripts/check_batch_parity.py \
      --model /opt/zen/work/models/Qwen3.5-9B-FP8 --max-new 48 --levels 1,2,4

`--decode1-max-rows 0` を渡すとサーバ側の per-row decode1 を切り、バッチ GEMM の
まま比較できる（A/B 用。server.py にそのまま渡す）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

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


async def ask(session, base: str, prompt: str, max_tokens: int):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "enable_thinking": False}
    t0 = time.perf_counter()
    async with session.post(base + "/v1/chat/completions", json=body) as resp:
        data = await resp.json()
    dt = time.perf_counter() - t0
    if resp.status != 200:
        return {"status": resp.status, "body": data, "sec": dt}
    choice = data["choices"][0]
    return {
        "status": resp.status,
        "text": choice["message"]["content"],
        "finish": choice["finish_reason"],
        "usage": data["usage"],
        "sec": dt,
    }


def short(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)[:160]


async def run_level(session, base: str, prompt: str, n: int, max_tokens: int,
                    concurrent: bool) -> list[dict]:
    if concurrent:
        return list(await asyncio.gather(
            *[ask(session, base, prompt, max_tokens) for _ in range(n)]))
    out = []
    for _ in range(n):
        out.append(await ask(session, base, prompt, max_tokens))
    return out


async def run(base: str, args, max_tokens: int) -> int:
    import aiohttp

    failures: list[str] = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        for prompt in ([args.prompt] if args.prompt else PROMPTS):
            serial = await run_level(session, base, prompt, 1, max_tokens, False)
            ref = serial[0]
            if ref.get("status") != 200:
                failures.append(f"{prompt!r}: 直列が失敗 {short(ref)}")
                print(f"FAIL  直列 {prompt!r}: {short(ref)}")
                continue
            ref_tokens = ref["usage"]["completion_tokens"]
            serial_sec = ref["sec"]
            print(f"基準  {prompt!r}: {ref_tokens} tok / {serial_sec:.2f}s "
                  f"(TPOT {1000.0 * serial_sec / max(1, ref_tokens):.1f} ms)")
            print(f"      出力 {ref['text']!r}")
            for level in args.levels:
                for concurrent in (True, False):
                    label = "並列" if concurrent else "直列"
                    t0 = time.perf_counter()
                    rows = await run_level(session, base, prompt, level, max_tokens,
                                           concurrent)
                    wall = time.perf_counter() - t0
                    bad = [r for r in rows
                           if r.get("status") != 200
                           or r.get("text") != ref["text"]
                           or r.get("finish") != ref["finish"]]
                    total_tok = sum(r.get("usage", {}).get("completion_tokens", 0)
                                    for r in rows)
                    agg = total_tok / wall if wall > 0 else 0.0
                    ok = not bad
                    print(f"{'PASS' if ok else 'FAIL'}  {label} {level} 並列: "
                          f"一致 {len(rows) - len(bad)}/{len(rows)} / "
                          f"wall {wall:.2f}s / aggregate {agg:.1f} tok/s "
                          f"(基準比 {wall / (serial_sec * level):.2f}x)")
                    for i, r in enumerate(bad):
                        idx = rows.index(r)
                        print(f"      row[{idx}] {short(r)}")
                        print(f"      基準 {ref['text']!r}")
                        a, b = ref["text"], r.get("text", "")
                        at = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                                  min(len(a), len(b)))
                        print(f"      最初の相違 at {at}: ref={a[max(0, at-12):at+12]!r} "
                              f"got={b[max(0, at-12):at+12]!r}")
                    if level > 1 and (bad or agg <= 0.0):
                        failures.append(f"{prompt!r}: {label} {level} 並列不一致")
    if failures:
        print("\n不一致:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nすべて一致")
    return 0


async def main_async(args) -> int:
    base = f"http://127.0.0.1:{args.port}"
    log = tempfile.NamedTemporaryFile(prefix="shelf_batch_", suffix=".log", delete=False)
    cmd = [sys.executable, "server.py", "--engine", args.engine, "--model", args.model,
           "--model-name", args.model_name, "--port", str(args.port),
           "--gpu", str(args.gpu), "--pipeline-parallel", str(args.pp),
           "--prefill-rows", str(args.prefill_rows),
           "--max-inflight", str(max(args.levels))]
    if args.decode1_max_rows is not None:
        cmd += ["--decode1-max-rows", str(args.decode1_max_rows)]
    if not args.stop_at_eos:
        cmd += ["--no-eos"]
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
        rc = await run(base, args, args.max_new)
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
    parser = argparse.ArgumentParser(description="バッチ decode の数値一貫性（N01）")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--engine", default="build/hip-rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--levels", default="2,4",
                        help="比較する並列数（カンマ区切り）")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--prompt", default="", help="このプロンプトだけで比較する")
    parser.add_argument("--decode1-max-rows", type=int, default=None,
                        help="サーバへ渡す（0 = per-row decode1 を切る）")
    parser.add_argument("--stop-at-eos", action="store_true",
                        help="EOS で止める（既定は --no-eos で max_new まで生成）")
    parser.add_argument("--kv-in-host", action="store_true")
    parser.add_argument("--online-kv-pages", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-wait", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    args.levels = [int(x) for x in args.levels.split(",") if x]
    if not args.levels:
        raise SystemExit("--levels が空")
    if not args.port:
        args.port = free_port()
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(args.engine)
    if not args.model_name:
        args.model_name = os.path.basename(args.model.rstrip("/"))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
