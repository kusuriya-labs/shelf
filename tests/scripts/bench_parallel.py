#!/usr/bin/env python3
"""N07: 並列時の実効性能測定（Aggregate tok/s・TTFT・TPOT・p95・混在負荷）。

エンジンを JSONL で直接駆動し、レベル 1 / 2 / 4 で
  * 最初のトークンまでの時間（TTFT）
  * トークン間時間（TPOT）の中央値・p95
  * Aggregate tok/s（全トークン ÷ wall）
を測る。`--mixed` では「生成中の要求」の後から長い prefill を投入し、
既存 decode のトークン間隔がどれだけ伸びるか（head-of-line blocking）を見る。

使い方:
  python3 tests/scripts/bench_parallel.py --model models/Qwen3.5-9B-FP8 --levels 1,2,4
  python3 tests/scripts/bench_parallel.py --model ... --mixed --prefix-blocks 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class Engine:
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.tokens: dict[str, list[float]] = {}
        self.tasks: list[asyncio.Task] = []
        self.counter = 0

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        # pong は reader を起動する前に直接読む（取り合いを避ける）。
        await self.send({"op": "ping"})
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine が起動前に終了した")
            text = line.decode("utf-8", "replace").strip()
            if text and json.loads(text).get("event") == "pong":
                break
        self.tasks.append(asyncio.create_task(self._reader()))

    async def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            ev = json.loads(text)
            rid = ev.get("request_id")
            if ev.get("event") == "token":
                self.tokens.setdefault(rid, []).append(time.perf_counter())
            elif ev.get("event") in ("done", "error"):
                fut = self.pending.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(ev)

    async def send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def begin(self, ids: list[int], max_new: int) -> tuple[str, asyncio.Future]:
        self.counter += 1
        rid = f"r{self.counter}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        self.tokens[rid] = []
        await self.send({"op": "generate", "request_id": rid, "input_ids": ids,
                         "max_new_tokens": max_new})
        return rid, fut

    async def generate(self, ids: list[int], max_new: int) -> tuple[str, dict]:
        rid, fut = await self.begin(ids, max_new)
        return rid, await fut

    async def close(self) -> int:
        if self.proc is None:
            return 0
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            rc = await asyncio.wait_for(self.proc.wait(), timeout=120)
        except asyncio.TimeoutError:
            self.proc.kill()
            rc = await self.proc.wait()
        for t in self.tasks:
            t.cancel()
        return rc


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def make_prompt(tag: int, n: int) -> list[int]:
    base = 1000 + (tag % 64) * 700
    return [base + (j * 11) % 4000 for j in range(n)]


async def level_run(args, level: int, tag: int) -> dict:
    eng = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                  "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                  "--max-context", str(args.max_context), "--no-eos"]
                 + (["--prefix-cache-blocks", str(args.prefix_blocks)]
                    if args.prefix_blocks else [])
                 + getattr(args, "extra", []))
    await eng.start()
    try:
        prompts = [make_prompt(tag + i, args.prompt_tokens) for i in range(level)]
        if args.warm:
            # prefix cache を温める（同じプロンプトを直列で 1 回ずつ、1 トークンだけ）。
            for p in prompts:
                await eng.generate(p, 1)
        t0 = time.perf_counter()
        pend = [await eng.begin(p, args.max_new) for p in prompts]
        done = await asyncio.gather(*[f for _, f in pend])
        wall = time.perf_counter() - t0
        ttfts, tpots = [], []
        total_tokens = 0
        for rid, ev in zip([r for r, _ in pend], done):
            ts = eng.tokens.get(rid, [])
            if not ts:
                continue
            ttfts.append(ts[0] - t0)
            total_tokens += len(ts)
            if len(ts) >= 2:
                gaps = [b - a for a, b in zip(ts, ts[1:])]
                tpots.extend(gaps)
        return {
            "level": level, "wall": wall, "tokens": total_tokens,
            "aggregate": total_tokens / wall if wall > 0 else 0.0,
            "ttft_med": statistics.median(ttfts) if ttfts else 0.0,
            "ttft_p95": percentile(ttfts, 0.95),
            "tpot_med": statistics.median(tpots) if tpots else 0.0,
            "tpot_p95": percentile(tpots, 0.95),
        }
    finally:
        await eng.close()


async def mixed_run(args, tag: int) -> dict:
    eng = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                  "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                  "--max-context", str(args.max_context), "--no-eos"]
                 + (["--prefix-cache-blocks", str(args.prefix_blocks)]
                    if args.prefix_blocks else [])
                 + getattr(args, "extra", []))
    await eng.start()
    try:
        # 先行の生成（短い prompt で長めに回す）
        rid, fut = await eng.begin(make_prompt(tag, args.prompt_tokens), args.max_new)
        await asyncio.sleep(args.mixed_delay)
        # 後から長い prefill を投入（同じ max_new は短く）
        long_ids = make_prompt(tag + 100, args.long_prompt_tokens)
        rid2, fut2 = await eng.begin(long_ids, 4)
        done = await asyncio.gather(fut, fut2)
        ts = eng.tokens.get(rid, [])
        gaps = [b - a for a, b in zip(ts, ts[1:])] if len(ts) >= 2 else []
        return {
            "stream_gaps_max": max(gaps) if gaps else 0.0,
            "stream_gaps_p95": percentile(gaps, 0.95),
            "stream_tokens": len(ts),
            "long_ttft": (eng.tokens[rid2][0] - ts[-1]) if eng.tokens.get(rid2) and ts else 0.0,
        }
    finally:
        await eng.close()


async def main_async(args) -> int:
    print(f"model={os.path.basename(args.model)} prompt={args.prompt_tokens} "
          f"max_new={args.max_new} prefix={args.prefix_blocks}")
    print("level   wall(s)  tokens  agg tok/s  TTFT med/p95 (ms)  TPOT med/p95 (ms)")
    tag = 0
    for level in args.levels:
        res = await level_run(args, level, tag)
        tag += level + 3
        print(f"{level:>5}  {res['wall']:7.2f}  {res['tokens']:6d}  {res['aggregate']:8.1f}  "
              f"{res['ttft_med'] * 1000:7.1f} / {res['ttft_p95'] * 1000:7.1f}  "
              f"{res['tpot_med'] * 1000:7.1f} / {res['tpot_p95'] * 1000:7.1f}")
        if args.repeat > 1:
            for _ in range(args.repeat - 1):
                r2 = await level_run(args, level, tag)
                tag += level + 3
                print(f"{'':>5}  {r2['wall']:7.2f}  {r2['tokens']:6d}  "
                      f"{r2['aggregate']:8.1f}  "
                      f"{r2['ttft_med'] * 1000:7.1f} / {r2['ttft_p95'] * 1000:7.1f}  "
                      f"{r2['tpot_med'] * 1000:7.1f} / {r2['tpot_p95'] * 1000:7.1f}")
    if args.mixed:
        res = await mixed_run(args, tag)
        print(f"mixed: stream tokens={res['stream_tokens']} "
              f"max gap={res['stream_gaps_max'] * 1000:.1f} ms "
              f"p95={res['stream_gaps_p95'] * 1000:.1f} ms "
              f"(長い prefill の TTFT {res['long_ttft'] * 1000:.1f} ms)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="N07: 並列時の実効性能測定")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--engine", default="build/hip-rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prefix-blocks", type=int, default=0)
    parser.add_argument("--warm", action="store_true",
                        help="prefix cache を温めてから測る（--prefix-blocks と併用）")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--mixed-delay", type=float, default=0.3)
    parser.add_argument("--long-prompt-tokens", type=int, default=1024)
    parser.add_argument("--engine-arg", action="append", default=[],
                        help="エンジンへそのまま渡す引数（例: --no-interleave）")
    parser.add_argument("--no-interleave", action="store_true",
                        help="N06 の交互実行を無効化する（比較用）")
    args = parser.parse_args(argv)
    args.levels = [int(x) for x in args.levels.split(",") if x]
    args.extra = list(args.engine_arg) + (["--no-interleave"] if args.no_interleave else [])
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(os.path.join(ROOT, args.engine))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
