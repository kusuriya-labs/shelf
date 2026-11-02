#!/usr/bin/env python3
"""N05: 長時間負荷・異常系の検証。

モード:
  soak   : エンジンへ多数の要求を流し続ける（直列・4 並列・cancel・prefix hit を混在）。
           VRAM の増加、エラー、HSA fault、正常終了を確認する。
  crash  : 生成中にエンジンを SIGKILL し、待っている要求がハングせずエラーで返ること、
           新しいエンジン（再起動相当）が正常に動くことを確認する。
  oom    : `--vram-budget-mib` を極小にして起動し、クラッシュではなく
           きれいな insufficient_memory で終了することを確認する。

使い方:
  python3 tests/scripts/check_stability.py --model models/Qwen3.5-9B-FP8 --mode soak --requests 300
  python3 tests/scripts/check_stability.py --model ... --mode crash
  python3 tests/scripts/check_stability.py --model ... --mode oom
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROMPT_TOKENS = 32
MAX_NEW = 4


class Engine:
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.logs: list[str] = []
        self.bad_lines = 0
        self.tasks: list[asyncio.Task] = []
        self.counter = 0

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.tasks.append(asyncio.create_task(self._drain()))
        await self.send({"op": "ping"})
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine が起動前に終了した")
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            ev = json.loads(text)
            if ev.get("event") == "pong":
                self.tasks.append(asyncio.create_task(self._reader()))
                return

    async def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                self.bad_lines += 1
                continue
            if ev.get("event") not in ("done", "error"):
                continue   # token などは終端ではない
            rid = ev.get("request_id")
            fut = self.pending.pop(rid, None)
            if fut is not None and not fut.done():
                fut.set_result(ev)

    async def _drain(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            self.logs.append(line.decode("utf-8", "replace").rstrip("\n"))

    async def send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def begin(self, ids: list[int], max_new: int) -> tuple[str, asyncio.Future]:
        self.counter += 1
        rid = f"r{self.counter}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.send({"op": "generate", "request_id": rid, "input_ids": ids,
                         "max_new_tokens": max_new})
        return rid, fut

    async def generate(self, ids: list[int], max_new: int) -> dict:
        _, fut = await self.begin(ids, max_new)
        return await fut

    async def cancel(self, rid: str) -> None:
        await self.send({"op": "cancel", "request_id": rid})

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
            try:
                await asyncio.wait_for(t, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()
        return rc

    def faults(self) -> list[str]:
        return [l for l in self.logs
                if "hsa_status_error" in l.lower() or "memory fault" in l.lower()]


failures: list[str] = []


def check(cond: bool, what: str, detail: str = "") -> None:
    if cond:
        print(f"PASS  {what}")
    else:
        print(f"FAIL  {what}{(' / ' + detail) if detail else ''}")
        failures.append(what)


def vram_mib(gpu: int) -> float | None:
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=10)
        data = json.loads(out.stdout)
        for key, val in data.items():
            if f"GPU[{gpu}]" in key or f"card{gpu}" in key.lower():
                return float(val.get("VRAM Total Used Memory (B)", 0)) / (1 << 20)
        # 形式違いのフォールバック: 最初の GPU
        for val in data.values():
            if isinstance(val, dict) and "VRAM Total Used Memory (B)" in val:
                return float(val["VRAM Total Used Memory (B)"]) / (1 << 20)
    except Exception:  # noqa: BLE001
        return None
    return None


def prompt_for(i: int, n: int) -> list[int]:
    base = 1000 + (i % 32) * 500
    return [base + (j * 13) % 4000 for j in range(n)]


async def mode_soak(args) -> int:
    engine = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                     "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                     "--max-context", str(args.max_context), "--no-eos",
                     "--prefix-cache-blocks", "32"]
                    + (["--kv-in-host"] if args.kv_in_host else [])
                    + (["--online-kv-pages", str(args.online_kv_pages)]
                       if args.online_kv_pages else []))
    print(f"起動: {' '.join(engine.argv)}")
    expected: dict[tuple, list[int]] = {}
    mismatches = 0
    await engine.start()
    done = cancelled = errors = 0
    per_req: list[float] = []
    vram: list[float] = []
    t0 = time.perf_counter()
    iters = max(1, args.requests // 4)
    issued = 0
    try:
        for it in range(iters):
            # 2 回に 1 回は同じプロンプト（prefix hit を混ぜる）。
            ids = prompt_for(it // 2, args.prompt_tokens)
            pend = [await engine.begin(ids, MAX_NEW) for _ in range(4)]
            issued += 4
            if it % 8 == 0:
                await asyncio.sleep(0.01)
                await engine.cancel(pend[1][0])
            t = time.perf_counter()
            evs = await asyncio.gather(*[f for _, f in pend])
            per_req.append((time.perf_counter() - t) / 4.0)
            key = tuple(ids)
            for ev in evs:
                if ev.get("event") == "error":
                    errors += 1
                elif ev.get("finish_reason") == "cancelled":
                    cancelled += 1
                else:
                    done += 1
                    got = ev.get("generated_ids")
                    if key in expected:
                        if got != expected[key]:
                            mismatches += 1
                            if mismatches <= 3:
                                print(f"  !! 同一プロンプトで不一致: {got} vs {expected[key]}")
                    else:
                        expected[key] = got
            if (it + 1) % 25 == 0:
                used = vram_mib(args.gpu)
                if used is not None:
                    vram.append(used)
                print(f"  {it + 1} iter: done={done} cancelled={cancelled} errors={errors} "
                      f"vram={used if used is not None else '-'} MiB", flush=True)
    finally:
        rc = await engine.close()
    wall = time.perf_counter() - t0
    print(f"soak: {done} 完了 / {cancelled} cancel / {errors} error, wall {wall:.1f}s, "
          f"rc={rc}, 行数={len(engine.logs)}")
    check(errors == 0, f"エラーがない（{errors}）")
    check(mismatches == 0, f"同じプロンプトは常に同じ列（不一致 {mismatches}）")
    check(rc == 0, f"正常終了（rc={rc}）")
    check(not engine.faults(), "HSA fault / memory fault がない",
          "; ".join(engine.faults()[:3]))
    check(engine.bad_lines == 0, "stdout は JSONL のみ")
    check(done + cancelled == issued, f"要求がすべて終端した（{done + cancelled}/{issued}）")
    if vram:
        base = vram[0]
        growth = max(vram) - base
        print(f"VRAM: 初回 {base:.0f} MiB / 最大 {max(vram):.0f} MiB / 増加 {growth:.0f} MiB")
        check(growth <= args.max_vram_growth_mib,
              f"VRAM の増加が {args.max_vram_growth_mib} MiB 以下（{growth:.0f}）")
    return 0 if not failures else 1


async def mode_crash(args) -> int:
    engine = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                     "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                     "--max-context", str(args.max_context), "--no-eos"])
    await engine.start()
    ids = prompt_for(0, args.prompt_tokens)
    pend = [await engine.begin(ids, 512) for _ in range(4)]
    await asyncio.sleep(0.5)
    pid = engine.proc.pid
    os.kill(pid, 9)
    outcomes = []
    for _, fut in pend:
        try:
            ev = await asyncio.wait_for(fut, timeout=30.0)
            outcomes.append(ev.get("event"))
        except asyncio.TimeoutError:
            outcomes.append("timeout")
    rc = await engine.close()
    print(f"crash: kill -9 後のイベント={outcomes} rc={rc}")
    check("timeout" not in outcomes, "kill 後も待っている要求がハングしない")
    check(all(o in ("error", "done") for o in outcomes),
          "待っている要求がエラー（または done）で返る", str(outcomes))
    check(not engine.faults(), "異常終了で HSA fault が出ない")

    # 再起動相当: 新しいプロセスで普通に動く
    engine2 = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                      "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                      "--max-context", str(args.max_context), "--no-eos"])
    await engine2.start()
    ev = await engine2.generate(ids, MAX_NEW)
    rc2 = await engine2.close()
    check(ev.get("event") == "done" and ev.get("generated_ids"), "再起動後に生成できる")
    check(rc2 == 0, f"再起動側も正常終了（rc={rc2}）")
    return 0 if not failures else 1


def mode_oom(args) -> int:
    engine = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                     "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                     "--max-context", str(args.max_context), "--vram-budget-mib", "1"])
    proc = subprocess.run(engine.argv, input=b"", capture_output=True, timeout=600)
    out = proc.stderr.decode("utf-8", "replace")
    print(f"oom: rc={proc.returncode}")
    print("\n".join(out.splitlines()[-6:]))
    check(proc.returncode != 0, "極小予算では起動しない（非 0 で終了）")
    check(any(k in out for k in ("予算", "不足", "budget", "insufficient_memory")),
          "予算不足の理由が stderr に出る")
    check("HSA_STATUS_ERROR_MEMORY_FAULT" not in out, "クラッシュ（HSA fault）ではない")
    return 0 if not failures else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="N05: 長時間負荷・異常系の検証")
    parser.add_argument("--mode", choices=["soak", "crash", "oom"], default="soak")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--engine", default="build/hip-rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--max-vram-growth-mib", type=float, default=256.0)
    parser.add_argument("--kv-in-host", action="store_true", help="soak: KV を host に置く")
    parser.add_argument("--online-kv-pages", type=int, default=0,
                        help="soak: online KV のミラー数")
    args = parser.parse_args(argv)
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(os.path.join(ROOT, args.engine))
    if args.mode == "soak":
        return asyncio.run(mode_soak(args))
    if args.mode == "crash":
        return asyncio.run(mode_crash(args))
    return mode_oom(args)


if __name__ == "__main__":
    raise SystemExit(main())
