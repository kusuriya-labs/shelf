#!/usr/bin/env python3
"""N04: メモリ経路の組み合わせ検証。

Host KV・Online KV・Prefix Cache・重みの Offload を組み合わせて起動し、
同じプロンプトで

  A. 直列 1 本（基準）
  B. 4 並列（バッチ decode）
  C. 直列の 2 回目（prefix 有効時は hit 経路）
  D. 4 並列のうち 2 本を cancel し、その後に直列で再実行（スロット再利用）

を回し、**生成列が全構成で一致**すること、エラー・HSA fault が出ないこと、
終了時に確保量（device / host_weights / host_kv）が報告されることを確認する。

使い方:
  python3 tests/scripts/check_memory_paths.py --model /opt/zen/work/models/Qwen3.5-9B-FP8
  python3 tests/scripts/check_memory_paths.py --model ... --only baseline,prefix
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

PROMPT_TOKENS = 96
MAX_NEW = 8

CONFIGS: list[tuple[str, list[str]]] = [
    ("baseline", []),
    ("host-kv", ["--kv-in-host"]),
    ("host-kv+online2", ["--kv-in-host", "--online-kv-pages", "2"]),
    ("host-kv+online4", ["--kv-in-host", "--online-kv-pages", "4"]),
    ("prefix", ["--prefix-cache-blocks", "16"]),
    ("prefix+host-kv", ["--prefix-cache-blocks", "16", "--kv-in-host"]),
    ("prefix+host-kv+online4",
     ["--prefix-cache-blocks", "16", "--kv-in-host", "--online-kv-pages", "4"]),
    ("offload", ["--offload-weights-mib", "1024", "--allow-host-weights"]),
    ("all",
     ["--prefix-cache-blocks", "16", "--kv-in-host", "--online-kv-pages", "4",
      "--offload-weights-mib", "1024", "--allow-host-weights"]),
]


class Engine:
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.logs: list[str] = []
        self.bad_lines = 0
        self.tasks: list[asyncio.Task] = []
        self.counter = 0

    async def start(self) -> str:
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
                return "ok"

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
        rid, fut = await self.begin(ids, max_new)
        return await fut

    async def cancel(self, rid: str) -> None:
        await self.send({"op": "cancel", "request_id": rid})

    async def close(self) -> int:
        if self.proc is None:
            return 0
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            rc = await asyncio.wait_for(self.proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            self.proc.kill()
            rc = await self.proc.wait()
        for t in self.tasks:
            try:
                await asyncio.wait_for(t, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()
        return rc

    def ready_line(self) -> str:
        for line in reversed(self.logs):
            if "shelf-compute: ready" in line:
                return line.strip()
        return ""

    def errors(self) -> list[str]:
        bad = []
        for line in self.logs:
            low = line.lower()
            if ("hsa_status_error" in low or "memory fault" in low or
                    "shelf-compute: 起動条件" in line):
                bad.append(line)
        return bad


failures: list[str] = []


def check(cond: bool, what: str, detail: str = "") -> None:
    if cond:
        print(f"PASS  {what}")
    else:
        print(f"FAIL  {what}{(' / ' + detail) if detail else ''}")
        failures.append(what)


async def run_config(name: str, extra: list[str], args, ids: list[int]) -> dict:
    engine = Engine([args.engine, "--model", args.model, "--gpu", str(args.gpu),
                     "--prefill-rows", str(args.prefill_rows), "--max-batch", "4",
                     "--max-context", str(args.max_context), "--no-eos"] + extra)
    print(f"---- {name}: {' '.join(extra) if extra else '(既定)'}")
    await engine.start()
    out: dict = {"name": name, "ready": "", "errors": [], "rc": None}
    try:
        ref = await engine.generate(ids, MAX_NEW)
        out["serial"] = ref.get("generated_ids")
        conc = list(await asyncio.gather(*[engine.generate(ids, MAX_NEW) for _ in range(4)]))
        out["parallel"] = [c.get("generated_ids") for c in conc]
        warm = await engine.generate(ids, MAX_NEW)
        out["warm"] = warm.get("generated_ids")
        pend = [await engine.begin(ids, MAX_NEW) for _ in range(4)]
        await asyncio.sleep(0.05)
        await engine.cancel(pend[0][0])
        await engine.cancel(pend[2][0])
        done = await asyncio.gather(*[fut for _, fut in pend])
        out["cancelled"] = [d.get("finish_reason") for d in done]
        after = await engine.generate(ids, MAX_NEW)
        out["after_cancel"] = after.get("generated_ids")
        out["bad_lines"] = engine.bad_lines
        out["ready"] = engine.ready_line()
        out["errors"] = engine.errors()
    finally:
        out["rc"] = await engine.close()
    return out


def is_moe(model_dir: str) -> bool:
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        return bool(text.get("num_experts"))
    except Exception:  # noqa: BLE001
        return False


async def main_async(args) -> int:
    moe = is_moe(args.model)
    ids = [1000 + (j * 13) % 4000 for j in range(args.prompt_tokens)]
    wanted = set(args.only.split(",")) if args.only else None
    results: list[dict] = []
    for name, extra in CONFIGS:
        if wanted is not None and name not in wanted:
            continue
        res = await run_config(name, extra, args, ids)
        results.append(res)
        ref = results[0]["serial"]
        # MoE で重みを offload すると CPU/GPU の分担が変わり、数値が変わり得る
        # （N11 の既知差）。その構成は「構成内の一致」だけを見る。
        cross = not (moe and "--offload-weights-mib" in extra)
        own = res["serial"]
        if cross:
            check(own == ref, f"{name}: 直列が基準と一致", f"{own} vs {ref}")
        check(all(x == own for x in res["parallel"]), f"{name}: 4 並列が直列と一致",
              f"{res['parallel']}")
        check(res["warm"] == own, f"{name}: 2 回目（prefix hit）が一致")
        check(res["after_cancel"] == own, f"{name}: cancel 直後の再実行が一致")
        check(res["rc"] == 0, f"{name}: 正常終了 (rc={res['rc']})")
        check(not res["errors"], f"{name}: HSA fault / 起動条件エラーなし")
        check(res["bad_lines"] == 0, f"{name}: stdout は JSONL のみ")
        print(f"      {res['ready']}")
    print("\n== メモリ確保量（起動ログ）")
    for res in results:
        print(f"  {res['name']:<24} {res['ready']}")
    if failures:
        print("\n不一致:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nすべて一致")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="N04: メモリ経路の組み合わせ検証")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--engine", default="build/hip-rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--only", default="", help="実行する構成名（カンマ区切り）")
    args = parser.parse_args(argv)
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(os.path.join(ROOT, args.engine))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
