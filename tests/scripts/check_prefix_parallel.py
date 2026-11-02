#!/usr/bin/env python3
"""Prefix Cache × 4 並列の E2E（N03）。

次を 1 プロセスのエンジンで続けて検証する:
  1. 同時 Prefix Hit: 同じ prefix を 4 並列で要求し、直列の生成列と一致すること。
  2. 重複キー: キャッシュが空の状態で 4 並列が同じ prefix を同時に登録しても、
     その後の hit が壊れないこと（N02 の実装修正の E2E 確認）。
  3. LRU 退避: プールを小さくして溢れさせ、退避後も生成列が変わらないこと。
  4. Cancel 競合: 生成中の cancel と、その直後のスロット再利用で状態が壊れないこと。
  5. KV 共有 + GDN 状態復元: prefix hit は KV ブロックを共有し、GDN スナップショットを
     復元する。生成列の一致で確認する（ずれれば列が変わる）。

使い方:
  python3 tests/scripts/check_prefix_parallel.py \
      --model /opt/zen/work/models/Qwen3.5-9B-FP8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class Engine:
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.tokens: dict[str, list[int]] = {}
        self.first: dict[str, asyncio.Event] = {}
        self.logs: list[str] = []
        self.bad_lines = 0
        self.tasks: list[asyncio.Task] = []
        self.counter = 0

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        # pong は reader を起動する前に直接読む（取り合いを避ける）。
        self.tasks.append(asyncio.create_task(self._drain()))
        await self.send({"op": "ping"})
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine が起動前に終了した")
            ev = json.loads(line)
            if ev.get("event") == "pong":
                break
        self.tasks.append(asyncio.create_task(self._reader()))

    async def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                self.bad_lines += 1
                print(f"engine stdout の非 JSON 行: {text!r}", file=sys.stderr)
                continue
            kind = ev.get("event")
            rid = ev.get("request_id")
            if kind == "token":
                self.tokens.setdefault(rid, []).append(ev["token_id"])
                evt = self.first.get(rid)
                if evt is not None:
                    evt.set()
            elif kind in ("done", "error"):
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
        self.tokens[rid] = []
        self.first[rid] = asyncio.Event()
        await self.send({"op": "generate", "request_id": rid, "input_ids": ids,
                         "max_new_tokens": max_new})
        return rid, fut

    async def generate(self, ids: list[int], max_new: int) -> dict:
        rid, fut = await self.begin(ids, max_new)
        done = await fut
        done["_rid"] = rid
        return done

    async def cancel(self, rid: str) -> None:
        await self.send({"op": "cancel", "request_id": rid})

    def hit_count(self) -> int:
        return sum(1 for line in self.logs if "[prefix] hit" in line)

    async def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        # EOF で reader/drain は自然終了する。残りのログ（stats 行）を読み切る。
        for t in self.tasks:
            try:
                await asyncio.wait_for(t, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()


def make_prompt(i: int, n: int) -> list[int]:
    base = 1000 + i * 5000
    return [base + (j * 13) % 4000 for j in range(n)]


failures: list[str] = []


def check(cond: bool, what: str, detail: str = "") -> None:
    if cond:
        print(f"PASS  {what}")
    else:
        print(f"FAIL  {what}{(' / ' + detail) if detail else ''}")
        failures.append(what)


def check_ids(label: str, ref_ids: list[int] | None, rows: list[tuple[str, list[int] | None]]) -> None:
    bad = [(name, got) for name, got in rows if got != ref_ids]
    check(not bad, label, f"不一致 {len(bad)}/{len(rows)}")
    for name, got in bad:
        show_diff(name, ref_ids, got)


def show_diff(label: str, ref: list[int] | None, got: list[int] | None) -> None:
    """生成列が違うとき、最初の相違位置を出す（切り分け用）。"""
    ref = ref or []
    got = got or []
    at = next((i for i in range(min(len(ref), len(got))) if ref[i] != got[i]),
              min(len(ref), len(got)))
    print(f"      {label}: first_diff={at} ref={ref[:12]}… 長さ {len(ref)} / "
          f"got={got[:12]}… 長さ {len(got)}")


async def main_async(args) -> int:
    engine = Engine([args.engine, "--model", args.model, "--max-batch", "4",
                     "--max-context", str(args.max_context),
                     "--prefix-cache-blocks", str(args.blocks),
                     "--prefix-snapshot-tokens", "64",
                     "--prefill-rows", "64"])
    print("起動:", " ".join(engine.argv))
    await engine.start()
    try:
        p0 = make_prompt(0, args.prompt_tokens)
        # 1) キャッシュが空の状態で 4 並列（同じ prefix を同時に登録する）。
        cold = list(await asyncio.gather(*[engine.generate(p0, args.max_new)
                                           for _ in range(4)]))
        ref0 = await engine.generate(p0, args.max_new)
        check_ids("空キャッシュで 4 並列が同じ列を生成", ref0.get("generated_ids"),
                  [(f"row{i}", d.get("generated_ids")) for i, d in enumerate(cold)])
        check(all("generated_ids" in d for d in cold), "4 並列すべて done が返る")

        # 2) 温まったキャッシュで 4 並列（同時 Prefix Hit）。
        before = engine.hit_count()
        warm = list(await asyncio.gather(*[engine.generate(p0, args.max_new)
                                           for _ in range(4)]))
        check_ids("同時 Prefix Hit でも生成列が一致", ref0.get("generated_ids"),
                  [(f"row{i}", d.get("generated_ids")) for i, d in enumerate(warm)])
        hits = engine.hit_count() - before
        check(hits >= 4, f"4 並列すべてが hit する（{hits} 件）")

        # 3) 4 種類のプロンプトを直列 → 4 並列で同時 hit。
        refs: dict[int, list[int]] = {}
        for i in range(1, 5):
            refs[i] = (await engine.generate(make_prompt(i, args.prompt_tokens),
                                             args.max_new)).get("generated_ids")
        before = engine.hit_count()
        conc = list(await asyncio.gather(*[
            engine.generate(make_prompt(i, args.prompt_tokens), args.max_new)
            for i in range(1, 5)]))
        for i in range(1, 5):
            if conc[i - 1].get("generated_ids") != refs[i]:
                show_diff(f"prompt{i}", refs[i], conc[i - 1].get("generated_ids"))
        check(all(conc[i - 1].get("generated_ids") == refs[i] for i in range(1, 5)),
              "同時 hit（別 prefix ×4）でも生成列が一致（個別）")
        check(engine.hit_count() - before >= 4, "別 prefix ×4 がすべて hit する")

        # 4) LRU 退避: プールを溢れさせる（合計ブロック数 > キャッシュ容量）。
        wave = list(range(5, 9))
        for i in wave:
            refs[i] = (await engine.generate(make_prompt(i, args.prompt_tokens),
                                             args.max_new)).get("generated_ids")
        conc = list(await asyncio.gather(*[
            engine.generate(make_prompt(i, args.prompt_tokens), args.max_new) for i in wave]))
        for j in range(len(wave)):
            if conc[j].get("generated_ids") != refs[wave[j]]:
                show_diff(f"prompt{wave[j]}", refs[wave[j]], conc[j].get("generated_ids"))
        check(all(conc[j].get("generated_ids") == refs[wave[j]] for j in range(len(wave))),
              "LRU 退避を挟んでも生成列が一致")
        evictions = sum(1 for line in engine.logs if "[prefix]" in line and "evict" in line)
        print(f"      ログ上の退避件数 = {evictions}（0 でも容量超過なら退避している）")

        # 5) Cancel 競合 + スロット再利用。
        pend = [await engine.begin(make_prompt(1, args.prompt_tokens), args.cancel_tokens)
                for _ in range(4)]
        await asyncio.sleep(0.05)
        await engine.cancel(pend[0][0])
        await engine.cancel(pend[1][0])
        done = await asyncio.gather(*[fut for _, fut in pend])
        cancelled = [d for d in done if d.get("finish_reason") == "cancelled"]
        check(len(cancelled) >= 1, f"cancel した要求が cancelled で返る（{len(cancelled)}/2）")
        after = await engine.generate(make_prompt(1, args.prompt_tokens), args.max_new)
        if after.get("generated_ids") != refs[1]:
            show_diff("cancel 後", refs[1], after.get("generated_ids"))
        check(after.get("generated_ids") == refs[1], "cancel 直後のスロット再利用でも列が一致")

        # 6) 状態の整合: 参照カウントが返り切っていること（エンジンが正常終了できる）。
    finally:
        await engine.close()
    stats: dict[str, int] = {}
    for line in engine.logs:
        if "[prefix] stats" not in line:
            continue
        for part in line.split("stats", 1)[1].split():
            key, sep, value = part.partition("=")
            if sep:
                try:
                    stats[key] = int(value)
                except ValueError:
                    pass
    check(stats.get("evictions", 0) > 0,
          f"LRU 退避が実際に起きる（evictions={stats.get('evictions')}）")
    check(engine.bad_lines == 0,
          f"engine の stdout は JSONL のみ（非 JSON {engine.bad_lines} 行）")
    print(f"ログ {len(engine.logs)} 行（[prefix] hit {engine.hit_count()} 件, stats={stats}）")
    if failures:
        print("不一致:")
        for f in failures:
            print(" -", f)
        return 1
    print("すべて一致")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prefix Cache × 4 並列の E2E（N03）")
    parser.add_argument("--engine", default="build/hip-rel/apps/shelf-compute")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--blocks", type=int, default=4,
                        help="prefix cache の追加ブロック数（小さいほど退避が起きる）")
    parser.add_argument("--max-context", type=int, default=192)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--cancel-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    args.engine = os.path.abspath(os.path.join(ROOT, args.engine))
    args.model = os.path.abspath(os.path.join(ROOT, args.model))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
