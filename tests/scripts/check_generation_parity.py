#!/usr/bin/env python3
"""生成の決定性・数値回帰ゲート（T08）: JSONL を直接駆動して生成列を比較する。

各「実行」を 1 プロセスで 2 回繰り返し、次を比較する:
  * 同一プロセスの連続要求（1 回目と 2 回目）
  * PP（pipeline-parallel）の違い
  * Prefill チャンク（`--prefill-rows`）の違い
  * 同じ base モデルの量子化違い（FP8 / MXFP4 など）

使い方:
  python3 tests/scripts/check_generation_parity.py \
      --config model=models/Qwen3.5-9B-FP8,pp=1,prefill=64 \
      --config model=models/Qwen3.5-9B-FP8,pp=2,prefill=64 \
      --config model=models/Qwen3.8-27B-FP8,label=27B-FP8 \
      --config model=models/Qwen3.8-27B-Quark-AWQ-MXFP4,label=27B-MXFP4

`--prompt` を渡すと transformers の tokenizer で ID 化する（無ければ `--ids`）。
比較は**生成トークン列の完全一致**。違えば差分位置を出す。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

failures: list[str] = []


def parse_config(text: str) -> dict:
    cfg: dict = {}
    for part in text.split(","):
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(f"--config の書式が不正: {part!r}")
        cfg[key.strip()] = value.strip()
    if "model" not in cfg:
        raise SystemExit("--config に model= が必要")
    cfg.setdefault("pp", "1")
    cfg.setdefault("prefill", "64")
    cfg.setdefault("act", "auto")
    cfg.setdefault("offload", "0")
    cfg.setdefault("label", os.path.basename(cfg["model"].rstrip("/")))
    return cfg


async def run_once(argv: list[str], prompt_ids: list[int], max_new_tokens: int,
                   repeats: int) -> tuple[list[list[int]], list[float]]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    async def send(obj):
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        await proc.stdin.drain()

    async def read_event():
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("エンジンが終了した")
        return json.loads(line.decode())

    results: list[list[int]] = []
    times: list[float] = []
    try:
        await send({"op": "ping"})
        while True:
            ev = await read_event()
            if ev.get("event") == "pong":
                break
        for r in range(repeats):
            t0 = time.perf_counter()
            await send({"op": "generate", "request_id": f"r{r}",
                        "input_ids": prompt_ids, "max_new_tokens": max_new_tokens})
            ids: list[int] = []
            while True:
                ev = await read_event()
                kind = ev.get("event")
                if kind == "token":
                    ids.append(int(ev["token_id"]))
                elif kind == "done":
                    assert ev.get("finish_reason") in ("stop", "length"), ev
                    assert ev.get("generated_ids") == ids, (ev, ids)
                    results.append(ids)
                    times.append(time.perf_counter() - t0)
                    break
                elif kind == "error":
                    raise RuntimeError(f"engine error: {ev}")
        await send({"op": "cancel", "request_id": "none"})
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    return results, times


def decode_ids(model: str, ids: list[int]) -> str:
    try:
        from transformers import AutoTokenizer
    except Exception:  # noqa: BLE001
        return ""
    try:
        tok = AutoTokenizer.from_pretrained(model)
        return tok.decode(ids, skip_special_tokens=True)
    except Exception as exc:  # noqa: BLE001
        return f"(decode 失敗: {exc})"


def first_diff(a: list[int], b: list[int]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"位置 {i}: {x} vs {y}"
    if len(a) != len(b):
        return f"長さ {len(a)} vs {len(b)}"
    return ""


async def main_async(args) -> int:
    if args.prompt:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_model or args.configs[0]["model"])
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=True, enable_thinking=False)["input_ids"]
        prompt_ids = [int(x) for x in prompt_ids]
    elif args.ids:
        prompt_ids = [int(x) for x in args.ids.split(",") if x]
    else:
        raise SystemExit("--prompt か --ids のどちらかが必要")

    print(f"prompt tokens = {len(prompt_ids)}, max_new_tokens = {args.max_new_tokens}")

    runs: list[dict] = []
    for cfg in args.configs:
        argv = [args.engine, "--model", cfg["model"],
                "--pipeline-parallel", cfg["pp"], "--prefill-rows", cfg["prefill"],
                "--act-quant", cfg["act"], "--offload-weights-mib", cfg["offload"]]
        label = (f"{cfg['label']}(pp={cfg['pp']},prefill={cfg['prefill']},"
                 f"act={cfg['act']},offload={cfg['offload']}MiB)")
        print(f"\n== {label} を実行")
        outs, times = await run_once(argv, prompt_ids, args.max_new_tokens, args.repeats)
        runs.append({"label": label, "model": cfg["model"], "outs": outs})
        for i, ids in enumerate(outs):
            text = decode_ids(cfg["model"], ids) if args.decode else ""
            per_tok = times[i] * 1000.0 / max(1, args.max_new_tokens)
            head = f"   {i + 1} 回目: {ids}"
            print(f"{head} {text!r}  ({times[i] * 1000:.0f} ms / {per_tok:.1f} ms/tok)"
                  if text else
                  f"{head}  ({times[i] * 1000:.0f} ms / {per_tok:.1f} ms/tok)")

    # 同一プロセス内の再現性
    print("\n== 同一プロセスの連続要求")
    for run in runs:
        same = all(o == run["outs"][0] for o in run["outs"][1:])
        print(f"{'PASS' if same else 'FAIL'}  {run['label']}: {len(run['outs'])} 回で一致")
        if not same:
            failures.append(f"{run['label']} の連続要求が不一致")

    # 実行間の一致（PP・prefill・量子化）
    if len(runs) > 1:
        print("\n== 実行間の一致")
        base = runs[0]
        for run in runs[1:]:
            a, b = base["outs"][0], run["outs"][0]
            same = a == b
            diff = "" if same else first_diff(a, b)
            print(f"{'PASS' if same else 'FAIL'}  {base['label']} vs {run['label']}"
                  + (f"（{diff}）" if diff else ""))
            if not same:
                failures.append(f"{base['label']} vs {run['label']}: {diff}")

    print()
    if failures:
        print(f"{len(failures)} 件不一致")
        return 1
    print("すべて一致")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成の決定性・数値回帰ゲート（T08）")
    parser.add_argument("--engine", default="build/hip_rel/apps/shelf-compute")
    parser.add_argument("--config", action="append", default=[], required=True,
                        help="model=PATH[,pp=N][,prefill=N][,label=NAME]")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--decode", action="store_true", help="tokenizer で本文も表示")
    args = parser.parse_args(argv)
    args.engine = os.path.abspath(args.engine)
    args.configs = [parse_config(c) for c in args.config]
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
