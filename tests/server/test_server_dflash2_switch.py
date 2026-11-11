#!/usr/bin/env python3
"""同時 request の増減に伴う tree と chain の切り替えを検証する。"""

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ServerHarness


async def run_requests(server, log):
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def wait_trace(predicate):
            for _ in range(1200):
                rounds = [(int(a), int(b), int(c)) for a, b, c in re.findall(
                    r"DFLASH2_TREE_ROUND speculative=(\d+) budget=(\d+) decoding=(\d+)",
                    log.read_text())]
                if predicate(rounds):
                    return rounds
                await asyncio.sleep(0.05)
            raise AssertionError(f"round trace timeout: {rounds[-20:]}")

        async def request(prompt, count, temperature=0):
            body = {"model": "shelf", "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature, "max_tokens": count, "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            content, finish, usage = [], None, None
            async with session.post(server.base_url + "/chat/completions", json=body) as response:
                if response.status != 200:
                    raise AssertionError(f"HTTP {response.status}: {await response.text()}")
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[5:])
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        content.append(choice.get("delta", {}).get("content", ""))
                        finish = choice.get("finish_reason") or finish
            text = "".join(content)
            if finish != "length" or not usage or usage["completion_tokens"] != count:
                raise AssertionError(f"finish={finish}, usage={usage}")
            if len(text.strip()) < 40 or "\ufffd" in text:
                raise AssertionError(f"broken output: {text!r}")
            return text

        first = asyncio.create_task(request(
            "Write a detailed tutorial on Python programming with many working examples and explanations.", 384))
        await wait_trace(lambda rounds: (1, 10, 1) in rounds)
        second = asyncio.create_task(request(
            "Explain in detail how a compiler parses and evaluates arithmetic expressions.", 96))
        await wait_trace(lambda rounds: (2, 8, 2) in rounds)
        await second
        if first.done():
            raise AssertionError("first request finished before the second")
        await wait_trace(lambda rounds: any(x == (1, 10, 1) for x in rounds[
            max(i for i, x in enumerate(rounds) if x == (2, 8, 2)) + 1:]))
        await first
        first = asyncio.create_task(request(
            "Write a detailed tutorial on algorithms with many examples and explanations.", 384))
        before = len(await wait_trace(lambda rounds: bool(rounds)))
        await wait_trace(lambda rounds: (1, 10, 1) in rounds[before:])
        second = asyncio.create_task(request(
            "Explain several methods for managing a project with examples.", 96, 0.7))
        await wait_trace(lambda rounds: (1, 8, 2) in rounds[before:])
        await second
        if first.done():
            raise AssertionError("first request finished before the normal decode request")
        await wait_trace(lambda rounds: any(x == (1, 10, 1) for x in rounds[
            max(i for i, x in enumerate(rounds) if x == (1, 8, 2)) + 1:]))
        await first


def main():
    draft = os.environ["SHELF_DFLASH2_MODEL_DIR"]
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "server.log"
        server = ServerHarness(log_path=log, max_seq_len=1024, arena_gib=31,
                               max_concurrent_requests=2, dflash2_model_dir=draft,
                               dflash2_tree_budget=None, startup_timeout=420,
                               env={"SHELF_DFLASH2_TREE_TRACE_ROUND": "1"})
        try:
            server.start()
            asyncio.run(run_requests(server, log))
        finally:
            server.close()
    print("PASS: tree -> chain -> tree with speculative and normal decode requests")


if __name__ == "__main__":
    main()
