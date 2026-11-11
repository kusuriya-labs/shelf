#!/usr/bin/env python3
"""構造化出力と DFlash2 を同時に使い、auto tree の切り替えを確認する。"""
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

TRACE = re.compile(r"DFLASH2_TREE_ROUND speculative=(\d+) budget=(\d+) decoding=(\d+)")
SCHEMA = {"type": "object", "properties": {
    "color": {"type": "string", "enum": ["red", "blue"]},
    "enabled": {"type": "boolean"},
    "kind": {"type": "string", "enum": ["first", "second"]},
}, "required": ["color", "enabled", "kind"], "additionalProperties": False}


async def exercise(server, log):
    def rounds():
        return [tuple(map(int, m)) for m in TRACE.findall(log.read_text())]

    async def wait_for(predicate):
        for _ in range(1200):
            seen = rounds()
            if predicate(seen):
                return seen
            await asyncio.sleep(0.05)
        raise AssertionError(f"round trace timeout: {rounds()[-20:]}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def generate(body):
            async with session.post(server.base_url + "/chat/completions", json=body) as response:
                if response.status != 200:
                    raise AssertionError(f"HTTP {response.status}: {await response.text()}")
                return await response.json()

        base = {"model": "shelf", "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        ordinary = asyncio.create_task(generate(dict(base, max_tokens=768, messages=[{
            "role": "user", "content": "Write a detailed tutorial on Python programming with many working examples and explanations."}])))
        await wait_for(lambda seen: (1, 10, 1) in seen)
        offset = len(rounds())
        constrained = await generate(dict(base, max_tokens=128, messages=[{
            "role": "user", "content": "Return a JSON object with color, enabled and kind."}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "example", "schema": SCHEMA, "strict": True}}))
        choice = constrained["choices"][0]
        assert choice["finish_reason"] == "stop", choice
        value = json.loads(choice["message"]["content"])
        assert set(value) == {"color", "enabled", "kind"}, value
        assert value["color"] in ("red", "blue") and isinstance(value["enabled"], bool), value
        assert value["kind"] in ("first", "second"), value
        seen = rounds()[offset:]
        assert (1, 8, 2) in seen, seen
        result = await ordinary
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length"), choice
        assert len(choice["message"]["content"].strip()) >= 40, choice
        # The unconstrained request can finish before the grammar matcher. Start
        # another after both finish to verify auto returns to tree mode.
        after = len(rounds())
        followup = asyncio.create_task(generate(dict(base, max_tokens=128, messages=[{
            "role": "user", "content": "Explain Python functions with examples."}])))
        await wait_for(lambda all_rounds: (1, 10, 1) in all_rounds[after:])
        followup_result = await followup
        assert followup_result["choices"][0]["message"]["content"].strip(), followup_result


def main():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "server.log"
        server = ServerHarness(log_path=log, max_seq_len=1536, arena_gib=31,
                               max_concurrent_requests=2,
                               dflash2_model_dir=os.environ["SHELF_DFLASH2_MODEL_DIR"],
                               dflash2_tree_budget=None, startup_timeout=420,
                               env={"SHELF_DFLASH2_TREE_TRACE_ROUND": "1"})
        try:
            server.start()
            asyncio.run(exercise(server, log))
        finally:
            server.close()
    print("PASS: grammar normal decode alongside speculative tree/chain/tree")


if __name__ == "__main__":
    main()
