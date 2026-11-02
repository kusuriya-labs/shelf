#!/usr/bin/env python3
"""Gate 7A: AsyncComputeClient cancellation と reuse の acceptance."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    compute_binary,
    model_dir,
)

SERVER_SRC = Path(__file__).resolve().parents[2] / "src" / "apps" / "server"
sys.path.insert(0, str(SERVER_SRC))

from serverlib import compute_client  # noqa: E402

PROMPT = [248041, 77091]


def make_argv() -> list[str]:
    return [
        str(compute_binary()),
        "--model-dir", str(model_dir()),
        "--serve-stdio",
        "--max-seq-len", "512",
        "--arena-gib", "auto",
        "--page-tokens", "16",
        "--device", os.environ.get("PHASESHIFT_TEST_DEVICE", "0"),
        "--kv-cache-dtype", "bf16",
    ]


async def run(checker: Checker) -> None:
    client = compute_client.AsyncComputeClient(make_argv(), max_inflight=1)
    await client.start()
    try:
        pid = client._proc.pid

        # Stream を 3 token で放棄する。client は cancel と drain を行い、
        # single-flight slot を解放しなければならない。
        stream = client.generate_stream(PROMPT, 400, temperature=0.0)
        seen = 0
        async for event in stream:
            if "token_id" in event:
                seen += 1
                if seen == 3:
                    break
        await stream.aclose()
        checker.check("stream-abandoned-after-tokens", seen == 3, f"seen={seen}")

        done = await client.generate(PROMPT, 8, temperature=0.0)
        checker.check("reuse-after-abandon",
                      done.get("finish_reason") == "length"
                      and len(done.get("generated_ids", [])) == 8,
                      repr(done))
        checker.check("compute-pid-stable", client._proc.pid == pid,
                      f"{pid} -> {client._proc.pid}")

        # 複数の stream を順に放棄しても slot は返る。
        for round_index in range(3):
            repeat = client.generate_stream(PROMPT, 400, temperature=0.0)
            count = 0
            async for event in repeat:
                if "token_id" in event:
                    count += 1
                    if count == 2:
                        break
            await repeat.aclose()
            checker.check(f"abandon-round-{round_index}", count == 2, f"count={count}")

        done2 = await client.generate(PROMPT, 4, temperature=0.0)
        checker.check("reuse-after-repeated-abandon",
                      done2.get("finish_reason") == "length"
                      and len(done2.get("generated_ids", [])) == 4,
                      repr(done2))
        checker.check("compute-pid-stable-2", client._proc.pid == pid,
                      f"{pid} -> {client._proc.pid}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        try:
            await client.shutdown()
        except Exception as exc:  # noqa: BLE001
            checker.check("shutdown", False, repr(exc))


def main() -> int:
    checker = Checker("compute-cancel")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    asyncio.run(run(checker))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
