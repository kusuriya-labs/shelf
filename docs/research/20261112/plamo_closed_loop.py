#!/usr/bin/env python3
import argparse
import asyncio
import json
import itertools
import random
import os
import signal
import subprocess
import time

import aiohttp
from plamo_batch import CASES


def workload(lengths):
    cases = []
    for language in ("ja", "en"):
        examples = [case for case in CASES if case[0] == language]
        for length in lengths:
            for offset in range(4):
                texts = [examples[(offset + i) % len(examples)][2] for i in range(length)]
                cases.append((language, examples[0][1], " ".join(texts)))
    random.Random(17).shuffle(cases)
    return cases


async def run(args):
    if args.concurrency < 1 or args.warmup < 0 or args.duration <= 0:
        raise ValueError("invalid concurrency or measurement duration")
    cases = workload(args.lengths)
    command = [args.server, "--model-dir", args.model, "--port", str(args.port),
               "--max-seq-len", "2048", "--max-concurrent-requests", str(args.concurrency)]
    if args.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.kv_cache_dtype != "bf16":
        command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    with open(args.log, "w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
                                             connector=aiohttp.TCPConnector(limit=0)) as session:
                url = f"http://127.0.0.1:{args.port}"
                for _ in range(6000):
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited: {proc.returncode}")
                    try:
                        async with session.get(url + "/v1/models") as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("server startup timeout")
                started = time.monotonic()
                ready = [asyncio.Event() for _ in range(args.concurrency)]
                start_measure = None
                end_measure = None
                records = []
                next_request = itertools.count()

                async def worker(worker_id):
                    while True:
                        index = next(next_request) % len(cases)
                        src, dst, text = cases[index]
                        body = {"model": "shelf", "messages": [{"role": "user", "content": text}],
                                "temperature": 0, "max_tokens": 320,
                                "chat_template_kwargs": {"source_lang": src, "target_lang": dst}}
                        begin = time.monotonic()
                        async with session.post(url + "/v1/chat/completions", json=body) as response:
                            result = await response.json()
                            if response.status != 200:
                                raise RuntimeError(f"HTTP {response.status}: {result}")
                        finished = time.monotonic()
                        usage = result["usage"]
                        if start_measure is not None and start_measure <= finished < end_measure:
                            records.append({"case": index, "worker": worker_id,
                                            "started_s": begin - start_measure,
                                            "finished_s": finished - start_measure,
                                            "input_tokens": usage["prompt_tokens"],
                                            "output_tokens": usage["completion_tokens"],
                                            "finish_reason": result["choices"][0]["finish_reason"]})
                        ready[worker_id].set()
                        if end_measure is not None and finished >= end_measure:
                            return

                tasks = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
                try:
                    ready_task = asyncio.gather(*(event.wait() for event in ready))
                    done, _ = await asyncio.wait([ready_task, *tasks], return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    if ready_task not in done:
                        raise RuntimeError("worker stopped before warmup completed")
                    await asyncio.sleep(args.warmup)
                    start_measure = time.monotonic()
                    end_measure = start_measure + args.duration
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                tokens = sum(record["output_tokens"] for record in records)
                print(json.dumps({"concurrency": args.concurrency, "warmup_after_first_completion_s": args.warmup,
                                  "elapsed_before_measurement_s": start_measure - started,
                                  "measurement_s": args.duration, "max_num_batched_tokens": args.max_num_batched_tokens,
                                  "lengths": args.lengths,
                                  "kv_cache_dtype": args.kv_cache_dtype,
                                  "completion_tokens": tokens,
                                  "completion_tokens_per_s": tokens / args.duration,
                                  "completed_requests": len(records), "records": records},
                                 ensure_ascii=False), flush=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup", type=float, default=20)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--kv-cache-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--port", type=int, default=18232)
    parser.add_argument("--log", default="/tmp/plamo-closed-loop-server.log")
    args = parser.parse_args()
    if not args.lengths or any(length < 1 for length in args.lengths):
        parser.error("lengths must be positive")
    asyncio.run(run(args))
