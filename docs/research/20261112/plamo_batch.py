#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import signal
import subprocess
import time

import aiohttp

CASES = [
    ("ja", "en", "今日は良い天気ですね。午後は公園を散歩しましょう。"),
    ("en", "ja", "The train will arrive in five minutes. Please wait behind the yellow line."),
    ("ja", "en", "ご注文いただいた商品は明日発送する予定です。配送状況はメールに記載された追跡番号からご確認ください。"),
    ("en", "ja", "Thank you for your message. We have reviewed your request and will send you an updated estimate by Friday afternoon."),
    ("ja", "en", "会議の開始時刻が変更になりました。資料を事前に確認し、質問があれば担当者までお知らせください。オンラインで参加する方には別途リンクをお送りします。"),
    ("en", "ja", "The museum is closed on Mondays and national holidays. Visitors who have purchased tickets online can enter through the north entrance. Please keep your confirmation email until the end of your visit."),
    ("ja", "en", "新しいシステムの導入にあたり、来週から操作説明会を開催します。各部署の代表者は参加希望日を選び、金曜日までに申し込んでください。詳細な手順書は社内サイトで公開する予定です。"),
    ("en", "ja", "Our team compared several approaches before choosing the current design. The first version was easier to build, but it required more memory and became slower as the number of requests increased. We will continue to monitor performance after release."),
]

async def main(args):
    proc = subprocess.Popen([args.server, "--model-dir", args.model, "--port", str(args.port),
        "--max-seq-len", "2048", "--max-concurrent-requests", str(args.concurrency)],
        stdout=subprocess.DEVNULL, stderr=open(args.log, "w"), start_new_session=True)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {proc.returncode}")
                try:
                    async with session.get(f"http://127.0.0.1:{args.port}/v1/models") as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("server startup timeout")
            async def request(i):
                source, target, text = CASES[i % len(CASES)]
                body = {"model": "shelf", "messages": [{"role": "user", "content": text}],
                        "temperature": 0, "max_tokens": 192,
                        "chat_template_kwargs": {"source_lang": source, "target_lang": target}}
                async with session.post(f"http://127.0.0.1:{args.port}/v1/chat/completions", json=body) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {result}")
                    return result
            await request(0)
            if args.warmup_batch:
                warmup_rounds = max(1, (len(CASES) + args.concurrency - 1) // args.concurrency)
                for round_index in range(warmup_rounds):
                    await asyncio.gather(*(request(round_index * args.concurrency + i)
                                           for i in range(args.concurrency)))
            start = time.monotonic()
            rounds = max(1, (len(CASES) + args.concurrency - 1) // args.concurrency)
            results = []
            for round_index in range(rounds):
                results.extend(await asyncio.gather(*(request(round_index * args.concurrency + i)
                                                       for i in range(args.concurrency))))
            elapsed = time.monotonic() - start
            counts = [r["usage"] for r in results]
            tokens = sum(x["completion_tokens"] for x in counts)
            print(json.dumps({"concurrency": args.concurrency, "rounds": rounds, "elapsed_s": elapsed,
                "completion_tokens": tokens, "completion_tokens_per_s": tokens / elapsed,
                "input_tokens": [x["prompt_tokens"] for x in counts],
                "output_tokens": [x["completion_tokens"] for x in counts],
                "finish_reasons": [r["choices"][0]["finish_reason"] for r in results],
                "outputs": [r["choices"][0]["message"]["content"] for r in results]}, ensure_ascii=False), flush=True)
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
    parser.add_argument("--warmup-batch", action="store_true")
    parser.add_argument("--port", type=int, default=18231)
    parser.add_argument("--log", default="/tmp/plamo-batch-server.log")
    asyncio.run(main(parser.parse_args()))
