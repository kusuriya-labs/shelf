#!/usr/bin/env python3
import argparse
import json
import queue
import subprocess
import threading
import time

from transformers import AutoTokenizer
from plamo_batch import CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--log", required=True)
    parser.add_argument("--warmup-batch", action="store_true")
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
               add_generation_prompt=True, source_lang=source, target_lang=target)["input_ids"]
               for source, target, text in CASES]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(prompts):
            parser.error("--case-index is out of range")
        prompts = [prompts[args.case_index]]
    command = [args.compute, "--model-dir", args.model, "--serve-stdio", "--device", "0",
               "--max-seq-len", "2048", "--max-concurrent-requests", str(args.concurrency),
               "--prefix-cache", "0", "--temperature", "0", "--kv-cache-dtype", "bf16"]
    events = queue.Queue()
    with open(args.log, "w") as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=log, text=True, bufsize=1)

        def read():
            for line in proc.stdout:
                try:
                    events.put((time.monotonic(), json.loads(line)))
                except json.JSONDecodeError:
                    continue
            events.put((time.monotonic(), {"event": "eof"}))

        threading.Thread(target=read, daemon=True).start()

        def send(value):
            proc.stdin.write(json.dumps(value) + "\n")
            proc.stdin.flush()

        def next_event():
            timestamp, event = events.get(timeout=1200)
            if event.get("event") in ("error", "eof"):
                raise RuntimeError(str(event))
            return timestamp, event

        def generate(rid, ids, count):
            send({"op": "generate", "request_id": rid, "input_ids": ids,
                  "max_new_tokens": count, "temperature": 0, "seed": rid})

        try:
            send({"op": "ping"})
            while next_event()[1].get("event") != "pong":
                pass
            generate(10000, prompts[0], 8)
            while True:
                _, event = next_event()
                if event.get("request_id") == 10000 and event.get("event") == "done":
                    break
            if args.warmup_batch:
                for i in range(args.concurrency):
                    generate(20000 + i, prompts[i % len(prompts)], args.tokens)
                warmed = set()
                while len(warmed) < args.concurrency:
                    _, event = next_event()
                    if event.get("event") == "done" and event.get("request_id", 0) >= 20000:
                        warmed.add(event["request_id"])
            first, done, times = {}, {}, {}
            for i in range(args.concurrency):
                generate(i + 1, prompts[i % len(prompts)], args.tokens)
            while len(done) < args.concurrency:
                timestamp, event = next_event()
                rid = event.get("request_id")
                if event.get("event") == "token":
                    first.setdefault(rid, timestamp)
                    times.setdefault(rid, []).append(timestamp)
                elif event.get("event") == "done":
                    done[rid] = timestamp
            if len(first) != args.concurrency:
                raise RuntimeError("missing token events")
            start, end = max(first.values()), min(done.values())
            count = sum(sum(start <= timestamp <= end for timestamp in values) for values in times.values())
            if end <= start:
                raise RuntimeError("no interval with all requests active")
            print(json.dumps({"concurrency": args.concurrency, "max_new_tokens": args.tokens,
                              "observed_tokens": sum(map(len, times.values())),
                              "case_index": args.case_index, "warmup_batch": args.warmup_batch,
                              "window_s": end - start, "window_tokens": count,
                              "decode_tokens_per_s": count / (end - start),
                              "prompt_tokens": [len(ids) for ids in prompts]}), flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
