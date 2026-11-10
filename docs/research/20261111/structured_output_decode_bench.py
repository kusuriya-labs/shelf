#!/usr/bin/env python3
"""Compare stdio decode throughput; optionally replay tokens through per-step masks."""
import argparse
import base64
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "docs/research/20261110/batch_decode_scripts/rows_cases.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary")
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--mask-replay", action="store_true")
    p.add_argument("--xgrammar", action="store_true")
    a = p.parse_args()
    cases = [c for c in json.loads(CASES.read_text()) if len(c["input_ids"]) >= 16]
    proc = subprocess.Popen([a.binary, "--model-dir", os.environ["SHELF_MODEL_DIR"], "--serve-stdio",
        "--device", "0", "--max-seq-len", "2048", "--max-concurrent-requests", str(a.n),
        "--prefix-cache", "0", "--temperature", "0"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(obj):
        proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n"); proc.stdin.flush()
    def receive():
        line = proc.stdout.readline()
        if not line: raise RuntimeError("compute exited")
        event = json.loads(line)
        if event.get("event") == "error": raise RuntimeError(str(event))
        return event
    def mask(token):
        data = bytearray(248320 // 8); data[token // 8] |= 1 << (token % 8)
        return base64.b64encode(data).decode("ascii")
    send({"op":"ping"})
    while receive().get("event") != "pong": pass
    def run(start, replay=None, grammar=None):
        sequences = {}; stamps = {}; done = set()
        for i in range(a.n):
            req = {"op":"generate", "request_id":start+i, "input_ids":cases[i % len(cases)]["input_ids"],
                   "max_new_tokens":a.tokens, "temperature":0}
            if replay is not None:
                req.update(grammar=True, bitmask=mask(replay[i][0]))
            elif grammar is not None:
                req.update(grammar=True, bitmask=grammar.next_mask())
            send(req)
        while len(done) < a.n:
            event = receive(); rid = event.get("request_id")
            if event.get("event") == "token":
                i = rid-start; sequences.setdefault(i, []).append(event["token_id"])
                stamps.setdefault(i, []).append(time.monotonic())
                k = len(sequences[i])
                if replay is not None and k < len(replay[i]):
                    send({"op":"mask", "request_id":rid, "bitmask":mask(replay[i][k])})
                elif grammar is not None and grammar.accept(event["token_id"]):
                    send({"op":"mask", "request_id":rid, "bitmask":grammar.next_mask()})
            elif event.get("event") == "done": done.add(rid)
        lo = max(v[0] for v in stamps.values()); hi = min(v[-1] for v in stamps.values())
        count = sum(sum(lo <= t <= hi for t in times) for times in stamps.values())
        intervals = [b-a for values in stamps.values() for a,b in zip(values,values[1:])]
        return sequences, {"n":a.n,"total_tokens":sum(map(len,sequences.values())),
            "all_active_tok_s":count/(hi-lo), "mean_step_ms":statistics.mean(intervals)*1000,
            "p95_step_ms":sorted(intervals)[int(0.95*(len(intervals)-1))]*1000}
    # Warm-up is measured separately and excluded from both runs.
    send({"op":"generate","request_id":900,"input_ids":cases[0]["input_ids"],"max_new_tokens":8})
    while True:
        e = receive()
        if e.get("request_id") == 900 and e.get("event") == "done": break
    tokens, plain = run(100)
    print(json.dumps({"plain":plain}), flush=True)
    if a.mask_replay:
        assert a.n == 1
        _, constrained = run(200, tokens)
        print(json.dumps({"mask_replay":constrained}), flush=True)
    if a.xgrammar or a.mask_replay:
        assert a.n == 1
        sys.path[:0] = [str(ROOT / "src/apps/server"), str(ROOT / "src/apps/common")]
        from transformers import AutoTokenizer
        from serverlib.structured_output import GrammarFactory
        tokenizer = AutoTokenizer.from_pretrained(os.environ["SHELF_MODEL_DIR"], trust_remote_code=True)
        grammar = GrammarFactory(tokenizer, 248320).create(
            {"type":"string", "enum":["hello " * 200]})
        _, constrained = run(300, grammar=grammar)
        print(json.dumps({"xgrammar":constrained}), flush=True)
    send({"op":"shutdown"}); proc.stdin.close(); proc.wait(timeout=60)

if __name__ == "__main__": main()
