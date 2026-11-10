#!/usr/bin/env python3
"""単発 DFlash2 tree の greedy 出力を target および chain と比較する。"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import compute_binary, model_dir, prompt_ids

PROMPTS = [
    "List the shell commands to find the five largest files in a directory.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Explain carefully why the square root of two is irrational, step by step.",
]


def run(ids, mode):
    cmd = [str(compute_binary()), "--model-dir", str(model_dir()),
           "--input-ids-file", "/dev/stdin", "--max-new-tokens", "256",
           "--max-seq-len", "512", "--arena-gib", "auto", "--device",
           os.environ.get("SHELF_TEST_DEVICE", "0"), "--temperature", "0",
           "--top-p", "1", "--top-k", "0", "--seed", "0"]
    if mode != "target":
        cmd += ["--dflash2-model-dir", os.environ["SHELF_DFLASH2_MODEL_DIR"],
                "--dflash2-drafts", "7"]
    if mode == "tree":
        cmd += ["--dflash2-tree-extra", "0"]
    proc = subprocess.run(cmd, input="\n".join(map(str, ids)) + "\n", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if proc.returncode:
        raise RuntimeError(f"{mode} failed: {proc.stderr[-4000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("GENERATED_IDS="):
            return [int(x) for x in line.split("=", 1)[1].split(",") if x.strip()]
    raise RuntimeError(f"{mode}: GENERATED_IDS missing")


def main():
    for prompt in PROMPTS:
        ids = prompt_ids(prompt)
        outputs = {mode: run(ids, mode) for mode in ("target", "chain", "tree")}
        for mode in ("chain", "tree"):
            expected, actual = outputs["target"], outputs[mode]
            if expected != actual:
                mismatch = next((i for i, (a, b) in enumerate(zip(expected, actual))
                                 if a != b), min(len(expected), len(actual)))
                raise AssertionError(f"{mode} differs at token {mismatch}: "
                                     f"target={expected[mismatch:mismatch+4]} "
                                     f"actual={actual[mismatch:mismatch+4]} prompt={prompt}")
        print(f"PASS: {prompt}", flush=True)


if __name__ == "__main__":
    main()
