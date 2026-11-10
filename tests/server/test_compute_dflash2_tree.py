#!/usr/bin/env python3
"""単発 DFlash2 tree の greedy 出力を target および chain と比較する。"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import compute_binary, model_dir, prompt_ids

PROMPTS = [
    "Write a detailed, long tutorial with many shell command examples for finding, sorting, inspecting, and safely managing large files in a directory.",
    "Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests.",
    "Explain in detail why the square root of two is irrational, then develop several different proofs and discuss related mathematical examples at length.",
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
    if mode.startswith("tree"):
        cmd += ["--dflash2-tree-extra", mode[4:]]
    proc = subprocess.run(cmd, input="\n".join(map(str, ids)) + "\n", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if proc.returncode:
        raise RuntimeError(f"{mode} failed: {proc.stderr[-4000:]}")
    metrics = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                   if line.startswith("DFLASH2_") and "=" in line)
    if mode.startswith("tree"):
        rounds = int(metrics["DFLASH2_ROUNDS"])
        elapsed = float(metrics["DFLASH2_ROUND_MS"])
        print(f"MEASURE {mode} rounds={rounds} accepted={metrics['DFLASH2_ACCEPTED_DRAFTS']} "
              f"round_ms={elapsed / rounds if rounds else 0:.3f}", flush=True)
    for line in proc.stdout.splitlines():
        if line.startswith("GENERATED_IDS="):
            return [int(x) for x in line.split("=", 1)[1].split(",") if x.strip()]
    raise RuntimeError(f"{mode}: GENERATED_IDS missing")


def main():
    for prompt_index, prompt in enumerate(PROMPTS):
        ids = prompt_ids(prompt)
        outputs = {mode: run(ids, mode) for mode in ("target", "chain", "tree0", "tree1", "tree2", "tree4")}
        if len(outputs["target"]) != 256:
            raise AssertionError(f"expected 256 generated tokens, got {len(outputs['target'])}: {prompt}")
        for mode in ("chain", "tree0", "tree1", "tree2", "tree4"):
            expected, actual = outputs["target"], outputs[mode]
            if expected != actual:
                mismatch = next((i for i, (a, b) in enumerate(zip(expected, actual))
                                 if a != b), min(len(expected), len(actual)))
                raise AssertionError(f"{mode} differs at token {mismatch}: "
                                     f"target={expected[mismatch:mismatch+4]} "
                                     f"actual={actual[mismatch:mismatch+4]} prompt={prompt}")
        if prompt_index == 0:
            for mode in ("tree0", "tree1", "tree2", "tree4"):
                if run(ids, mode) != outputs["target"]:
                    raise AssertionError(f"{mode} repeat differs for {prompt}")
        print(f"PASS: {prompt} ({len(outputs['target'])} tokens)", flush=True)


if __name__ == "__main__":
    main()
