#!/usr/bin/env python3
"""単発 DFlash2 tree の greedy 出力を target および chain と比較する。"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import compute_binary, model_dir, prompt_ids

PROMPTS = [
    "Write a detailed, long tutorial with many shell command examples for finding, sorting, inspecting, and safely managing large files in a directory.",
    "Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests.",
    "Explain in detail why the square root of two is irrational, then develop several different proofs and discuss related mathematical examples at length.",
]


def run(ids, mode, logits_path=None):
    cmd = [str(compute_binary()), "--model-dir", str(model_dir()),
           "--input-ids-file", "/dev/stdin", "--max-new-tokens", "256",
           "--max-seq-len", "512", "--arena-gib", "auto", "--device",
           os.environ.get("SHELF_TEST_DEVICE", "0"), "--temperature", "0",
           "--top-p", "1", "--top-k", "0", "--seed", "0"]
    if logits_path is not None:
        cmd += ["--dump-logits", str(logits_path)]
    if mode != "target":
        cmd += ["--dflash2-model-dir", os.environ["SHELF_DFLASH2_MODEL_DIR"],
                "--dflash2-drafts", "7"]
    if mode.startswith("tree"):
        cmd += ["--dflash2-tree-extra", mode[4:]]
    if mode.startswith("budget"):
        cmd += ["--dflash2-tree-budget", mode[6:]]
    proc = subprocess.run(cmd, input="\n".join(map(str, ids)) + "\n", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if proc.returncode:
        raise RuntimeError(f"{mode} failed: {proc.stderr[-4000:]}")
    metrics = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                   if line.startswith("DFLASH2_") and "=" in line)
    if mode.startswith(("tree", "budget")):
        rounds = int(metrics["DFLASH2_ROUNDS"])
        elapsed = float(metrics["DFLASH2_ROUND_MS"])
        print(f"MEASURE {mode} rounds={rounds} accepted={metrics['DFLASH2_ACCEPTED_DRAFTS']} "
              f"round_ms={elapsed / rounds if rounds else 0:.3f}", flush=True)
    for line in proc.stdout.splitlines():
        if line.startswith("GENERATED_IDS="):
            tokens = [int(x) for x in line.split("=", 1)[1].split(",") if x.strip()]
            if logits_path is None:
                return tokens
            metadata = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                            if line.startswith("LOGITS_") and "=" in line)
            vocab = int(metadata["LOGITS_VOCAB"])
            steps = [int(x) for x in metadata["LOGITS_STEP_ROWS"].split(",")]
            raw = np.memmap(logits_path, dtype=np.float32, mode="r").reshape(-1, vocab)
            offsets = np.cumsum([0] + steps[:-1])
            prefill_chunks = int(next(line.split("=", 1)[1] for line in proc.stdout.splitlines()
                                      if line.startswith("PREFILL_CHUNKS=")))
            offsets = offsets[prefill_chunks:]
            if len(offsets) != len(tokens):
                raise AssertionError(f"logit steps {len(offsets)} != tokens {len(tokens)}")
            margins = []
            for offset, token in zip(offsets, tokens):
                row = raw[offset]
                if int(np.argmax(row)) != token:
                    raise AssertionError("logit row does not match target token")
                top = np.partition(row, -2)[-2:]
                margins.append(float(top.max() - top.min()))
            return tokens, margins
    raise RuntimeError(f"{mode}: GENERATED_IDS missing")


def main():
    failures = []
    for prompt in PROMPTS:
        ids = prompt_ids(prompt)
        with tempfile.TemporaryDirectory() as tmp:
            target, margins = run(ids, "target", Path(tmp) / "logits.bin")
        outputs = {mode: run(ids, mode) for mode in ("chain", "tree0", "tree1", "tree2", "tree4", "budget8", "budget10", "budget12", "budget16", "budget24")}
        if len(target) != 256:
            raise AssertionError(f"expected 256 generated tokens, got {len(target)}: {prompt}")
        if outputs["budget8"] != outputs["chain"]:
            raise AssertionError(f"budget8 differs from chain: {prompt}")
        for mode in ("chain", "tree0", "tree1", "tree2", "tree4", "budget10", "budget12", "budget16", "budget24"):
            actual = outputs[mode]
            if len(actual) != len(target):
                raise AssertionError(f"{mode} length differs: {len(actual)}")
            mismatches = [i for i, (a, b) in enumerate(zip(target, actual)) if a != b]
            if mismatches:
                first = mismatches[0]
                print(f"{mode} mismatches={len(mismatches)} first={first} "
                      f"first_margin={margins[first]:.6f}", flush=True)
                if margins[first] > 0.1:
                    failures.append(f"{mode} first mismatch at {first}: "
                                    f"target margin {margins[first]:.6f} > 0.1; "
                                    f"prompt={prompt}")
        print(f"CHECKED: {prompt} ({len(target)} tokens)", flush=True)
    if failures:
        raise AssertionError("\n".join(failures))
    print("PASS: all prompts", flush=True)


if __name__ == "__main__":
    main()
