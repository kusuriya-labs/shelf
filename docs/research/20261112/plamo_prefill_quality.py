#!/usr/bin/env python3
import argparse
import json
import os
import struct
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    cases = torch.load(args.reference, map_location="cpu", weights_only=True)
    case = next(item for item in cases if item["name"] == "general-corpus-0")
    ids, logits = case["ids"], case["logits"]
    context, count, vocab = 256, 30, logits.shape[1]
    if len(ids) <= context + count or logits.shape[0] != len(ids):
        raise ValueError("reference needs at least 287 tokens")
    tokens, baseline = root / "input.tokens", root / "reference.dump"
    tokens.write_bytes(b"PSKLDTOK" + struct.pack("<IQ", 1, len(ids)) + struct.pack(f"<{len(ids)}i", *ids))
    with baseline.open("wb") as output:
        output.write(struct.pack("<8sIIQII", b"PSQUAL01", 1, vocab, count, 0, context + 1))
        for index in range(context, context + count):
            output.write(struct.pack("<iI", ids[index + 1], index + 1))
            output.write(logits[index].contiguous().numpy().tobytes())
    results = {}
    for variant in ("base", "b", "fused"):
        env = os.environ.copy()
        env.pop("SHELF_EXL3_MUL24_PREFILL_AB", None)
        env.pop("SHELF_EXL3_INT8_DENSE", None)
        if variant == "b":
            env["SHELF_EXL3_MUL24_PREFILL_AB"] = "B"
        elif variant == "fused":
            env["SHELF_EXL3_INT8_DENSE"] = "0"
        command = [args.bench, "quality", "--model-dir", args.model, "--tokens-file", str(tokens),
                   "--context", str(context), "--positions", str(count), "--compare", str(baseline),
                   "--dump", str(root / f"{variant}.dump")]
        run = subprocess.run(command, env=env, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError(f"{variant}: shelf-bench exited {run.returncode}\n{run.stdout}\n{run.stderr}")
        results[variant] = dict(line.split("=", 1) for line in run.stdout.splitlines()
                                if line.startswith("QUALITY_") and "=" in line)
        print(json.dumps({variant: results[variant]}), flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
