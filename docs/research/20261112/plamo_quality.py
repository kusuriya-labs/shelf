#!/usr/bin/env python3
import argparse
import json
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
    results = []
    for case in cases:
        name = case["name"]
        ids = case["ids"]
        logits = case["logits"].contiguous()
        if len(ids) < 3 or logits.shape[0] != len(ids):
            raise ValueError(f"invalid reference case: {name}")
        vocab = logits.shape[1]
        tokens = root / f"{name}.tokens"
        baseline = root / f"{name}.baseline"
        tokens.write_bytes(b"PSKLDTOK" + struct.pack("<IQ", 1, len(ids)) + struct.pack(f"<{len(ids)}i", *ids))
        with baseline.open("wb") as output:
            output.write(struct.pack("<8sIIQII", b"PSQUAL01", 1, vocab, len(ids) - 2, 0, 2))
            for index in range(1, len(ids) - 1):
                output.write(struct.pack("<iI", ids[index + 1], index + 1))
                output.write(logits[index].numpy().tobytes())
        cmd = [args.bench, "quality", "--model-dir", args.model, "--tokens-file", str(tokens),
               "--context", "1", "--compare", str(baseline), "--dump", str(root / f"{name}.candidate")]
        run = subprocess.run(cmd, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError(f"{name}: shelf-bench exited {run.returncode}\n{run.stdout}\n{run.stderr}")
        metrics = dict(line.split("=", 1) for line in run.stdout.splitlines() if line.startswith("QUALITY_") and "=" in line)
        results.append({"name": name, "tokens": len(ids), "metrics": metrics})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
