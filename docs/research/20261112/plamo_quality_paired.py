#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path

import torch


def agreements(case, path):
    ids = case["ids"]
    with path.open("rb") as source:
        header = source.read(32)
        magic, version, vocab, count, _, score_start = struct.unpack("<8sIIQII", header)
        if magic != b"PSQUAL01" or version != 1 or count != len(ids) - 2 or score_start != 2:
            raise ValueError(f"invalid candidate dump: {path}")
        result = []
        for index in range(1, len(ids) - 1):
            target, position = struct.unpack("<iI", source.read(8))
            raw = bytearray(source.read(vocab * 4))
            if target != ids[index + 1] or position != index + 1 or len(raw) != vocab * 4:
                raise ValueError(f"invalid candidate row: {path} index {index}")
            row = torch.frombuffer(raw, dtype=torch.float32)
            result.append(row.argmax().item() == case["logits"][index].argmax().item())
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    groups = {"ja-en": [], "en-ja": []}
    for case in torch.load(args.reference, map_location="cpu", weights_only=True):
        name = case["name"]
        group = ("ja-en" if name.startswith("heldout-ja-en-") else
                 "en-ja" if name.startswith("heldout-en-ja-") else None)
        if group is None:
            continue
        before = agreements(case, args.baseline / f"{name}.candidate")
        after = agreements(case, args.candidate / f"{name}.candidate")
        losses = sum(a and not b for a, b in zip(before, after))
        gains = sum(not a and b for a, b in zip(before, after))
        groups[group].append((len(before), losses, gains))
    output = {}
    for group, cases in groups.items():
        positions = sum(case[0] for case in cases)
        losses = sum(case[1] for case in cases)
        gains = sum(case[2] for case in cases)
        if not positions:
            raise ValueError(f"no cases for {group}")
        discordant = losses + gains
        p = sum(math.comb(discordant, k) for k in range(losses, discordant + 1)) / 2 ** discordant
        differences = [gain - loss for _, loss, gain in cases]
        permutation_sums = [0]
        for difference in differences:
            permutation_sums = [v + sign * difference for v in permutation_sums for sign in (-1, 1)]
        observed = sum(differences)
        permutation_p = sum(value <= observed for value in permutation_sums) / len(permutation_sums)
        output[group] = {"cases": len(cases), "positions": positions,
                         "lost_agreements": losses, "gained_agreements": gains,
                         "one_sided_mcnemar_p": p, "one_sided_case_permutation_p": permutation_p}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
