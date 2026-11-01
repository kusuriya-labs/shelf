#!/usr/bin/env python3
"""2 つの logits ダンプ（f32）を比較する（T08/T09 の T2 判定用）。

`shelf-run --out DIR` が書く `DIR/logits_f32.bin` を読み、次を出す:
  * max / mean の絶対差、相対 L2
  * 各側の top-1 / top-2 と margin（top1 - top2）
  * argmax が一致するか

使い方:
  python3 tests/scripts/compare_logits.py /tmp/run_a /tmp/run_b
"""

from __future__ import annotations

import argparse
import array
import json
import os


def load(path: str) -> array.array:
    with open(os.path.join(path, "logits_f32.bin"), "rb") as f:
        out = array.array("f")
        out.frombytes(f.read())
    return out


def top2(values) -> tuple[int, float, float]:
    i1 = i2 = -1
    v1 = v2 = float("-inf")
    for i, v in enumerate(values):
        if v > v1:
            i2, v2 = i1, v1
            i1, v1 = i, v
        elif v > v2:
            i2, v2 = i, v
    return i1, v1, v2


def compare(a_path: str, b_path: str) -> dict:
    a = load(a_path)
    b = load(b_path)
    if len(a) != len(b):
        raise SystemExit(f"長さが違う: {len(a)} vs {len(b)}")
    n = len(a)
    max_abs = max(abs(x - y) for x, y in zip(a, b))
    mean_abs = sum(abs(x - y) for x, y in zip(a, b)) / n
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    diff_norm = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    i1a, v1a, v2a = top2(a)
    i1b, v1b, v2b = top2(b)
    return {
        "vocab": n,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rel_l2": diff_norm / (norm_a or 1.0),
        "a": {"top1": i1a, "top1_value": v1a, "top2_value": v2a, "margin": v1a - v2a},
        "b": {"top1": i1b, "top1_value": v1b, "top2_value": v2b, "margin": v1b - v2b},
        "argmax_equal": i1a == i1b,
        "top1_value_diff": abs(v1a - v1b),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="logits ダンプの比較")
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args(argv)
    result = compare(args.a, args.b)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
