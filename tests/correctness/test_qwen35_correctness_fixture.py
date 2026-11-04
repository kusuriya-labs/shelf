#!/usr/bin/env python3
"""Qwen3.8-27B-FP8 の CPU f64 fixture と correctness path を突き合わせる。

correctness path は W8A8（activation も E4M3）で動くため、f64 oracle と
bit 一致はしない。ここでは次を固定する。

- hidden_01..hidden_08 の相対 L2 が 2e-2 以下
- greedy token が fixture と一致する。fixture の top1-top2 margin が
  0.5 以下で、生成 token が fixture の top5 に入る場合だけ分岐を許す
- 許可分岐より前の token は完全一致

model dir は SHELF_MODEL_DIR、build dir は SHELF_BUILD_DIR で指定する。
どちらかが無い場合は exit 77 で skip する。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "qwen35"
FIXTURE_JSON = FIXTURE_DIR / "qwen38_27b_fp8_cpu_oracle.json"
FIXTURE_HIDDEN = FIXTURE_DIR / "qwen38_27b_fp8_cpu_oracle_hidden.safetensors"
TAP_COUNT = 8
TOKEN_CHECK = 64
REL_TOL = 2e-2
MEAN_ABS_TOL = 2e-2
MARGIN_TOL = 0.5


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    model_dir = os.environ.get("SHELF_MODEL_DIR")
    if not model_dir:
        print("SHELF_MODEL_DIR is not set", file=sys.stderr)
        return 77
    build_dir = Path(os.environ.get("SHELF_BUILD_DIR", REPO_ROOT / "build"))
    compute = build_dir / "shelf-compute"
    if not compute.exists():
        print(f"shelf-compute not found: {compute}", file=sys.stderr)
        return 77
    if not FIXTURE_JSON.exists() or not FIXTURE_HIDDEN.exists():
        print("correctness fixture is missing", file=sys.stderr)
        return 77

    fixture = json.loads(FIXTURE_JSON.read_text())
    case = fixture["case"]
    input_ids = [int(v) for v in case["input_ids"]]
    expected = [int(v) for v in case["generated_ids"]]
    steps = case["steps"]
    hidden_size = int(fixture["model"]["geometry"]["hidden_size"])
    hidden_ref = load_file(str(FIXTURE_HIDDEN))

    tp = int(os.environ.get("SHELF_CORRECTNESS_TP", "1"))
    devices = os.environ.get(
        "SHELF_CORRECTNESS_DEVICES",
        os.environ.get("SHELF_TEST_DEVICE", "0"),
    )

    with tempfile.TemporaryDirectory(prefix="shelf_correctness_fixture_") as tmp:
        tmp_path = Path(tmp)
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("\n".join(str(v) for v in input_ids))
        hidden_dir = tmp_path / "hidden"
        hidden_dir.mkdir()
        env = os.environ.copy()
        env["SHELF_QWEN35_KERNEL_MODE"] = "correctness"
        env["SHELF_DEBUG_HIDDEN_FILE"] = str(hidden_dir)
        command = [
            str(compute),
            "--model-dir", model_dir,
            "--input-ids-file", str(ids_path),
            "--max-new-tokens", str(TOKEN_CHECK),
            "--max-seq-len", "128",
            "--tensor-parallel", str(tp),
            "--device", devices,
            "--kv-cache-dtype", "bf16",
            "--prefix-cache", "0",
            "--max-concurrent-requests", "1",
            "--working-set-reserve-gib", "1",
            "--hidden-taps", ",".join(str(i) for i in range(TAP_COUNT)),
        ]
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        if run.returncode != 0:
            sys.stderr.write(run.stdout[-4000:])
            sys.stderr.write(run.stderr[-4000:])
            return fail(f"shelf-compute exited with {run.returncode}")

        match = re.search(r"^GENERATED_IDS=([0-9,]*)$", run.stdout, re.MULTILINE)
        if match is None:
            return fail("GENERATED_IDS was not reported")
        generated = [int(v) for v in match.group(1).split(",") if v]

        checked = min(len(generated), len(expected), TOKEN_CHECK)
        prefix = 0
        for i in range(checked):
            if generated[i] == expected[i]:
                prefix += 1
                continue
            step = steps[i] if i < len(steps) else {}
            top5 = [int(v[0]) for v in step.get("top5", [])]
            margin = float(step.get("margin", 0.0))
            if margin <= MARGIN_TOL and generated[i] in top5:
                print(
                    f"accepted low-margin divergence at step {i}: "
                    f"got {generated[i]} expected {expected[i]} margin {margin:.6f}"
                )
                break
            return fail(
                f"token {i} differs: got {generated[i]} expected {expected[i]} "
                f"(margin {margin:.6f})"
            )

        if prefix < 8:
            return fail(f"greedy prefix too short: {prefix} tokens")

        worst_rel = 0.0
        worst_tap = 0
        worst_mean = 0.0
        for tap in range(TAP_COUNT):
            path = hidden_dir / f"prefill_L{tap}.bin"
            if not path.exists():
                return fail(f"hidden tap {tap} was not dumped")
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size % hidden_size:
                return fail(f"hidden tap {tap} has an unexpected size")
            got = raw.reshape(-1, hidden_size)[:16]
            ref = hidden_ref[f"hidden_{tap + 1:02d}"][: got.shape[0]]
            if ref.shape != got.shape:
                return fail(f"hidden tap {tap} shape {got.shape} != {ref.shape}")
            rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
            mean_abs = float(np.abs(got - ref).mean())
            print(f"hidden L{tap}: rel_l2={rel:.6g} mean_abs={mean_abs:.6g}")
            if rel > worst_rel:
                worst_rel = rel
                worst_tap = tap
            worst_mean = max(worst_mean, mean_abs)
            if rel > REL_TOL or mean_abs > MEAN_ABS_TOL:
                return fail(
                    f"hidden tap {tap} outside tolerance: "
                    f"rel_l2={rel:.6g} mean_abs={mean_abs:.6g}"
                )

        print(
            f"PASS: token prefix {prefix}, "
            f"worst hidden rel_l2 {worst_rel:.6g} at L{worst_tap}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
