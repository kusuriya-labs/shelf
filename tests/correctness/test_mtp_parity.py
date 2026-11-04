#!/usr/bin/env python3
"""MTP drafter の GENERATED_IDS が target-only greedy と一致することを固定する。

- 同じ prompt / sampling で `shelf-compute` を 2 回実行する。
- `--mtp` ありとなしの GENERATED_IDS が完全一致することを要求する。
- model dir は SHELF_MODEL_DIR_MTP（優先）または SHELF_MODEL_DIR、
  build dir は SHELF_BUILD_DIR で指定する。無い場合は exit 77 で skip する。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_JSON = REPO_ROOT / "tests" / "fixtures" / "qwen35" / "qwen38_27b_fp8_cpu_oracle.json"
MAX_NEW_TOKENS = 16
MTP_DRAFTS = 4


def run(compute: Path, model_dir: str, ids_path: Path, device: str, extra: list[str]):
    command = [
        str(compute),
        "--model-dir", model_dir,
        "--input-ids-file", str(ids_path),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--max-seq-len", "128",
        "--max-num-batched-tokens", "32",
        "--device", device,
        "--prefix-cache", "0",
        *extra,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:])
        sys.stderr.write(completed.stderr[-4000:])
        return None, completed.returncode
    return completed, 0


def generated_ids(stdout: str) -> list[int] | None:
    match = re.search(r"^GENERATED_IDS=([0-9,]*)$", stdout, re.MULTILINE)
    if match is None:
        return None
    return [int(v) for v in match.group(1).split(",") if v]


def main() -> int:
    model_dir = os.environ.get("SHELF_MODEL_DIR_MTP") or os.environ.get("SHELF_MODEL_DIR")
    if not model_dir:
        print("SHELF_MODEL_DIR_MTP is not set", file=sys.stderr)
        return 77
    build_dir = Path(os.environ.get("SHELF_BUILD_DIR", REPO_ROOT / "build"))
    compute = build_dir / "shelf-compute"
    if not compute.exists():
        print(f"shelf-compute not found: {compute}", file=sys.stderr)
        return 77
    device = os.environ.get("SHELF_TEST_DEVICE", "0")

    input_ids = [248041, 77091]
    if FIXTURE_JSON.exists():
        fixture = json.loads(FIXTURE_JSON.read_text())
        input_ids = [int(v) for v in fixture["case"]["input_ids"]]

    with tempfile.TemporaryDirectory(prefix="shelf_mtp_parity_") as tmp:
        ids_path = Path(tmp) / "ids.txt"
        ids_path.write_text("\n".join(str(v) for v in input_ids))

        target, code = run(compute, model_dir, ids_path, device, [])
        if target is None:
            return code
        mtp, code = run(
            compute, model_dir, ids_path, device,
            ["--mtp", "--mtp-drafts", str(MTP_DRAFTS)],
        )
        if mtp is None:
            return code

        target_ids = generated_ids(target.stdout)
        mtp_ids = generated_ids(mtp.stdout)
        if target_ids is None or mtp_ids is None:
            print("GENERATED_IDS was not reported", file=sys.stderr)
            return 1
        if target_ids != mtp_ids:
            print(f"FAIL: target={target_ids}", file=sys.stderr)
            print(f"FAIL: mtp   ={mtp_ids}", file=sys.stderr)
            return 1
        rounds = re.search(r"^MTP_ROUNDS=([0-9]+)$", mtp.stdout, re.MULTILINE)
        if rounds is None or int(rounds.group(1)) == 0:
            print("FAIL: MTP did not execute a speculative round", file=sys.stderr)
            return 1
        print(f"MTP_PARITY: PASS ({len(target_ids)} tokens)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
