#!/usr/bin/env python3
"""MTP drafter の GENERATED_IDS が target-only greedy と一致することを固定する。

- 2048 token の prose prompt を使い、partial accept を含む条件で 64 token を生成する。
- 同じ prompt / sampling で `shelf-compute` を 2 回実行する。
- `--mtp` ありとなしの GENERATED_IDS が完全一致することを要求する。
- partial accept の compact log replay が実際に走ったことも確認する。
- model dir は SHELF_MODEL_DIR_MTP（優先）または SHELF_MODEL_DIR、
  build dir は SHELF_BUILD_DIR で指定する。無い場合は exit 77 で skip する。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_SOURCE = REPO_ROOT / "tests" / "data" / "mtp_perf" / "prose.txt"
PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 64
MTP_DRAFTS = 7


def run(compute: Path, model_dir: str, ids_path: Path, device: str, extra: list[str]):
    command = [
        str(compute),
        "--model-dir", model_dir,
        "--input-ids-file", str(ids_path),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--max-seq-len", "4096",
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


def metric(stdout: str, key: str) -> int:
    match = re.search(rf"^{key}=([0-9]+)$", stdout, re.MULTILINE)
    return int(match.group(1)) if match is not None else 0


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
    if not PROMPT_SOURCE.exists():
        print(f"prompt source not found: {PROMPT_SOURCE}", file=sys.stderr)
        return 77
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not available", file=sys.stderr)
        return 77
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as error:
        print(f"tokenizer load failed: {error}", file=sys.stderr)
        return 1

    tp = int(os.environ.get("SHELF_TP", "1"))
    device = os.environ.get("SHELF_TEST_DEVICE", "0,1" if tp > 1 else "0")
    tp_args = ["--tensor-parallel", str(tp)] if tp > 1 else []
    text = PROMPT_SOURCE.read_text(encoding="utf-8")
    base_ids = tokenizer.encode(text)
    if not base_ids:
        print("prompt source produced no tokens", file=sys.stderr)
        return 1
    input_ids: list[int] = []
    while len(input_ids) < PROMPT_TOKENS:
        input_ids.extend(base_ids)
    input_ids = input_ids[:PROMPT_TOKENS]

    with tempfile.TemporaryDirectory(prefix="shelf_mtp_parity_") as tmp:
        ids_path = Path(tmp) / "ids.txt"
        ids_path.write_text("\n".join(str(v) for v in input_ids))

        target, code = run(compute, model_dir, ids_path, device, tp_args)
        if target is None:
            return code
        mtp, code = run(
            compute, model_dir, ids_path, device,
            ["--mtp", "--mtp-drafts", str(MTP_DRAFTS), *tp_args],
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
        rounds = metric(mtp.stdout, "MTP_ROUNDS")
        restores = metric(mtp.stdout, "MTP_GDN_RESTORES")
        if rounds == 0:
            print("FAIL: MTP did not execute a speculative round", file=sys.stderr)
            return 1
        if restores == 0:
            print("FAIL: MTP test did not exercise partial accept", file=sys.stderr)
            return 1
        print(
            f"MTP_PARITY: PASS ({len(target_ids)} tokens, rounds={rounds}, "
            f"gdn_restores={restores})"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
