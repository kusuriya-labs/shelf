#!/usr/bin/env python3
"""Run the FP8 KV quality gate inside CTest. Skips (77) without a model dir."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    model_dir = os.environ.get("SHELF_MODEL_DIR")
    if not model_dir:
        print("SHELF_MODEL_DIR is not set", file=sys.stderr)
        return 77
    build_dir = Path(os.environ.get("SHELF_BUILD_DIR", REPO_ROOT / "build"))
    device = os.environ.get("SHELF_QUALITY_DEVICE", "0")
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "quality_gate.py"),
        "run",
        "--model-dir", model_dir,
        "--build-dir", str(build_dir),
        "--work-dir", str(build_dir / "quality"),
        "--device", device,
    ]
    command.extend(sys.argv[1:])
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
