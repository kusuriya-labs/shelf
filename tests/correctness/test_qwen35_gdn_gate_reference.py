#!/usr/bin/env python3
"""GDN の gate 活性を f64 参照で固定する test。

fixture は合成 Qwen3.5 checkpoint で、`output_gate_type` を**宣言しない**。
参照実装（transformers の Qwen3_5RMSNormGated）は GDN の gate を silu に固定しているため、
Shelf も config key に依存せず silu を emission しなければならない。gate が sigmoid に
落ちると层 0（GDN）の出力が rel_l2 0.9 規模で逸脱するので、この test はその回帰を捕まえる。

fixture の再生成: tools/qwen35_gdn_tiny_fixture.py（外部の qwen35-f64-oracle が必要）。
build dir は SHELF_BUILD_DIR、無ければ repo の build/ を使う。
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
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "qwen35" / "qwen35_gdn_tiny"
TAP = 0
REL_TOL = 2e-2
MEAN_ABS_TOL = 2e-2


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    fixture = json.loads((FIXTURE_DIR / "oracle.json").read_text())
    hidden_ref = load_file(str(FIXTURE_DIR / "oracle_hidden.safetensors"))
    geometry = fixture["model"]["geometry"]
    hidden_size = int(geometry["hidden_size"])
    input_ids = [int(v) for v in fixture["case"]["input_ids"]]
    if fixture["model"].get("declares_output_gate_type"):
        return fail("the fixture must not declare output_gate_type")
    if "output_gate_type" in json.loads((FIXTURE_DIR / "config.json").read_text())["text_config"]:
        return fail("the fixture config declares output_gate_type")

    build_dir = Path(os.environ.get("SHELF_BUILD_DIR", REPO_ROOT / "build"))
    compute = build_dir / "shelf-compute"
    if not compute.exists():
        print(f"shelf-compute not found: {compute}", file=sys.stderr)
        return 77

    with tempfile.TemporaryDirectory(prefix="shelf_gdn_gate_") as tmp:
        tmp_path = Path(tmp)
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("\n".join(str(v) for v in input_ids))
        hidden_dir = tmp_path / "hidden"
        hidden_dir.mkdir()
        env = os.environ.copy()
        env["SHELF_QWEN35_KERNEL_MODE"] = "correctness"
        env["SHELF_DEBUG_HIDDEN_FILE"] = str(hidden_dir)
        run = subprocess.run(
            [
                str(compute),
                "--model-dir", str(FIXTURE_DIR),
                "--input-ids-file", str(ids_path),
                "--max-new-tokens", "1",
                "--device", os.environ.get("SHELF_TEST_DEVICE", "0"),
                "--arena-gib", os.environ.get("SHELF_GDN_GATE_ARENA_GIB", "2"),
                "--max-concurrent-requests", "1",
                "--prefix-cache", "0",
                "--hidden-taps", str(TAP),
            ],
            capture_output=True, text=True, env=env,
        )
        tap_path = hidden_dir / f"prefill_L{TAP}.bin"
        if not tap_path.exists():
            sys.stderr.write(run.stdout[-3000:])
            sys.stderr.write(run.stderr[-3000:])
            return fail(f"hidden tap {TAP} was not dumped (exit {run.returncode})")

        got = np.fromfile(tap_path, dtype=np.float32).reshape(-1, hidden_size)
        ref = np.asarray(hidden_ref[f"hidden_{TAP + 1:02d}"], dtype=np.float64)[: got.shape[0]]
        if ref.shape != got.shape:
            return fail(f"hidden tap {TAP} shape {got.shape} != {ref.shape}")

        rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
        mean_abs = float(np.abs(got - ref).mean())
        print(f"gdn layer {TAP}: rel_l2={rel:.6g} mean_abs={mean_abs:.6g}")
        if rel > REL_TOL or mean_abs > MEAN_ABS_TOL:
            return fail(
                "GDN layer output is outside the f64 reference band. "
                f"rel_l2={rel:.6g} mean_abs={mean_abs:.6g} "
                "(a sigmoid GDN gate instead of silu lands near rel_l2 0.9)"
            )

        match = re.search(r"^GENERATED_IDS=([0-9,]*)$", run.stdout, re.MULTILINE)
        if match is None:
            return fail("GENERATED_IDS was not reported")
        generated = [int(v) for v in match.group(1).split(",") if v]
        expected = int(fixture["case"]["prompt_last_argmax"])
        if not generated:
            return fail("no token generated")
        if generated[0] != expected:
            return fail(
                f"prompt continuation argmax differs: got {generated[0]} expected {expected} "
                f"(oracle top1-top2 margin {fixture['case']['prompt_margin_last']:.3f})"
            )
        print(f"PASS: rel_l2 {rel:.6g}, prompt argmax {generated[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
