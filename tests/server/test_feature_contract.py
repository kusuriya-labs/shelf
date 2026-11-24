#!/usr/bin/env python3
"""機能契約の Fail Fast テスト。

未対応の機能組み合わせが警告だけで継続せず、起動時に拒否されることを確認する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support  # noqa: E402

COMPUTE = support.compute_binary()
MODEL = support.model_dir()


def run_compute(extra, model_dir=MODEL, device="0"):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(" ".join(map(str, support.prompt_ids("What is 2+2?"))))
        prompt = f.name
    try:
        args = [str(COMPUTE), "--model-dir", str(model_dir),
                "--input-ids-file", prompt, "--max-new-tokens", "1",
                "--max-seq-len", "128", "--device", device,
                "--kv-cache-dtype", "bf16", "--temperature", "0"] + extra
        return subprocess.run(args, capture_output=True, text=True, timeout=180)
    finally:
        os.unlink(prompt)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        print(detail[:500])
    return ok


def test_pp_prefix_cache_rejected():
    result = run_compute(["--pipeline-parallel", "2", "--prefix-cache", "1"],
                         device="0,1")
    ok = result.returncode != 0 and "does not support prefix cache" in result.stderr
    return check("pp + prefix cache is rejected", ok, result.stderr)


def test_unknown_quant_method_rejected():
    with tempfile.TemporaryDirectory() as d:
        temp = Path(d)
        config = json.loads((MODEL / "config.json").read_text())
        config["quantization_config"] = {"quant_method": "awq"}
        (temp / "config.json").write_text(json.dumps(config))
        result = run_compute([], model_dir=temp)
    ok = result.returncode != 0 and "unsupported HF quant_method" in result.stderr
    return check("unknown quant method is rejected", ok, result.stderr)


def main():
    ok = True
    ok &= test_pp_prefix_cache_rejected()
    ok &= test_unknown_quant_method_rejected()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
