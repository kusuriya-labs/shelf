#!/usr/bin/env python3
"""gfx1201 kernel ISA リソース契約のチェック。

build ディレクトリの compile_commands.json から実際のビルドフラグを取り、
対象 translation unit を assembly へコンパイルして、kernel ごとに
レジスタspill と private segment (local memory) 使用量を読む。
FP8 kernel は spill 0 / private segment 0 が契約である。
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

KERNELS = {
    "src/shelf/kernels/optimized/linear/fp8.hip": "gemm_fp8_",
    "src/shelf/kernels/optimized/activation_quantize.hip": "activation_quantize_fp8_",
    "src/shelf/kernels/optimized/attention/paged_decode.hip": "fp8_impl",
}

LIMITS = {
    "gemm_fp8_": {"vgpr_spill": 0, "sgpr_spill": 0, "private": 0},
    "activation_quantize_fp8_": {"vgpr_spill": 0, "sgpr_spill": 0, "private": 0},
    "fp8_impl": {"vgpr_spill": 0, "sgpr_spill": 0, "private": 0},
}

META = {
    "vgpr": r"\.vgpr_count:\s+(\d+)",
    "vgpr_spill": r"\.vgpr_spill_count:\s+(\d+)",
    "sgpr": r"\.sgpr_count:\s+(\d+)",
    "sgpr_spill": r"\.sgpr_spill_count:\s+(\d+)",
    "private": r"\.private_segment_fixed_size:\s+(\d+)",
    "lds": r"\.group_segment_fixed_size:\s+(\d+)",
    "name": r"\.name:\s+(\S+)",
}


def build_command(compile_db, source, source_root):
    for entry in compile_db:
        if entry["file"].endswith(source):
            args = shlex.split(entry.get("command") or " ".join(entry["arguments"]))
            out = []
            i = 0
            while i < len(args):
                a = args[i]
                if a == "-o":
                    i += 2
                    continue
                if a in ("-c", "-x"):
                    i += 2 if a == "-x" else 1
                    continue
                if a.endswith(".o") or a.endswith(".hip") or a.endswith(".cpp"):
                    i += 1
                    continue
                out.append(a)
                i += 1
            src = Path(entry["file"])
            if not src.is_absolute():
                src = Path(entry.get("directory", source_root)) / src
            out += ["-c", "-x", "hip", str(src)]
            return out
    return None


def demangle(names):
    if not names:
        return []
    res = subprocess.run(["c++filt"], input="\n".join(names),
                         capture_output=True, text=True)
    return res.stdout.splitlines()


def analyze(source, command, workdir, keep):
    with tempfile.NamedTemporaryFile(suffix=".s", dir=workdir, delete=False) as tmp:
        asm_path = tmp.name
    cmd = command + ["-S", "-o", asm_path]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        Path(asm_path).unlink(missing_ok=True)
        return None, proc.stderr
    text = Path(asm_path).read_text()
    if not keep:
        Path(asm_path).unlink(missing_ok=True)
    else:
        print(f"  asm: {asm_path}")
    rows = []
    for block in re.findall(r"  - \.args:.*?\.wavefront_size: \d+", text, re.S):
        got = {}
        for key, pattern in META.items():
            m = re.search(pattern, block)
            if m:
                got[key] = int(m.group(1)) if key != "name" else m.group(1)
        if "name" in got:
            rows.append(got)
    names = demangle([r["name"] for r in rows])
    for row, name in zip(rows, names):
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|\()", name)
        row["short"] = m.group(1) if m else name
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--list", action="store_true", help="全 kernel の資源値を表示")
    ap.add_argument("--keep-asm", action="store_true", help="assembly を残してパスを表示")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    db_path = build_dir / "compile_commands.json"
    if not db_path.is_file():
        print(f"FAIL: {db_path} が見つからない。先に build を行うこと", file=sys.stderr)
        return 1
    compile_db = json.loads(db_path.read_text())

    failures = []
    print(f"{'kernel':<62} {'VGPR':>5} {'spill':>6} {'SGPR':>5} {'spill':>6} {'priv':>5} {'LDS':>6}")
    for source, prefix in KERNELS.items():
        command = build_command(compile_db, source, build_dir)
        if command is None:
            failures.append(f"{source}: compile_commands.json にエントリが無い")
            continue
        rows, err = analyze(source, command, str(build_dir), args.keep_asm)
        if rows is None:
            failures.append(f"{source}: compile failed\n{err}")
            continue
        for row in rows:
            if prefix not in row["name"]:
                continue
            limits = LIMITS[prefix]
            bad = [key for key, cap in limits.items() if row.get(key, 0) > cap]
            short = row["short"].split("(")[0].split("::")[-1]
            if not args.list and bad:
                short = row["short"]
            print(f"{short[:62]:<62} {row.get('vgpr', 0):>5} {row.get('vgpr_spill', 0):>6} "
                  f"{row.get('sgpr', 0):>5} {row.get('sgpr_spill', 0):>6} "
                  f"{row.get('private', 0):>5} {row.get('lds', 0):>6}"
                  f"{'' if not bad else '  <-- ' + ','.join(bad)}")
            if bad:
                failures.append(f"{source}: {row['short']} が契約超過 {bad}")

    if failures:
        print("", file=sys.stderr)
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("FP8 kernel spill / private segment contract: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
