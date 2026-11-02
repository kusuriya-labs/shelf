#!/usr/bin/env python3
"""Required acceptance runner.

Builds and runs the self-contained required test suite.
Any skip/failure is an acceptance failure.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_report(repo_root: Path) -> str:
    lines = []
    lines.append("=== Required Acceptance Report ===")
    lines.append("")
    lines.append("Git revision: " + (
        subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"))
    lines.append("Date: " + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    lines.append("uname -a: " + platform.uname().__str__())
    lines.append("")
    rocm_root = Path("/opt/rocm")
    version_file = rocm_root / ".version"
    rocm_version = version_file.read_text().strip() if version_file.is_file() else "unknown"
    lines.append(f"ROCM_VERSION: {rocm_version}")

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        for candidate in (rocm_root / "core-7.14" / "bin" / "hipcc",
                          rocm_root / "bin" / "hipcc"):
            if candidate.is_file():
                hipcc = str(candidate)
                break
    if hipcc:
        hip_version = subprocess.run(
            [hipcc, "--version"], capture_output=True, text=True,
        ).stdout.splitlines()
        lines.append(f"HIP_VERSION: {hip_version[0] if hip_version else 'unknown'}")
    else:
        lines.append("HIP_VERSION: unknown")

    cmake = shutil.which("cmake")
    lines.append("CMake version: " + (
        subprocess.run([cmake, "--version"], capture_output=True, text=True)
        .stdout.splitlines()[0] if cmake else "unknown"))

    ninja = shutil.which("ninja")
    lines.append("Ninja version: " + (
        subprocess.run([ninja, "--version"], capture_output=True, text=True)
        .stdout.strip() if ninja else "unknown"))
    lines.append("")

    rocm_smi = shutil.which("rocm-smi")
    gpu_count = 0
    if rocm_smi:
        gpu_out = subprocess.run(
            [rocm_smi, "--showid"], capture_output=True, text=True,
        ).stdout
        gpu_count = gpu_out.count("GPU[")
    lines.append(f"GPU count: {gpu_count}")

    if rocm_smi:
        gpu_info = subprocess.run(
            [rocm_smi, "--showid", "--showproduct.name"],
            capture_output=True, text=True,
        ).stdout
        lines.append(gpu_info)
    lines.append("")

    import os
    lines.append(f"CMAKE_HIP_ARCHITECTURES: {os.environ.get('CMAKE_HIP_ARCHITECTURES', 'not set')}")
    return "\n".join(lines) + "\n"


def parse_junit(xml_path: Path, report_path: Path) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    failed_tests = []
    skipped_tests = []

    for testcase in root.findall(".//testcase"):
        name = testcase.get("name", "")
        total += 1
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped_elem = testcase.find("skipped")
        if failure is not None:
            failed += 1
            failed_tests.append((name, failure.text or ""))
        elif error is not None:
            errors += 1
            failed_tests.append((name, error.text or ""))
        elif skipped_elem is not None:
            skipped += 1
            skipped_tests.append((name, skipped_elem.text or ""))
        else:
            passed += 1

    with open(report_path, "a") as f:
        f.write("\n=== Test Summary ===\n")
        f.write(f"Total: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write("\n")

        if skipped_tests:
            f.write("=== Required tests SKIPPED (treated as FAILURE) ===\n")
            for name, msg in skipped_tests:
                f.write(f"  SKIP: {name}\n")
                if msg.strip():
                    f.write(f"        {msg.strip()}\n")
            f.write("\n")

        if failed_tests:
            f.write("=== Required tests FAILED ===\n")
            for name, msg in failed_tests:
                f.write(f"  FAIL: {name}\n")
                if msg.strip():
                    f.write(f"        {msg.strip()}\n")
            f.write("\n")

        if skipped_tests or failed_tests:
            f.write("=== Final Result: ACCEPTANCE FAILED ===\n")
            return 1
        f.write("=== Final Result: ACCEPTANCE PASSED ===\n")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--arch", default=None,
                        help="HIP architecture to build (default: $CMAKE_HIP_ARCHITECTURES or gfx1201)")
    args = parser.parse_args()

    arch = args.arch or os.environ.get("CMAKE_HIP_ARCHITECTURES") or "gfx1201"

    build_dir = Path(args.build_dir).resolve()
    report_path = build_dir / "required-acceptance.txt"
    ctest_xml = build_dir / "required-ctest.xml"

    build_dir.mkdir(parents=True, exist_ok=True)

    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        cmake_cmd = [
            "cmake", "-S", str(REPO_ROOT), "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPHASESHIFT_ROCM_ROOT=/opt/rocm",
            "-DCMAKE_PREFIX_PATH=/opt/rocm",
            "-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm",
            f"-DCMAKE_HIP_ARCHITECTURES={arch}",
            "-DPHASESHIFT_BUILD_TESTS=ON",
            "-DPHASESHIFT_BUILD_OPTIONAL_TESTS=OFF",
        ]
        result = subprocess.run(cmake_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: cmake configure failed")
            print(result.stdout)
            print(result.stderr)
            return 1

    if not args.no_build:
        result = subprocess.run(
            [
                "cmake", "--build", str(build_dir),
                "--target", "phaseshift-required-tests",
                "--parallel",
            ],
            capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: cmake build failed")
            print(result.stdout)
            print(result.stderr)
            return 1

    with open(report_path, "w") as f:
        f.write(env_report(REPO_ROOT))
        f.write("\n=== Build result: SUCCESS ===\n")
        f.write("\n=== Running required tests ===\n")

    ctest_cmd = [
        "ctest", "--test-dir", str(build_dir),
        "-L", "required",
        "--output-on-failure",
        "--output-junit", str(ctest_xml),
    ]
    result = subprocess.run(ctest_cmd, capture_output=True, text=True)

    with open(report_path, "a") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if not ctest_xml.is_file():
        print("ERROR: ctest JUnit output not produced")
        return 1

    return parse_junit(ctest_xml, report_path)


if __name__ == "__main__":
    sys.exit(main())
