#!/usr/bin/env python3
"""Resident model session runner for the heavy GPU test executables.

Starts one phaseshift-model-host per session, then launches each
selected test as its own worker process with the resident socket in the
environment.  The host owns the compact resident weight block; workers own
only ephemeral state and are killed as a process group on deadline.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from resident_session import (  # noqa: E402
    DISABLE_ENV,
    SOCKET_ENV,
    ResidentProtocolError,
    ResidentSession,
    dflash2_dirs_from_env,
)

DRAFT_ONLY_TESTS = [
    "test_dflash2_gate3_real",
    "test_dflash2_gate3_perf",
    "test_dflash2_gate4_real",
    "test_dflash2_gate4_perf",
    "test_dflash2_gate5_real",
    "test_dflash2_gate5_perf",
    "test_dflash2_gate51_trace",
]

TARGET_DRAFT_TESTS = [
    "test_dflash2_gate6_real",
    "test_dflash2_gate6_perf",
    "test_dflash2_gate7_real",
    "test_dflash2_gate7_perf",
    "test_dflash2_gate8_live",
    "test_dflash2_gate9_e2e",
    "test_dflash2_gate11_profile",
]

TARGET_ONLY_TESTS = [
    "test_dflash2_target_taps",
    "test_dflash2_gate51_target_fixture",
    "test_dflash2_gate11c_perf",
    "test_dflash2_gate11d_verify_profile",
    "test_dflash2_gate11h_verify_profile",
]

RESIDENT_INFRA_TESTS = [
    "test_resident_model_ipc",
]

TP_TESTS = [
    "test_qwen35_tp_e2e",
    "test_dflash2_tp_e2e",
]

GROUPS = {
    "dflash2-draft": DRAFT_ONLY_TESTS,
    "dflash2-full": TARGET_DRAFT_TESTS,
    "qwen35-target": TARGET_ONLY_TESTS,
    "resident": RESIDENT_INFRA_TESTS,
    "tp": TP_TESTS,
    "host": DRAFT_ONLY_TESTS + TARGET_DRAFT_TESTS + TARGET_ONLY_TESTS + RESIDENT_INFRA_TESTS,
    "all": DRAFT_ONLY_TESTS + TARGET_DRAFT_TESTS + TARGET_ONLY_TESTS
    + RESIDENT_INFRA_TESTS + TP_TESTS,
}

PROFILE_DRAFT = "dflash2-draft"
PROFILE_TARGET = "qwen35-target"
PROFILE_FULL = "dflash2-full"
PROFILE_TP = "qwen35-tp"
PROFILE_NONE = "none"


def profile_for(test: str) -> str:
    if test in DRAFT_ONLY_TESTS:
        return PROFILE_DRAFT
    if test in TARGET_DRAFT_TESTS:
        return PROFILE_FULL
    if test in TARGET_ONLY_TESTS:
        return PROFILE_TARGET
    if test in TP_TESTS:
        return PROFILE_TP
    return PROFILE_NONE


def dirs_for_profile(profile: str) -> Tuple[List[str], List[str]]:
    from resident_session import existing_dir

    target = existing_dir(
        os.environ.get("PHASESHIFT_SERVER_MODEL_DIR"),
        os.environ.get("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET"),
        os.environ.get("PHASESHIFT_MODEL_DIR"),
    )
    drafts = dflash2_dirs_from_env()
    if profile == PROFILE_DRAFT:
        return [], drafts
    if profile == PROFILE_TARGET:
        return target, []
    if profile == PROFILE_FULL:
        return target, drafts
    if profile == PROFILE_TP:
        return [], []
    return target, drafts


def resolve_tests(names: Sequence[str], build_dir: Path) -> List[str]:
    out: List[str] = []
    for name in names:
        binary = build_dir / "tests" / name
        if not binary.is_file():
            raise SystemExit(f"test binary not found: {binary}")
        out.append(name)
    return out


def run_worker(
    build_dir: Path,
    name: str,
    env: Dict[str, str],
    timeout: float,
) -> Tuple[int, float, bool, str]:
    binary = build_dir / "tests" / name
    started = time.time()
    proc = subprocess.Popen(
        [str(binary)],
        cwd=str(build_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    output = ""
    try:
        output, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            output, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            output = ""
        rc = -9
    return rc, time.time() - started, timed_out, output or ""


def tail(text: str, lines: int = 30) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", default=None,
                        help="comma separated GPU indices (default: --device)")
    parser.add_argument("--staging-gib", type=int, default=30)
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--group", default=None, choices=sorted(GROUPS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-group-by-profile", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--inject-timeout", action="store_true")
    parser.add_argument("--fault-timeout", type=float, default=10.0)
    parser.add_argument("--check-fail-closed", action="store_true")
    parser.add_argument("--disable-resident", action="store_true")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    if args.list:
        for group, tests in GROUPS.items():
            print(f"{group}: {len(tests)}")
            for test in tests:
                print(f"  {test}")
        return 0

    names: List[str] = list(args.test)
    if args.group:
        names.extend(GROUPS[args.group])
    if not names:
        raise SystemExit("pass --test NAME or --group NAME")

    seen = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    tests = resolve_tests(ordered, build_dir)

    if not args.no_group_by_profile and not args.inject_timeout:
        profile_order = [PROFILE_DRAFT, PROFILE_TARGET, PROFILE_FULL,
                         PROFILE_TP, PROFILE_NONE]
        tests.sort(key=lambda name: profile_order.index(profile_for(name)))

    devices = [int(part) for part in (args.devices or str(args.device)).split(",")
               if part.strip()]
    first_profile = profile_for(tests[0])
    warmup_dirs, warmup_drafts = dirs_for_profile(first_profile)

    sessions: List[ResidentSession] = []
    for device in devices:
        sessions.append(ResidentSession(
            build_dir=build_dir,
            device=device,
            staging_gib=args.staging_gib,
            warmup_model_dirs=warmup_dirs,
            warmup_dflash2_dirs=warmup_drafts,
        ))

    print(f"===== resident session start devices={','.join(map(str, devices))}")
    for session in sessions:
        session.start()
        print(f"host ready device={session.device} socket={session.socket_path} "
              f"disk_load_count={session.disk_load_count()}")

    def socket_mapping() -> str:
        return ",".join(f"{index}={session.socket_path}"
                        for index, session in enumerate(sessions))

    def total_loads() -> int:
        return sum(session.disk_load_count() for session in sessions)

    def total_attach() -> int:
        return sum(session.attach_count() for session in sessions)

    def all_health() -> bool:
        return all(session.health() for session in sessions)

    base_loads = total_loads()
    print(f"session ready disk_load_count={base_loads} attach_count={total_attach()}")
    atexit.register(lambda: [host.shutdown() for host in sessions])

    if args.check_fail_closed:
        for session in sessions:
            if session.proc is not None:
                session.proc.kill()
                session.proc.wait(timeout=15)
                session.proc = None
        if all_health():
            print("FAIL: health check still healthy after host death")
            return 1
        print("PASS: health check failed after host death (session aborts fail-closed)")
        return 0

    env = dict(os.environ)
    env[SOCKET_ENV] = socket_mapping()
    env["PHASESHIFT_TEST_STATS_SOCKET"] = sessions[0].socket_path
    if args.disable_resident:
        env[DISABLE_ENV] = "1"
    env.setdefault("PHASESHIFT_TEST_DEVICE", str(devices[0]))

    results: List[Tuple[str, int, float, bool]] = []
    abort = False
    session_started = time.time()

    if args.inject_timeout:
        worker = "test_resident_timeout_worker"
        if (build_dir / "tests" / worker).is_file():
            print(f"----- fault injection: {worker} deadline={args.fault_timeout}s")
            fault_env = dict(env)
            fault_env["PHASESHIFT_TEST_SLEEP_SEC"] = "3600"
            loads_before = total_loads()
            rc, elapsed, timed_out, output = run_worker(
                build_dir, worker, fault_env, args.fault_timeout)
            print(output.strip())
            if not timed_out:
                print(f"FAIL: fault worker exited rc={rc} before the deadline")
                return 1
            if not all_health():
                print("FAIL: host unhealthy after worker SIGKILL")
                return 1
            loads_after = total_loads()
            if loads_after != loads_before:
                print(f"FAIL: disk_load_count changed {loads_before} -> {loads_after}")
                return 1
            print(f"PASS: host survived worker SIGKILL disk_load_count={loads_after} "
                  f"attach_count={total_attach()} kill_ms={elapsed * 1000:.0f}")
        else:
            print(f"SKIP: {worker} not built")

    current_profile = None
    for name in tests:
        profile = profile_for(name)
        if profile != current_profile:
            keep_dirs, _ = dirs_for_profile(profile)
            released: List[str] = []
            for host in sessions:
                released.extend(host.release_except(keep_dirs))
            if released:
                print(f"----- profile {current_profile} -> {profile}: released "
                      f"{', '.join(sorted(set(released)))}")
            if not all_health():
                print("FAIL: host unhealthy during profile switch; aborting session")
                abort = True
                break
            current_profile = profile
            print(f"----- profile {profile} disk_load_count={total_loads()}")

        loads_before = total_loads()
        if profile == PROFILE_TP:
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        else:
            env["HIP_VISIBLE_DEVICES"] = str(devices[0])
        print(f"----- run {name}")
        rc, elapsed, timed_out, output = run_worker(build_dir, name, env, args.timeout)
        results.append((name, rc, elapsed, timed_out))
        print(output.rstrip())
        status = "TIMEOUT" if timed_out else f"rc={rc}"
        print(f"----- {name}: {status} {elapsed:.1f}s "
              f"disk_load_count {loads_before} -> {total_loads()}")
        if rc == 77 and not timed_out:
            print(f"----- {name}: SKIP")

        if timed_out or rc not in (0, 77):
            healthy = all_health()
            if not healthy:
                print("FAIL: host unhealthy after worker failure; aborting session")
                abort = True
                break

    total = time.time() - session_started
    final_loads = total_loads()
    passed = sum(1 for _, rc, _, timed in results if rc == 0 and not timed)
    skipped = sum(1 for _, rc, _, timed in results if rc == 77 and not timed)
    failed = len(results) - passed - skipped

    print("===== resident session summary")
    print(f"tests={len(results)} passed={passed} skipped={skipped} failed={failed}"
          f"{' aborted=1' if abort else ''}")
    print(f"disk_load_count {base_loads} -> {final_loads}")
    print(f"attach_count={total_attach()}")
    print(f"wall_time_s={total:.1f}")
    for name, rc, elapsed, timed_out in results:
        marker = "SKIP" if rc == 77 else ("TIMEOUT" if timed_out else
                                          ("PASS" if rc == 0 else "FAIL"))
        print(f"  {marker} {name} {elapsed:.1f}s rc={rc}")

    for session in sessions:
        session.shutdown()
    if abort:
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
