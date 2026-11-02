#!/usr/bin/env python3
"""compute_pids はこのテストが起動した compute だけを返す."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import OWNER_ENV, Checker, compute_pids  # noqa: E402

SLEEPER = "import time; time.sleep(60)"


def spawn(args: list[str], *, owned: bool) -> subprocess.Popen:
    env = dict(os.environ)
    if not owned:
        env.pop(OWNER_ENV, None)
    return subprocess.Popen([sys.executable, "-c", SLEEPER, *args], env=env)


def main() -> int:
    checker = Checker("compute-pids-scope")
    procs = []
    try:
        owned = spawn(["phaseshift-compute", "--serve-stdio"], owned=True)
        procs.append(owned)
        foreign = spawn(["phaseshift-compute", "--serve-stdio"], owned=False)
        procs.append(foreign)
        wrong_op = spawn(["phaseshift-compute", "--serve-once"], owned=True)
        procs.append(wrong_op)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if all(proc.poll() is None for proc in procs):
                break
            time.sleep(0.01)

        pids = compute_pids()
        checker.check("owned-compute-counted", owned.pid in pids, repr(pids))
        checker.check("foreign-compute-ignored", foreign.pid not in pids, repr(pids))
        checker.check("non-serve-stdio-ignored", wrong_op.pid not in pids, repr(pids))
        checker.check("exactly-owned", pids == sorted([owned.pid]), repr(pids))
    except Exception as exc:  # noqa: BLE001 - report as test failure
        checker.check("exception", False, repr(exc))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
