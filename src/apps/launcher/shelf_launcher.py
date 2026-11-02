#!/usr/bin/env python3
"""Shelf launcher: resident model host orchestration.

    ./build/shelf-launcher compute --model-dir /path/to/model --tp 2 --device 0,1 ...

Starts one shelf-model-host per tensor parallel rank, waits until every
host answers a protocol ping, then runs the requested child with --model-host
pointing at those hosts.  Every host is shut down when the child exits.

Invariant: the target model is always loaded through the model hosts. Other
draft weights stay local to the child.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(sys.argv[0]).resolve().parent

CHILD_BINARIES = {
    "compute": "shelf-compute",
    "server": "shelf-server",
    "bench": "shelf-bench",
}

TP_SUBCOMMANDS = ("pp", "quality")

RESERVED_FLAGS = ("--model-dir", "--model-host", "--device", "--tp")


def fail(message: str) -> None:
    print(f"shelf-launcher: {message}", file=sys.stderr)
    raise SystemExit(2)


def resolve_binary(name: str, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            fail(f"{name} not found: {path}")
        return path
    for directory in (_HERE, _HERE.parent):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found is None:
        fail(f"{name} not found; pass --host-bin or add it to PATH")
    return Path(found)


def socket_spec(sockets: list[Path], tp: int) -> str:
    if tp == 1:
        return str(sockets[0])
    return ",".join(f"{rank}={path}" for rank, path in enumerate(sockets))


def child_flag(rest: list[str], flag: str) -> str | None:
    for index, arg in enumerate(rest):
        if arg == flag and index + 1 < len(rest):
            return rest[index + 1]
        prefix = flag + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def parse_devices(text: str, tp: int) -> list[int]:
    devices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            fail(f"bad --device entry: {part}")
        devices.append(int(part))
    if len(devices) != tp:
        example = ",".join(str(rank) for rank in range(tp))
        fail(f"--device expects {tp} entries for --tp {tp}, got {len(devices)} "
             f"(example: --device {example})")
    if len(set(devices)) != len(devices):
        fail(f"--device entries must be unique: {text}")
    return devices


class ModelHost:
    def __init__(
        self,
        binary: Path,
        socket: Path,
        device: int,
        tp: int,
        rank: int,
        model_dir: str,
        staging_gib: int,
        log_path: Path,
        allow_any_dir: bool,
        warmup: list[tuple[str, str]],
    ) -> None:
        self.binary = binary
        self.socket = socket
        self.device = device
        self.tp = tp
        self.rank = rank
        self.model_dir = model_dir
        self.staging_gib = staging_gib
        self.log_path = log_path
        self.allow_any_dir = allow_any_dir
        self.warmup = warmup
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        argv = [
            str(self.binary),
            "--socket", str(self.socket),
            "--device", str(self.device),
            "--staging-gib", str(self.staging_gib),
            "--warmup-tp-size", str(self.tp),
            "--warmup-tp-rank", str(self.rank),
        ]
        for kind, directory in self.warmup:
            argv += [f"--warmup-{kind}", directory]
        if self.allow_any_dir:
            argv.append("--allow-any-dir")
        env = dict(os.environ)
        env.setdefault("SHELF_ARENA_VMM", "1")
        log = open(self.log_path, "w")
        try:
            self.proc = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        finally:
            log.close()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                rc = self.proc.returncode if self.proc else -1
                fail(f"model host rank{self.rank} exited rc={rc}; see {self.log_path}")
            if self.socket.exists() and self.ping():
                return
            time.sleep(0.2)
        fail(f"model host rank{self.rank} not ready within {timeout_s:.0f}s; see {self.log_path}")

    def ping(self) -> bool:
        try:
            result = subprocess.run(
                [str(self.binary), "--socket", str(self.socket), "--ctl", "ping"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            subprocess.run(
                [str(self.binary), "--socket", str(self.socket), "--ctl", "shutdown"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None
        try:
            self.socket.unlink()
        except OSError:
            pass


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="shelf-launcher",
        description=(
            "Orchestrate resident model hosts for shelf-compute, "
            "shelf-server and shelf-bench. The target model is always "
            "loaded through the model hosts; a draft model is always loaded "
            "locally by the child. Launcher options come before <child>; "
            "everything after it is passed to the child unchanged."))
    parser.add_argument("child", choices=sorted(CHILD_BINARIES),
                        help="child entrypoint to run")
    parser.add_argument("--model-dir", required=True,
                        help="target model directory (loaded into the hosts)")
    parser.add_argument("--device", default="0",
                        help="comma separated GPU devices, one per rank; "
                             "count must match --tp (default 0)")
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2),
                        help="tensor parallel size; one host per rank (default 1)")
    parser.add_argument("--host-staging-gib", type=int, default=30,
                        help="host staging arena in GiB (default 30)")
    parser.add_argument("--host-timeout-s", type=float, default=600.0,
                        help="host ready timeout in seconds (default 600)")
    parser.add_argument("--socket-dir", default="/tmp",
                        help="directory for the host sockets (default /tmp)")
    parser.add_argument("--log-dir", default="/tmp",
                        help="directory for the host logs (default /tmp)")
    parser.add_argument("--host-bin", default=None,
                        help="shelf-model-host path")
    parser.add_argument("--allow-any-dir", action="store_true",
                        help="let the hosts acquire models outside --model-dir")
    return parser.parse_known_args(argv)


def build_child_argv(
    opts: argparse.Namespace,
    rest: list[str],
    child_binary: Path,
    sockets: list[Path],
    devices: list[int],
) -> list[str]:
    argv = [str(child_binary)]
    tail = list(rest)
    if opts.child == "bench":
        if not tail:
            fail("bench requires a subcommand (pp | tg | quality | batch | ...)")
        argv.append(tail.pop(0))
    tp_target = opts.child != "bench" or argv[1] in TP_SUBCOMMANDS
    device_arg = (",".join(str(device) for device in devices)
                  if tp_target else str(devices[0]))
    argv += [
        "--model-dir", str(Path(opts.model_dir).expanduser().resolve()),
        "--model-host", socket_spec(sockets, opts.tp),
        "--device", device_arg,
    ]
    if opts.tp > 1 and tp_target:
        argv += ["--tp", str(opts.tp)]
    argv += tail
    return argv


def main(argv: list[str] | None = None) -> int:
    opts, rest = parse_args(list(sys.argv[1:] if argv is None else argv))

    model_dir = Path(opts.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        fail(f"model directory not found: {model_dir}")
    opts.model_dir = str(model_dir)

    for arg in rest:
        flag = arg.split("=", 1)[0]
        if flag in RESERVED_FLAGS:
            fail(f"{flag} is owned by the launcher; pass it before <child>")

    devices = parse_devices(opts.device, opts.tp)

    host_binary = resolve_binary("shelf-model-host", opts.host_bin)
    child_binary = resolve_binary(CHILD_BINARIES[opts.child], None)

    socket_dir = Path(opts.socket_dir).expanduser().resolve()
    socket_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(opts.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    sockets = [socket_dir / f"rank{rank}.sock" for rank in range(opts.tp)]
    hosts: list[ModelHost] = []
    for rank, socket in enumerate(sockets):
        try:
            socket.unlink()
        except FileNotFoundError:
            pass
        warmup = [("qwen35", str(model_dir))]
        host = ModelHost(
            binary=host_binary,
            socket=socket,
            device=devices[rank],
            tp=opts.tp,
            rank=rank,
            model_dir=str(model_dir),
            staging_gib=opts.host_staging_gib,
            log_path=log_dir / f"shelf-launcher-host-{os.getpid()}-{rank}.log",
            allow_any_dir=opts.allow_any_dir,
            warmup=warmup,
        )
        host.start()
        hosts.append(host)

    child: subprocess.Popen | None = None
    rc = 1
    try:
        for host in hosts:
            host.wait_ready(opts.host_timeout_s)

        child_argv = build_child_argv(opts, rest, child_binary, sockets, devices)
        print("launcher: hosts ready; running " + " ".join(child_argv),
              file=sys.stderr)
        child = subprocess.Popen(child_argv)

        def forward(signum: int, _frame: object) -> None:
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signum)
                except OSError:
                    pass

        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        rc = child.wait()
    finally:
        for host in hosts:
            host.shutdown()

    if rc < 0:
        return 128 - rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
