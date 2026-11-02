"""Resident model host session helpers shared by the GPU test runner and the
server regression runner.

The wire protocol is the test-only PhaseShift resident protocol:
24-byte little-endian header (magic, version, command, status, payload_size)
followed by the payload.
"""

from __future__ import annotations

import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x31575350
VERSION = 1

CMD_HELLO = 1
CMD_ACQUIRE = 2
CMD_RELEASE = 3
CMD_HEALTH = 4
CMD_STATS = 5
CMD_SHUTDOWN = 6

STATUS_OK = 0
STATUS_ERROR = 1
STATUS_NOT_FOUND = 2
STATUS_UNHEALTHY = 3

HEADER = struct.Struct("<IIIIQ")

SOCKET_ENV = "PHASESHIFT_MODEL_HOST_SOCKET"
DISABLE_ENV = "PHASESHIFT_DISABLE_RESIDENT_MODEL"


def default_host_binary(build_dir: Path) -> Path:
    for path in (build_dir / "phaseshift-model-host",
                 build_dir / "tests" / "phaseshift-model-host"):
        if path.is_file():
            return path
    return build_dir / "phaseshift-model-host"


def existing_dir(*values: Optional[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in out:
                out.append(resolved)
    return out


def dflash2_dirs_from_env() -> List[str]:
    return existing_dir(os.environ.get("PHASESHIFT_MODEL_DIR_DFLASH2"))


class ResidentProtocolError(RuntimeError):
    pass


class ResidentProtocolClient:
    def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
        import socket as socket_module

        self._timeout = timeout
        self._sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(socket_path)
        self._buffer = bytearray()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "ResidentProtocolClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ResidentProtocolError("resident host closed the connection")
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def call(self, command: int, payload: bytes = b"") -> Tuple[int, bytes]:
        self._sock.sendall(HEADER.pack(MAGIC, VERSION, command, STATUS_OK, len(payload)) + payload)
        magic, version, _cmd, status, size = HEADER.unpack(self._read_exact(HEADER.size))
        if magic != MAGIC or version != VERSION:
            raise ResidentProtocolError("bad resident response header")
        body = self._read_exact(size) if size else b""
        return status, body

    def stats(self) -> str:
        status, body = self.call(CMD_STATS)
        if status != STATUS_OK:
            raise ResidentProtocolError("stats rejected")
        return body.decode("utf-8", "replace")

    def health(self) -> bool:
        status, _body = self.call(CMD_HEALTH)
        return status == STATUS_OK

    def release(self, key: str) -> bool:
        status, _body = self.call(CMD_RELEASE, key.encode("utf-8"))
        return status in (STATUS_OK, STATUS_NOT_FOUND)

    def shutdown(self) -> bool:
        status, _body = self.call(CMD_SHUTDOWN)
        return status == STATUS_OK


_TOP_LEVEL_KEYS = frozenset({
    "proto_version", "device", "last_health", "disk_load_count", "attach_count",
    "entries", "resident_bytes", "socket",
})


def parse_stats(text: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    entries: List[Dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("entry "):
            entry: Dict[str, str] = {}
            for token in line[len("entry "):].split():
                if "=" in token:
                    key, _, value = token.partition("=")
                    entry[key] = value
            if entry:
                entries.append(entry)
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key in _TOP_LEVEL_KEYS:
            values[key] = value
    values["_entries"] = entries
    return values


def stat_int(stats: Dict[str, object], key: str) -> int:
    try:
        return int(str(stats.get(key, "0")))
    except ValueError:
        return 0


class ResidentSession:
    def __init__(
        self,
        build_dir: Path,
        device: int = 0,
        staging_gib: int = 30,
        socket_path: Optional[str] = None,
        warmup_model_dirs: Optional[Sequence[str]] = None,
        warmup_dflash2_dirs: Optional[Sequence[str]] = None,
        allow_any_dir: bool = True,
        health_timeout_ms: int = 5000,
        ready_timeout: float = 300.0,
    ) -> None:
        self.build_dir = Path(build_dir)
        self.device = device
        self.staging_gib = staging_gib
        self.socket_path = socket_path or f"/tmp/phaseshift-resident-{os.getpid()}-{device}.sock"
        self.warmup_model_dirs = list(warmup_model_dirs or [])
        self.warmup_dflash2_dirs = list(warmup_dflash2_dirs or [])
        self.allow_any_dir = allow_any_dir
        self.health_timeout_ms = health_timeout_ms
        self.ready_timeout = ready_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = Path(f"/tmp/phaseshift-resident-host-{os.getpid()}.log")

    def start(self) -> None:
        host = default_host_binary(self.build_dir)
        if not host.is_file():
            raise ResidentProtocolError(f"resident host binary not found: {host}")
        argv = [
            str(host),
            "--socket", self.socket_path,
            "--device", str(self.device),
            "--staging-gib", str(self.staging_gib),
            "--health-timeout-ms", str(self.health_timeout_ms),
        ]
        if self.allow_any_dir:
            argv.append("--allow-any-dir")
        for model_dir in self.warmup_model_dirs:
            argv.extend(["--warmup-qwen35", model_dir])
        for dflash_dir in self.warmup_dflash2_dirs:
            argv.extend(["--warmup-dflash2", dflash_dir])

        env = dict(os.environ)
        env.setdefault("PHASESHIFT_ARENA_VMM", "1")
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                log.close()
                raise ResidentProtocolError(
                    f"resident host exited rc={self.proc.returncode}; see {self.log_path}")
            if Path(self.socket_path).exists():
                try:
                    with ResidentProtocolClient(self.socket_path, timeout=5.0) as client:
                        client.stats()
                    log.close()
                    return
                except (OSError, ResidentProtocolError):
                    pass
            time.sleep(0.2)
        log.close()
        raise ResidentProtocolError(f"resident host not ready; see {self.log_path}")

    def env(self) -> Dict[str, str]:
        return {SOCKET_ENV: self.socket_path}

    def stats(self) -> Dict[str, object]:
        with ResidentProtocolClient(self.socket_path) as client:
            return parse_stats(client.stats())

    def disk_load_count(self) -> int:
        return stat_int(self.stats(), "disk_load_count")

    def attach_count(self) -> int:
        return stat_int(self.stats(), "attach_count")

    def health(self) -> bool:
        try:
            with ResidentProtocolClient(self.socket_path) as client:
                return client.health()
        except (OSError, ResidentProtocolError):
            return False

    def release_except(self, keep_model_dirs: Sequence[str]) -> List[str]:
        keep = {str(Path(d).resolve()) for d in keep_model_dirs}
        released: List[str] = []
        with ResidentProtocolClient(self.socket_path) as client:
            status, body = client.call(CMD_STATS)
            if status != STATUS_OK:
                return released
            entries = parse_stats(body.decode("utf-8", "replace")).get("_entries", [])
            if not isinstance(entries, list):
                entries = []
            for entry in entries:
                key = entry.get("key", "")
                if not key:
                    continue
                parts = key.split(":")
                model_dir = parts[1] if len(parts) > 1 else ""
                if model_dir in keep:
                    continue
                if client.release(key):
                    released.append(model_dir)
        return released

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            with ResidentProtocolClient(self.socket_path, timeout=5.0) as client:
                client.shutdown()
        except (OSError, ResidentProtocolError):
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
            os.unlink(self.socket_path)
        except OSError:
            pass
