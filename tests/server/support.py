"""Shared helpers for the Shelf Server acceptance scripts.

These scripts drive real processes (shelf-compute, shelf-server).
They never import the Shelf C++ API.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO_ROOT / "src" / "apps" / "server" / "shelf_server.py"
COMMON_DIR = REPO_ROOT / "src" / "apps" / "common"


def build_dir() -> Path:
    return Path(os.environ.get("SHELF_BUILD_DIR", REPO_ROOT / "build"))


def compute_binary() -> Path:
    env = os.environ.get("SHELF_COMPUTE")
    if env:
        return Path(env)
    return build_dir() / "shelf-compute"


def server_binary() -> Path:
    env = os.environ.get("SHELF_SERVER")
    if env:
        return Path(env)
    return build_dir() / "shelf-server"


def model_dir() -> Path:
    env = (
        os.environ.get("SHELF_SERVER_MODEL_DIR")
        or os.environ.get("SHELF_MODEL_DIR")
    )
    if env:
        return Path(env)
    return REPO_ROOT / "models" / "Qwen3.8-27B-FP8"


_PROCESSOR_CACHE: dict[str, object] = {}


def load_processor():
    if "processor" not in _PROCESSOR_CACHE:
        if str(COMMON_DIR) not in sys.path:
            sys.path.insert(0, str(COMMON_DIR))
        from shelf_chat import codec

        _PROCESSOR_CACHE["processor"] = codec.load_processor(str(model_dir()))
    return _PROCESSOR_CACHE["processor"]


def prompt_ids(text: str) -> list[int]:
    processor = load_processor()
    from shelf_chat import codec

    return codec.prompt_ids(
        processor, [{"role": "user", "content": text}], enable_thinking=False)


def decode_ids(generated_ids) -> str:
    processor = load_processor()
    from shelf_chat import codec

    return codec.decode_generated(processor, generated_ids)


def generation_stop_tokens() -> list[int]:
    """Generation stop tokens from generation_config.json (text_config fallback)."""
    generation_config = model_dir() / "generation_config.json"
    if generation_config.is_file():
        config = json.loads(generation_config.read_text())
        eos = config.get("eos_token_id") if isinstance(config, dict) else None
        if isinstance(eos, bool):
            eos = None
        if isinstance(eos, int):
            return [int(eos)]
        if isinstance(eos, list):
            tokens = [int(t) for t in eos if isinstance(t, int) and not isinstance(t, bool)]
            if tokens:
                return tokens
    with open(model_dir() / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config", config)
    eos = text_config.get("eos_token_id")
    return [int(eos)] if eos is not None else []


def generation_stop_probe() -> dict:
    """Developer diagnostic for the generation stop token contract."""
    processor = load_processor()
    with open(model_dir() / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config", config)
    tokenizer = getattr(processor, "tokenizer", processor)
    model_eos = text_config.get("eos_token_id")
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    stop_tokens = generation_stop_tokens()

    def token_repr(token_id):
        if token_id is None:
            return None
        try:
            return tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:  # noqa: BLE001 - diagnostic only
            return None

    return {
        "model_config_eos": model_eos,
        "tokenizer_eos_metadata": tokenizer_eos,
        "model_eos_token": token_repr(model_eos),
        "tokenizer_eos_token": token_repr(tokenizer_eos),
        "generation_stop_tokens": stop_tokens,
        "generation_stop_token_names": [token_repr(t) for t in stop_tokens],
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ensure_chat_import():
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))
    from shelf_chat import codec  # noqa: E402

    return codec


class Failure(RuntimeError):
    pass


class Checker:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"[PASS] {self.name}:{label}")
        else:
            self.failed += 1
            print(f"[FAIL] {self.name}:{label} :: {detail}")

    def done(self) -> int:
        print(f"{self.name}: passed={self.passed} failed={self.failed}")
        return 0 if self.failed == 0 else 1


def http_json(url: str, payload: dict, timeout: float = 600.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_status(url: str, payload: dict, timeout: float = 600.0):
    """POST JSON and return ``(status, body_text)`` without raising on 4xx/5xx."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def http_sse(url: str, payload: dict, timeout: float = 600.0):
    """Yield each SSE ``data:`` payload (excluding the terminal ``[DONE]``)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)


class ComputeHarness:
    """Spawns shelf-compute --serve-stdio and speaks the JSONL protocol."""

    def __init__(self, log_path: Path | None = None, arena_gib: int = 24,
                 max_seq_len: int = 256, device: int = 0,
                 max_concurrent_requests: int = 1,
                 batch_trace: bool = False,
                 graph_debug: bool = False,
                 kv_cache_dtype: str = "bf16",
                 dflash2_model_dir: str | None = None,
                 dflash2_drafts: int = 7,
                 model_path: Path | str | None = None,
                 working_set_reserve_gib: int | None = None,
                 tp: int = 1,
                 devices: str | None = None,
                 auto_arena: bool = False):
        self.log_path = log_path
        self.arena_gib = arena_gib
        self.auto_arena = auto_arena
        self.max_seq_len = max_seq_len
        self.device = device
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_trace = batch_trace
        self.graph_debug = graph_debug
        self.kv_cache_dtype = kv_cache_dtype
        self.dflash2_model_dir = dflash2_model_dir
        self.dflash2_drafts = dflash2_drafts
        self.model_path = Path(model_path) if model_path is not None else None
        self.working_set_reserve_gib = working_set_reserve_gib
        self.tp = tp
        self.devices = devices
        self.proc: subprocess.Popen | None = None
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        binary = compute_binary()
        if not binary.is_file():
            raise Failure(f"shelf-compute not found: {binary}")
        arena_gib = self.arena_gib if self.auto_arena else test_arena_gib(self.arena_gib)
        self.arena_gib = arena_gib
        if self.tp > 1 and not self.devices:
            raise Failure("devices is required when tp > 1")
        device_arg = self.devices if self.tp > 1 else str(_test_device(self.device))
        argv = [
            str(binary),
            "--model-dir", str(self.model_path or model_dir()),
            "--serve-stdio",
            "--max-seq-len", str(self.max_seq_len),
            "--device", device_arg,
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--max-concurrent-requests", str(self.max_concurrent_requests),
        ]
        if not self.auto_arena:
            argv += ["--arena-gib", str(arena_gib)]
        if self.tp > 1:
            argv += ["--tp", str(self.tp)]
        reserve = _test_working_set_reserve_gib(self.working_set_reserve_gib)
        if reserve is not None:
            argv += ["--working-set-reserve-gib", str(reserve)]
        if self.dflash2_model_dir is not None:
            argv += ["--dflash2-model-dir", str(self.dflash2_model_dir),
                     "--dflash2-drafts", str(self.dflash2_drafts)]
        env = dict(os.environ)
        if self.batch_trace:
            env["SHELF_BATCH_TRACE"] = "1"
        if self.graph_debug:
            env["SHELF_GRAPH_DEBUG"] = "1"
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    def _pump_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as handle:
                    handle.write(line)

    def wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if "SHELF_SERVE_READY" in self.stderr_lines:
                return
            if self.proc is not None and self.proc.poll() is not None:
                raise Failure(
                    f"compute exited early rc={self.proc.returncode}: "
                    + "\n".join(self.stderr_lines[-20:]))
            time.sleep(0.2)
        raise Failure("compute did not become ready")

    def send(self, payload: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def send_batch(self, payloads: list[dict]) -> None:
        """Write several commands in one flush.

        The serve loop drains every buffered line before running a step, so a
        single atomic write lets a test issue a command and its cancel together
        without racing the first prefill.
        """
        assert self.proc is not None and self.proc.stdin is not None
        text = "".join(json.dumps(payload) + "\n" for payload in payloads)
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def recv(self, timeout: float = 300.0) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        result: list[str] = []

        def _read():
            result.append(self.proc.stdout.readline())

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise Failure("timed out waiting for compute event")
        line = result[0].strip()
        if not line:
            raise Failure("compute closed stdout")
        return json.loads(line)

    def count_load_events(self) -> int:
        return sum(1 for line in self.stderr_lines if "SHELF_SERVE_READY" in line)

    def batch_steps(self) -> list[tuple[int, int]]:
        steps = []
        for line in self.stderr_lines:
            if not line.startswith("SHELF_BATCH_STEP"):
                continue
            parts = dict(
                token.split("=") for token in line.split()[1:] if "=" in token)
            steps.append((int(parts.get("requests", 0)), int(parts.get("tokens", 0))))
        return steps

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.send({"op": "shutdown"})
            self.recv(timeout=30)
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False



def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise Failure(f"{name} must be an integer: {raw!r}") from exc


def _test_working_set_reserve_gib(configured: int | None):
    if configured is not None:
        return configured
    return _env_int("SHELF_TEST_WORKING_SET_RESERVE_GIB")


def _test_device(default: int) -> int:
    """Resolve the GPU device for a harness.

    A shared multi-GPU host can have device 0 busy; ``SHELF_TEST_DEVICE``
    lets a run pin the suite to a free device without editing every test.
    """
    raw = os.environ.get("SHELF_TEST_DEVICE")
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def test_arena_gib(default: int = 31) -> int:
    """Arena size for default-profile server tests.

    ``--arena-gib`` sizes the single GPU pool that also holds the weights, so the
    server default (16) cannot load the 27B target. Tests that exercise the
    production default profile still size the arena for the host via
    ``SHELF_TEST_ARENA_GIB``.
    """
    raw = os.environ.get("SHELF_TEST_ARENA_GIB")
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


class ServerHarness:
    """Starts the shelf-server process and waits for readiness."""

    def __init__(self, log_path: Path | None = None, env: dict | None = None,
                 startup_timeout: float = 240.0, max_seq_len: int = 512,
                 arena_gib: int = 24, device: int = 0,
                 max_concurrent_requests: int = 1,
                 use_defaults: bool = False,
                 kv_cache_dtype: str = "bf16",
                 dflash2_model_dir: str | None = None,
                 dflash2_drafts: int = 7,
                 arena_override: int | None = None,
                 working_set_reserve_gib: int | None = None):
        self.log_path = log_path or (Path("/tmp") / f"shelf-server-{os.getpid()}.log")
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self.startup_timeout = startup_timeout
        self.max_seq_len = max_seq_len
        self.arena_gib = arena_gib
        self.device = device
        self.max_concurrent_requests = max_concurrent_requests
        self.use_defaults = use_defaults
        self.kv_cache_dtype = kv_cache_dtype
        self.dflash2_model_dir = dflash2_model_dir
        self.dflash2_drafts = dflash2_drafts
        self.arena_override = arena_override
        self.working_set_reserve_gib = working_set_reserve_gib
        self.proc: subprocess.Popen | None = None
        self.port = free_port()
        self.base_url = ""

    def start(self) -> None:
        binary = server_binary()
        if not binary.is_file():
            raise Failure(f"shelf-server not found: {binary}")
        argv = [
            str(binary),
            "--model-dir", str(model_dir()),
            "--port", str(self.port),
            "--device", str(_test_device(self.device)),
        ]
        if not self.use_defaults:
            arena_gib = test_arena_gib(self.arena_gib)
            self.arena_gib = arena_gib
            argv += [
                "--max-seq-len", str(self.max_seq_len),
                "--arena-gib", str(arena_gib),
                "--max-concurrent-requests", str(self.max_concurrent_requests),
                "--kv-cache-dtype", self.kv_cache_dtype,
            ]
            reserve = _test_working_set_reserve_gib(self.working_set_reserve_gib)
            if reserve is not None:
                argv += ["--working-set-reserve-gib", str(reserve)]
        elif self.arena_override is not None:
            argv += ["--arena-gib", str(self.arena_override)]
        if self.dflash2_model_dir is not None:
            argv += ["--dflash2-model-dir", str(self.dflash2_model_dir),
                     "--dflash2-drafts", str(self.dflash2_drafts)]
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            argv, stdout=self._log, stderr=subprocess.STDOUT, env=self.env)
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise Failure(f"server exited rc={self.proc.returncode}")
            try:
                payload = http_get_json(f"{self.base_url}/models")
                if any(m.get("id") == "shelf" for m in payload.get("data", [])):
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        raise Failure("server did not become ready")

    def close(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None
        try:
            self._log.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


OWNER_ENV = "SHELF_TEST_OWNER"
os.environ[OWNER_ENV] = str(os.getpid())


def _owned_compute(entry: Path) -> bool:
    try:
        environ = (entry / "environ").read_bytes().split(b"\x00")
    except OSError:
        return False
    marker = f"{OWNER_ENV}={os.environ[OWNER_ENV]}".encode()
    return marker in environ


def compute_pids() -> list[int]:
    """PIDs of live shelf-compute --serve-stdio processes this test owns.

    A shared GPU host can run a resident compute for someone else. Every compute
    process started through this module inherits ``OWNER_ENV`` from the test
    process, so the scan ignores processes owned by other test runs.
    """
    pids = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if "shelf-compute" not in cmdline or "--serve-stdio" not in cmdline:
            continue
        if not _owned_compute(entry):
            continue
        pids.append(int(entry.name))
    return sorted(pids)


def cli_generated_ids(messages: list[dict], max_new_tokens: int,
                      device: int | None = None,
                      enable_thinking: bool = False) -> list[int]:
    """Tokenize with the chat codec and run compute for parity checks.

    ``device`` defaults to ``SHELF_TEST_DEVICE`` when set; the parity run
    happens after the server under test has exited, so the test device is free.
    """
    codec = ensure_chat_import()
    processor = codec.load_processor(str(model_dir()))
    device = _test_device(1) if device is None else device
    ids = codec.prompt_ids(processor, messages, enable_thinking=enable_thinking)
    binary = compute_binary()
    argv = [
        str(binary),
        "--model-dir", str(model_dir()),
        "--input-ids-file", "/dev/stdin",
        "--max-new-tokens", str(max_new_tokens),
        "--max-seq-len", "512",
        "--arena-gib", str(test_arena_gib(24)),
        "--device", str(device),
        "--temperature", "0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--seed", "0",
    ]
    reserve = _test_working_set_reserve_gib(None)
    if reserve is not None:
        argv += ["--working-set-reserve-gib", str(reserve)]
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        stdout, stderr = proc.communicate("\n".join(str(t) for t in ids) + "\n", timeout=600)
        if proc.returncode != 0:
            raise Failure(f"compute parity run failed: {stderr[-2000:]}")
        for line in stdout.splitlines():
            if line.startswith("GENERATED_IDS="):
                return [int(x) for x in line[len("GENERATED_IDS="):].split(",") if x.strip()]
    raise Failure("GENERATED_IDS not found in compute output")
