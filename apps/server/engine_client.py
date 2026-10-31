"""shelf server — エンジン（`shelf-serve`）との JSONL クライアント。

移植元: `references/phaseshift-fp8/src/apps/server/serverlib/compute_client.py`
（自作品）。shelf の契約は `draft_docs/engine_contract.md` §1/§2 で、差分は次の 4 点:

* `request_id` は **文字列**（参照は int）
* `min_p` を送る（参照には無い。契約 §3 の 5 個目）
* `stats` op は無い（KV 統計は未実装）
* `done` に `seed` が必ず入る（参照は返さない）

1 プロセス 1 stdin 書き込み・1 stdout 読み取り。`request_id` ごとにメールボックスへ
振り分ける。制御系（`pong`）は control キューへ。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


class EngineError(RuntimeError):
    """エンジンが返した error イベント。"""

    def __init__(self, code: str, message: str, request_id: Optional[str] = None,
                 param: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id
        self.param = param


class EngineProtocolError(RuntimeError):
    """プロトコル違反（行が壊れている・期待したイベントが来ない等）。"""


class EngineCancelled(RuntimeError):
    """cancel により終端した要求。"""


@dataclass
class Capabilities:
    backend: str = "?"
    model: str = "?"
    max_context: int = 0
    features: list[str] = field(default_factory=list)

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "Capabilities":
        caps = event.get("capabilities") or {}
        return cls(
            backend=str(caps.get("backend", "?")),
            model=str(caps.get("model", "?")),
            max_context=int(caps.get("max_context", 0) or 0),
            features=[str(x) for x in (caps.get("features") or [])],
        )


@dataclass
class Done:
    request_id: str
    generated_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int
    seed: int


class EngineClient:
    """`shelf-serve` の 1 プロセスを駆動する（非同期）。"""

    def __init__(self, argv: list[str], max_inflight: int = 1, log=None):
        self._argv = list(argv)
        self._max_inflight = max(1, int(max_inflight))
        self._log = log
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._control: asyncio.Queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._max_inflight)
        self._counter = 0
        self._closing = False
        self.capabilities: Optional[Capabilities] = None

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> Capabilities:
        if self._proc is not None:
            return self.capabilities or Capabilities()
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        caps = await self.ping()
        self.capabilities = caps
        return caps

    async def close(self) -> None:
        self._closing = True
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None and not proc.stdin.is_closing():
            try:
                await self._send({"op": "shutdown"})
            except Exception:  # noqa: BLE001 - 終了経路では握りつぶす
                pass
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # --------------------------------------------------------------- plumbing
    async def _send(self, obj: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EngineError("state_error", "engine is not running")
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            proc.stdin.write(line.encode("utf-8"))
            await proc.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        while True:
            raw = await stream.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                # 壊れた行はプロトコル違反。購読側へ流す（プロセスは落とさない）。
                for queue in list(self._mailboxes.values()):
                    queue.put_nowait(EngineProtocolError(f"bad json line: {exc}"))
                continue
            rid = event.get("request_id")
            if rid is None:
                self._control.put_nowait(event)   # pong / shutdown
                continue
            queue = self._mailboxes.get(str(rid))
            if queue is None:
                continue   # 終端後に届いた残り（破棄）
            kind = event.get("event")
            if kind == "error":
                queue.put_nowait(
                    EngineError(event.get("code", "internal_error"),
                                event.get("message", ""), str(rid), event.get("param", ""))
                )
            else:
                queue.put_nowait(event)

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            raw = await self._proc.stderr.readline()
            if not raw:
                break
            if self._log is not None:
                self._log(raw.decode("utf-8", "replace").rstrip())

    # ---------------------------------------------------------------- control
    async def ping(self, timeout: float = 60.0) -> Capabilities:
        async with self._control_lock:
            await self._send({"op": "ping"})
            while True:
                try:
                    event = await asyncio.wait_for(self._control.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise EngineProtocolError("ping timed out") from exc
                if isinstance(event, Exception):
                    raise event
                if event.get("event") == "pong":
                    return Capabilities.from_event(event)

    # --------------------------------------------------------------- requests
    async def cancel(self, request_id: str) -> None:
        try:
            await self._send({"op": "cancel", "request_id": request_id})
        except EngineError:
            pass

    async def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """`token` イベントを逐次返し、最後に `done`（dict）を返す。

        `error` は `EngineError`、`cancel` による終端は `EngineCancelled` を送出する。
        """
        async with self._semaphore:
            self._counter += 1
            request_id = str(self._counter)
            queue: asyncio.Queue = asyncio.Queue()
            self._mailboxes[request_id] = queue
            try:
                payload: dict[str, Any] = {
                    "op": "generate",
                    "request_id": request_id,
                    "input_ids": list(input_ids),
                    "max_new_tokens": int(max_new_tokens),
                }
                # 既定値は送らない（エンジン側の既定 = greedy に委ねる）。
                if temperature != 0.0:
                    payload["temperature"] = float(temperature)
                if top_p != 1.0:
                    payload["top_p"] = float(top_p)
                if top_k != 0:
                    payload["top_k"] = int(top_k)
                if min_p != 0.0:
                    payload["min_p"] = float(min_p)
                if seed is not None:
                    payload["seed"] = int(seed)
                await self._send(payload)

                while True:
                    item = await queue.get()
                    if isinstance(item, EngineError):
                        raise item
                    if isinstance(item, EngineProtocolError):
                        raise item
                    kind = item.get("event")
                    if kind == "token":
                        yield item
                        continue
                    if kind == "done":
                        if item.get("finish_reason") == "cancelled":
                            raise EngineCancelled(request_id)
                        return
                    # 未知のイベントは無視（前方互換）
            finally:
                self._mailboxes.pop(request_id, None)
