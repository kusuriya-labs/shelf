"""shelf server — エンジン（`shelf-compute`）との JSONL クライアント。

移植元: `references/phaseshift-fp8/src/apps/server/serverlib/compute_client.py`
（自作品）。shelf の契約は `draft_docs/engine_contract.md` §1/§2 で、差分は次の 4 点:

* `request_id` は **文字列**（参照は int）
* `min_p` を送る（参照には無い。契約 §3 の 5 個目）
* `stats` op は無い（KV 統計は未実装）
* `done` に `seed` が必ず入る（参照は返さない）

1 プロセス 1 stdin 書き込み・1 stdout 読み取り。`request_id` ごとにメールボックスへ
振り分ける。制御系（`pong`）は control キューへ。

**done の伝搬（T02）**: `generate()` はトークンイベントを dict で逐次返し、
最後に `Done`（finish_reason / usage / seed）を 1 つ返す。呼び出し側は
これを HTTP / SSE へ写す。

**中断（T03）**: 消費側が途中で止めた（HTTP 切断・キャンセル）ときは
`finally` で `cancel` を送り、terminal イベントまで drain してから mailbox を
解放する（契約 §4）。子プロセスが終了したら待っている全要求を `EngineError` で
起こす（無期限に待たない）。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


class EngineError(RuntimeError):
    """エンジンが返した error イベント（またはエンジンの異常終了）。"""

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
    """`done` イベント（`engine_contract.md` §2）。"""

    request_id: str
    generated_ids: list[int]
    finish_reason: str          # stop / length / cancelled / error
    prompt_tokens: int
    generated_tokens: int
    seed: int

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "Done":
        ids = event.get("generated_ids")
        if not isinstance(ids, list):
            raise EngineProtocolError("done に generated_ids（配列）が無い")
        reason = event.get("finish_reason")
        if reason not in ("stop", "length", "cancelled", "error"):
            raise EngineProtocolError(f"done の finish_reason が不正: {reason!r}")
        return cls(
            request_id=str(event.get("request_id", "")),
            generated_ids=[int(x) for x in ids],
            finish_reason=str(reason),
            prompt_tokens=int(event.get("prompt_tokens", 0) or 0),
            generated_tokens=int(event.get("generated_tokens", len(ids)) or 0),
            seed=int(event.get("seed", 0) or 0),
        )

    def as_event(self) -> dict[str, Any]:
        """JSONL と同じ形（テスト・ログ用）。"""
        return {
            "event": "done",
            "request_id": self.request_id,
            "generated_ids": list(self.generated_ids),
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "seed": self.seed,
        }


class EngineClient:
    """`shelf-compute` の 1 プロセスを駆動する（非同期）。"""

    def __init__(self, argv: list[str], max_inflight: int = 1, log=None):
        self._argv = list(argv)
        self._max_inflight = max(1, int(max_inflight))
        self._log = log
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._wait_task: Optional[asyncio.Task] = None
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._control: asyncio.Queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._max_inflight)
        self._counter = 0
        self._closing = False
        self._exit_error: Optional[EngineError] = None
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
        self._exit_error = None
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        self._wait_task = asyncio.create_task(self._watch_exit())
        caps = await self.ping()
        self.capabilities = caps
        return caps

    async def close(self) -> None:
        self._closing = True
        proc = self._proc
        if proc is None:
            return
        # 契約に `shutdown` op は無い。stdin を閉じる（EOF）とエンジンは
        # `shutdown` イベントを出して終了する。
        if proc.stdin is not None and not proc.stdin.is_closing():
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        for task in (self._stdout_task, self._stderr_task, self._wait_task):
            if task is not None:
                task.cancel()
        self._proc = None
        self._wake_all(EngineError("state_error", "engine is closed"))

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # --------------------------------------------------------------- plumbing
    async def _send(self, obj: dict[str, Any]) -> None:
        if not self.is_alive():
            raise EngineError("state_error", "engine is not running")
        proc = self._proc
        assert proc is not None
        if proc.stdin is None:
            raise EngineError("state_error", "engine stdin is closed")
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            try:
                proc.stdin.write(line.encode("utf-8"))
                await proc.stdin.drain()
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise EngineError("state_error", f"engine stdin is closed: {exc}") from exc

    def _wake_all(self, exc: Exception) -> None:
        """待っている購読側を 1 度だけ起こす（二重に積まない）。"""
        for queue in list(self._mailboxes.values()):
            queue.put_nowait(exc)
        for _ in range(len(self._mailboxes) + 1):
            self._control.put_nowait(exc)

    async def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        stream = proc.stdout
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
        # stdout が閉じた = エンジンが終了した。待っている要求を必ず起こす。
        self._wake_all(EngineError("state_error", "engine exited before terminal event"))

    async def _watch_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        rc = await proc.wait()
        if self._closing:
            return
        err = EngineError("state_error", f"engine exited (code {rc})")
        self._exit_error = err
        self._wake_all(err)

    async def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        while True:
            raw = await proc.stderr.readline()
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
        """cancel を送る（冪等）。送れないときは何もしない。"""
        try:
            await self._send({"op": "cancel", "request_id": request_id})
        except EngineError:
            pass

    async def _abort(self, request_id: str, queue: asyncio.Queue) -> None:
        """中断した要求をエンジン側でも止め、terminal まで drain する。

        直列エンジンなので、drain しないと次の要求が前の生成の後ろに並ぶ。
        エンジンが落ちている・応答しない場合は待ち切らない（上限つき）。
        """
        try:
            await self.cancel(request_id)
        except asyncio.CancelledError:
            raise
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return
            if isinstance(item, (EngineError, EngineProtocolError)):
                return
            if isinstance(item, dict) and item.get("event") == "done":
                return

    async def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: Optional[int] = None,
    ) -> AsyncIterator[Any]:
        """`token` イベント（dict）を逐次返し、最後に `Done` を返す。

        `error` イベントは `EngineError`、プロトコル違反は `EngineProtocolError`。
        `finish_reason == "cancelled"` も `Done` として返す（HTTP 層が判断する）。
        消費側が途中で止めた場合は `cancel` を送って終端まで drain する。
        """
        async with self._semaphore:
            if not self.is_alive():
                raise EngineError("state_error", "engine is not running")
            self._counter += 1
            request_id = str(self._counter)
            queue: asyncio.Queue = asyncio.Queue()
            self._mailboxes[request_id] = queue
            finished = False
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
                        finished = True
                        yield Done.from_event(item)
                        return
                    # 未知のイベントは無視（前方互換）
            finally:
                if not finished:
                    await self._abort(request_id, queue)
                self._mailboxes.pop(request_id, None)
