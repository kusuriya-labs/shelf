"""Async JSONL client for ``shelf-compute --serve-stdio``.

One compute process is started per server instance and held for its lifetime.
The protocol is one JSON object per line. A single stdout reader task routes
events to per-request mailboxes by ``request_id``; events without a request ID
(pong / shutdown) go to a control queue. Exactly one stdout consumer and one
stdin writer exist per process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Optional


class ComputeError(RuntimeError):
    """Raised when the compute process fails or reports an error event."""


class ComputeInvalidArgument(ComputeError):
    """Raised when compute rejects the request before generation."""


class ComputeProtocolError(ComputeError):
    """Raised when the compute process violates the JSONL protocol."""


class ComputeCancelled(ComputeError):
    """Raised when a generation ended because this client cancelled it."""


@dataclass(frozen=True)
class ComputeCapabilities:
    prefix_cache: bool = False

    @classmethod
    def from_pong(cls, pong: dict) -> "ComputeCapabilities":
        caps = pong.get("capabilities") or {}
        if not isinstance(caps, dict):
            caps = {}
        return cls(prefix_cache=bool(caps.get("prefix_cache", False)))


def _non_negative_int(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass(frozen=True)
class ComputeKVStats:
    page_tokens: int
    total_pages: int
    used_pages: int
    free_pages: int
    capacity_tokens: int
    free_tokens: int
    active_requests: int
    queued_requests: int
    banker_safe: bool
    overcommit_pages: int

    @classmethod
    def from_event(cls, event: dict) -> Optional["ComputeKVStats"]:
        kv = event.get("kv")
        if not isinstance(kv, dict):
            return None
        fields = {}
        for name in (
            "page_tokens",
            "total_pages",
            "used_pages",
            "free_pages",
            "capacity_tokens",
            "free_tokens",
            "active_requests",
            "queued_requests",
            "overcommit_pages",
        ):
            value = _non_negative_int(kv.get(name))
            if value is None:
                return None
            fields[name] = value
        banker_safe = kv.get("banker_safe")
        if not isinstance(banker_safe, bool):
            return None
        return cls(banker_safe=banker_safe, **fields)

    def to_wire(self) -> dict:
        return {
            "page_tokens": self.page_tokens,
            "total_pages": self.total_pages,
            "used_pages": self.used_pages,
            "free_pages": self.free_pages,
            "capacity_tokens": self.capacity_tokens,
            "free_tokens": self.free_tokens,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "banker_safe": self.banker_safe,
            "overcommit_pages": self.overcommit_pages,
        }


class AsyncComputeClient:
    def __init__(self, argv: list[str], max_inflight: int = 4, log=None):
        self._argv = list(argv)
        self._log = log if log is not None else sys.stderr
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._mailboxes: dict[int, asyncio.Queue] = {}
        self._control: asyncio.Queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._next_request_id = 1
        self._max_inflight = max(1, int(max_inflight))
        self._semaphore = asyncio.Semaphore(self._max_inflight)
        self._inflight = 0
        self._queued = 0
        self._closing = False
        self._shutdown_started = False
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self.capabilities = ComputeCapabilities()

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._proc is not None:
            raise ComputeError("compute process already started")
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        pong = await self.ping()
        self.capabilities = ComputeCapabilities.from_pong(pong)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _read_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()

            if not line:
                await self._broadcast_error(ComputeError(
                    f"compute process exited with code {self._proc.returncode}"))
                return

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                await self._broadcast_error(
                    ComputeProtocolError(f"invalid compute JSON: {exc}"))
                return

            request_id = event.get("request_id")

            if request_id is None:
                await self._control.put(event)
                continue

            async with self._state_lock:
                mailbox = self._mailboxes.get(int(request_id))

            if mailbox is not None:
                await mailbox.put(event)

    async def _read_stderr(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        trace_path = os.environ.get("SHELF_COMPUTE_LOG")
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace")
                self._log.write("[compute] " + text)
                self._log.flush()
                if trace_path:
                    try:
                        with open(trace_path, "a", encoding="utf-8") as handle:
                            handle.write(text)
                    except OSError:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - losing stderr must be visible
            self._log.write(f"[compute] stderr reader failed: {exc}\n")
            self._log.flush()

    async def _broadcast_error(self, exc: Exception) -> None:
        async with self._state_lock:
            mailboxes = list(self._mailboxes.values())
        for mailbox in mailboxes:
            await mailbox.put(exc)
        await self._control.put(exc)

    # ----------------------------------------------------------------- protocol

    async def _send(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ComputeError("compute process is not running")

        line = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"

        async with self._write_lock:
            if self._proc is None or self._proc.stdin is None:
                raise ComputeError("compute process is not running")
            try:
                self._proc.stdin.write(line)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise ComputeError("compute process closed its stdin") from exc

    @staticmethod
    def _error_from_event(event: dict) -> ComputeError:
        message = event.get("message", "compute error")
        if event.get("code") == "invalid_argument":
            return ComputeInvalidArgument(message)
        return ComputeError(message)

    async def ping(self) -> dict:
        async with self._control_lock:
            await self._send({"op": "ping"})
            while True:
                try:
                    event = await asyncio.wait_for(self._control.get(), timeout=300.0)
                except asyncio.TimeoutError as exc:
                    raise ComputeProtocolError("compute ping timed out") from exc
                if isinstance(event, Exception):
                    raise event
                if event.get("event") == "pong":
                    return event

    async def kv_stats(self, timeout: float = 5.0) -> Optional[ComputeKVStats]:
        if not self.is_alive():
            return None
        async with self._control_lock:
            await self._send({"op": "stats"})
            while True:
                try:
                    event = await asyncio.wait_for(self._control.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise ComputeProtocolError("compute stats timed out") from exc
                if isinstance(event, Exception):
                    raise event
                stats = ComputeKVStats.from_event(event)
                if stats is not None:
                    return stats

    # ----------------------------------------------------------------- requests

    async def _begin_request(self) -> tuple[int, asyncio.Queue]:
        self._queued += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._queued -= 1
        async with self._state_lock:
            if self._closing:
                self._semaphore.release()
                raise ComputeError("compute client is shutting down")
            request_id = self._next_request_id
            self._next_request_id += 1
            mailbox: asyncio.Queue = asyncio.Queue()
            self._mailboxes[request_id] = mailbox
            self._inflight += 1
        return request_id, mailbox

    async def _end_request(self, request_id: int) -> None:
        async with self._state_lock:
            self._mailboxes.pop(request_id, None)
            self._inflight -= 1
        self._semaphore.release()

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def queued(self) -> int:
        return self._queued

    async def _send_generate(
        self,
        request_id: int,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        seed: int,
        prefix_cache_checkpoint_position: int = 0,
    ) -> None:
        await self._send({
            "op": "generate",
            "request_id": request_id,
            "input_ids": list(input_ids),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "top_k": int(top_k),
            "seed": int(seed),
            "prefix_cache_checkpoint_position": int(prefix_cache_checkpoint_position),
        })

    async def cancel(self, request_id: int) -> None:
        try:
            await self._send({"op": "cancel", "request_id": request_id})
        except ComputeError:
            pass

    async def _drain_terminal(self, mailbox: asyncio.Queue, timeout: float = 30.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            try:
                event = await asyncio.wait_for(mailbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return
            if isinstance(event, Exception):
                return
            if event.get("event") in ("done", "error"):
                return

    async def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        seed: int = 0,
        prefix_cache_checkpoint_position: int = 0,
    ) -> dict:
        request_id, mailbox = await self._begin_request()
        sent = False
        completed = False
        try:
            await self._send_generate(
                request_id, input_ids, max_new_tokens, temperature,
                top_p, min_p, top_k, seed, prefix_cache_checkpoint_position)
            sent = True
            while True:
                event = await mailbox.get()
                if isinstance(event, Exception):
                    raise event
                kind = event.get("event")
                if kind == "error":
                    completed = True
                    raise self._error_from_event(event)
                if kind == "done":
                    completed = True
                    if event.get("finish_reason") == "cancelled":
                        raise ComputeCancelled(f"request {request_id} was cancelled")
                    return event
        finally:
            if sent and not completed:
                await self.cancel(request_id)
                await self._drain_terminal(mailbox)
            await self._end_request(request_id)

    async def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        seed: int = 0,
        prefix_cache_checkpoint_position: int = 0,
    ) -> AsyncIterator[dict]:
        """Yield ``{"token_id": N}`` events, then one ``{"done": event}``.

        Each stream owns its request mailbox. If the consumer abandons the
        stream, only this request is cancelled and its terminal event is drained
        before the in-flight slot is released. Other requests are unaffected.
        """
        request_id, mailbox = await self._begin_request()

        sent = False
        completed = False

        try:
            await self._send_generate(
                request_id, input_ids, max_new_tokens, temperature,
                top_p, min_p, top_k, seed, prefix_cache_checkpoint_position)

            sent = True

            while True:
                event = await mailbox.get()

                if isinstance(event, Exception):
                    raise event

                kind = event.get("event")

                if kind == "token":
                    yield {
                        "token_id": int(event["token_id"])
                    }

                elif kind == "error":
                    completed = True
                    raise self._error_from_event(event)

                elif kind == "done":
                    completed = True
                    yield {
                        "done": event
                    }
                    return

        finally:
            if sent and not completed:
                await self.cancel(request_id)
                await self._drain_terminal(mailbox)

            await self._end_request(request_id)

    # ----------------------------------------------------------------- shutdown

    async def shutdown(self, timeout: float = 10.0) -> None:
        async with self._state_lock:
            self._closing = True
            if self._shutdown_started:
                return
            self._shutdown_started = True

        proc = self._proc
        if proc is None or proc.returncode is not None:
            return

        try:
            await self._send({"op": "shutdown"})
        except ComputeError:
            pass
        else:
            try:
                while True:
                    event = await asyncio.wait_for(self._control.get(), timeout=timeout)
                    if isinstance(event, Exception):
                        break
                    if event.get("event") == "shutdown":
                        break
            except asyncio.TimeoutError:
                pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        self._stdout_task = None
        self._stderr_task = None
        self._proc = None
