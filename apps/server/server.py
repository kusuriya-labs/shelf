"""shelf server — OpenAI 互換の HTTP 面（パス A）。

移植元: `references/phaseshift-fp8/src/apps/server/phaseshift_server.py`（自作品）。
shelf の契約は `draft_docs/serve_protocol.md`（エラー envelope・code → HTTP）と
`draft_docs/engine_contract.md`（エンジンの JSONL）。**このファイルはビルド非結線**。

設計の要点:
* 受付時の検証（未知フィールド・型・範囲・構造）は**エンジンに渡す前**に 400 で返す
  （`serve_protocol.md` §2-2）。エンジン由来のエラーだけを `openai_errors.from_engine` で写す
* クライアントへ返すメッセージからはソース位置を落とす（§1）
* ストリームは SSE（`data: {...}\\n\\n` → 最後に `data: [DONE]`）
* **done を正とする（T02）**: finish_reason / seed / usage はエンジンの `done` から写す。
  HTTP 側で推測しない
* **文字列の復元（T05）**: SSE の増分は「受け取ったトークン列全体の decode」と
  「送出済み文字列」の差分で決める。トークンごとの decode は BPE 境界で壊れる
* トークナイザは差し替え可能（テストはフェイクを注入する）
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol

from aiohttp import web

from engine_client import Done, EngineCancelled, EngineClient, EngineError, EngineProtocolError
import openai_errors as oe

# 受付で許可するフィールド（これ以外は 400。拡張はここに足す）
_ALLOWED_CHAT_FIELDS = {
    "model", "messages", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "top_k", "min_p", "seed", "stream", "stop", "n",
    # shelf 拡張（`min_p` と同じ扱い）。思考の ON/OFF を chat template に渡す。
    "enable_thinking", "chat_template_kwargs",
}

# メッセージ 1 件で許可するフィールド。tool / マルチモーダルは未対応なので
# 「黙って捨てる」のではなく `unsupported_parameter` で 400 にする。
_ALLOWED_MESSAGE_FIELDS = {"role", "content"}
_ALLOWED_ROLES = {"system", "user", "assistant"}
# `chat_template_kwargs` で受けるキー（Qwen 系の思考モードだけ）。
_ALLOWED_TEMPLATE_KWARGS = {"enable_thinking"}

# 受付の上限（`serve_protocol.md` §4 の 413 と、入力サイズの上限）。
_MAX_BODY_BYTES = 16 * 1024 * 1024

@dataclass
class OpsState:
    """T18: 運用の観測値とライフサイクルの状態（プロセス内で 1 個）。

    ミドルウェアが更新し、`/metrics` と `/readyz` が読む。テストからも同じ
    オブジェクトを渡して検証できる。
    """

    started_at: float = field(default_factory=time.time)
    ready: bool = False
    shutting_down: bool = False
    inflight: int = 0
    requests_total: Counter = field(default_factory=Counter)   # (path, status)
    errors_total: Counter = field(default_factory=Counter)     # code
    queue_rejected: int = 0
    timeouts: int = 0
    auth_failures: int = 0
    tokens_generated: int = 0
    duration_sum: float = 0.0
    duration_count: int = 0

    def record(self, path: str, status: int, code: str, seconds: float) -> None:
        self.requests_total[(path, int(status))] += 1
        if code:
            self.errors_total[code] += 1
        self.duration_sum += seconds
        self.duration_count += 1

_MAX_MESSAGES = 4096
_MAX_TOTAL_CHARS = 4 * 1024 * 1024
_MAX_STOP_SEQUENCES = 4
_MAX_STOP_CHARS = 256


def _ids_from_template_output(out: Any) -> list[int]:
    """`apply_chat_template(tokenize=True)` の戻りを token id 列へ正規化する。

    transformers 5 系は `BatchEncoding`（`input_ids` を持つ）を返す。将来の
    版でリスト / 入れ子 / Encoding の配列 / tensor を返しても受ける。
    """
    if hasattr(out, "keys") and "input_ids" in list(out.keys()):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if hasattr(first, "ids"):      # Encoding の配列
            return [int(x) for x in first.ids]
        if isinstance(first, (list, tuple)):   # バッチ次元つき
            out = first
    return [int(x) for x in out]


class Tokenizer(Protocol):
    """chat messages → token id 列、token id 列 → 文字列。"""

    def prompt_ids(self, messages: list[dict], *, add_generation_prompt: bool = True,
                   enable_thinking: Optional[bool] = None) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class HfTokenizer:
    """HF `transformers` の AutoProcessor を使う実装（遅延 import）。

    chat template は参照の `phaseshift_chat/codec.py` と同じ考え方（`apply_chat_template`
    に messages を渡す）。思考モードの制御は `enable_thinking` をテンプレートへ渡す。
    """

    def __init__(self, model_dir: str, trust_remote_code: bool = False):
        from transformers import AutoProcessor  # 遅延 import（テストでは使わない）

        self._proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    def prompt_ids(self, messages: list[dict], *, add_generation_prompt: bool = True,
                   enable_thinking: Optional[bool] = None) -> list[int]:
        # Qwen の chat template は `enable_thinking` を受ける（未指定ならテンプレート既定）。
        kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        out = self._proc.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True, **kwargs
        )
        return _ids_from_template_output(out)

    def decode(self, ids: list[int]) -> str:
        tok = getattr(self._proc, "tokenizer", self._proc)
        return tok.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 受付時の検証（serve_protocol §2-2）
# ---------------------------------------------------------------------------
def _bad(message: str, param: Optional[str] = None) -> oe.ApiError:
    return oe.bad_request(message, param)


def _unsupported(message: str, param: str) -> oe.ApiError:
    return oe.ApiError(400, message, "unsupported_parameter", param)


def _number(body: dict, name: str, default: float, lo: float, hi: float) -> float:
    if name not in body or body[name] is None:
        return default
    v = body[name]
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise _bad(f"{name} は数値", name)
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise _bad(f"{name} は有限の値", name)
    if not (lo <= v <= hi):
        raise _bad(f"{name} は [{lo}, {hi}] の範囲", name)
    return v


def _integer(body: dict, name: str, default: int, lo: int, hi: int) -> int:
    if name not in body or body[name] is None:
        return default
    v = body[name]
    if not isinstance(v, int) or isinstance(v, bool):
        raise _bad(f"{name} は整数", name)
    if not (lo <= v <= hi):
        raise _bad(f"{name} は [{lo}, {hi}] の範囲", name)
    return int(v)


def _validate_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise _bad("messages は空でない配列", "messages")
    if len(messages) > _MAX_MESSAGES:
        raise _bad(f"messages は最大 {_MAX_MESSAGES} 件", "messages")
    total = 0
    out: list[dict] = []
    for i, message in enumerate(messages):
        where = f"messages[{i}]"
        if not isinstance(message, dict):
            raise _bad(f"{where} はオブジェクト", where)
        unknown = sorted(k for k in message if k not in _ALLOWED_MESSAGE_FIELDS)
        if unknown:
            raise _unsupported(f"{where}.{unknown[0]} は未対応です", f"{where}.{unknown[0]}")
        role = message.get("role")
        if not isinstance(role, str):
            raise _bad(f"{where}.role は文字列", f"{where}.role")
        if role not in _ALLOWED_ROLES:
            raise _unsupported(
                f"{where}.role '{role}' は未対応（{sorted(_ALLOWED_ROLES)} のみ）",
                f"{where}.role")
        content = message.get("content")
        if not isinstance(content, str):
            raise _unsupported(f"{where}.content は文字列のみ（マルチモーダルは未対応）",
                               f"{where}.content")
        total += len(content)
        if total > _MAX_TOTAL_CHARS:
            raise oe.ApiError(413, "入力テキストが大きすぎます", "request_too_large", "messages")
        out.append({"role": role, "content": content})
    if not out:
        raise _bad("messages は空でない配列", "messages")
    return out


def _validate_stop(stop: Any) -> list[str]:
    """形だけ検証する（値そのものは未対応なので 400 を返す）。"""
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list):
        raise _bad("stop は文字列または文字列の配列", "stop")
    if len(stop) > _MAX_STOP_SEQUENCES:
        raise _bad(f"stop は最大 {_MAX_STOP_SEQUENCES} 件", "stop")
    for i, item in enumerate(stop):
        if not isinstance(item, str):
            raise _bad(f"stop[{i}] は文字列", f"stop[{i}]")
        if not item:
            raise _bad(f"stop[{i}] が空", f"stop[{i}]")
        if len(item) > _MAX_STOP_CHARS:
            raise _bad(f"stop[{i}] は最大 {_MAX_STOP_CHARS} 文字", f"stop[{i}]")
    return list(stop)


def _validate_template_kwargs(kwargs: Any) -> dict[str, Any]:
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise _bad("chat_template_kwargs はオブジェクト", "chat_template_kwargs")
    unknown = sorted(k for k in kwargs if k not in _ALLOWED_TEMPLATE_KWARGS)
    if unknown:
        raise _unsupported(f"chat_template_kwargs.{unknown[0]} は未対応です",
                           f"chat_template_kwargs.{unknown[0]}")
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if not isinstance(value, bool):
            raise _bad(f"chat_template_kwargs.{key} は真偽値", f"chat_template_kwargs.{key}")
        out[key] = value
    return out


def validate_chat_body(body: Any) -> tuple[dict, list[dict]]:
    """受付時の検証。`(正規化したパラメータ, messages)` を返すか ApiError を投げる。"""
    if not isinstance(body, dict):
        raise _bad("リクエスト本文は JSON オブジェクト")
    unknown = sorted(k for k in body if k not in _ALLOWED_CHAT_FIELDS)
    if unknown:
        raise _bad(f"未知のフィールド '{unknown[0]}'", unknown[0])

    if "model" in body and body["model"] is not None and not isinstance(body["model"], str):
        raise _bad("model は文字列", "model")

    messages = _validate_messages(body.get("messages"))

    # `max_tokens` と `max_completion_tokens` はどちらか一方（両方なら同値のみ）。
    if "max_tokens" in body and body["max_tokens"] is not None and \
            "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        if body["max_tokens"] != body["max_completion_tokens"]:
            raise _bad("max_tokens と max_completion_tokens が異なります", "max_tokens")
    token_key = "max_tokens" if "max_tokens" in body else "max_completion_tokens"
    max_new_tokens = _integer(
        {token_key: body.get(token_key)} if token_key in body else {}, token_key, 256, 1, 32768)

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise _bad("stream は真偽値", "stream")

    # `n` は 1 だけ受ける（複数生成は未対応。黙って 1 にしない）。
    n = _integer(body, "n", 1, 1, 128)
    if n != 1:
        raise _unsupported("n > 1 は未対応です", "n")

    # `stop` は未対応（停止は EOS と max_tokens で行う）。
    if body.get("stop") is not None:
        _validate_stop(body["stop"])
        raise _unsupported("stop（文字列の停止列）は未対応です", "stop")

    template_kwargs = _validate_template_kwargs(body.get("chat_template_kwargs"))
    enable_thinking: Optional[bool] = None
    if "enable_thinking" in body and body["enable_thinking"] is not None:
        if not isinstance(body["enable_thinking"], bool):
            raise _bad("enable_thinking は真偽値", "enable_thinking")
        enable_thinking = bool(body["enable_thinking"])
    if "enable_thinking" in template_kwargs:
        if enable_thinking is not None and enable_thinking != template_kwargs["enable_thinking"]:
            raise _bad("enable_thinking が chat_template_kwargs と矛盾しています",
                       "chat_template_kwargs.enable_thinking")
        enable_thinking = bool(template_kwargs["enable_thinking"])

    seed: Optional[int] = None
    if "seed" in body and body["seed"] is not None:
        seed = _integer(body, "seed", 0, 0, (1 << 64) - 1)

    params: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": _number(body, "temperature", 0.0, 0.0, 100.0),
        "top_p": _number(body, "top_p", 1.0, 1e-6, 1.0),
        "min_p": _number(body, "min_p", 0.0, 0.0, 1.0),
        "top_k": _integer(body, "top_k", 0, 0, 1 << 31),
        "stream": stream,
        "seed": seed,
        "enable_thinking": enable_thinking,
    }
    return params, messages


# ---------------------------------------------------------------------------
# 文字列の復元（T05）
# ---------------------------------------------------------------------------
class IncrementalDecoder:
    """トークン列全体の decode から増分を確定する。

    BPE はトークン境界が文字境界と一致しない（日本語の 1 文字が複数トークンに
    またがる）。1 トークンずつ decode すると途中で `U+FFFD` が出るため、
    **受け取った全 ID** を毎回 decode し、送出済み文字列との差分だけを返す。

    末尾が `U+FFFD`（= 未完成の UTF-8）の差分は次トークンまで保留する。
    最終的に `finish()` が残りを返すので、非ストリームの
    `decode(全 ID)` と**同じ文字列**になる。
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._text = ""

    def push(self, token_id: int) -> str:
        self._ids.append(int(token_id))
        return self._diff(final=False)

    def finish(self) -> str:
        return self._diff(final=True)

    @property
    def text(self) -> str:
        return self._text

    def _diff(self, final: bool) -> str:
        full = self._tokenizer.decode(self._ids) if self._ids else ""
        if not full.startswith(self._text):
            # decode は決定的なので通常は起きない。起きた場合は共通接頭辞より
            # 後ろを送る（文字を落とさない側に倒す）。
            keep = 0
            limit = min(len(full), len(self._text))
            while keep < limit and full[keep] == self._text[keep]:
                keep += 1
            self._text = full[:keep]
        delta = full[len(self._text):]
        if not final and delta.endswith("\ufffd"):
            return ""   # 未完成の UTF-8 は次トークンで確定させる
        self._text += delta
        return delta


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _chunk(model: str, request_id: str, delta: dict, finish: Optional[str] = None,
           *, usage: Optional[dict] = None, seed: Optional[int] = None) -> dict:
    obj: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if seed is not None:
        obj["seed"] = int(seed)
    if usage is not None:
        obj["usage"] = usage
    return obj


def _finish_reason(reason: str) -> str:
    """エンジンの finish_reason を OpenAI の語へ写す（`engine_contract.md` §2）。"""
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "length"
    if reason == "cancelled":
        raise EngineCancelled("要求はキャンセルされました")
    raise EngineProtocolError(f"done の finish_reason が不正: {reason!r}")


def _usage(done: Done) -> dict:
    return {
        "prompt_tokens": int(done.prompt_tokens),
        "completion_tokens": int(done.generated_tokens),
        "total_tokens": int(done.prompt_tokens) + int(done.generated_tokens),
    }


def _client_gone(request: web.Request) -> bool:
    transport = request.transport
    return transport is None or transport.is_closing()


async def _read_json(request: web.Request) -> Any:
    if request.content_type != "application/json":
        raise oe.ApiError(415, "Content-Type は application/json が必要です",
                          "unsupported_media_type")
    length = request.content_length
    if length is not None and length > _MAX_BODY_BYTES:
        raise oe.ApiError(413, "リクエスト本文が大きすぎます", "request_too_large")
    try:
        return await request.json()
    except web.HTTPRequestEntityTooLarge:
        raise oe.ApiError(413, "リクエスト本文が大きすぎます", "request_too_large")
    except (json.JSONDecodeError, ValueError):
        raise _bad("本文が JSON でない")


def make_app(client: EngineClient, tokenizer: Tokenizer, model_name: str, *,
             api_key: str = "", request_timeout: float = 0.0, max_inflight: int = 4,
             max_queue: int = 64, state: Optional["OpsState"] = None) -> web.Application:
    """HTTP 面を作る。T18 の運用機能（認証・制限・監視・ライフサイクル）を含む。

    * `api_key` が空でなければ `/healthz` `/readyz` 以外に
      `Authorization: Bearer` か `X-API-Key` を要求する（401）。
    * `request_timeout > 0` のとき、1 要求の総時間を制限する（504 / SSE は error + [DONE]）。
    * `max_inflight + max_queue` を超える同時要求は 429（エンジンは直列〜max_inflight）。
    * `state.ready` が false の間は `/v1/*` を 503（起動直後・エンジン死亡・停止中）。
    """
    app = web.Application(client_max_size=_MAX_BODY_BYTES)
    # 既存の呼び出し（テスト・組み込み）は「起動済み」として扱う。
    # 実サーバは state を渡し、listen 開始後に ready を立てる。
    ops = state if state is not None else OpsState(ready=True)
    exempt = {"/healthz", "/readyz"}
    limit = max(1, int(max_inflight)) + max(0, int(max_queue))

    def _authorized(request: web.Request) -> bool:
        if not api_key:
            return True
        got = ""
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            got = auth[7:].strip()
        elif request.headers.get("X-API-Key"):
            got = request.headers.get("X-API-Key", "").strip()
        return hmac.compare_digest(got, api_key)

    @web.middleware
    async def guard_mw(request: web.Request, handler):
        t0 = time.monotonic()
        path = request.path
        status = 500
        code = ""
        counted = False
        try:
            if path not in exempt and not _authorized(request):
                ops.auth_failures += 1
                exc = oe.ApiError(401, "API key が無効です", "invalid_api_key")
                status, code = exc.status, exc.code
                return web.json_response(
                    exc.body(), status=exc.status,
                    headers={"WWW-Authenticate": "Bearer"})
            if path == "/metrics":
                resp = await handler(request)
                status = resp.status
                return resp
            if path.startswith("/v1/"):
                if ops.shutting_down:
                    raise oe.ApiError(503, "サーバは停止処理中です", "service_unavailable")
                if not ops.ready or not client.is_alive():
                    raise oe.ApiError(503, "エンジンが準備できていません",
                                      "service_unavailable")
                if ops.inflight >= limit:
                    ops.queue_rejected += 1
                    raise oe.ApiError(429, "同時要求が多すぎます", "rate_limit_exceeded")
                ops.inflight += 1
                counted = True
            resp = await handler(request)
            status = resp.status
            return resp
        except web.HTTPRequestEntityTooLarge:
            api = oe.ApiError(413, "リクエスト本文が大きすぎます", "request_too_large")
            status, code = api.status, api.code
            return web.json_response(api.body(), status=api.status)
        except oe.ApiError as exc:
            status, code = exc.status, exc.code
            return web.json_response(exc.body(), status=exc.status)
        except EngineError as exc:
            api = oe.from_engine(exc.code, oe.traceback_free(exc.message), exc.param)
            status, code = api.status, api.code
            return web.json_response(api.body(), status=api.status)
        except EngineCancelled:
            status, code = 400, "cancelled"
            return web.json_response(
                oe.ApiError(400, "要求はキャンセルされました", "cancelled").body(), status=400)
        except EngineProtocolError as exc:
            status, code = 500, "internal_error"
            return web.json_response(
                oe.ApiError(500, oe.traceback_free(str(exc)), "internal_error").body(),
                status=500)
        except Exception:  # noqa: BLE001 - 未処理例外は 500 のみ（§2-3）
            status, code = 500, "internal_error"
            return web.json_response(
                oe.ApiError(500, "内部エラー", "internal_error").body(), status=500)
        finally:
            if counted:
                ops.inflight -= 1
            ops.record(path, status, code, time.monotonic() - t0)

    app.middlewares.append(guard_mw)

    async def healthz(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def readyz(_: web.Request) -> web.Response:
        ok = ops.ready and not ops.shutting_down and client.is_alive()
        body: dict[str, Any] = {"status": "ready" if ok else "not_ready"}
        caps = client.capabilities
        if caps is not None:
            body["backend"] = caps.backend
            body["max_context"] = caps.max_context
        return web.json_response(body, status=200 if ok else 503)

    async def metrics(_: web.Request) -> web.Response:
        lines = [
            "# HELP shelf_up プロセスが生きている（常に 1）。",
            "# TYPE shelf_up gauge",
            "shelf_up 1",
            "# HELP shelf_ready 要求を受け付けられる状態か。",
            "# TYPE shelf_ready gauge",
            f"shelf_ready {1 if (ops.ready and not ops.shutting_down and client.is_alive()) else 0}",
            "# HELP shelf_engine_alive エンジン子プロセスが生きているか。",
            "# TYPE shelf_engine_alive gauge",
            f"shelf_engine_alive {1 if client.is_alive() else 0}",
            "# TYPE shelf_inflight gauge",
            f"shelf_inflight {ops.inflight}",
            "# TYPE shelf_requests_total counter",
        ]
        for (path, status), n in sorted(ops.requests_total.items()):
            lines.append(f'shelf_requests_total{{path="{path}",status="{status}"}} {n}')
        lines.append("# TYPE shelf_errors_total counter")
        for code_name, n in sorted(ops.errors_total.items()):
            lines.append(f'shelf_errors_total{{code="{code_name}"}} {n}')
        lines += [
            "# TYPE shelf_queue_rejected_total counter",
            f"shelf_queue_rejected_total {ops.queue_rejected}",
            "# TYPE shelf_request_timeouts_total counter",
            f"shelf_request_timeouts_total {ops.timeouts}",
            "# TYPE shelf_auth_failures_total counter",
            f"shelf_auth_failures_total {ops.auth_failures}",
            "# TYPE shelf_tokens_generated_total counter",
            f"shelf_tokens_generated_total {ops.tokens_generated}",
            "# TYPE shelf_request_duration_seconds summary",
            f"shelf_request_duration_seconds_sum {ops.duration_sum:.6f}",
            f"shelf_request_duration_seconds_count {ops.duration_count}",
            "# TYPE shelf_uptime_seconds gauge",
            f"shelf_uptime_seconds {max(0.0, time.time() - ops.started_at):.3f}",
        ]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [
                {"id": model_name, "object": "model", "owned_by": "shelf"}]})

    def _check_context(input_ids: list[int], max_new_tokens: int) -> None:
        caps = client.capabilities
        if caps is None or not caps.max_context:
            return
        if len(input_ids) + max_new_tokens <= caps.max_context:
            return
        raise oe.ApiError(
            400,
            f"入力 {len(input_ids)} + 生成 {max_new_tokens} が最大文脈 "
            f"{caps.max_context} を超えます",
            "context_length_exceeded", "messages")

    async def _complete(request: web.Request, gen: AsyncIterator[Any],
                        request_id: str, deadline: Optional[float]) -> web.Response:
        ids: list[int] = []
        done: Optional[Done] = None
        try:
            async with asyncio.timeout_at(deadline):
                async for ev in gen:
                    if isinstance(ev, Done):
                        done = ev
                        continue
                    if _client_gone(request):
                        raise EngineCancelled(request_id)
                    ids.append(int(ev["token_id"]))
        except TimeoutError:
            ops.timeouts += 1
            raise oe.ApiError(504, "要求がタイムアウトしました", "timeout") from None
        if done is None:
            raise EngineProtocolError("done が来ないままストリームが終了した")
        ops.tokens_generated += int(done.generated_tokens)
        finish = _finish_reason(done.finish_reason)
        if done.generated_ids != ids:
            raise EngineProtocolError("token 列と done.generated_ids が一致しない")
        text = tokenizer.decode(done.generated_ids) if done.generated_ids else ""
        return web.json_response({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "seed": int(done.seed),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
            "usage": _usage(done),
        })

    async def _stream(request: web.Request, gen: AsyncIterator[Any],
                      request_id: str, deadline: Optional[float]) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                     "Connection": "keep-alive"})
        await resp.prepare(request)

        async def send(obj: Any) -> None:
            try:
                await resp.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            except (ConnectionResetError, BrokenPipeError, RuntimeError) as exc:
                raise EngineCancelled(request_id) from exc
            if _client_gone(request):
                raise EngineCancelled(request_id)

        decoder = IncrementalDecoder(tokenizer)
        done: Optional[Done] = None
        try:
            async with asyncio.timeout_at(deadline):
                await send(_chunk(model_name, request_id,
                                  {"role": "assistant", "content": ""}))
                async for ev in gen:
                    if isinstance(ev, Done):
                        done = ev
                        continue
                    delta = decoder.push(int(ev["token_id"]))
                    if delta:
                        await send(_chunk(model_name, request_id, {"content": delta}))
                if done is None:
                    raise EngineProtocolError("done が来ないままストリームが終了した")
                finish = _finish_reason(done.finish_reason)
                tail = decoder.finish()
                if tail:
                    await send(_chunk(model_name, request_id, {"content": tail}))
                if decoder.text != tokenizer.decode(done.generated_ids):
                    raise EngineProtocolError("SSE の復元文字列が非ストリームと一致しない")
                await send(_chunk(model_name, request_id, {}, finish=finish,
                                  usage=_usage(done), seed=done.seed))
                await resp.write(b"data: [DONE]\n\n")
                ops.tokens_generated += int(done.generated_tokens)
        except TimeoutError:
            # ヘッダは送信済み。SSE の作法として error を 1 つ流して [DONE] で閉じる。
            ops.timeouts += 1
            try:
                await send({"error": {"message": "要求がタイムアウトしました",
                                      "type": "server_error", "code": "timeout"}})
                await resp.write(b"data: [DONE]\n\n")
            except EngineCancelled:
                pass
        except EngineCancelled:
            # クライアント切断・キャンセル: エラー応答は作らない（§2-4）。
            if not _client_gone(request):
                raise
        finally:
            try:
                await resp.write_eof()
            except (ConnectionResetError, BrokenPipeError, RuntimeError):
                pass
        return resp

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await _read_json(request)
        params, messages = validate_chat_body(body)
        asked = body.get("model")
        if asked is not None and asked != model_name:
            raise oe.ApiError(404, f"model '{asked}' は無い", "model_not_found", "model")

        input_ids = tokenizer.prompt_ids(messages,
                                         enable_thinking=params["enable_thinking"])
        _check_context(input_ids, params["max_new_tokens"])
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        deadline = (asyncio.get_running_loop().time() + request_timeout
                    if request_timeout > 0 else None)
        gen = client.generate(
            input_ids,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            min_p=params["min_p"],
            seed=params["seed"],
        )
        try:
            if not params["stream"]:
                return await _complete(request, gen, request_id, deadline)
            return await _stream(request, gen, request_id, deadline)
        finally:
            # 途中で止めた場合（切断・例外）はエンジン側の cancel まで面倒を見る。
            await gen.aclose()

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    return app


async def _run(args: argparse.Namespace) -> int:
    state = OpsState()
    client = EngineClient(
        [args.engine, "--model", args.model, "--gpu", str(args.gpu),
         "--pipeline-parallel", str(args.pipeline_parallel), "--prefill-rows",
         str(args.prefill_rows), "--act-quant", args.act_quant,
         "--offload-weights-mib", str(args.offload_weights_mib),
         "--prefix-cache-blocks", str(args.prefix_cache_blocks),
         "--max-batch", str(args.max_batch)]
        + (["--kv-in-host"] if args.kv_in_host else [])
        + (["--online-kv-pages", str(args.online_kv_pages)] if args.online_kv_pages else []),
        max_inflight=args.max_inflight,
        log=lambda line: print(f"[engine] {line}", file=sys.stderr),
    )
    caps = await client.start()
    print(f"shelf-server: engine ready backend={caps.backend} max_context={caps.max_context}",
          file=sys.stderr)
    tokenizer = HfTokenizer(args.model, trust_remote_code=args.trust_remote_code)
    app = make_app(client, tokenizer, args.model_name or args.model.rsplit("/", 1)[-1],
                   api_key=args.api_key, request_timeout=args.request_timeout,
                   max_inflight=args.max_inflight, max_queue=args.max_queue,
                   state=state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    state.ready = True
    print(f"shelf-server: listening on http://{args.host}:{args.port}", file=sys.stderr)
    if args.api_key:
        print("shelf-server: API key required", file=sys.stderr)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:   # Windows など
            pass
    try:
        await stop.wait()
    finally:
        # T18: 新規受付を止め、処理中の要求を猶予時間まで待ってから畳む。
        state.shutting_down = True
        state.ready = False
        deadline = time.monotonic() + max(0.0, args.shutdown_grace)
        while state.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        print(
            f"shelf-server: shutdown (inflight={state.inflight}, "
            f"requests={sum(state.requests_total.values())}, "
            f"timeouts={state.timeouts}, errors={sum(state.errors_total.values())})",
            file=sys.stderr)
        await runner.cleanup()
        await client.close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="shelf-server", description="OpenAI 互換の HTTP 面（パス A）")
    p.add_argument("--engine", default="./build/hip/apps/shelf-compute")
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pipeline-parallel", type=int, default=1)
    p.add_argument("--prefill-rows", type=int, default=64)
    p.add_argument("--prefix-cache-blocks", type=int, default=0,
                   help="T16: prefix cache の追加ブロック数（0 = 無効）")
    p.add_argument("--kv-in-host", action="store_true",
                   help="T14: KV プールを pinned host に置く")
    p.add_argument("--online-kv-pages", type=int, default=0,
                   help="T14: pinned KV の直近 N ページを VRAM にミラーする（0 = 無効。スロットごとに窓を持つ）")
    p.add_argument("--max-batch", type=int, default=4,
                   help="T15: 連続バッチングの最大スロット数（既定 4）")
    p.add_argument("--offload-weights-mib", type=int, default=0,
                   help="T14: 重みを pinned host へ落とす量（0 = なし）")
    p.add_argument("--act-quant", default="auto", choices=["auto", "fp4", "fp8"],
                   help="MXFP4 重みの活性化量子化（auto = checkpoint の宣言）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-inflight", type=int, default=4,
                   help="同時にエンジンへ流す要求の上限（T15。既定 4）")
    p.add_argument("--max-queue", type=int, default=64,
                   help="T18: 実行待ちに許す要求数（超過は 429。既定 64）")
    p.add_argument("--api-key", default=os.environ.get("SHELF_API_KEY", ""),
                   help="T18: API key（空なら認証なし。環境変数 SHELF_API_KEY も可）")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="T18: 1 要求の総時間（秒。0 = 無効。既定 600）")
    p.add_argument("--shutdown-grace", type=float, default=10.0,
                   help="T18: SIGTERM/SIGINT 後に処理中の要求を待つ秒数（既定 10）")
    args = p.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
