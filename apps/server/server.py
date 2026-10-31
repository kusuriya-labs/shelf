"""shelf server — OpenAI 互換の HTTP 面（パス A）。

移植元: `references/phaseshift-fp8/src/apps/server/phaseshift_server.py`（自作品）。
shelf の契約は `draft_docs/serve_protocol.md`（エラー envelope・code → HTTP）と
`draft_docs/engine_contract.md`（エンジンの JSONL）。**このファイルはビルド非結線**。

設計の要点:
* 受付時の検証（未知フィールド・型・範囲）は**エンジンに渡す前**に 400 で返す
  （`serve_protocol.md` §2-2）。エンジン由来のエラーだけを `openai_errors.from_engine` で写す
* クライアントへ返すメッセージからはソース位置を落とす（§1）
* ストリームは SSE（`data: {...}\\n\\n` → 最後に `data: [DONE]`）
* トークナイザは差し替え可能（テストはフェイクを注入する）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Any, AsyncIterator, Optional, Protocol

from aiohttp import web

from engine_client import EngineCancelled, EngineClient, EngineError, EngineProtocolError
import openai_errors as oe

# 受付で許可するフィールド（これ以外は 400。拡張はここに足す）
_ALLOWED_CHAT_FIELDS = {
    "model", "messages", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "top_k", "min_p", "seed", "stream", "stop", "n",
    # shelf 拡張（`min_p` と同じ扱い）。思考の ON/OFF を chat template に渡す。
    "enable_thinking", "chat_template_kwargs",
}


class Tokenizer(Protocol):
    """chat messages → token id 列、token id 列 → 文字列。"""

    def prompt_ids(self, messages: list[dict], *, add_generation_prompt: bool = True,
                   enable_thinking: Optional[bool] = None) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class HfTokenizer:
    """HF `transformers` の AutoProcessor を使う実装（遅延 import）。

    chat template は参照の `phaseshift_chat/codec.py` と同じ考え方（`apply_chat_template`
    に messages を渡す）。思考モードの制御は未移植（既定のまま）。
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
        ids = self._proc.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True, **kwargs
        )
        if isinstance(ids, dict):  # BatchEncoding の場合
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def decode(self, ids: list[int]) -> str:
        tok = getattr(self._proc, "tokenizer", self._proc)
        return tok.decode(ids, skip_special_tokens=True)


def validate_chat_body(body: Any) -> tuple[dict, dict]:
    """受付時の検証。`(正規化したパラメータ, messages)` を返すか ApiError を投げる。"""
    if not isinstance(body, dict):
        raise oe.bad_request("リクエスト本文は JSON オブジェクト")
    unknown = sorted(k for k in body if k not in _ALLOWED_CHAT_FIELDS)
    if unknown:
        raise oe.bad_request(f"未知のフィールド '{unknown[0]}'", unknown[0])
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise oe.bad_request("messages は空でない配列", "messages")

    def num(name: str, default: float, lo: float, hi: float) -> float:
        if name not in body or body[name] is None:
            return default
        v = body[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise oe.bad_request(f"{name} は数値", name)
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            raise oe.bad_request(f"{name} は有限の値", name)
        if not (lo <= v <= hi):
            raise oe.bad_request(f"{name} は [{lo}, {hi}] の範囲", name)
        return v

    params: dict[str, Any] = {
        "max_new_tokens": int(num("max_tokens" if "max_tokens" in body else "max_completion_tokens", 256, 1, 32768)),
        "temperature": num("temperature", 0.0, 0.0, 100.0),
        "top_p": num("top_p", 1.0, 1e-6, 1.0),
        "min_p": num("min_p", 0.0, 0.0, 1.0),
        "stream": bool(body.get("stream", False)),
    }
    if "top_k" in body and body["top_k"] is not None:
        tk = body["top_k"]
        if not isinstance(tk, int) or isinstance(tk, bool) or tk < 0:
            raise oe.bad_request("top_k は 0 以上の整数", "top_k")
        params["top_k"] = int(tk)
    else:
        params["top_k"] = 0
    if "enable_thinking" in body and body["enable_thinking"] is not None:
        if not isinstance(body["enable_thinking"], bool):
            raise oe.bad_request("enable_thinking は真偽値", "enable_thinking")
        params["enable_thinking"] = bool(body["enable_thinking"])
    else:
        params["enable_thinking"] = None
    if "seed" in body and body["seed"] is not None:
        sd = body["seed"]
        if not isinstance(sd, int) or isinstance(sd, bool) or sd < 0:
            raise oe.bad_request("seed は 0 以上の整数", "seed")
        params["seed"] = int(sd)
    else:
        params["seed"] = None
    return params, {"messages": messages}


def _chunk(model: str, request_id: str, delta: dict, finish: Optional[str] = None) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def make_app(client: EngineClient, tokenizer: Tokenizer, model_name: str) -> web.Application:
    app = web.Application()

    @web.middleware
    async def error_mw(request: web.Request, handler):
        try:
            return await handler(request)
        except oe.ApiError as exc:
            return web.json_response(exc.body(), status=exc.status)
        except EngineError as exc:
            api = oe.from_engine(exc.code, oe.traceback_free(exc.message), exc.param)
            return web.json_response(api.body(), status=api.status)
        except EngineCancelled:
            return web.json_response(
                oe.ApiError(400, "要求はキャンセルされました", "cancelled").body(), status=400)
        except EngineProtocolError as exc:
            return web.json_response(
                oe.ApiError(500, oe.traceback_free(str(exc)), "internal_error").body(),
                status=500)
        except Exception as exc:  # noqa: BLE001 - 未処理例外は 500 のみ（§2-3）
            return web.json_response(
                oe.ApiError(500, "内部エラー", "internal_error").body(), status=500)

    app.middlewares.append(error_mw)

    async def healthz(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [
                {"id": model_name, "object": "model", "owned_by": "shelf"}]})

    async def chat(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise oe.bad_request("本文が JSON でない")
        params, _ = validate_chat_body(body)
        asked = body.get("model")
        if asked is not None and asked != model_name:
            raise oe.ApiError(404, f"model '{asked}' は無い", "model_not_found", "model")

        input_ids = tokenizer.prompt_ids(body["messages"],
                                         enable_thinking=params.get("enable_thinking"))
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        gen = client.generate(
            input_ids,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            min_p=params["min_p"],
            seed=params["seed"],
        )

        if not params["stream"]:
            ids: list[int] = []
            async for ev in gen:
                ids.append(int(ev["token_id"]))
            text = tokenizer.decode(ids) if ids else ""
            return web.json_response({
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if not ids else "length",
                }],
                "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(ids),
                          "total_tokens": len(input_ids) + len(ids)},
            })

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                     "Connection": "keep-alive"})
        await resp.prepare(request)

        async def send(obj: Any) -> None:
            await resp.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

        await send(_chunk(model_name, request_id, {"role": "assistant", "content": ""}))
        pending: list[int] = []
        try:
            async for ev in gen:
                pending.append(int(ev["token_id"]))
                # まとめて 1 チャンク（毎トークン 1 行だと過剰。8 個ずつ）
                if len(pending) >= 8:
                    await send(_chunk(model_name, request_id,
                                      {"content": tokenizer.decode(pending)}))
                    pending = []
            if pending:
                await send(_chunk(model_name, request_id, {"content": tokenizer.decode(pending)}))
            await send(_chunk(model_name, request_id, {}, finish="stop"))
            await resp.write(b"data: [DONE]\n\n")
        finally:
            await resp.write_eof()
        return resp

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    return app


async def _run(args: argparse.Namespace) -> int:
    client = EngineClient(
        [args.engine, "--model", args.model, "--gpu", str(args.gpu),
         "--pipeline-parallel", str(args.pipeline_parallel), "--prefill-rows",
         str(args.prefill_rows)],
        log=lambda line: print(f"[engine] {line}", file=sys.stderr),
    )
    caps = await client.start()
    print(f"shelf-server: engine ready backend={caps.backend} max_context={caps.max_context}",
          file=sys.stderr)
    tokenizer = HfTokenizer(args.model, trust_remote_code=args.trust_remote_code)
    app = make_app(client, tokenizer, args.model_name or args.model.rsplit("/", 1)[-1])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f"shelf-server: listening on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        await asyncio.Event().wait()
    finally:
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
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
