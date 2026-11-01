"""HTTP 面の表駆動テスト（フェイクエンジン・実ポート不要）。

`serve_protocol.md` §3 の各行と、受付時の 400（エンジンに届けない）を検証する。

T02: finish_reason / seed / usage はエンジンの `done` から写す。
T05: SSE の増分はトークン列全体の decode から確定し、非ストリームと一致する。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from aiohttp.test_utils import TestClient, TestServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine_client import EngineClient  # noqa: E402
import server as srv  # noqa: E402

FAKE = [sys.executable, os.path.join(HERE, "fake_engine.py")]
MODEL = "test-model"


class FakeTokenizer:
    """1 メッセージ = 1 トークン（テスト用の最小実装）。"""

    def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
        return [10 + i for i in range(len(messages))]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class FixedTokenizer(FakeTokenizer):
    """プロンプトの末尾を固定する（フェイクエンジンの合図を選ぶ）。"""

    def __init__(self, last: int):
        self.last = last

    def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
        return [1, self.last]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class ByteTokenizer:
    """UTF-8 のバイト境界がトークン境界とずれるトークナイザ（T05 用）。

    `[1..10]` の decode は "abcdefg日!"。8 と 9 が「日」の 2 バイトに分かれており、
    8 個ずつ decode すると途中で U+FFFD が出る（旧実装の失敗）。
    """

    _VOCAB = {
        1: b"a", 2: b"b", 3: b"c", 4: b"d", 5: b"e", 6: b"f", 7: b"g",
        8: b"\xe6\x97",   # 「日」の 1 バイト目まで
        9: b"\xa5",       # 残り
        10: b"!",
    }

    def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
        return [0]

    def decode(self, ids):
        raw = b"".join(self._VOCAB.get(int(i), b"?") for i in ids)
        return raw.decode("utf-8", "replace")


def client_for(tokenizer=None):
    client = EngineClient(FAKE)
    app = srv.make_app(client, tokenizer or FakeTokenizer(), MODEL)
    return TestClient(TestServer(app)), client


async def _post(body, tokenizer=None, headers=None):
    tc, client = client_for(tokenizer)
    await client.start()
    await tc.start_server()
    try:
        resp = await tc.post("/v1/chat/completions", json=body, headers=headers)
        return resp.status, await resp.json()
    finally:
        await tc.close()
        await client.close()


def post(body, tokenizer=None, headers=None):
    return asyncio.run(_post(body, tokenizer, headers))


def _msg(text: str = "hi") -> list[dict]:
    return [{"role": "user", "content": text}]


@pytest.mark.parametrize("body,status,code,param", [
    # ---- 未知フィールド・型（受付時。エンジンに届かない）----
    ({"messages": _msg(), "nope": 1}, 400, "invalid_argument", "nope"),
    ({"messages": "hi"}, 400, "invalid_argument", "messages"),
    ({"messages": []}, 400, "invalid_argument", "messages"),
    ({"messages": [1]}, 400, "invalid_argument", "messages[0]"),
    ({"messages": [{"content": "hi"}]}, 400, "invalid_argument", "messages[0].role"),
    ({"messages": [{"role": "tool", "content": "hi"}]},
     400, "unsupported_parameter", "messages[0].role"),
    ({"messages": [{"role": "user", "content": 5}]},
     400, "unsupported_parameter", "messages[0].content"),
    ({"messages": [{"role": "user", "content": "hi", "name": "x"}]},
     400, "unsupported_parameter", "messages[0].name"),
    # ---- サンプリング ----
    ({"messages": _msg(), "top_p": 0}, 400, "invalid_argument", "top_p"),
    ({"messages": _msg(), "min_p": 2}, 400, "invalid_argument", "min_p"),
    ({"messages": _msg(), "top_k": -1}, 400, "invalid_argument", "top_k"),
    ({"messages": _msg(), "top_k": 1.5}, 400, "invalid_argument", "top_k"),
    ({"messages": _msg(), "temperature": "x"}, 400, "invalid_argument", "temperature"),
    ({"messages": _msg(), "max_tokens": 0}, 400, "invalid_argument", "max_tokens"),
    ({"messages": _msg(), "max_tokens": 1.5}, 400, "invalid_argument", "max_tokens"),
    ({"messages": _msg(), "max_tokens": 4, "max_completion_tokens": 5},
     400, "invalid_argument", "max_tokens"),
    ({"messages": _msg(), "seed": -1}, 400, "invalid_argument", "seed"),
    # ---- 明示的に未対応（黙って無視しない）----
    ({"messages": _msg(), "stream": "yes"}, 400, "invalid_argument", "stream"),
    ({"messages": _msg(), "n": 2}, 400, "unsupported_parameter", "n"),
    ({"messages": _msg(), "stop": "END"}, 400, "unsupported_parameter", "stop"),
    ({"messages": _msg(), "stop": [1]}, 400, "invalid_argument", "stop[0]"),
    ({"messages": _msg(), "chat_template_kwargs": {"foo": True}},
     400, "unsupported_parameter", "chat_template_kwargs.foo"),
    ({"messages": _msg(), "chat_template_kwargs": {"enable_thinking": 1}},
     400, "invalid_argument", "chat_template_kwargs.enable_thinking"),
    ({"messages": _msg(), "enable_thinking": "yes"},
     400, "invalid_argument", "enable_thinking"),
    # ---- モデル不一致 ----
    ({"model": "other", "messages": _msg()}, 404, "model_not_found", "model"),
])
def test_error_table(body, status, code, param):
    got_status, payload = post(body)
    assert got_status == status
    err = payload["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["code"] == code
    assert err["param"] == param
    assert err["type"] == ("invalid_request_error" if status < 500 else "server_error")


def test_engine_error_maps_to_http():
    """エンジン由来の code は表どおりに写す（受付ではない）。"""
    class ErrTokenizer(FakeTokenizer):
        def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
            return [999]

    status, payload = post({"messages": _msg(), "max_tokens": 1}, tokenizer=ErrTokenizer())
    assert status == 503
    assert payload["error"]["code"] == "overloaded"
    assert payload["error"]["type"] == "server_error"


def test_content_type_is_checked():
    body = {"messages": _msg()}
    status, payload = post(body, headers={"Content-Type": "text/plain"})
    assert status == 415
    assert payload["error"]["code"] == "unsupported_media_type"


def test_context_length_exceeded():
    class LongTokenizer(FakeTokenizer):
        def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
            return list(range(2000))

    status, payload = post({"messages": _msg(), "max_tokens": 256},
                           tokenizer=LongTokenizer())
    assert status == 400
    assert payload["error"]["code"] == "context_length_exceeded"


def test_non_stream_completion_uses_done():
    status, payload = post({"model": MODEL, "messages": _msg(), "max_tokens": 3, "seed": 7})
    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["seed"] == 7
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "<11><12><13>"
    assert choice["finish_reason"] == "length"
    # done の usage をそのまま写す（HTTP 側で数え直さない）。
    assert payload["usage"] == {"prompt_tokens": 1, "completion_tokens": 3,
                                "total_tokens": 4}


def test_non_stream_finish_reason_stop():
    status, payload = post({"messages": _msg(), "max_tokens": 8},
                           tokenizer=FixedTokenizer(998))
    assert status == 200
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == "<999><1000>"
    assert payload["usage"]["completion_tokens"] == 2


async def _stream_lines(body, tokenizer=None):
    tc, client = client_for(tokenizer)
    await client.start()
    await tc.start_server()
    try:
        resp = await tc.post("/v1/chat/completions", json=body)
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        lines = []
        async for raw in resp.content:
            text = raw.decode().strip()
            if text.startswith("data:"):
                lines.append(text[len("data:"):].strip())
        return lines
    finally:
        await tc.close()
        await client.close()


def stream(body, tokenizer=None):
    return asyncio.run(_stream_lines(body, tokenizer))


def test_stream_completion_sse():
    lines = stream({"model": MODEL, "messages": _msg(), "max_tokens": 4, "stream": True})
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # 増分の連結 = 非ストリームの本文
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert joined == "<11><12><13><14>"
    final = chunks[-1]["choices"][0]
    assert final["finish_reason"] == "length"
    assert chunks[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 4,
                                   "total_tokens": 5}
    assert "seed" in chunks[-1]


def test_stream_finish_reason_stop():
    chunks = [json.loads(x) for x in
              stream({"messages": _msg(), "max_tokens": 8, "stream": True},
                     tokenizer=FixedTokenizer(998))[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 2


def test_stream_unicode_restoration_matches_non_stream():
    """T05: 日本語（トークン境界がバイト境界とずれる）でも非ストリームと一致する。"""
    body = {"messages": _msg(), "max_tokens": 10}
    status, payload = post(dict(body), tokenizer=ByteTokenizer())
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "abcdefg日!"

    chunks = [json.loads(x) for x in
              stream(dict(body, stream=True), tokenizer=ByteTokenizer())[:-1]]
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert joined == "abcdefg日!"
    assert "\ufffd" not in joined


def test_stream_usage_and_seed_from_done():
    chunks = [json.loads(x) for x in
              stream({"messages": _msg(), "max_tokens": 2, "stream": True, "seed": 3})[:-1]]
    assert chunks[-1]["seed"] == 3
    assert chunks[-1]["usage"]["prompt_tokens"] == 1
    assert chunks[-1]["usage"]["completion_tokens"] == 2


def test_models_endpoint():
    async def go():
        tc, client = client_for()
        await client.start()
        await tc.start_server()
        try:
            resp = await tc.get("/v1/models")
            return resp.status, await resp.json()
        finally:
            await tc.close()
            await client.close()

    status, payload = asyncio.run(go())
    assert status == 200
    assert payload["data"][0]["id"] == MODEL


def test_disconnect_cancels_engine(tmp_path, monkeypatch):
    """HTTP 切断でエンジンへ cancel が届く（T03）。"""
    log = tmp_path / "cancel.log"
    monkeypatch.setenv("FAKE_CANCEL_LOG", str(log))

    class SlowTokenizer(FakeTokenizer):
        def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
            return [777]   # フェイクの「遅い生成」

        def decode(self, ids):
            return "x" * len(ids)   # 毎トークン空でない差分にする

    async def go():
        tc, client = client_for(SlowTokenizer())
        await client.start()
        await tc.start_server()
        try:
            resp = await tc.post("/v1/chat/completions", json={
                "messages": _msg(), "max_tokens": 64, "stream": True})
            assert resp.status == 200
            await resp.content.readline()   # 最初の chunk（role）
            resp.close()
            for _ in range(150):
                if log.exists() and log.read_text(encoding="utf-8").strip():
                    break
                await asyncio.sleep(0.02)
        finally:
            await tc.close()
            await client.close()

    asyncio.run(go())
    assert log.exists(), "cancel がエンジンへ届かなかった"
    assert log.read_text(encoding="utf-8").strip()


def test_body_size_limit_returns_413():
    """本文が上限を超えたら 413（`serve_protocol.md` §4）。受付で読み捨てる。"""
    payload = (b'{"messages":[{"role":"user","content":"' +
               b"a" * (srv._MAX_BODY_BYTES + 1) + b'"}]}')

    async def go():
        tc, client = client_for()
        await client.start()
        await tc.start_server()
        try:
            resp = await tc.post("/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
            return resp.status, await resp.json()
        finally:
            await tc.close()
            await client.close()

    status, body = asyncio.run(go())
    assert status == 413
    assert body["error"]["code"] == "request_too_large"
    assert body["error"]["type"] == "invalid_request_error"
