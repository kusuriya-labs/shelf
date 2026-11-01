"""T18: 運用機能（認証・同時要求制限・タイムアウト・readiness・metrics・停止）。

フェイクエンジン（`fake_engine.py`）で HTTP 面だけを検証する。GPU もモデルも不要。
"""

from __future__ import annotations

import asyncio
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
MSG = [{"role": "user", "content": "hi"}]


class MarkerTokenizer:
    """プロンプト末尾を固定してフェイクエンジンの合図を選ぶ。"""

    def __init__(self, last: int = 0):
        self.last = last

    def prompt_ids(self, messages, *, add_generation_prompt=True, enable_thinking=None):
        return [1, self.last]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def build(tokenizer=None, *, die_after_ping=False, **kwargs):
    argv = list(FAKE) + (["--die-after-ping"] if die_after_ping else [])
    client = EngineClient(argv)
    state = srv.OpsState()
    app = srv.make_app(client, tokenizer or MarkerTokenizer(), MODEL, state=state, **kwargs)
    return client, state, TestClient(TestServer(app))


async def start(tc, client, state, *, ready=True):
    await client.start()
    await tc.start_server()
    state.ready = ready
    return tc


def test_healthz_is_always_ok_and_readyz_tracks_state():
    async def run():
        client, state, tc = build()
        await start(tc, client, state, ready=False)
        try:
            assert (await tc.get("/healthz")).status == 200
            r = await tc.get("/readyz")
            assert r.status == 503
            assert (await r.json())["status"] == "not_ready"
            state.ready = True
            r = await tc.get("/readyz")
            assert r.status == 200
            assert (await r.json())["status"] == "ready"
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_v1_returns_503_before_ready_and_while_shutting_down():
    async def run():
        client, state, tc = build()
        await start(tc, client, state, ready=False)
        try:
            r = await tc.post("/v1/chat/completions", json={"messages": MSG})
            assert r.status == 503
            assert (await r.json())["error"]["code"] == "service_unavailable"

            state.ready = True
            state.shutting_down = True
            r = await tc.post("/v1/chat/completions", json={"messages": MSG})
            assert r.status == 503
            assert (await tc.get("/readyz")).status == 503
            assert (await tc.get("/healthz")).status == 200
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_api_key_is_required_for_v1_and_metrics():
    async def run():
        client, state, tc = build(api_key="s3cret")
        await start(tc, client, state)
        try:
            r = await tc.post("/v1/chat/completions", json={"messages": MSG})
            assert r.status == 401
            body = await r.json()
            assert body["error"]["code"] == "invalid_api_key"
            assert r.headers.get("WWW-Authenticate") == "Bearer"
            assert state.auth_failures == 1

            bad = {"Authorization": "Bearer nope"}
            assert (await tc.post("/v1/chat/completions", json={"messages": MSG},
                                  headers=bad)).status == 401

            ok = {"Authorization": "Bearer s3cret"}
            r = await tc.post("/v1/chat/completions", json={"messages": MSG}, headers=ok)
            assert r.status == 200

            # 代替ヘッダ（X-API-Key）も受ける
            r = await tc.post("/v1/chat/completions", json={"messages": MSG},
                              headers={"X-API-Key": "s3cret"})
            assert r.status == 200

            # healthz / readyz は認証不要、metrics は必要
            assert (await tc.get("/healthz")).status == 200
            assert (await tc.get("/readyz")).status == 200
            assert (await tc.get("/metrics")).status == 401
            assert (await tc.get("/metrics", headers=ok)).status == 200
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_queue_limit_returns_429():
    async def run():
        client, state, tc = build(tokenizer=MarkerTokenizer(777),
                                  max_inflight=1, max_queue=1)
        await start(tc, client, state)
        slow = {"messages": MSG, "max_tokens": 40}
        try:
            first = asyncio.create_task(tc.post("/v1/chat/completions", json=slow))
            for _ in range(100):
                if state.inflight >= 1:
                    break
                await asyncio.sleep(0.01)
            second = asyncio.create_task(tc.post("/v1/chat/completions", json=slow))
            for _ in range(100):
                if state.inflight >= 2:
                    break
                await asyncio.sleep(0.01)
            assert state.inflight == 2

            r = await tc.post("/v1/chat/completions", json=slow)
            assert r.status == 429
            assert (await r.json())["error"]["code"] == "rate_limit_exceeded"
            assert state.queue_rejected == 1

            r1 = await first
            r2 = await second
            assert r1.status == 200 and r2.status == 200
            assert state.inflight == 0
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_non_stream_timeout_returns_504_and_cancels(tmp_path, monkeypatch):
    async def run():
        log = tmp_path / "cancel.log"
        monkeypatch.setenv("FAKE_CANCEL_LOG", str(log))
        client, state, tc = build(tokenizer=MarkerTokenizer(777), request_timeout=0.15)
        await start(tc, client, state)
        try:
            r = await tc.post("/v1/chat/completions",
                              json={"messages": MSG, "max_tokens": 64})
            assert r.status == 504
            assert (await r.json())["error"]["code"] == "timeout"
            assert state.timeouts == 1
            assert state.inflight == 0
            for _ in range(100):
                if log.exists() and log.read_text().strip():
                    break
                await asyncio.sleep(0.01)
            assert log.exists() and log.read_text().strip(), "エンジンへ cancel が届く"
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_stream_timeout_ends_with_error_and_done():
    async def run():
        client, state, tc = build(tokenizer=MarkerTokenizer(777), request_timeout=0.15)
        await start(tc, client, state)
        try:
            r = await tc.post("/v1/chat/completions",
                              json={"messages": MSG, "max_tokens": 64, "stream": True})
            assert r.status == 200
            text = await r.text()
            assert '"code": "timeout"' in text
            assert text.rstrip().endswith("data: [DONE]")
            assert state.timeouts == 1
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_metrics_exposes_counters():
    async def run():
        client, state, tc = build()
        await start(tc, client, state)
        try:
            assert (await tc.post("/v1/chat/completions",
                                  json={"messages": MSG, "max_tokens": 3})).status == 200
            # 受付エラーも数える
            assert (await tc.post("/v1/chat/completions",
                                  json={"messages": "bad"})).status == 400
            r = await tc.get("/metrics")
            assert r.status == 200
            text = await r.text()
            assert "shelf_up 1" in text
            assert "shelf_ready 1" in text
            assert 'shelf_requests_total{path="/v1/chat/completions",status="200"} 1' in text
            assert 'shelf_requests_total{path="/v1/chat/completions",status="400"} 1' in text
            assert 'shelf_errors_total{code="invalid_argument"} 1' in text
            assert "shelf_tokens_generated_total 3" in text
            assert "shelf_request_duration_seconds_count" in text
            assert "shelf_uptime_seconds" in text
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())


def test_engine_death_marks_not_ready():
    async def run():
        client, state, tc = build(die_after_ping=True)
        await client.start()
        await tc.start_server()
        state.ready = True
        try:
            for _ in range(200):
                if not client.is_alive():
                    break
                await asyncio.sleep(0.01)
            assert not client.is_alive()
            r = await tc.get("/readyz")
            assert r.status == 503
            r = await tc.post("/v1/chat/completions", json={"messages": MSG})
            assert r.status == 503
        finally:
            await tc.close()
            await client.close()

    asyncio.run(run())
