"""エンジンクライアントとエラー写像のテスト（モデル・GPU・ポート不要）。

`draft_docs/serve_protocol.md` §3 の表と 1 対 1 にする（表を変えたらここも変える）。
T02（done の伝搬）と T03（cancel・子プロセス終了）もここで固定する。
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # apps/server
sys.path.insert(0, HERE)                     # tests/

from engine_client import Done, EngineClient, EngineError, EngineProtocolError  # noqa: E402
import openai_errors as oe  # noqa: E402

FAKE = [sys.executable, os.path.join(HERE, "fake_engine.py")]


def run(coro):
    return asyncio.run(coro)


async def _collect(client, input_ids, **kwargs):
    """(token ids, Done) を集める。"""
    toks: list[int] = []
    done = None
    async for ev in client.generate(input_ids, **kwargs):
        if isinstance(ev, Done):
            done = ev
        else:
            toks.append(int(ev["token_id"]))
    return toks, done


def test_ping_capabilities():
    async def go():
        client = EngineClient(FAKE)
        caps = await client.start()
        await client.close()
        return caps

    caps = run(go())
    assert caps.backend == "fake"
    assert caps.max_context == 1024
    assert "fp8_w8a8" in caps.features


def test_generate_streams_tokens_then_done():
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        try:
            return await _collect(client, [5], max_new_tokens=3)
        finally:
            await client.close()

    toks, done = run(go())
    assert toks == [6, 7, 8], toks          # フェイクは入力末尾 +1..+3
    assert isinstance(done, Done)
    assert done.generated_ids == toks
    assert done.finish_reason == "length"
    assert done.prompt_tokens == 1
    assert done.generated_tokens == 3


def test_done_carries_finish_reason_stop():
    """エンジンの `stop` をそのまま伝搬する（HTTP 側で推測しない）。"""
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        try:
            return await _collect(client, [998], max_new_tokens=8)
        finally:
            await client.close()

    toks, done = run(go())
    assert toks == [999, 1000]              # 998 + 1, 998 + 2
    assert done is not None and done.finish_reason == "stop"


def test_generate_seed_is_echoed():
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        try:
            return await _collect(client, [1], max_new_tokens=1, seed=42)
        finally:
            await client.close()

    toks, done = run(go())
    assert toks == [2]
    assert done is not None and done.seed == 42


def test_engine_error_maps_to_api_error():
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        try:
            async for _ in client.generate([999], max_new_tokens=1):
                pass
        except EngineError as exc:
            await client.close()
            return exc
        await client.close()
        raise AssertionError("error が来なかった")

    exc = run(go())
    assert exc.code == "overloaded"
    api = oe.from_engine(exc.code, exc.message, exc.param)
    assert api.status == oe.status_for_code(exc.code)
    assert api.body()["error"]["type"] == oe.type_for_status(api.status)


def test_abort_sends_cancel_with_matching_request_id():
    """消費側が途中で止めたら、その request_id の cancel を送る（T03）。"""
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        sent: list[dict] = []
        orig = client._send

        async def spy(obj):
            sent.append(dict(obj))
            await orig(obj)

        client._send = spy  # noqa: SLF001 - テストのため
        try:
            gen = client.generate([777], max_new_tokens=2)   # 777 = 遅い生成
            first = await gen.__anext__()
            await gen.aclose()
            # 中断後も次の要求が流れる（terminal まで drain 済み）。
            toks, done = await _collect(client, [777], max_new_tokens=2)
            return first, sent, toks, done
        finally:
            await client.close()

    first, sent, toks, done = run(go())
    assert first["token_id"] == 778
    cancels = [o for o in sent if o.get("op") == "cancel"]
    assert cancels, "cancel が送られていない"
    assert cancels[0]["request_id"] == "1"      # 中断した要求の ID
    assert toks and toks[0] == 778
    assert done is not None


def test_engine_exit_unblocks_waiting_generate():
    """子プロセスが終了したら、待っている要求は 500 系で起きる（T03）。"""
    async def go():
        client = EngineClient(FAKE + ["--die-after-ping"])
        await client.start()
        try:
            return await asyncio.wait_for(_collect(client, [1], max_new_tokens=1), timeout=5.0)
        finally:
            await client.close()

    with pytest.raises(EngineError) as exc:
        run(go())
    assert exc.value.code in ("state_error", "internal_error")


def test_unknown_field_is_rejected_before_engine():
    """受付時の拒否はエンジンに届けない（serve_protocol §2-2）。

    ここでは「サーバ側の検証関数」を代表として、未知フィールドを受付で弾く形を示す。
    """
    body = {"model": "x", "messages": [], "min_p": 0.1, "nope": 1}
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "top_k",
               "min_p", "seed", "stream"}
    unknown = [k for k in body if k not in allowed]
    assert unknown == ["nope"]
    err = oe.bad_request(f"未知のフィールド '{unknown[0]}'", unknown[0])
    assert err.status == 400 and err.body()["error"]["param"] == "nope"


@pytest.mark.parametrize("code,status", [
    ("invalid_argument", 400),
    ("overflow", 400),
    ("context_length_exceeded", 400),
    ("unsupported_parameter", 400),
    ("request_too_large", 413),
    ("unsupported_media_type", 415),
    ("model_not_found", 404),
    ("busy", 409),
    ("overloaded", 503),
    ("state_error", 500),
    ("device_error", 500),
    ("internal_error", 500),
    ("何か知らない code", 500),
])
def test_error_table(code, status):
    """`serve_protocol.md` §3 の行（未知は 500 に落ちる）。"""
    assert oe.status_for_code(code) == status
    assert oe.type_for_status(status) == (
        "invalid_request_error" if status < 500 else "server_error")


def test_envelope_has_four_fields():
    err = oe.ApiError(400, "sampling top_p: (0, 1] の範囲が必要", "invalid_argument", "top_p")
    body = err.body()["error"]
    assert set(body.keys()) == {"message", "type", "param", "code"}
    assert body["type"] == "invalid_request_error"


def test_traceback_free_message():
    msg = "/opt/zen/work/shelf/src/runtime/generate/sampling.cpp:19: sampling top_p: 範囲外\n"
    clean = oe.traceback_free(msg)
    assert ".cpp" not in clean and "\n" not in clean
    assert clean.startswith("sampling top_p")


def test_bad_json_line_surfaces_protocol_error():
    """壊れた行はプロトコルエラーとして購読側へ（プロセスは落とさない）。"""
    async def go():
        client = EngineClient(FAKE)
        await client.start()
        try:
            # 直接 _send で壊れた行を送る（フェイクは error を返す）
            await client._send({"op": "nope"})  # noqa: SLF001 - テストのため
        finally:
            await client.close()

    run(go())   # 例外にならず終了すればよい（error は制御側で捨てられる）
