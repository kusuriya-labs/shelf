"""受付時検証（`serve_protocol.md` §2-2 / §4）の単体テスト（T04）。

HTTP を通さずに `validate_chat_body` を直接呼ぶ。表は「拒否する条件」と
「返す param / code」の 1 対 1。
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import openai_errors as oe  # noqa: E402
import server as srv  # noqa: E402


def _msg(text: str = "hi", role: str = "user") -> list[dict]:
    return [{"role": role, "content": text}]


def _expect(body, status: int, code: str, param):
    with pytest.raises(oe.ApiError) as exc:
        srv.validate_chat_body(body)
    err = exc.value
    assert err.status == status
    assert err.code == code
    assert err.param == param


def test_accepts_minimal_body():
    params, messages = srv.validate_chat_body({"messages": _msg("こんにちは")})
    assert messages == _msg("こんにちは")
    assert params["max_new_tokens"] == 256
    assert params["temperature"] == 0.0 and params["top_p"] == 1.0
    assert params["top_k"] == 0 and params["min_p"] == 0.0
    assert params["stream"] is False
    assert params["seed"] is None
    assert params["enable_thinking"] is None


@pytest.mark.parametrize("body,status,code,param", [
    ("not a dict", 400, "invalid_argument", None),
    ({"messages": _msg(), "nope": 1}, 400, "invalid_argument", "nope"),
    ({"model": 1, "messages": _msg()}, 400, "invalid_argument", "model"),
    # messages の構造
    ({"messages": "hi"}, 400, "invalid_argument", "messages"),
    ({"messages": []}, 400, "invalid_argument", "messages"),
    ({"messages": [1]}, 400, "invalid_argument", "messages[0]"),
    ({"messages": [{"role": "user"}]}, 400, "unsupported_parameter", "messages[0].content"),
    ({"messages": [{"role": "user", "content": None}]},
     400, "unsupported_parameter", "messages[0].content"),
    ({"messages": [{"role": None, "content": "x"}]},
     400, "invalid_argument", "messages[0].role"),
    ({"messages": [{"role": "tool", "content": "x"}]},
     400, "unsupported_parameter", "messages[0].role"),
    ({"messages": [{"role": "user", "content": "x", "tool_calls": []}]},
     400, "unsupported_parameter", "messages[0].tool_calls"),
    # stream の型
    ({"messages": _msg(), "stream": 1}, 400, "invalid_argument", "stream"),
    ({"messages": _msg(), "stream": "true"}, 400, "invalid_argument", "stream"),
    ({"messages": _msg(), "stream": None}, 400, "invalid_argument", "stream"),
    # stop（形は検証し、値は未対応）
    ({"messages": _msg(), "stop": "END"}, 400, "unsupported_parameter", "stop"),
    ({"messages": _msg(), "stop": ["a", "b"]}, 400, "unsupported_parameter", "stop"),
    ({"messages": _msg(), "stop": [""]}, 400, "invalid_argument", "stop[0]"),
    ({"messages": _msg(), "stop": 5}, 400, "invalid_argument", "stop"),
    ({"messages": _msg(), "stop": ["a", "b", "c", "d", "e"]},
     400, "invalid_argument", "stop"),
    # n
    ({"messages": _msg(), "n": 0}, 400, "invalid_argument", "n"),
    ({"messages": _msg(), "n": 2}, 400, "unsupported_parameter", "n"),
    ({"messages": _msg(), "n": 1.0}, 400, "invalid_argument", "n"),
    # chat_template_kwargs
    ({"messages": _msg(), "chat_template_kwargs": []},
     400, "invalid_argument", "chat_template_kwargs"),
    ({"messages": _msg(), "chat_template_kwargs": {"x": True}},
     400, "unsupported_parameter", "chat_template_kwargs.x"),
    ({"messages": _msg(), "chat_template_kwargs": {"enable_thinking": "no"}},
     400, "invalid_argument", "chat_template_kwargs.enable_thinking"),
    ({"messages": _msg(), "enable_thinking": True,
      "chat_template_kwargs": {"enable_thinking": False}},
     400, "invalid_argument", "chat_template_kwargs.enable_thinking"),
    # 数値
    ({"messages": _msg(), "temperature": -1}, 400, "invalid_argument", "temperature"),
    ({"messages": _msg(), "temperature": float("nan")},
     400, "invalid_argument", "temperature"),
    ({"messages": _msg(), "top_p": 1.5}, 400, "invalid_argument", "top_p"),
    ({"messages": _msg(), "min_p": -0.1}, 400, "invalid_argument", "min_p"),
    ({"messages": _msg(), "top_k": 2**40}, 400, "invalid_argument", "top_k"),
    ({"messages": _msg(), "seed": -1}, 400, "invalid_argument", "seed"),
    ({"messages": _msg(), "seed": 1 << 64}, 400, "invalid_argument", "seed"),
    ({"messages": _msg(), "max_tokens": 0}, 400, "invalid_argument", "max_tokens"),
    ({"messages": _msg(), "max_tokens": 40000}, 400, "invalid_argument", "max_tokens"),
    ({"messages": _msg(), "max_tokens": 3, "max_completion_tokens": 4},
     400, "invalid_argument", "max_tokens"),
])
def test_rejections(body, status, code, param):
    _expect(body, status, code, param)


def test_message_count_limit():
    messages = [{"role": "user", "content": "x"}] * (srv._MAX_MESSAGES + 1)
    _expect({"messages": messages}, 400, "invalid_argument", "messages")


def test_total_content_size_limit():
    big = "a" * (srv._MAX_TOTAL_CHARS // 2 + 1)
    messages = [{"role": "user", "content": big}, {"role": "user", "content": big}]
    _expect({"messages": messages}, 413, "request_too_large", "messages")


def test_chat_template_kwargs_enable_thinking():
    params, _ = srv.validate_chat_body(
        {"messages": _msg(), "chat_template_kwargs": {"enable_thinking": False}})
    assert params["enable_thinking"] is False

    params, _ = srv.validate_chat_body({"messages": _msg(), "enable_thinking": True})
    assert params["enable_thinking"] is True

    # 同じ値なら矛盾ではない。
    params, _ = srv.validate_chat_body(
        {"messages": _msg(), "enable_thinking": True,
         "chat_template_kwargs": {"enable_thinking": True}})
    assert params["enable_thinking"] is True


def test_max_completion_tokens_alone():
    params, _ = srv.validate_chat_body(
        {"messages": _msg(), "max_completion_tokens": 12})
    assert params["max_new_tokens"] == 12

    params, _ = srv.validate_chat_body(
        {"messages": _msg(), "max_tokens": 12, "max_completion_tokens": 12})
    assert params["max_new_tokens"] == 12
