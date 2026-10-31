"""shelf server — OpenAI 互換のエラー envelope（`serve_protocol.md` §1〜§3）。

エンジンの `code`（`shelf-serve` が出す文字列）を HTTP status へ写す**唯一の表**。
`shelf::Status::Code` → `code` の写像はエンジン側（`src/api/protocol.cpp`）が持つので、
ここは「code → HTTP」だけを知る（Server が shelf の型を知らない = パス A の利点）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# `serve_protocol.md` §3 の表（エンジンが出しうる code → HTTP）
_CODE_TO_STATUS: dict[str, int] = {
    "invalid_argument": 400,
    "overflow": 400,
    "context_length_exceeded": 400,
    "unsupported_parameter": 400,
    "model_not_found": 404,
    "busy": 409,
    "overloaded": 503,
    "state_error": 500,
    "device_error": 500,
    "internal_error": 500,
}


def status_for_code(code: str) -> int:
    return _CODE_TO_STATUS.get(code, 500)


def type_for_status(status: int) -> str:
    # §1: type は status から機械生成する（500 未満 = invalid_request_error）
    return "invalid_request_error" if int(status) < 500 else "server_error"


@dataclass
class ApiError(Exception):
    """クライアントへ返すエラー。envelope の 4 フィールドを保持する。"""

    status: int
    message: str
    code: str
    param: Optional[str] = None

    @property
    def type(self) -> str:
        return type_for_status(self.status)

    def body(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


def bad_request(message: str, param: Optional[str] = None,
                code: str = "invalid_argument") -> ApiError:
    """受付時の拒否（`serve_protocol.md` §2-2: エンジンに届ける前に 400）。"""
    return ApiError(400, message, code, param)


def from_engine(code: str, message: str, param: Optional[str] = None) -> ApiError:
    """エンジン由来の error イベントを API エラーへ（表はこのファイルが正本）。"""
    return ApiError(status_for_code(code), message, code, param or None)


def traceback_free(message: str) -> str:
    """クライアント向けメッセージからソース位置・改行を落とす（§1）。"""
    text = " ".join(str(message).split())
    # "<path>:<line>: " 形式の接頭辞を落とす
    head, sep, rest = text.partition(": ")
    if sep and "/" in head and head.rsplit(":", 1)[-1].isdigit():
        text = rest
    if len(text) > 400:
        text = text[:400] + "…"
    return text
