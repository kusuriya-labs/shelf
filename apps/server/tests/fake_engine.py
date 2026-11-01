"""テスト用のフェイクエンジン（`shelf-compute` の代役）。

stdin/stdout で `engine_contract.md` §1/§2 を話す。モデルも GPU も使わないので、
CI（`pytest`）でサーバ側の経路をそのまま検証できる。

テスト用の合図（`input_ids` の末尾）:
  * 999 — `overloaded` の error イベント
  * 998 — 2 トークン出して `finish_reason = "stop"`
  * 777 — 遅い生成（cancel を途中で受ける。T03 の検証用）
それ以外は `max_new_tokens` 個のトークンを出して `finish_reason = "length"`。

CLI: `--die-after-ping` で pong の直後に終了する（子プロセス終了の検証用）。
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time

_TOKENS_CAP = 64
_buf = b""
_deferred: list[str] = []
_cancelled: set[str] = set()


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _readline() -> str | None:
    """1 行読む（ブロック）。EOF は None。"""
    global _buf
    while b"\n" not in _buf:
        chunk = os.read(0, 4096)
        if not chunk:
            return None
        _buf += chunk
    line, _, _buf = _buf.partition(b"\n")
    return line.decode("utf-8", "replace")


def _poll_lines() -> list[str]:
    """未読の行を読む（ブロックしない）。"""
    global _buf
    while True:
        ready, _, _ = select.select([0], [], [], 0)
        if not ready:
            break
        chunk = os.read(0, 4096)
        if not chunk:
            break
        _buf += chunk
    out: list[str] = []
    while b"\n" in _buf:
        line, _, _buf = _buf.partition(b"\n")
        text = line.decode("utf-8", "replace").strip()
        if text:
            out.append(text)
    return out


def _log_cancel(request_id: str) -> None:
    """テストから観測できるよう、cancel をファイルへ記録する（環境変数で有効化）。"""
    path = os.environ.get("FAKE_CANCEL_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{request_id}\n")
    except OSError:
        pass


def _handle_control(line: str) -> bool:
    """生成中に届いた行を処理する。True = shutdown。"""
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        _emit({"event": "error", "code": "invalid_argument", "message": "bad json"})
        return False
    op = req.get("op")
    if op == "cancel":
        rid = str(req.get("request_id"))
        _cancelled.add(rid)
        _log_cancel(rid)
        return False
    if op == "ping":
        _emit(_pong())
        return False
    if op == "shutdown":
        return True
    _deferred.append(line)
    return False


def _pong() -> dict:
    return {
        "event": "pong",
        "capabilities": {
            "backend": "fake",
            "model": "fake-model",
            "max_context": 1024,
            "features": ["device_execution", "fp8_w8a8"],
        },
    }


def _done(req: dict, rid: str, ids: list[int], reason: str) -> None:
    _emit({
        "event": "done",
        "request_id": rid,
        "generated_ids": ids,
        "finish_reason": reason,
        "prompt_tokens": len(req.get("input_ids") or []),
        "generated_tokens": len(ids),
        "seed": int(req.get("seed", 0) or 0),
    })


def _generate(req: dict) -> bool:
    """1 要求を実行する。True = shutdown。"""
    rid = str(req.get("request_id"))
    ids_in = req.get("input_ids") or [0]
    marker = ids_in[-1] if ids_in else 0
    # `input_ids == [999]` はエラー注入の合図（テスト用の規約）。
    if ids_in == [999] or marker == 999:
        _emit({"event": "error", "request_id": rid, "code": "overloaded",
               "message": "engine: KV の予算が足りない"})
        return False
    if rid in _cancelled:
        _done(req, rid, [], "cancelled")
        _cancelled.discard(rid)
        return False

    if marker == 998:
        count, reason, slow = 2, "stop", False
    elif marker == 777:
        count, reason, slow = min(int(req.get("max_new_tokens", 3)), _TOKENS_CAP), "length", True
    else:
        count, reason, slow = min(int(req.get("max_new_tokens", 3)), _TOKENS_CAP), "length", False

    out: list[int] = []
    for i in range(count):
        token = marker + i + 1
        out.append(token)
        _emit({"event": "token", "request_id": rid, "token_id": token})
        if slow:
            time.sleep(0.01)
            for line in _poll_lines():
                if _handle_control(line):
                    _done(req, rid, out, "cancelled")
                    return True
            if rid in _cancelled:
                _cancelled.discard(rid)
                _done(req, rid, out, "cancelled")
                return False
    _done(req, rid, out, reason)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fake-engine")
    parser.add_argument("--die-after-ping", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    while True:
        if _deferred:
            line = _deferred.pop(0)
        else:
            line = _readline()
            if line is None:
                break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"event": "error", "code": "invalid_argument", "message": "bad json"})
            continue
        op = req.get("op")
        if op == "ping":
            _emit(_pong())
            if args.die_after_ping:
                return 0
            continue
        if op == "cancel":
            rid = str(req.get("request_id"))
            _cancelled.add(rid)
            _log_cancel(rid)
            continue
        if op == "shutdown":
            break
        if op != "generate":
            _emit({"event": "error", "code": "invalid_argument", "message": f"unknown op {op}"})
            continue
        if _generate(req):
            break
    _emit({"event": "shutdown"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
