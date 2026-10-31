"""テスト用のフェイクエンジン（`shelf-serve` の代役）。

stdin/stdout で `engine_contract.md` §1/§2 を話す。モデルも GPU も使わないので、
CI（`pytest`）でサーバ側の経路をそのまま検証できる。
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    tokens = 3
    for line in sys.stdin:
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
            _emit({
                "event": "pong",
                "capabilities": {
                    "backend": "fake",
                    "model": "fake-model",
                    "max_context": 1024,
                    "features": ["device_execution", "fp8_w8a8"],
                },
            })
            continue
        if op == "cancel":
            # 直列なので「次の generate を cancelled にする」だけ
            cancelled.add(str(req.get("request_id")))
            continue
        if op != "generate":
            _emit({"event": "error", "code": "invalid_argument", "message": f"unknown op {op}"})
            continue

        rid = str(req.get("request_id"))
        # `input_ids == [999]` はエラー注入の合図（テスト用の規約）。
        if req.get("input_ids") == [999]:
            _emit({"event": "error", "request_id": rid, "code": "overloaded",
                   "message": "engine: KV の予算が足りない"})
            continue
        if rid in cancelled:
            _emit({"event": "done", "request_id": rid, "generated_ids": [],
                   "finish_reason": "cancelled", "prompt_tokens": len(req.get("input_ids", [])),
                   "generated_tokens": 0, "seed": int(req.get("seed", 0) or 0)})
            cancelled.discard(rid)
            continue

        ids: list[int] = []
        base = (req.get("input_ids") or [0])[-1]
        for i in range(tokens):
            tok = base + i + 1
            ids.append(tok)
            _emit({"event": "token", "request_id": rid, "token_id": tok})
        _emit({
            "event": "done",
            "request_id": rid,
            "generated_ids": ids,
            "finish_reason": "length",
            "prompt_tokens": len(req.get("input_ids") or []),
            "generated_tokens": len(ids),
            "seed": int(req.get("seed", 0) or 0),
        })
    _emit({"event": "shutdown"})
    return 0


cancelled: set[str] = set()


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
