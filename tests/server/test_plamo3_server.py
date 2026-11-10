#!/usr/bin/env python3
"""Real PLaMo 3 server smoke test; set SHELF_PLAMO_MODEL_DIR to enable."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import Checker, ServerHarness, http_get_json, http_json, http_post_status, http_sse


def main():
    model = Path(os.environ.get("SHELF_PLAMO_MODEL_DIR", ""))
    if not (model / "config.json").is_file():
        print("SKIP: SHELF_PLAMO_MODEL_DIR is not a model directory")
        return 77
    os.environ["SHELF_MODEL_DIR"] = str(model)
    check = Checker("plamo3-server")
    try:
        with ServerHarness(max_seq_len=2048) as server:
            check.check("models", bool(http_get_json(server.base_url + "/models")["data"]))
            meta = http_get_json(server.base_url + "/meta")
            check.check("no-reasoning", not meta["reasoning"]["enabled"], repr(meta))
            cases = [
                ("今日は良い天気ですね。", {"source_lang": "ja", "target_lang": "en", "context": "天気の会話", "glossary": [{"source": "天気", "target": "weather"}]}),
                ("The train will arrive in five minutes.", {"source_lang": "en", "target_lang": "ja", "form": "polite"}),
            ]
            for source, args in cases:
                body = {"model": "shelf", "messages": [{"role": "user", "content": source}],
                        "temperature": 0, "max_tokens": 48, "chat_template_kwargs": args}
                result = http_json(server.base_url + "/chat/completions", body)
                choice = result["choices"][0]
                content = choice["message"]["content"]
                check.check("translation", bool(content.strip()) and choice["finish_reason"] == "stop", repr(result))
                chunks = list(http_sse(server.base_url + "/chat/completions", {**body, "stream": True}))
                text = "".join(c.get("delta", {}).get("content", "") for x in chunks for c in x.get("choices", []))
                check.check("stream-parity", text == content, repr((text, content)))
            body["messages"][0]["content"] = "hello " * 4000
            status, detail = http_post_status(server.base_url + "/chat/completions", body)
            check.check("context-limit", status == 400 and "context" in detail.lower(), detail)
    except Exception as exc:
        check.check("exception", False, repr(exc))
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
