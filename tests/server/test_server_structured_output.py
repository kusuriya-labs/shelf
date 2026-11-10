#!/usr/bin/env python3
"""End-to-end structured Chat Completions, including streaming and rejection."""
import json
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ServerHarness, http_json, http_post_status, http_sse

SCHEMAS = [
    {"type":"object", "properties":{"kind":{"type":"string", "enum":["red", "blue"]},
      "nested":{"type":"object", "properties":{"ok":{"type":"boolean"}},
                "required":["ok"], "additionalProperties":False},
      "optional":{"type":"integer"}}, "required":["kind", "nested"], "additionalProperties":False},
    {"type":"array", "items":{"type":"string", "enum":["yes", "no"]}},
]


def check(value, schema):
    if schema["type"] == "object":
        assert isinstance(value, dict)
        assert set(schema.get("required", [])) <= set(value)
        if schema.get("additionalProperties") is False: assert set(value) <= set(schema.get("properties", {}))
        for key, v in value.items():
            if key in schema.get("properties", {}): check(v, schema["properties"][key])
    elif schema["type"] == "array":
        assert isinstance(value, list)
        for item in value: check(item, schema["items"])
    elif schema["type"] == "string": assert isinstance(value, str)
    elif schema["type"] == "integer": assert isinstance(value, int) and not isinstance(value, bool)
    elif schema["type"] == "boolean": assert isinstance(value, bool)
    if "enum" in schema: assert value in schema["enum"]


def main():
    if importlib.util.find_spec("xgrammar") is None:
        print("SKIP: xgrammar is not installed")
        return 77
    with ServerHarness() as server:
        url = server.base_url + "/chat/completions"
        base = {"model":"shelf", "messages":[{"role":"user", "content":"Return a short JSON value matching the requested format."}],
                "chat_template_kwargs":{"enable_thinking":False}, "max_tokens":128}
        for i, schema in enumerate(SCHEMAS):
            for stream in (False, True):
                body = dict(base, temperature=0.8 if stream else 0, stream=stream,
                            response_format={"type":"json_schema", "json_schema":{"name":"test", "schema":schema, "strict":True}})
                if stream:
                    events = list(http_sse(url, body))
                    text = "".join(e["choices"][0].get("delta", {}).get("content", "") for e in events if e.get("choices"))
                else:
                    text = http_json(url, body)["choices"][0]["message"]["content"]
                check(json.loads(text), schema)
                print("schema", i, "stream", stream, "OK", flush=True)
        body = dict(base, temperature=0.8, response_format={"type":"json_object"})
        assert isinstance(json.loads(http_json(url, body)["choices"][0]["message"]["content"]), dict)
        body["stream"] = True
        events = list(http_sse(url, body))
        text = "".join(e["choices"][0].get("delta", {}).get("content", "") for e in events if e.get("choices"))
        assert isinstance(json.loads(text), dict)
        for stream in (False, True):
            bad = {"type":"string", "pattern":"a+"}
            body = dict(base, stream=stream, response_format={"type":"json_schema", "json_schema":{"name":"x", "schema":bad, "strict":True}})
            status, raw = http_post_status(url, body)
            assert status == 400 and json.loads(raw)["error"]["code"] == "unsupported_parameter", raw
        print("structured output passed")
    return 0

if __name__ == "__main__": raise SystemExit(main())
