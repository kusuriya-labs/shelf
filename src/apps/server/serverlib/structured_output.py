"""Per-request XGrammar state for the explicitly supported JSON Schema subset."""
from __future__ import annotations

import base64
import json

from .openai_protocol import OpenAIProtocolError


def validate_schema(schema, path="response_format.json_schema.schema"):
    def fail(message):
        raise OpenAIProtocolError(message, param=path, code="unsupported_parameter")
    if not isinstance(schema, dict):
        fail("schema must be an object")
    allowed = {"type", "properties", "required", "items", "enum", "additionalProperties", "title", "description"}
    unknown = set(schema) - allowed
    if unknown:
        fail(f"unsupported schema keyword: {sorted(unknown)[0]}")
    typ = schema.get("type")
    if typ not in ("object", "array", "string", "number", "integer", "boolean", "null"):
        fail("schema type must be a single supported type")
    for key in ("title", "description"):
        if key in schema and not isinstance(schema[key], str):
            fail(f"{key} must be a string")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        fail("enum must be a non-empty array")
    if typ == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list) or any(not isinstance(x, str) for x in required) or len(set(required)) != len(required) or not set(required) <= set(properties):
            fail("invalid object properties or required list")
        if not isinstance(schema.get("additionalProperties", True), bool):
            fail("additionalProperties must be a boolean")
        for name, child in properties.items():
            if not isinstance(name, str): fail("property names must be strings")
            validate_schema(child, path + ".properties." + name)
    elif "properties" in schema or "required" in schema or "additionalProperties" in schema:
        fail("object keywords require type object")
    if typ == "array":
        if "items" not in schema: fail("array items schema is required")
        validate_schema(schema["items"], path + ".items")
    elif "items" in schema:
        fail("items requires type array")


class GrammarFactory:
    def __init__(self, processor, vocab_size):
        self.processor = processor
        self.vocab_size = vocab_size
        self.compiler = None

    def create(self, schema):
        try:
            import xgrammar as xgr
            if not self.vocab_size:
                raise ValueError("model vocabulary size is unavailable")
            if self.compiler is None:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=self.vocab_size)
                self.compiler = xgr.GrammarCompiler(info)
            grammar = self.compiler.compile_json_schema(json.dumps(schema), strict_mode=False)
            return GrammarState(xgr, grammar, self.vocab_size)
        except ImportError as exc:
            raise OpenAIProtocolError("structured output requires xgrammar", param="response_format", code="unsupported_parameter") from exc
        except Exception as exc:
            raise OpenAIProtocolError(f"cannot compile schema: {exc}", param="response_format", code="unsupported_parameter") from exc


class GrammarState:
    def __init__(self, xgr, grammar, vocab_size):
        self.matcher = xgr.GrammarMatcher(grammar)
        self.mask = xgr.allocate_token_bitmask(1, vocab_size)

    def next_mask(self):
        self.matcher.fill_next_token_bitmask(self.mask)
        return base64.b64encode(self.mask.numpy().tobytes()).decode("ascii")

    def accept(self, token):
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"compute emitted token {token} outside grammar")
        return not self.matcher.is_terminated()
