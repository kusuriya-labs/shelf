"""Per-request XGrammar state for the explicitly supported JSON Schema subset."""
from __future__ import annotations

import base64
import json
import math

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
    if "enum" in schema:
        # XGrammar 0.2.9 treats enum as an alternative to type, rather than
        # intersecting them. Reject cases that could admit invalid values.
        if typ in ("object", "array"):
            fail("enum for object or array is not supported")
        def matches_type(value):
            if typ == "string": return isinstance(value, str)
            if typ == "boolean": return isinstance(value, bool)
            if typ == "null": return value is None
            if typ in ("number", "integer"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False
                if isinstance(value, float) and not math.isfinite(value): return False
                return typ == "number" or isinstance(value, int) or value.is_integer()
            return False
        if not all(matches_type(value) for value in schema["enum"]):
            fail("enum values must match the schema type")
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


def tool_structural_tag(tools, choice, forced_name=None):
    """Build Qwen XML tags; preserve the declared order of schema properties.

    Non-strict functions use free text within their tags.  The caller must
    validate strict schemas and function names before compiling this structure.
    """
    if choice not in ("auto", "required", "named"):
        raise ValueError("invalid tool choice")
    tags = []
    for tool in tools:
        function = tool["function"]
        if forced_name is not None and function["name"] != forced_name:
            continue
        content = ({"type": "json_schema", "json_schema": function["parameters"],
                    "style": "qwen_xml", "any_order": False}
                   if function.get("strict") else {"type": "any_text"})
        tags.append({"type": "tag",
                     "begin": "<tool_call>\n<function=" + function["name"] + ">\n",
                     "content": content,
                     "end": "\n</function>\n</tool_call>"})
    if not tags:
        raise ValueError("no matching tool")
    if choice == "auto":
        fmt = {"type": "triggered_tags", "triggers": ["<tool_call>"], "tags": tags}
    else:
        fmt = {"type": "or", "elements": tags} if len(tags) > 1 else tags[0]
    return {"type": "structural_tag", "format": fmt}


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
        self.after_reasoning = False
        self.matcher = xgr.GrammarMatcher(grammar)
        self.mask = xgr.allocate_token_bitmask(1, vocab_size)

    def next_mask(self):
        self.matcher.fill_next_token_bitmask(self.mask)
        return base64.b64encode(self.mask.numpy().tobytes()).decode("ascii")

    def accept(self, token):
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"compute emitted token {token} outside grammar")
        return not self.matcher.is_terminated()
