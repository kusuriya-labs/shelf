"""Per-request XGrammar state for the explicitly supported JSON Schema subset."""
from __future__ import annotations

import base64
import json
import math
import re

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


def validate_tool_schema(schema, path):
    validate_schema(schema, path)
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise OpenAIProtocolError("strict tool parameters must be an object with additionalProperties=false", param=path, code="unsupported_parameter")
    def check(node):
        if any(isinstance(v, str) and "<" in v for v in node.get("enum", [])):
            raise OpenAIProtocolError("strict tool string enum cannot contain '<'", param=path, code="unsupported_parameter")
        for name, child in node.get("properties", {}).items():
            if "<" in name:
                raise OpenAIProtocolError("strict tool property cannot contain '<'", param=path, code="unsupported_parameter")
            check(child)
        if "items" in node:
            check(node["items"])
    check(schema)
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in schema.get("properties", {})):
        raise OpenAIProtocolError("invalid parameter name", param=path, code="unsupported_parameter")


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
                    "style": "qwen_xml", "any_order": False, "excludes": ["<"]}
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


def parse_qwen_strict_arguments(body, schema):
    """Decode XML parameter bodies using the schema, preserving string bytes.

    The caller must supply a grammar-validated function body. Unlike the
    generic tool parser, this does not strip whitespace from string values.
    """
    properties = schema.get("properties", {})
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("number is outside the supported range")
        return result
    result = {}
    position = 0
    for match in re.finditer(r"<parameter=([^>\s]+)>(.*?)</parameter>", body, re.DOTALL):
        if body[position:match.start()].strip():
            raise ValueError("unexpected text between parameters")
        name, value = match.groups()
        if name in result:
            raise ValueError("duplicate parameter")
        child = properties.get(name)
        if child is None:
            if not schema.get("additionalProperties", True):
                raise ValueError("unexpected parameter")
            # An unknown property has no type constraint; a string is valid.
            result[name] = value
        elif child["type"] == "string":
            # qwen_xml permits whitespace around enum alternatives inside a
            # parameter. Remove only that grammar whitespace, never whitespace
            # belonging to the declared enum value (or a free-form string).
            if "enum" in child and value not in child["enum"]:
                candidates = [item for item in child["enum"] if item.strip() == value.strip()]
                if not candidates:
                    raise ValueError("string is outside enum")
                value = candidates[0]
            result[name] = value
        else:
            result[name] = json.loads(value, parse_float=finite_float)
        position = match.end()
    if body[position:].strip():
        raise ValueError("unexpected text after parameters")
    if not set(schema.get("required", [])) <= result.keys():
        raise ValueError("missing required parameter")
    return result


class GrammarFactory:
    def __init__(self, processor, vocab_size):
        self.processor = processor
        self.vocab_size = vocab_size
        self.compiler = None

    def create(self, schema, *, tools=False):
        try:
            import xgrammar as xgr
            if not self.vocab_size:
                raise ValueError("model vocabulary size is unavailable")
            if self.compiler is None:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=self.vocab_size)
                self.compiler = xgr.GrammarCompiler(info)
            grammar = (self.compiler.compile_structural_tag(schema) if tools else
                       self.compiler.compile_json_schema(json.dumps(schema), strict_mode=False))
            return GrammarState(xgr, grammar, self.vocab_size)
        except ImportError as exc:
            raise OpenAIProtocolError("structured output requires xgrammar", param="tools" if tools else "response_format", code="unsupported_parameter") from exc
        except Exception as exc:
            raise OpenAIProtocolError(f"cannot compile schema: {exc}", param="tools" if tools else "response_format", code="unsupported_parameter") from exc


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
