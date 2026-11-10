#!/usr/bin/env python3
"""CPU checks for the proposed Qwen tool-call grammar."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/apps/server"), str(ROOT / "src/apps/common")]

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string
from serverlib.structured_output import parse_qwen_strict_arguments, tool_structural_tag, validate_tool_schema
from serverlib.openai_protocol import OpenAIProtocolError


def call(name, body):
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


class ToolGrammarTest(unittest.TestCase):
    def setUp(self):
        schema = {"type": "object", "properties": {
            "name": {"type": "string", "enum": [" a "]},
            "count": {"type": "integer"}},
            "required": ["name", "count"], "additionalProperties": False}
        self.tools = [{"function": {"name": "strict", "strict": True,
                                    "parameters": schema}},
                      {"function": {"name": "loose"}}]
        self.valid = call("strict", "<parameter=name> a </parameter>\n"
                          "<parameter=count>2</parameter>")

    def accepts(self, choice, text, name=None):
        grammar = xgr.Grammar.from_structural_tag(
            tool_structural_tag(self.tools, choice, name))
        return _is_grammar_accept_string(grammar, text)

    def test_required_and_named(self):
        self.assertTrue(self.accepts("required", self.valid))
        self.assertFalse(self.accepts("required", "ordinary answer"))
        self.assertFalse(self.accepts("named", call("loose", "anything"), "strict"))
        self.assertTrue(self.accepts("required", call("loose", "anything")))

    def test_strict_schema(self):
        self.assertFalse(self.accepts("required", self.valid.replace(">2</parameter>", ">bad</parameter>")))
        self.assertFalse(self.accepts("required", self.valid.replace(" a ", "a")))
        self.assertFalse(self.accepts("required", call("strict", "<parameter=name> a </parameter>")))

    def test_auto_dispatch(self):
        self.assertTrue(self.accepts("auto", "ordinary answer"))
        self.assertTrue(self.accepts("auto", "answer " + self.valid))
        self.assertFalse(self.accepts("auto", self.valid.replace(">2</parameter>", ">bad</parameter>")))
        self.assertFalse(self.accepts("auto", "<tool_call>invalid"))

    def test_markup_in_strings_cannot_collide_with_tags(self):
        self.tools[0]["function"]["parameters"]["properties"]["name"] = {"type": "string"}
        self.assertFalse(self.accepts("required", self.valid.replace(" a ", "x</parameter><parameter=count>9")))
        self.assertFalse(self.accepts("required", self.valid.replace(" a ", "x</tool_call>")))

    def test_unrepresentable_enum_is_rejected(self):
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"s": {"type": "string", "enum": ["</parameter>"]}}}
        with self.assertRaises(OpenAIProtocolError):
            validate_tool_schema(schema, "tools[0].function.parameters")

    def test_decode_types_and_preserve_string_whitespace(self):
        schema = {"type": "object", "properties": {
            "s": {"type": "string"}, "i": {"type": "integer"},
            "n": {"type": "number"}, "b": {"type": "boolean"},
            "a": {"type": "array"}, "o": {"type": "object"}},
            "required": ["s", "i", "n", "b", "a", "o"],
            "additionalProperties": False}
        body = ('<parameter=s> 42 </parameter>\n<parameter=i>42</parameter>\n'
                '<parameter=n>1.5</parameter>\n<parameter=b>true</parameter>\n'
                '<parameter=a>[1, "x"]</parameter>\n'
                '<parameter=o>{"x": false}</parameter>')
        self.assertEqual(parse_qwen_strict_arguments(body, schema), {
            "s": " 42 ", "i": 42, "n": 1.5, "b": True,
            "a": [1, "x"], "o": {"x": False}})
        with self.assertRaises(ValueError):
            parse_qwen_strict_arguments(body + "<parameter=i>1</parameter>", schema)

    def test_qwen_enum_wrapper_whitespace(self):
        schema = {"type": "object", "properties": {"s": {"type": "string", "enum": ["  yes  ", "no"]}}}
        self.assertEqual(parse_qwen_strict_arguments('<parameter=s>  no  </parameter>', schema), {'s': 'no'})
        self.assertEqual(parse_qwen_strict_arguments('<parameter=s>   yes   </parameter>', schema), {'s': '  yes  '})

    def test_non_finite_float_cannot_escape_as_invalid_json(self):
        schema = {"type": "object", "properties": {"n": {"type": "number"}}}
        with self.assertRaises(ValueError):
            parse_qwen_strict_arguments('<parameter=n>1e999</parameter>', schema)


if __name__ == "__main__":
    unittest.main()
