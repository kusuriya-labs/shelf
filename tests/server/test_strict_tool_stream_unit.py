#!/usr/bin/env python3
"""Exercise strict tool stream conversion without a model or GPU."""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/apps/server'), str(ROOT / 'src/apps/common')]
from serverlib.chat_service import ChatService, PreparedChat, ToolCallDelta, StreamDone


class StrictToolStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_typed_arguments_and_whitespace(self):
        schema = {'type': 'object', 'properties': {'s': {'type': 'string'},
                  'n': {'type': 'integer'}, 'b': {'type': 'boolean'}},
                  'required': ['s', 'n', 'b'], 'additionalProperties': False}
        tools = [{'function': {'name': 'f', 'strict': True, 'parameters': schema}}]
        prepared = PreparedChat(prompt_ids=[1], tools=tools, max_new_tokens=32, enable_thinking=False)
        request = SimpleNamespace(tool_choice='required', tools=tools, forced_tool_name=None)
        answer = ('<tool_call>\n<function=f>\n<parameter=s>  x  </parameter>\n'
                  '<parameter=n>12</parameter>\n<parameter=b>true</parameter>\n'
                  '</function>\n</tool_call>')
        service = ChatService.__new__(ChatService)
        service._processor = object()
        async def stream(*args):
            yield {'token_id': 7}
            yield {'done': {'generated_ids': [7], 'finish_reason': 'stop'}}
        service._stream_open = stream
        with patch('serverlib.chat_service.codec.split_generation', return_value=SimpleNamespace(answer=answer, reasoning='')):
            events = [event async for event in service._stream_strict_tools(prepared, request, {})]
        call = next(event for event in events if isinstance(event, ToolCallDelta))
        self.assertEqual(json.loads(call.arguments), {'s': '  x  ', 'n': 12, 'b': True})
        self.assertEqual(next(event for event in events if isinstance(event, StreamDone)).finish_reason, 'tool_calls')


if __name__ == '__main__':
    unittest.main()
