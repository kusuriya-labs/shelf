import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/apps/server'), str(ROOT / 'src/apps/common')]
from serverlib import hy4_tools
from serverlib.chat_service import ChatService, PreparedChat, ToolCallDelta, StreamDone, ReasoningDelta
from serverlib.openai_protocol import ChatRequest, encode_chat_completion, openai_messages_to_hf

TOOLS = [{'function': {'name': 'shell', 'parameters': {'type': 'object', 'properties': {
    'command': {'type': 'string'}, 'timeout': {'type': 'integer'}}}}}]
ANSWER = ('<tool_calls:opensource><tool_call:opensource>shell'
          '<arg_key:opensource>command</arg_key:opensource>'
          '<arg_value:opensource>cat README.md</arg_value:opensource>'
          '<arg_key:opensource>timeout</arg_key:opensource>'
          '<arg_value:opensource>30</arg_value:opensource>'
          '</tool_call:opensource></tool_calls:opensource>')


class Hy4ToolTest(unittest.TestCase):
    def test_arguments_and_multiple_calls(self):
        content, calls = hy4_tools.parse_answer('前' + ANSWER + '後', TOOLS)
        self.assertEqual(content, '前後')
        self.assertEqual(calls[0]['function']['arguments'], {'command': 'cat README.md', 'timeout': 30})
        inner = ANSWER[len(hy4_tools.CALLS_OPEN):-len(hy4_tools.CALLS_CLOSE)]
        self.assertEqual(len(hy4_tools.parse_answer(hy4_tools.CALLS_OPEN + inner * 2 + hy4_tools.CALLS_CLOSE, TOOLS)[1]), 2)

    def test_invalid_and_incomplete(self):
        for answer in (ANSWER[:-10], ANSWER.replace('shell', 'unknown', 1), ANSWER.replace('30</arg_value', 'bad</arg_value')):
            with self.assertRaises(ValueError):
                hy4_tools.parse_answer(answer, TOOLS)

    def test_tool_history_ids(self):
        messages = [{'role': 'assistant', 'tool_calls': [{'id': 'call_1', 'function': {'name': 'shell', 'arguments': '{"command":"ls"}'}}]},
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'file'}]
        converted = openai_messages_to_hf(messages, preserve_tool_ids=True)
        self.assertEqual(converted[0]['tool_calls'][0]['id'], 'call_1')
        self.assertEqual(converted[1]['tool_call_id'], 'call_1')
        self.assertNotIn('id', openai_messages_to_hf(messages)[0]['tool_calls'][0])

    def test_prepare_named_and_none(self):
        class Tokenizer:
            _shelf_hy4 = True
            def encode(self, text, **kwargs):
                self.prefix = text
                return [7]
        service = ChatService.__new__(ChatService)
        service._processor = Tokenizer()
        service._max_seq_len = 8192
        service._inference_budget_tokens = 100
        service._large_context_tokens = 0
        service._large_context_inference_budget_tokens = 0
        service._reasoning_close_token = -1
        request = ChatRequest(model='hy4', messages=[{'role': 'user', 'content': 'x'}], tools=TOOLS,
                              tool_choice='named', forced_tool_name='shell', stream=False, max_new_tokens=64,
                              temperature=0, top_p=1, min_p=0, top_k=0, seed=0)
        with patch('serverlib.chat_service.codec.prompt_ids', return_value=[1]) as prompt:
            prepared = service.prepare(request)
        self.assertEqual(prepared.tool_prefix, hy4_tools.CALLS_OPEN + hy4_tools.CALL_OPEN + 'shell')
        self.assertTrue(service._processor.prefix.startswith('</think:opensource>'))
        self.assertEqual(prompt.call_args.kwargs['tools'], TOOLS)
        from dataclasses import replace
        with patch('serverlib.chat_service.codec.prompt_ids', return_value=[1]) as prompt:
            prepared = service.prepare(replace(request, tool_choice='none', forced_tool_name=None))
        self.assertIsNone(prepared.tools)
        self.assertIsNone(prompt.call_args.kwargs['tools'])


class Hy4StreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream(self):
        service = ChatService.__new__(ChatService)
        service._processor = object()
        async def stream(*args):
            yield {'token_id': 1}
            yield {'done': {'generated_ids': [1], 'finish_reason': 'stop'}}
        service._stream_open = stream
        prepared = PreparedChat(prompt_ids=[1], tools=TOOLS, max_new_tokens=64)
        request = SimpleNamespace(tool_choice='auto', tools=TOOLS, forced_tool_name=None)
        with patch('serverlib.chat_service.codec.split_generation', return_value=SimpleNamespace(answer=ANSWER, reasoning='考え')):
            events = [event async for event in service._stream_hy4_tools(prepared, request, {})]
        self.assertEqual(next(x for x in events if isinstance(x, ReasoningDelta)).text, '考え')
        self.assertEqual(json.loads(next(x for x in events if isinstance(x, ToolCallDelta)).arguments)['command'], 'cat README.md')
        self.assertEqual(next(x for x in events if isinstance(x, StreamDone)).finish_reason, 'tool_calls')

    async def test_non_stream_completion(self):
        service = ChatService.__new__(ChatService)
        service._processor = SimpleNamespace(_shelf_hy4=True)
        prepared = PreparedChat(prompt_ids=[1], tools=TOOLS, max_new_tokens=64)
        service.prepare = lambda request: prepared
        async def stream(*args):
            yield {'done': {'generated_ids': [1], 'finish_reason': 'stop'}}
        service._stream_open = stream
        request = ChatRequest(model='hy4', messages=[], tools=TOOLS, tool_choice='auto', stream=False,
                              max_new_tokens=64, temperature=0, top_p=1, min_p=0, top_k=0, seed=0)
        with patch('serverlib.chat_service.codec.split_generation', return_value=SimpleNamespace(answer=ANSWER, reasoning='考え')):
            result = await service.complete(request)
        wire = encode_chat_completion(request, result)['choices'][0]
        self.assertEqual(wire['finish_reason'], 'tool_calls')
        self.assertEqual(wire['message']['reasoning_content'], '考え')
        self.assertEqual(json.loads(wire['message']['tool_calls'][0]['function']['arguments'])['timeout'], 30)


if __name__ == '__main__':
    unittest.main()
