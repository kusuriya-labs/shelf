#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src/apps/server"))
sys.path.insert(0, str(root / "src/apps/common"))

from aiohttp.test_utils import TestClient, TestServer
from shelf_server import create_app
from serverlib.chat_service import AssistantResult, ContentDelta, ReasoningDelta, StreamDone, ToolCallDelta
from serverlib.openai_protocol import openai_messages_to_hf


class FakeService:
    model_name = "shelf"
    inference_budget_ceiling = 128
    max_seq_len = 1024

    def __init__(self):
        self.closed = asyncio.Event()
        self.slow = False
        self.tool = False
        self.length = False
        self.request = None

    def prepare(self, chat):
        self.request = chat
        openai_messages_to_hf(chat.messages)
        return None

    async def complete(self, chat):
        self.prepare(chat)
        calls = [{"function": {"name": "weather", "arguments": {"city": "Tokyo"}}}] if self.tool else []
        return AssistantResult("" if calls else "hello", calls, "length" if self.length else "stop", 10, 4, 3, "thinking", 2)

    async def stream(self, chat, prepared):
        try:
            yield ReasoningDelta("think")
            if self.slow:
                await asyncio.sleep(0.1)
            if self.tool:
                yield ToolCallDelta(0, "call_0", "weather", '{"city":')
                yield ToolCallDelta(0, "call_0", None, '"Tokyo"}')
            else:
                yield ContentDelta("hel")
                yield ContentDelta("lo")
            yield StreamDone("length" if self.length else "stop", 10, 4, 3)
        finally:
            self.closed.set()

    async def shutdown(self):
        pass


class ResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = FakeService()
        self.client = TestClient(TestServer(create_app(self.service, self.service)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def post(self, **extra):
        return await self.client.post('/v1/responses', json={"model": "shelf", "input": "hello", **extra})

    async def stream_events(self, **extra):
        response = await self.post(stream=True, **extra)
        self.assertEqual(response.status, 200)
        chunks = (await response.text()).strip().split("\n\n")
        events = []
        for chunk in chunks:
            lines = chunk.splitlines()
            data = json.loads(lines[1][6:])
            self.assertEqual(lines[0], "event: " + data["type"])
            events.append(data)
        self.assertEqual([e['sequence_number'] for e in events], list(range(len(events))))
        return events

    async def test_basic_usage_and_incomplete(self):
        self.service.length = True
        response = await self.post(max_output_tokens=4, store=True, metadata={}, user="x")
        data = await response.json()
        self.assertEqual(response.status, 200)
        self.assertTrue(data['id'].startswith('resp_'))
        self.assertEqual(data['status'], 'incomplete')
        self.assertEqual(data['incomplete_details']['reason'], 'max_output_tokens')
        self.assertEqual(data['usage']['input_tokens_details']['cached_tokens'], 3)
        self.assertEqual(data['usage']['output_tokens_details']['reasoning_tokens'], 2)
        self.assertEqual([x['type'] for x in data['output']], ['reasoning', 'message'])

    async def test_input_history_and_tools(self):
        self.service.tool = True
        data = await (await self.post(input=[
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ], instructions="system", tools=[{"type": "function", "name": "weather", "parameters": {"type": "object"}}], reasoning={"effort": "low"})).json()
        self.assertEqual(data['output'][1]['name'], 'weather')
        self.assertEqual(data['output'][1]['call_id'], 'call_0')
        self.assertEqual(self.service.request.messages[0]['content'], 'system')
        self.assertEqual(self.service.request.messages[2]['reasoning_content'], 'think')
        self.assertEqual(self.service.request.messages[3]['tool_call_id'], 'c1')
        self.assertEqual(self.service.request.reasoning_effort, 'low')

    async def test_rejections(self):
        cases = [({"previous_response_id": "resp_1"}, "previous_response_id"),
                 ({"conversation": "c"}, "conversation"),
                 ({"text": {"format": {"type": "json_schema"}}}, "text.format"),
                 ({"tools": [{"type": "function", "name": "x", "strict": True}]}, "tools[0].strict"),
                 ({"tools": [{"type": "web_search"}]}, "tools[0].type"),
                 ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}, "input[0].content[0].image_url"),
                 ({"tool_choice": "required"}, "tool_choice"),
                 ({"unknown": 1}, "unknown"), ({"parallel_tool_calls": False}, "parallel_tool_calls"),
                 ({"truncation": "auto"}, "truncation"), ({"include": ["x"]}, "include"),
                 ({"max_output_tokens": "x"}, "max_output_tokens")]
        for extra, param in cases:
            with self.subTest(extra=extra):
                response = await self.post(**extra)
                self.assertEqual(response.status, 400)
                error = (await response.json())['error']
                self.assertEqual(error['param'], param)
                self.assertEqual(error['type'], 'invalid_request_error')

    async def test_text_stream(self):
        events = await self.stream_events()
        names = [e['type'] for e in events]
        self.assertEqual(names, ['response.created', 'response.in_progress', 'response.output_item.added', 'response.reasoning_text.delta', 'response.reasoning_text.done', 'response.output_item.done', 'response.output_item.added', 'response.content_part.added', 'response.output_text.delta', 'response.output_text.delta', 'response.output_text.done', 'response.content_part.done', 'response.output_item.done', 'response.completed'])
        self.assertEqual(events[-1]['response']['output'][1]['content'][0]['text'], 'hello')
        self.assertEqual(events[-1]['response']['usage']['input_tokens_details']['cached_tokens'], 3)

    async def test_tool_stream(self):
        self.service.tool = True
        events = await self.stream_events()
        names = [e['type'] for e in events]
        self.assertEqual(names.count('response.function_call_arguments.delta'), 2)
        self.assertLess(names.index('response.function_call_arguments.done'), len(names) - 1)
        call = events[-1]['response']['output'][1]
        self.assertEqual(json.loads(call['arguments']), {'city': 'Tokyo'})
        self.assertEqual(call['call_id'], 'call_0')

    async def test_stream_incomplete(self):
        self.service.length = True
        events = await self.stream_events()
        self.assertEqual(events[-1]['type'], 'response.incomplete')

    async def test_disconnect_closes_generator(self):
        self.service.slow = True
        response = await self.post(stream=True)
        await response.content.readline()
        response.close()
        await asyncio.wait_for(self.service.closed.wait(), 2)


if __name__ == '__main__':
    unittest.main()
