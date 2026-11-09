#!/usr/bin/env python3
"""stop の検証と fake compute による生成停止。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/apps/server"), str(ROOT / "src/apps/common"), str(Path(__file__).resolve().parent)]

from support import Checker
from serverlib.chat_service import ChatService, ContentDelta, ReasoningDelta, StreamDone, ToolCallDelta
from serverlib.openai_protocol import OpenAIProtocolError, parse_chat_request


class Processor:
    _shelf_reasoning_close_ids = (1000,)

    def apply_chat_template(self, *args, **kwargs):
        return [5000, 5001]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(x for x in ids if x < 256).decode("utf-8", errors="replace")


class Compute:
    def __init__(self, ids):
        self.ids = ids
        self.cancelled = 0
        self.completed = 0

    async def generate_stream(self, *args, **kwargs):
        finished = False
        try:
            for token in self.ids:
                await asyncio.sleep(0)
                yield {"token_id": token}
            finished = True
            self.completed += 1
            yield {"done": {"generated_ids": self.ids, "finish_reason": "length", "restored_tokens": 2}}
        finally:
            if not finished:
                self.cancelled += 1


def request(**kwargs):
    return parse_chat_request({"model": "shelf", "messages": [{"role": "user", "content": "hi"}], **kwargs})


async def run(checker):
    for value in ("", [], [""], ["a"] * 5, [1], 1, False, ["a", None]):
        try:
            request(stop=value)
        except OpenAIProtocolError as exc:
            checker.check(f"invalid-{value!r}", exc.param == "stop")
        else:
            checker.check(f"invalid-{value!r}", False)
    checker.check("valid-null", request(stop=None).stop == ())
    checker.check("valid-array", request(stop=["ab", "日本"]).stop == ("ab", "日本"))

    async def sample(text, stops, thinking=False):
        compute = Compute(text)
        service = ChatService(Processor(), compute, "shelf", 1000)
        req = request(stop=stops, chat_template_kwargs={"enable_thinking": thinking})
        events = [event async for event in service.stream(req)]
        return compute, events

    data = list("before STOP after".encode())
    compute, events = await sample(data, ["STOP", "other"])
    checker.check("stream-content", "".join(e.text for e in events if isinstance(e, ContentDelta)) == "before ", repr(events))
    checker.check("stream-cancel-usage", compute.cancelled == 1 and isinstance(events[-1], StreamDone) and events[-1].finish_reason == "stop" and events[-1].completion_tokens == len("before STOP"), repr(events))
    compute = Compute(data)
    service = ChatService(Processor(), compute, "shelf", 1000)
    result = await service.complete(request(stop="STOP", chat_template_kwargs={"enable_thinking": False}))
    checker.check("nonstream-parity", result.content == "before " and result.finish_reason == "stop" and result.completion_tokens == len("before STOP") and compute.cancelled == 1, repr(result))
    compute, events = await sample(list("abc ST".encode()), "STOP")
    checker.check("partial-flush", "".join(e.text for e in events if isinstance(e, ContentDelta)) == "abc ST" and events[-1].finish_reason == "length" and compute.completed == 1, repr(events))
    compute, events = await sample(list("日本語終端後".encode()), ["終端"])
    checker.check("unicode", "".join(e.text for e in events if isinstance(e, ContentDelta)) == "日本語" and compute.cancelled == 1, repr(events))
    compute, events = await sample(list("STOP in reasoning".encode()) + [1000] + list("answer STOP tail".encode()), "STOP", True)
    checker.check("reasoning-excluded", "".join(e.text for e in events if isinstance(e, ReasoningDelta)) == "STOP in reasoning" and "".join(e.text for e in events if isinstance(e, ContentDelta)) == "answer " and compute.cancelled == 1, repr(events))

    class FakeParser:
        def __init__(self):
            self.buffer = ""
            self.closed = False
        def feed(self, text):
            self.buffer += text
            if self.buffer == "<tool_call>STOP</tool_call>":
                self.closed = True
                return [{"type": "region_close", "field": "tool_calls", "value": {"function": {"name": "f", "arguments": {"value": "STOP"}}}}]
            return []
        def finalize(self):
            return ({"tool_calls": [{"function": {"name": "f", "arguments": {"value": "STOP"}}}]}, [])
    tool = {"type": "function", "function": {"name": "f"}}
    compute = Compute(list("<tool_call>STOP</tool_call>".encode()))
    service = ChatService(Processor(), compute, "shelf", 1000)
    with patch("serverlib.chat_service.codec.new_response_parser", return_value=FakeParser()):
        events = [e async for e in service.stream(request(stop="STOP", tools=[tool], chat_template_kwargs={"enable_thinking": False}))]
    checker.check("tool-markup-excluded", any(isinstance(e, ToolCallDelta) and "STOP" in e.arguments for e in events) and events[-1].finish_reason == "tool_calls" and compute.cancelled == 0, repr(events))

    shared = Compute(list("alpha STOP trailing".encode()))
    service = ChatService(Processor(), shared, "shelf", 1000)
    first, second = await asyncio.gather(service.complete(request(stop="STOP", chat_template_kwargs={"enable_thinking": False})), service.complete(request(stop="NEVER", chat_template_kwargs={"enable_thinking": False})))
    checker.check("concurrent-isolation", first.content == "alpha " and second.content == "alpha STOP trailing" and shared.cancelled == 1 and shared.completed == 1, repr((first, second)))


def main():
    checker = Checker("server-stop")
    asyncio.run(run(checker))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
