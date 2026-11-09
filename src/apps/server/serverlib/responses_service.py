from __future__ import annotations

import json
import uuid

from aiohttp import web

from .chat_service import ContentDelta, ReasoningDelta, StreamDone, ToolCallDelta
from .responses_protocol import response_object, usage


class ResponseStream:
    def __init__(self, chat):
        self.response = response_object(chat)
        self.sequence = 0
        self.current = None
        self.key = None

    def event(self, kind, **values):
        payload = {"type": kind, "sequence_number": self.sequence, **values}
        self.sequence += 1
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"event: {kind}\ndata: {data}\n\n".encode()

    def start(self):
        return [self.event("response.created", response=self.response.copy()),
                self.event("response.in_progress", response=self.response.copy())]

    def close_item(self):
        if self.current is None:
            return []
        item = self.current
        index = len(self.response["output"])
        base = {"item_id": item["id"], "output_index": index}
        events = []
        if item["type"] == "message":
            part = item["content"][0]
            events.append(self.event("response.output_text.done", **base, content_index=0, text=part["text"], logprobs=[]))
            events.append(self.event("response.content_part.done", **base, content_index=0, part=part.copy()))
        elif item["type"] == "reasoning":
            events.append(self.event("response.reasoning_text.done", **base, content_index=0, text=item["content"][0]["text"]))
        else:
            events.append(self.event("response.function_call_arguments.done", **base, arguments=item["arguments"]))
        if "status" in item:
            item["status"] = "completed"
        self.response["output"].append(item)
        events.append(self.event("response.output_item.done", output_index=index, item=item))
        self.current = None
        self.key = None
        return events

    def feed(self, delta):
        if isinstance(delta, ReasoningDelta):
            kind, key = "reasoning", "reasoning"
        elif isinstance(delta, ContentDelta):
            kind, key = "message", "message"
        elif isinstance(delta, ToolCallDelta):
            kind, key = "function_call", ("function_call", delta.index)
        else:
            return []
        events = []
        if self.key != key:
            events.extend(self.close_item())
            index = len(self.response["output"])
            if kind == "reasoning":
                item = {"id": "rs_" + uuid.uuid4().hex, "type": kind, "summary": [], "content": []}
            elif kind == "message":
                item = {"id": "msg_" + uuid.uuid4().hex, "type": kind, "status": "in_progress", "role": "assistant", "content": []}
            else:
                item = {"id": "fc_" + uuid.uuid4().hex, "type": kind, "status": "in_progress", "call_id": delta.call_id, "name": delta.name or "", "arguments": ""}
            self.current, self.key = item, key
            events.append(self.event("response.output_item.added", output_index=index, item=item.copy()))
            if kind == "message":
                part = {"type": "output_text", "text": "", "annotations": []}
                item["content"].append(part)
                events.append(self.event("response.content_part.added", item_id=item["id"], output_index=index, content_index=0, part=part.copy()))
            elif kind == "reasoning":
                item["content"].append({"type": "reasoning_text", "text": ""})
        item = self.current
        base = {"item_id": item["id"], "output_index": len(self.response["output"])}
        if kind == "function_call":
            if delta.name:
                item["name"] = delta.name
            item["arguments"] += delta.arguments
            if delta.arguments:
                events.append(self.event("response.function_call_arguments.delta", **base, delta=delta.arguments))
        else:
            item["content"][0]["text"] += delta.text
            if delta.text:
                extra = {"logprobs": []} if kind == "message" else {}
                events.append(self.event("response.output_text.delta" if kind == "message" else "response.reasoning_text.delta", **base, content_index=0, delta=delta.text, **extra))
        return events

    def finish(self, done):
        events = self.close_item()
        self.response["status"] = "incomplete" if done.finish_reason == "length" else "completed"
        if self.response["status"] == "incomplete":
            self.response["incomplete_details"] = {"reason": "max_output_tokens"}
        self.response["usage"] = usage(done.prompt_tokens, done.completion_tokens, done.cached_tokens)
        events.append(self.event("response." + self.response["status"], response=self.response))
        return events

    def fail(self, exc):
        self.response["status"] = "failed"
        self.response["error"] = {"code": getattr(exc, "code", "server_error"), "message": str(exc)}
        return self.event("response.failed", response=self.response)


async def stream_response(request, service, chat, prepared):
    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"})
    await response.prepare(request)
    state = ResponseStream(chat)
    events = service.stream(chat, prepared)
    try:
        for chunk in state.start():
            await response.write(chunk)
        async for event in events:
            chunks = state.finish(event) if isinstance(event, StreamDone) else state.feed(event)
            for chunk in chunks:
                await response.write(chunk)
        await response.write_eof()
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            await response.write(state.fail(exc))
            await response.write_eof()
        except (ConnectionResetError, BrokenPipeError):
            pass
    finally:
        await events.aclose()
    return response
