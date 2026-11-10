"""OpenAI ChatRequest to Shelf compute and back.

ChatService owns the transformation chain and nothing else:

    OpenAI ChatRequest
          -> HF messages
          -> HF chat template
          -> token IDs
          -> AsyncComputeClient
          -> generated token IDs
          -> GenerationRegionParser (reasoning / answer)
          -> Qwen response parser (answer only)
          -> internal assistant response

HTTP concerns live in ``shelf_server.py``; wire encoding lives in
``openai_protocol.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from shelf_chat import codec, reasoning

from .compute_client import ComputeCancelled, ComputeError, ComputeInvalidArgument
from .openai_protocol import (
    ChatRequest,
    ChatServiceError,
    OpenAIProtocolError,
    openai_messages_to_hf,
    thinking_budget_tokens,
)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
}


@dataclass(frozen=True)
class PreparedChat:
    prompt_ids: list[int]
    tools: Optional[list[dict]]
    max_new_tokens: int
    enable_thinking: bool = True
    reasoning_budget_tokens: int = 0
    reasoning_close_token: int = -1
    tool_prefix: str = ""


@dataclass(frozen=True)
class AssistantResult:
    content: str
    tool_calls: list[dict]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: Optional[int] = None
    reasoning_content: str = ""
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id: str
    name: Optional[str] = None
    arguments: str = ""


@dataclass(frozen=True)
class StreamDone:
    finish_reason: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


def _cached_tokens(done: dict) -> Optional[int]:
    value = done.get("restored_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def map_finish_reason(raw: str, *, has_tool_calls: bool) -> str:
    mapped = _FINISH_REASON_MAP.get(raw)
    if mapped is None:
        raise ChatServiceError(
            f"unsupported compute finish reason: {raw!r}",
            code="compute_error",
        )
    return "tool_calls" if mapped == "stop" and has_tool_calls else mapped


def _complete_tool_text(text: str) -> tuple[str, bool]:
    start = text.rfind("<tool_call>")
    if start > text.rfind("</tool_call>"):
        return text[:start], False
    marker = "<tool_call>"
    for size in range(len(marker) - 1, 4, -1):
        if text.endswith(marker[:size]):
            return text[:-size], False
    return text, True


def log_tokens(generated_ids) -> None:
    path = os.environ.get("SHELF_SERVER_TOKEN_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(",".join(str(int(t)) for t in generated_ids) + "\n")
    except OSError:
        pass


class ChatService:
    def __init__(self, processor, compute, model_name: str, max_seq_len: int,
                 inference_budget_tokens: int = 32768,
                 large_context_tokens: int = 0,
                 large_context_inference_budget_tokens: int = 0,
                 reasoning_close_token: int = -1):
        self._processor = processor
        self._compute = compute
        self._model_name = model_name
        self._max_seq_len = max_seq_len
        self._inference_budget_tokens = inference_budget_tokens
        self._large_context_tokens = large_context_tokens
        self._large_context_inference_budget_tokens = (
            large_context_inference_budget_tokens)
        self._reasoning_close_token = reasoning_close_token

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def inference_budget_tokens(self) -> int:
        return self._inference_budget_tokens

    @property
    def large_context_tokens(self) -> int:
        return self._large_context_tokens

    @property
    def large_context_inference_budget_tokens(self) -> int:
        return self._large_context_inference_budget_tokens

    @property
    def inference_budget_ceiling(self) -> int:
        return max(self._inference_budget_tokens,
                   self._large_context_inference_budget_tokens)

    @property
    def reasoning_close_token(self) -> int:
        return self._reasoning_close_token

    def inference_budget_for(self, prompt_tokens: int) -> int:
        if (self._large_context_inference_budget_tokens > 0 and
                self._large_context_tokens > 0 and
                prompt_tokens > self._large_context_tokens):
            return self._large_context_inference_budget_tokens
        return self._inference_budget_tokens

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def compute(self):
        return self._compute

    # ------------------------------------------------------------------ prepare

    def prepare(self, request: ChatRequest) -> PreparedChat:
        messages = openai_messages_to_hf(request.messages)

        tools = request.tools if request.tool_choice != "none" else None

        try:
            prompt_ids = codec.prompt_ids(
                self._processor,
                messages,
                tools=tools,
                enable_thinking=request.enable_thinking,
                reasoning_effort=request.reasoning_effort,
                preserve_thinking=request.preserve_thinking,
            )
        except OpenAIProtocolError:
            raise
        except Exception as exc:
            raise OpenAIProtocolError(
                f"chat template failed: {exc}",
                param="messages",
            ) from exc

        tool_prefix = ""
        if request.tool_choice in ("required", "named"):
            tool_prefix = "<tool_call>\n<function="
            if request.forced_tool_name is not None:
                tool_prefix += request.forced_tool_name + ">\n"
            prefix = ("</think>\n\n" if request.enable_thinking else "") + tool_prefix
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            prompt_ids += list(tokenizer.encode(prefix, add_special_tokens=False))

        if len(prompt_ids) >= self._max_seq_len:
            raise OpenAIProtocolError(
                "context length exceeded",
                param="messages",
            )

        remaining = self._max_seq_len - len(prompt_ids)

        budget = self.inference_budget_for(len(prompt_ids))
        max_new_tokens = min(request.max_new_tokens, budget, remaining)

        reasoning_budget_tokens = 0
        if request.enable_thinking and not tool_prefix and self._reasoning_close_token >= 0:
            reasoning_budget_tokens = thinking_budget_tokens(max_new_tokens)

        return PreparedChat(
            prompt_ids=prompt_ids,
            tools=tools,
            max_new_tokens=max_new_tokens,
            enable_thinking=request.enable_thinking and not tool_prefix,
            tool_prefix=tool_prefix,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_close_token=self._reasoning_close_token,
        )

    # ----------------------------------------------------------------- non-stream

    async def complete(self, request: ChatRequest) -> AssistantResult:
        prepared = self.prepare(request)
        if request.stop or prepared.tool_prefix:
            content, reasoning_parts, calls = [], [], []
            done = None
            async for event in self.stream(request, prepared):
                if isinstance(event, ContentDelta):
                    content.append(event.text)
                elif isinstance(event, ReasoningDelta):
                    reasoning_parts.append(event.text)
                elif isinstance(event, ToolCallDelta):
                    calls.append({"type": "function", "function": {
                        "name": event.name, "arguments": event.arguments}})
                elif isinstance(event, StreamDone):
                    done = event
            if done is None:
                raise ChatServiceError("compute ended without a done event", code="compute_error")
            return AssistantResult(
                content="".join(content), tool_calls=calls,
                finish_reason=done.finish_reason, prompt_tokens=done.prompt_tokens,
                completion_tokens=done.completion_tokens, cached_tokens=done.cached_tokens,
                reasoning_content="".join(reasoning_parts))

        done = await self._generate(prepared, request)

        generated_ids = [int(x) for x in done.get("generated_ids", [])]
        log_tokens(generated_ids)

        split = codec.split_generation(
            self._processor, generated_ids,
            thinking=prepared.enable_thinking)

        answer = prepared.tool_prefix + split.answer
        if prepared.tools:
            answer, complete = _complete_tool_text(answer)
            if not complete and done.get("finish_reason") != "length":
                raise ChatServiceError("incomplete tool call", code="tool_parse_failure")
        try:
            parsed = codec.parse_answer(
                self._processor,
                answer,
                tools=prepared.tools,
            )
        except codec.ToolParseError as exc:
            raise ChatServiceError(
                f"tool call parse failure: {exc}",
                code="tool_parse_failure",
            ) from exc
        except codec.ToolCallingUnavailable as exc:
            raise ChatServiceError(
                str(exc),
                code="tool_calling_unavailable",
                status=400,
            ) from exc

        tool_calls = parsed["tool_calls"]
        self._validate_forced_calls(request, tool_calls, str(done.get("finish_reason", "")))

        return AssistantResult(
            content=parsed["content"],
            tool_calls=tool_calls,
            finish_reason=map_finish_reason(
                str(done.get("finish_reason", "")), has_tool_calls=bool(tool_calls)),
            prompt_tokens=len(prepared.prompt_ids),
            completion_tokens=len(generated_ids),
            cached_tokens=_cached_tokens(done),
            reasoning_content=split.reasoning,
            reasoning_tokens=split.reasoning_tokens,
        )

    async def _generate(self, prepared: PreparedChat, request: ChatRequest) -> dict:
        try:
            return await self._compute.generate(
                prepared.prompt_ids,
                prepared.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                min_p=request.min_p,
                top_k=request.top_k,
                seed=request.seed,
                prefix_cache_checkpoint_position=0,
                reasoning_budget_tokens=prepared.reasoning_budget_tokens,
                reasoning_close_token=prepared.reasoning_close_token,
            )
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc), param=None) from exc
        except ComputeCancelled as exc:
            raise ChatServiceError(str(exc), code="cancelled") from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc

    # ------------------------------------------------------------------- streaming

    async def stream(self, request: ChatRequest,
                     prepared: Optional[PreparedChat] = None) -> AsyncIterator[Any]:
        if prepared is None:
            prepared = self.prepare(request)
        inner = self._stream_impl(prepared, request)
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()

    async def _stream_impl(self, prepared: PreparedChat,
                           request: ChatRequest) -> AsyncIterator[Any]:
        tracker = {"tokens": 0}
        inner = (self._stream_forced_tools(prepared, request, tracker) if prepared.tool_prefix
                 else self._stream_tools(prepared, request, tracker) if prepared.tools
                 else self._stream_text(prepared, request, tracker))
        pending = ""
        try:
            async for event in inner:
                if isinstance(event, ContentDelta) and request.stop:
                    pending += event.text
                    matches = [pending.find(stop) for stop in request.stop]
                    matches = [position for position in matches if position >= 0]
                    if matches:
                        prefix = pending[:min(matches)]
                        await inner.aclose()
                        if prefix:
                            yield ContentDelta(prefix)
                        yield StreamDone("stop", prompt_tokens=len(prepared.prompt_ids),
                                         completion_tokens=tracker["tokens"])
                        return
                    hold = max((size for stop in request.stop
                                for size in range(1, min(len(stop) - 1, len(pending)) + 1)
                                if pending.endswith(stop[:size])), default=0)
                    ready = pending[:-hold] if hold else pending
                    pending = pending[-hold:] if hold else ""
                    if ready:
                        yield ContentDelta(ready)
                else:
                    if pending and isinstance(event, (ToolCallDelta, StreamDone)):
                        yield ContentDelta(pending)
                        pending = ""
                    yield event
        finally:
            await inner.aclose()

    def _stream_open(self, prepared: PreparedChat, request: ChatRequest):
        return self._compute.generate_stream(
            prepared.prompt_ids,
            prepared.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
            seed=request.seed,
            prefix_cache_checkpoint_position=0,
            reasoning_budget_tokens=prepared.reasoning_budget_tokens,
            reasoning_close_token=prepared.reasoning_close_token,
        )

    async def _stream_text(self, prepared: PreparedChat,
                           request: ChatRequest, tracker: dict) -> AsyncIterator[Any]:
        regions = codec.new_region_parser(
            self._processor, thinking=prepared.enable_thinking)
        generated: list[int] = []

        stream = self._stream_open(prepared, request)
        try:
            async for event in stream:
                if "token_id" in event:
                    generated.append(int(event["token_id"]))
                    tracker["tokens"] = len(generated)
                    for delta in regions.feed(generated[-1:]):
                        yield _region_event(delta)
                elif "done" in event:
                    done = event["done"]
                    final_ids = [int(x) for x in done.get("generated_ids", generated)]
                    log_tokens(final_ids)
                    tracker["tokens"] = len(final_ids)
                    for delta in self._region_tail(regions, generated, final_ids):
                        yield _region_event(delta)
                    yield StreamDone(map_finish_reason(
                        str(done.get("finish_reason", "")), has_tool_calls=False),
                        prompt_tokens=len(prepared.prompt_ids),
                        completion_tokens=len(final_ids),
                        cached_tokens=_cached_tokens(done))
                    return
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc)) from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc
        finally:
            await stream.aclose()

    async def _stream_forced_tools(self, prepared: PreparedChat,
                                   request: ChatRequest, tracker: dict) -> AsyncIterator[Any]:
        generated = []
        done = None
        stream = self._stream_open(prepared, request)
        try:
            async for event in stream:
                if "token_id" in event:
                    generated.append(int(event["token_id"]))
                    tracker["tokens"] = len(generated)
                elif "done" in event:
                    done = event["done"]
                    break
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc)) from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc
        finally:
            await stream.aclose()
        if done is None:
            raise ChatServiceError("compute ended without a done event", code="compute_error")
        final_ids = [int(x) for x in done.get("generated_ids", generated)]
        tracker["tokens"] = len(final_ids)
        log_tokens(final_ids)
        answer = prepared.tool_prefix + codec.split_generation(
            self._processor, final_ids, thinking=False).answer
        close = answer.rfind("</tool_call>")
        calls = []
        if close >= 0:
            complete = answer[:close + len("</tool_call>")]
            if "</function>" not in complete:
                raise ChatServiceError("incomplete tool call", code="tool_parse_failure")
            try:
                calls = codec.parse_answer(self._processor, complete, tools=prepared.tools)["tool_calls"]
            except codec.ToolParseError as exc:
                raise ChatServiceError(f"tool call parse failure: {exc}", code="tool_parse_failure") from exc
        finish = str(done.get("finish_reason", ""))
        self._validate_forced_calls(request, calls, finish)
        for index, call in enumerate(calls):
            delta = _tool_call_delta(call, index)
            if delta is not None:
                yield delta
        yield StreamDone(map_finish_reason(finish, has_tool_calls=bool(calls)),
            prompt_tokens=len(prepared.prompt_ids), completion_tokens=len(final_ids),
            cached_tokens=_cached_tokens(done))

    async def _stream_tools(self, prepared: PreparedChat,
                            request: ChatRequest, tracker: dict) -> AsyncIterator[Any]:
        regions = codec.new_region_parser(
            self._processor, thinking=prepared.enable_thinking)
        parser = codec.new_response_parser(self._processor)
        state: dict[str, Any] = {"content": "", "calls": 0, "answer": ""}
        generated: list[int] = []
        done: dict | None = None

        stream = self._stream_open(prepared, request)
        try:
            async for event in stream:
                if "token_id" in event:
                    generated.append(int(event["token_id"]))
                    tracker["tokens"] = len(generated)
                    for delta in regions.feed(generated[-1:]):
                        for item in self._consume_region(delta, parser, state):
                            yield item
                elif "done" in event:
                    done = event["done"]
                    final_ids = [int(x) for x in done.get("generated_ids", generated)]
                    log_tokens(final_ids)
                    tracker["tokens"] = len(final_ids)
                    for delta in self._region_tail(regions, generated, final_ids):
                        for item in self._consume_region(delta, parser, state):
                            yield item

                    try:
                        message, final_events = parser.finalize()
                    except Exception as exc:
                        raise ChatServiceError(
                            f"streaming tool parse failed: {exc}",
                            code="tool_parse_failure",
                        ) from exc

                    _, complete = _complete_tool_text(state["answer"])
                    if not complete:
                        if done.get("finish_reason") != "length":
                            raise ChatServiceError("incomplete tool call", code="tool_parse_failure")
                        final_events = []
                        message = {"content": state["content"], "tool_calls": []}
                    if final_events:
                        for item in self._feed_parser(parser, "", state,
                                                      events=final_events):
                            yield item
                    for item in self._finalize_tool_calls(message, state):
                        yield item
                    break
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc)) from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc
        finally:
            await stream.aclose()

        if done is None:
            raise ChatServiceError("compute ended without a done event",
                                   code="compute_error")
        final_ids = [int(x) for x in done.get("generated_ids", generated)]
        yield StreamDone(map_finish_reason(
            str(done.get("finish_reason", "")), has_tool_calls=state["calls"] > 0),
            prompt_tokens=len(prepared.prompt_ids),
            completion_tokens=len(final_ids),
            cached_tokens=_cached_tokens(done))

    def _region_tail(self, regions, generated: list[int],
                     final_ids: list[int]) -> list[Any]:
        """Flush the incremental parser against the authoritative done tokens."""
        if final_ids[:len(generated)] == generated:
            deltas = regions.feed(final_ids[len(generated):])
            return list(deltas) + list(regions.finalize())

        split = codec.split_generation(
            self._processor, final_ids, thinking=regions.thinking)
        deltas = []
        for region, text in ((codec.REASONING, split.reasoning),
                             (codec.ANSWER, split.answer)):
            emitted = regions.emitted(region)
            if not text.startswith(emitted):
                raise ChatServiceError(
                    "compute token stream diverged from done.generated_ids",
                    code="compute_error")
            tail = text[len(emitted):]
            regions.resync(region, text)
            if tail:
                deltas.append(reasoning.RegionDelta(region, tail))
        regions.finalize()
        return deltas

    def _consume_region(self, delta, parser, state: dict):
        if delta.region == codec.REASONING:
            yield ReasoningDelta(delta.text)
            return
        state["answer"] += delta.text
        for item in self._feed_parser(parser, delta.text, state):
            yield item

    def _feed_parser(self, parser, text: str, state: dict, events=None):
        if events is None:
            events = parser.feed(text) if text else []
        for event in events:
            kind = event.get("type")
            if kind == "region_chunk" and event.get("field") == "content" \
                    and not event.get("dirty"):
                chunk = event.get("text", "")
                if not chunk:
                    continue
                delta = self._content_delta(state, chunk)
                if delta:
                    yield ContentDelta(delta)
            elif kind == "region_close" and event.get("field") == "tool_calls":
                call = _tool_call_delta(event.get("value"), state["calls"])
                if call is not None:
                    state["calls"] += 1
                    yield call

    def _validate_forced_name(self, request, name):
        if request is not None and request.tool_choice in ("required", "named"):
            allowed = {tool["function"]["name"] for tool in request.tools}
            if name not in allowed or (request.forced_tool_name is not None and name != request.forced_tool_name):
                raise ChatServiceError(f"unknown function in required tool call: {name}", code="tool_parse_failure")

    def _validate_forced_calls(self, request, calls, finish):
        if request.tool_choice in ("required", "named"):
            if not calls and finish != "length":
                raise ChatServiceError("required tool call was not generated", code="tool_parse_failure")
            for call in calls:
                self._validate_forced_name(request, call["function"]["name"])

    def _content_delta(self, state: dict, chunk: str) -> str:
        state["content"] += chunk
        return chunk

    def _finalize_tool_calls(self, message, state: dict):
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls[state["calls"]:]:
            delta = _tool_call_delta(call, state["calls"])
            if delta is not None:
                state["calls"] += 1
                yield delta
        content = message.get("content") or ""
        if content.startswith(state["content"]):
            tail = content[len(state["content"]):]
            if tail:
                state["content"] += tail
                yield ContentDelta(tail)


def _region_event(delta) -> Any:
    if delta.region == codec.REASONING:
        return ReasoningDelta(delta.text)
    return ContentDelta(delta.text)


def _tool_call_delta(value, index: int) -> Optional[ToolCallDelta]:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        payload = arguments
    else:
        payload = json.dumps(
            {} if arguments is None else arguments,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return ToolCallDelta(
        index=index,
        call_id=f"call_{index}",
        name=str(name),
        arguments=payload,
    )
