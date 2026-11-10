"""OpenAI Chat Completions の入力検証と応答エンコード。

対応するパラメータは明示的に列挙し、未対応・未知のキーは拒否する。
user、metadata、service_tier と parallel_tool_calls=true は受理するが使用しない。
store=true、構造化出力、strict tool、文法制約には対応しない。
required と関数名指定は生成 prompt に tool call の開始部分を補う。
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from shelf_chat.codec import DEFAULT_REASONING_EFFORT, REASONING_EFFORTS

CHAT_TEMPLATE_KWARGS = ("enable_thinking", "preserve_thinking")

DEFAULT_MAX_NEW_TOKENS = 4096

DONE = b"data: [DONE]\n\n"


class OpenAIProtocolError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str = "invalid_request",
        status: int = 400,
    ):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code
        self.status = status


class ChatServiceError(RuntimeError):
    def __init__(self, message: str, *, code: str = "internal_error", status: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tool_choice: str
    stream: bool

    max_new_tokens: int
    temperature: float
    top_p: float
    min_p: float
    top_k: int
    seed: int
    stop: tuple[str, ...] = ()
    include_usage: bool = False
    enable_thinking: bool = True
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    preserve_thinking: bool = True
    forced_tool_name: str | None = None


def error_body(message: str, *, param: str | None = None,
               code: str = "invalid_request",
               error_type: str = "invalid_request_error") -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def exception_body(exc) -> dict:
    return error_body(exc.message, param=exc.param, code=exc.code)


def sse_data(payload: dict) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {text}\n\n".encode("utf-8")


def _reject(body: dict, key: str, message: str, code: str = "unsupported_parameter") -> None:
    if body.get(key) is not None:
        raise OpenAIProtocolError(message, param=key, code=code)


def reject_unsupported(body: dict) -> None:
    allowed = {
        "model", "messages", "tools", "tool_choice", "stream", "stream_options",
        "max_tokens", "max_completion_tokens", "temperature", "top_p", "min_p",
        "top_k", "seed", "chat_template_kwargs", "reasoning_effort", "n",
        "user", "store", "metadata", "service_tier", "parallel_tool_calls",
        "frequency_penalty", "presence_penalty", "stop",
    }
    for key in body:
        if key not in allowed:
            raise OpenAIProtocolError(
                f"{key} is not supported", param=key, code="unsupported_parameter")
    if body.get("store") is not None and not isinstance(body["store"], bool):
        raise OpenAIProtocolError("store must be a boolean", param="store")
    if body.get("store") is True:
        raise OpenAIProtocolError("store=true is not supported", param="store", code="unsupported_parameter")
    for key in ("frequency_penalty", "presence_penalty"):
        value = body.get(key)
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, float)) or value != 0):
            raise OpenAIProtocolError(
                f"only {key}=0 is supported", param=key, code="unsupported_parameter")
    _reject(body, "response_format", "response_format is not supported")
    _reject(body, "reasoning", "reasoning is not supported")
    _reject(body, "text", "text is not supported")
    _reject(body, "logit_bias", "logit_bias is not supported")
    _reject(body, "logprobs", "logprobs is not supported")

    if body.get("parallel_tool_calls") is not None and body["parallel_tool_calls"] is not True:
        raise OpenAIProtocolError(
            "parallel_tool_calls=false is not supported",
            param="parallel_tool_calls",
            code="unsupported_parameter",
        )

    n = body.get("n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n != 1):
        raise OpenAIProtocolError(
            "only n=1 is supported",
            param="n",
            code="unsupported_parameter",
        )


def _max_new_tokens(body: dict, inference_budget_tokens: int) -> int:
    old = _integer(body, "max_tokens", None, low=1, high=(1 << 63) - 1)
    new = _integer(body, "max_completion_tokens", None, low=1, high=(1 << 63) - 1)
    if old is not None and new is not None and old != new:
        raise OpenAIProtocolError(
            "max_tokens and max_completion_tokens disagree",
            param="max_completion_tokens",
        )
    value = new if new is not None else old
    return min(value, int(inference_budget_tokens)) if value is not None else int(inference_budget_tokens)


ANSWER_RESERVE_MIN_TOKENS = 512


def answer_reserve_tokens(budget_tokens: int) -> int:
    if budget_tokens < 1024:
        return 0
    return min(max(ANSWER_RESERVE_MIN_TOKENS, budget_tokens // 4),
               budget_tokens // 2)


def thinking_budget_tokens(budget_tokens: int) -> int:
    reserve = answer_reserve_tokens(budget_tokens)
    if reserve == 0:
        return 0
    return max(1, budget_tokens - reserve)


def parse_tool_choice(value) -> tuple[str, str | None]:
    if value is None:
        return "auto", None
    if isinstance(value, str) and value in ("auto", "none", "required"):
        return value, None
    if isinstance(value, dict) and set(value) == {"type", "function"} and value.get("type") == "function":
        function = value.get("function")
        if isinstance(function, dict) and set(function) == {"name"} and isinstance(function["name"], str) and function["name"]:
            return "named", function["name"]
    raise OpenAIProtocolError("invalid tool_choice", param="tool_choice")


def normalize_tools(raw_tools) -> list[dict] | None:
    if raw_tools is None:
        return None

    if not isinstance(raw_tools, list):
        raise OpenAIProtocolError(
            "tools must be an array",
            param="tools",
        )

    result = []

    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise OpenAIProtocolError(
                "tool must be an object",
                param=f"tools[{index}]",
            )

        if tool.get("type") != "function":
            raise OpenAIProtocolError(
                "only function tools are supported",
                param=f"tools[{index}].type",
            )

        fn = tool.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            raise OpenAIProtocolError(
                "tool function name is required",
                param=f"tools[{index}].function.name",
            )

        if fn.get("strict") is not None and fn.get("strict") is not False:
            raise OpenAIProtocolError(
                "strict tool schemas are not supported",
                param=f"tools[{index}].function.strict",
                code="unsupported_parameter",
            )

        normalized = {
            "type": "function",
            "function": dict(fn),
        }
        normalized["function"].pop("strict", None)
        result.append(normalized)

    if not result:
        raise OpenAIProtocolError(
            "tools must not be empty",
            param="tools",
        )
    return result


def _number(body: dict, key: str, default: float, *, low: float | None = None,
            high: float | None = None) -> float:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenAIProtocolError(f"{key} must be a number", param=key)
    try:
        value = float(value)
    except OverflowError:
        raise OpenAIProtocolError(f"{key} must be finite", param=key) from None
    if not math.isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise OpenAIProtocolError(f"{key} must be a finite float32", param=key)
    if low is not None and value < low:
        raise OpenAIProtocolError(f"{key} must be >= {low}", param=key)
    if high is not None and value > high:
        raise OpenAIProtocolError(f"{key} must be <= {high}", param=key)
    return value


def _integer(body: dict, key: str, default: int | None, *, low: int | None = None,
             high: int | None = None) -> int | None:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAIProtocolError(f"{key} must be an integer", param=key)
    if low is not None and value < low:
        raise OpenAIProtocolError(f"{key} must be >= {low}", param=key)
    if high is not None and value > high:
        raise OpenAIProtocolError(f"{key} must be <= {high}", param=key)
    return value


def parse_reasoning_effort(body: dict) -> tuple[str, bool]:
    """Return ``(effort, explicitly_requested)``.

    Only the three effort levels the Qwen3.8 chat template defines are accepted.
    An unsupported value is an error, never a silent fallback to the default.
    """
    value = body.get("reasoning_effort")
    if value is None:
        return DEFAULT_REASONING_EFFORT, False
    if isinstance(value, bool) or not isinstance(value, str):
        raise OpenAIProtocolError(
            "reasoning_effort must be one of low, medium, xhigh",
            param="reasoning_effort",
            code="unsupported_parameter",
        )
    if value not in REASONING_EFFORTS:
        raise OpenAIProtocolError(
            f"reasoning_effort={value!r} is not supported; "
            "expected one of low, medium, xhigh",
            param="reasoning_effort",
            code="unsupported_parameter",
        )
    return value, True


def parse_chat_template_kwargs(body: dict) -> tuple[bool, bool]:
    """Return ``(enable_thinking, preserve_thinking)``.

    ``chat_template_kwargs`` is an explicit allow-list: a key that the server
    would ignore is rejected instead of silently dropped.
    """
    raw = body.get("chat_template_kwargs")
    if raw is None:
        return True, True
    if not isinstance(raw, dict):
        raise OpenAIProtocolError(
            "chat_template_kwargs must be an object",
            param="chat_template_kwargs",
        )

    for key in raw:
        if key not in CHAT_TEMPLATE_KWARGS:
            raise OpenAIProtocolError(
                f"chat_template_kwargs.{key} is not supported",
                param=f"chat_template_kwargs.{key}",
                code="unsupported_parameter",
            )

    enable_thinking = raw.get("enable_thinking", True)
    if not isinstance(enable_thinking, bool):
        raise OpenAIProtocolError(
            "chat_template_kwargs.enable_thinking must be a boolean",
            param="chat_template_kwargs.enable_thinking",
        )

    preserve_thinking = raw.get("preserve_thinking", True)
    if not isinstance(preserve_thinking, bool):
        raise OpenAIProtocolError(
            "chat_template_kwargs.preserve_thinking must be a boolean",
            param="chat_template_kwargs.preserve_thinking",
        )

    return enable_thinking, preserve_thinking


def _stream_options_include_usage(body: dict) -> bool:
    options = body.get("stream_options")
    if options is None:
        return False

    if not isinstance(options, dict):
        raise OpenAIProtocolError(
            "stream_options must be an object", param="stream_options")

    for key in options:
        if key != "include_usage":
            raise OpenAIProtocolError(
                f"stream_options.{key} is not supported",
                param=f"stream_options.{key}",
                code="unsupported_parameter",
            )

    flag = options.get("include_usage", False)
    if not isinstance(flag, bool):
        raise OpenAIProtocolError(
            "stream_options.include_usage must be a boolean",
            param="stream_options.include_usage",
        )
    return flag


def parse_stop(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if (not isinstance(value, list) or not 1 <= len(value) <= 4 or
            any(not isinstance(item, str) or not item for item in value)):
        raise OpenAIProtocolError("stop must be a non-empty string or an array of 1 to 4 non-empty strings", param="stop")
    return tuple(value)


def parse_chat_request(body: Any,
                       *,
                       inference_budget_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> ChatRequest:
    if not isinstance(body, dict):
        raise OpenAIProtocolError("request body must be a JSON object")

    reject_unsupported(body)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise OpenAIProtocolError("model is required", param="model")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAIProtocolError("messages must be a non-empty array", param="messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise OpenAIProtocolError(
                "message must be an object", param=f"messages[{index}]")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise OpenAIProtocolError("stream must be a boolean", param="stream")

    tool_choice, forced_tool_name = parse_tool_choice(body.get("tool_choice"))
    tools = normalize_tools(body.get("tools"))
    if tool_choice in ("required", "named"):
        if not tools or (forced_tool_name is not None and forced_tool_name not in {tool["function"]["name"] for tool in tools}):
            raise OpenAIProtocolError("tool_choice requires a matching function in tools", param="tool_choice", code="unsupported_parameter")
        if forced_tool_name is not None:
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", forced_tool_name) is None:
                raise OpenAIProtocolError("invalid function name", param="tool_choice")
    max_new_tokens = _max_new_tokens(body, inference_budget_tokens)

    temperature = _number(body, "temperature", 0.0, low=0.0)
    top_p = _number(body, "top_p", 1.0, low=0.0, high=1.0)
    if top_p == 0.0:
        raise OpenAIProtocolError("top_p must be > 0", param="top_p")
    min_p = _number(body, "min_p", 0.0, low=0.0, high=1.0)
    top_k = _integer(body, "top_k", 0, low=0, high=(1 << 32) - 1)
    seed = _integer(body, "seed", 0, low=-(1 << 63), high=(1 << 63) - 1)
    include_usage = _stream_options_include_usage(body)

    enable_thinking, preserve_thinking = parse_chat_template_kwargs(body)
    reasoning_effort, effort_given = parse_reasoning_effort(body)
    if effort_given and not enable_thinking:
        raise OpenAIProtocolError(
            "reasoning_effort cannot be combined with "
            "chat_template_kwargs.enable_thinking=false",
            param="reasoning_effort",
        )

    return ChatRequest(
        model=model,
        messages=list(messages),
        tools=tools,
        tool_choice=tool_choice,
        forced_tool_name=forced_tool_name,
        stream=stream,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        seed=seed,
        stop=parse_stop(body.get("stop")),
        include_usage=include_usage,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
    )


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                raise OpenAIProtocolError(
                    "only text message content is supported",
                    param="messages",
                )
        return "".join(parts)
    raise OpenAIProtocolError(
        "message content must be a string or an array of text parts",
        param="messages",
    )


def _assistant_reasoning(message: dict) -> str | None:
    """Normalize an assistant history reasoning field.

    ``reasoning_content`` is the Shelf wire name. The ``reasoning`` alias
    some clients echo is accepted when both are absent or agree; a conflicting
    pair is rejected instead of silently preferring one.
    """
    primary = message.get("reasoning_content")
    alias = message.get("reasoning")

    for key, value in (("reasoning_content", primary), ("reasoning", alias)):
        if value is not None and not isinstance(value, str):
            raise OpenAIProtocolError(
                f"assistant {key} must be a string",
                param="messages",
            )

    if primary is not None and alias is not None and primary != alias:
        raise OpenAIProtocolError(
            "assistant reasoning_content and reasoning disagree",
            param="messages",
        )

    return primary if primary is not None else alias


def openai_messages_to_hf(messages: list[dict]) -> list[dict]:
    result = []
    system_parts = []

    tool_names: dict[str, str] = {}

    for message in messages:
        role = message.get("role")

        if role not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise OpenAIProtocolError(
                f"unsupported message role: {role!r}",
                param="messages",
            )

        content = content_to_text(message.get("content"))

        if role in ("system", "developer"):
            if content:
                system_parts.append(content)
            continue

        if role == "assistant":
            hf = {
                "role": "assistant",
                "content": content,
            }

            reasoning_content = _assistant_reasoning(message)
            if reasoning_content is not None:
                hf["reasoning_content"] = reasoning_content

            calls = message.get("tool_calls") or []

            if calls:
                if not isinstance(calls, list):
                    raise OpenAIProtocolError(
                        "assistant tool_calls must be an array",
                        param="messages",
                    )
                converted = []

                for call in calls:
                    if not isinstance(call, dict):
                        raise OpenAIProtocolError(
                            "assistant tool call must be an object",
                            param="messages",
                        )
                    call_id = str(call.get("id") or "")
                    fn = call.get("function") or {}
                    name = fn.get("name") if isinstance(fn, dict) else None

                    if not name:
                        raise OpenAIProtocolError(
                            "assistant tool call is missing function.name",
                            param="messages",
                        )

                    args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"

                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError as exc:
                            raise OpenAIProtocolError(
                                f"invalid tool arguments: {exc}",
                                param="messages",
                            ) from exc

                    if not isinstance(args, dict):
                        raise OpenAIProtocolError(
                            "tool call arguments must decode to a JSON object",
                            param="messages",
                        )

                    converted.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args,
                        },
                    })

                    if call_id:
                        tool_names[call_id] = name

                hf["tool_calls"] = converted

            result.append(hf)
            continue

        if role == "tool":
            hf = {
                "role": "tool",
                "content": content,
            }

            call_id = message.get("tool_call_id")
            name = message.get("name")

            if not name and call_id:
                name = tool_names.get(str(call_id))

            if name:
                hf["name"] = str(name)

            result.append(hf)
            continue

        result.append({
            "role": role,
            "content": content,
        })

    if system_parts:
        result.insert(0, {
            "role": "system",
            "content": "\n\n".join(system_parts),
        })

    return result


def wire_tool_calls(calls: list[dict]) -> list[dict]:
    result = []

    for index, call in enumerate(calls):
        fn = call["function"]
        arguments = fn["arguments"]
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=False,
            )

        result.append({
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": fn["name"],
                "arguments": arguments,
            },
        })

    return result


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def encode_chat_completion(request: ChatRequest, result,
                           *, completion_id: str | None = None,
                           created: int | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant"}
    if result.tool_calls:
        message["content"] = (result.content if request.stop and result.finish_reason == "stop" else None)
        message["tool_calls"] = wire_tool_calls(result.tool_calls)
    else:
        message["content"] = result.content
    if result.reasoning_content:
        message["reasoning_content"] = result.reasoning_content

    usage: dict[str, Any] = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    if result.cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": result.cached_tokens}

    return {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": usage,
    }


def encode_stream_usage(request: ChatRequest, event,
                        *, completion_id: str, created: int) -> dict | None:
    from .chat_service import StreamDone

    if not request.include_usage:
        return None
    if not isinstance(event, StreamDone):
        return None
    if event.prompt_tokens is None or event.completion_tokens is None:
        return None

    usage: dict[str, Any] = {
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.prompt_tokens + event.completion_tokens,
    }
    if event.cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": event.cached_tokens}

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [],
        "usage": usage,
    }


def encode_stream_event(request: ChatRequest, event,
                        *, completion_id: str, created: int) -> dict | None:
    from .chat_service import (
        ContentDelta,
        ReasoningDelta,
        StreamDone,
        ToolCallDelta,
    )

    if isinstance(event, ReasoningDelta):
        if not event.text:
            return None
        delta: dict[str, Any] = {"reasoning_content": event.text}
    elif isinstance(event, ContentDelta):
        if not event.text:
            return None
        delta = {"content": event.text}
    elif isinstance(event, ToolCallDelta):
        entry: dict[str, Any] = {
            "index": event.index,
            "id": event.call_id,
            "type": "function",
            "function": {
                "name": event.name or "",
                "arguments": event.arguments,
            },
        }
        delta = {"tool_calls": [entry]}
    elif isinstance(event, StreamDone):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": event.finish_reason,
                }
            ],
        }
    else:
        return None

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ],
    }
