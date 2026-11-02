"""PhaseShift shared chat codec.

Bridges the Hugging Face Qwen processor (tokenizer + chat template) and the
PhaseShift compute engine's raw token-ID contract.

This module is the single source of truth for:

* processor loading
* chat-template prompt construction (system / user / assistant / tool)
* generation decoding
* tool-call response parsing

It is imported by ``phaseshift-server`` so that the
token sequence sent to ``phaseshift-compute`` never diverges between the two
entry points.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from . import reasoning

__all__ = [
    "ANSWER",
    "REASONING",
    "REASONING_EFFORTS",
    "DEFAULT_REASONING_EFFORT",
    "ToolParseError",
    "ToolCallingUnavailable",
    "TOOL_RESPONSE_TEMPLATE",
    "load_processor",
    "prompt_ids",
    "decode_generated",
    "tool_calling_available",
    "streaming_tool_calling_available",
    "probe_tool_calling",
    "probe_reasoning",
    "reasoning_available",
    "reasoning_capability",
    "reasoning_close_ids",
    "new_region_parser",
    "split_generation",
    "new_response_parser",
    "parse_answer",
    "strip_tool_markup",
]

ANSWER = reasoning.ANSWER
REASONING = reasoning.REASONING

REASONING_EFFORTS = ("low", "medium", "xhigh")
DEFAULT_REASONING_EFFORT = "xhigh"


class ToolParseError(RuntimeError):
    """Raised when a tool-calling generation cannot be parsed."""


class ToolCallingUnavailable(RuntimeError):
    """Raised when the active Transformers install cannot parse tool calls."""


# Qwen3.5 emits tool calls as::
#
#     <tool_call>
#     <function=NAME>
#     <parameter=KEY>
#     VALUE
#     </parameter>
#     </function>
#     </tool_call>
#
# This declarative response template is consumed by Transformers'
# ``parse_response``. It is intentionally a template (not a hand-written
# regex loop): the parser in transformers.utils.chat_parsing owns the scanning,
# overlap handling, and structured-value coercion.
# The parser is fed the already separated ANSWER region and is created with an
# empty ``prefix``, so the ``start_anchor`` below is not used to truncate a
# Prompt anymore. It is kept because a response template must declare one.
_THINK_CLOSE = "".join(
    chr(code) for code in (0x3C, 0x2F, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x3E))
_THINK_OPEN = "".join(
    chr(code) for code in (0x3C, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x3E))

TOOL_RESPONSE_TEMPLATE: dict[str, Any] = {
    "version": 1,
    "start_anchor": _THINK_CLOSE,
    "defaults": {"role": "assistant", "content": ""},
    "fields": {
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\s]+)>",
            "close": "</tool_call>",
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
            },
            "repeats": True,
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def load_processor(model_dir: str, trust_remote_code: bool = True):
    """Load the Hugging Face processor for ``model_dir``."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)


def _flatten_ids(ids: Any) -> list[int]:
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return list(ids[0])
    return list(ids)


def prompt_ids(
    processor,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    preserve_thinking: bool = True,
) -> list[int]:
    """Render ``messages`` through the model chat template and return token IDs.

    ``tools`` is passed through to the chat template unchanged; no Prompt-magic
    or tool-description synthesis is performed here. ``enable_thinking``,
    ``reasoning_effort`` and ``preserve_thinking`` are forwarded to the model's
    own chat template, which is the source of truth for how a thinking level is
    expressed in the Prompt.
    """
    return _render(
        processor, messages, tools, enable_thinking, True,
        reasoning_effort, preserve_thinking)


def _render(processor, messages, tools, enable_thinking, add_generation_prompt,
            reasoning_effort, preserve_thinking):
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "preserve_thinking": preserve_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    ids = processor.apply_chat_template(messages, **kwargs)
    return _flatten_ids(ids)


def decode_generated(processor, generated_ids: Iterable[int]) -> str:
    """Decode generated token IDs exactly like the PhaseShift CLI does."""
    return processor.decode(list(generated_ids), skip_special_tokens=True)


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _tag_ids_from_prompt(processor, text: str, *, thinking: bool) -> tuple[int, ...]:
    """Find the token span the chat template itself uses for ``text``."""
    ids = prompt_ids(
        processor, [{"role": "user", "content": "probe"}],
        enable_thinking=thinking)
    for width in range(1, 5):
        if width > len(ids):
            break
        for start in range(len(ids) - width + 1):
            window = ids[start:start + width]
            if decode_generated(processor, window) == text:
                return tuple(int(token) for token in window)
    return ()


def reasoning_close_ids(processor) -> tuple[int, ...]:
    """Token IDs the tokenizer assigns to the reasoning close tag.

    The tokenizer encoding is used when it round-trips through decode; the
    rendered chat Prompt is the fallback, because that Prompt is what the model
    actually continues from.
    """
    cached = getattr(processor, "_phaseshift_reasoning_close_ids", None)
    if cached is not None:
        return cached

    ids = _tokenizer(processor).encode(_THINK_CLOSE, add_special_tokens=False)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    close = tuple(int(token) for token in ids) if ids else ()
    if not close or decode_generated(processor, close) != _THINK_CLOSE:
        close = _tag_ids_from_prompt(processor, _THINK_CLOSE, thinking=False)
    if not close:
        raise ToolCallingUnavailable(
            "tokenizer produced no tokens for the reasoning close tag")
    try:
        processor._phaseshift_reasoning_close_ids = close
    except Exception:  # noqa: BLE001 - best-effort cache only
        pass
    return close


def new_region_parser(processor, *, thinking: bool):
    """Create the per-request reasoning/answer splitter."""
    return reasoning.GenerationRegionParser(
        lambda ids: decode_generated(processor, ids),
        reasoning_close_ids(processor),
        thinking=thinking,
    )


def split_generation(processor, generated_ids: Iterable[int], *,
                     thinking: bool) -> reasoning.RegionText:
    """Split one complete generation into reasoning and answer text."""
    ids = [int(token) for token in generated_ids]
    return reasoning.split_regions(
        lambda chunk: decode_generated(processor, chunk),
        ids,
        reasoning_close_ids(processor),
        thinking=thinking,
    )


def strip_tool_markup(text: str) -> str:
    """Remove Qwen tool-call markup that leaked into plain content."""
    if not text or _TOOL_OPEN not in text:
        return text
    out = []
    i = 0
    while True:
        start = text.find(_TOOL_OPEN, i)
        if start == -1:
            out.append(text[i:])
            break
        end = text.find(_TOOL_CLOSE, start)
        if end == -1:
            out.append(text[i:start])
            break
        out.append(text[i:start])
        i = end + len(_TOOL_CLOSE)
    return "".join(out).strip()


def _parse_response(processor, text: str) -> dict[str, Any]:
    return processor.parse_response(text, TOOL_RESPONSE_TEMPLATE, prefix="")


def _response_parser_factory(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "get_response_parser"):
        return tokenizer
    if hasattr(processor, "get_response_parser"):
        return processor
    raise AttributeError("processor has no get_response_parser")


def new_response_parser(processor):
    """Create one stateful streaming parser for a single generation.

    A parser must never be shared across requests: it carries the region state
    (and captured function names) of exactly one assistant turn.

    The parser is fed the already separated ANSWER region only, so it is built
    with an empty prefix. That is what removes the old dependency on the chat
    Prompt carrying a ``</think>`` start anchor: with thinking enabled the close
    tag is generated, not pre-filled.
    """
    factory = _response_parser_factory(processor)
    return factory.get_response_parser(
        response_template=TOOL_RESPONSE_TEMPLATE, prefix="")


def probe_tool_calling(processor) -> dict[str, Any]:
    """Probe the installed Transformers for Qwen3.5 tool support.

    Returns a capability dict. Never raises; callers decide how to report a
    missing capability.
    """
    capability = {
        "chat_template_tools": False,
        "response_parser": False,
        "streaming_response_parser": False,
        "detail": "",
    }

    def note(message: str) -> None:
        capability["detail"] = (capability["detail"] + "; " if capability["detail"] else "") + message

    probe_tools = [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "description": "noop",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }
    ]
    try:
        prompt_ids(
            processor, [{"role": "user", "content": "ping"}], tools=probe_tools)
        capability["chat_template_tools"] = True
    except Exception as exc:  # noqa: BLE001 - capability probe reports, never raises
        note(f"chat template tools: {exc}")

    probe_text = (
        "<tool_call>\n<function=probe>\n<parameter=x>\n1\n</parameter>\n"
        "</function>\n</tool_call>")

    try:
        parsed = _parse_response(processor, probe_text)
        capability["response_parser"] = bool(parsed.get("tool_calls"))
    except Exception as exc:  # noqa: BLE001
        note(f"response parser: {exc}")

    try:
        parser = new_response_parser(processor)
        events = parser.feed(probe_text)
        message, final_events = parser.finalize()
        events = list(events) + list(final_events)
        names = []
        args_ok = True
        for event in events:
            if event.get("type") == "region_close" and event.get("field") == "tool_calls":
                function = (event.get("value") or {}).get("function") or {}
                if function.get("name"):
                    names.append(function["name"])
                if not isinstance(function.get("arguments"), dict):
                    args_ok = False
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name"):
                names.append(function["name"])
        capability["streaming_response_parser"] = bool(names) and args_ok
    except Exception as exc:  # noqa: BLE001
        note(f"streaming response parser: {exc}")

    return capability


def _capability(processor) -> dict[str, Any]:
    cached = getattr(processor, "_phaseshift_tool_capability", None)
    if cached is None:
        cached = probe_tool_calling(processor)
        try:
            processor._phaseshift_tool_capability = cached
        except Exception:  # noqa: BLE001 - best-effort cache only
            pass
    return cached


def tool_calling_available(processor) -> bool:
    cached = _capability(processor)
    return bool(cached["chat_template_tools"] and cached["response_parser"])


def streaming_tool_calling_available(processor) -> bool:
    cached = _capability(processor)
    return bool(
        cached["chat_template_tools"] and cached["streaming_response_parser"])


def _parse_tool_calls(processor, final_text: str):
    """Parse tool calls from the final ANSWER region text.

    Returns ``(tool_calls, content)``. Raises ``ToolParseError`` when tool-call
    markup is present but could not be parsed into a call.
    """
    if not tool_calling_available(processor):
        raise ToolCallingUnavailable(
            "installed Transformers cannot parse Qwen3.5 tool calls")

    try:
        parsed = _parse_response(processor, final_text)
    except Exception as exc:  # noqa: BLE001
        if _TOOL_OPEN in final_text:
            raise ToolParseError(f"failed to parse tool call: {exc}") from exc
        return [], strip_tool_markup(final_text)

    tool_calls = []
    for call in parsed.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        arguments = fn.get("arguments")
        if not name:
            if _TOOL_OPEN in final_text:
                raise ToolParseError("parsed tool call is missing a function name")
            continue
        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else {"value": arguments}
        tool_calls.append(
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        )

    if not tool_calls and _TOOL_OPEN in final_text:
        raise ToolParseError("tool-call markup present but no tool call parsed")

    return tool_calls, strip_tool_markup(parsed.get("content", ""))


def parse_answer(
    processor,
    answer_text: str,
    tools: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Parse one already separated ANSWER region.

    The reasoning region is never passed here, so tool markup produced while
    thinking cannot be promoted to a tool call. Returns
    ``{"content": str, "tool_calls": list}`` where each tool call uses the
    OpenAI shape ``{"type": "function", "function": {"name", "arguments"}}``.

    Raises ``ToolParseError`` when a tool request produced tool-call-like output
    that the response template could not parse. Raw markup is never silently
    promoted to ``content``.
    """
    if not tools:
        return {"content": strip_tool_markup(answer_text), "tool_calls": []}
    tool_calls, content = _parse_tool_calls(processor, answer_text)
    return {"content": content, "tool_calls": tool_calls}


def probe_reasoning(processor) -> dict[str, Any]:
    """Probe the loaded chat template for Qwen3.8 reasoning support.

    A template that honours ``enable_thinking`` leaves the assistant turn inside
    ``<think>`` when thinking is enabled, and closes the block in the Prompt when
    it is disabled. ``reasoning_effort`` support is reported when the rendered
    Prompt changes between two accepted effort levels.
    """
    capability: dict[str, Any] = {
        "thinking": False,
        "effort": False,
        "detail": "",
    }

    def note(message: str) -> None:
        detail = capability["detail"]
        capability["detail"] = (detail + "; " if detail else "") + message

    messages = [{"role": "user", "content": "ping"}]
    try:
        close_ids = reasoning_close_ids(processor)
        open_ids = _tag_ids_from_prompt(processor, _THINK_OPEN, thinking=True)

        thinking_ids = prompt_ids(processor, messages, enable_thinking=True)
        answer_ids = prompt_ids(processor, messages, enable_thinking=False)

        def contains(ids, pattern) -> bool:
            width = len(pattern)
            if width == 0:
                return False
            return any(tuple(ids[index:index + width]) == pattern
                       for index in range(len(ids) - width + 1))

        capability["thinking"] = (
            bool(close_ids)
            and bool(open_ids)
            and contains(thinking_ids, open_ids)
            and not contains(thinking_ids, close_ids)
            and contains(answer_ids, close_ids)
        )
        if not capability["thinking"]:
            note("chat template does not open <think> for generation")

        rendered = {}
        for effort in REASONING_EFFORTS:
            rendered[effort] = prompt_ids(
                processor, messages, enable_thinking=True,
                reasoning_effort=effort)
        capability["effort"] = len({tuple(ids) for ids in rendered.values()}) > 1
        if not capability["effort"]:
            note("chat template ignores reasoning_effort")
    except Exception as exc:  # noqa: BLE001 - capability probe reports, never raises
        note(str(exc))

    return capability


def reasoning_available(processor) -> bool:
    cached = getattr(processor, "_phaseshift_reasoning_capability", None)
    if cached is None:
        cached = probe_reasoning(processor)
        try:
            processor._phaseshift_reasoning_capability = cached
        except Exception:  # noqa: BLE001 - best-effort cache only
            pass
    return bool(cached["thinking"] and cached["effort"])


def reasoning_capability(processor) -> dict[str, Any]:
    cached = getattr(processor, "_phaseshift_reasoning_capability", None)
    if cached is None:
        cached = probe_reasoning(processor)
        try:
            processor._phaseshift_reasoning_capability = cached
        except Exception:  # noqa: BLE001 - best-effort cache only
            pass
    return cached
