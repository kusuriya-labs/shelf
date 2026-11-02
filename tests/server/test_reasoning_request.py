#!/usr/bin/env python3
"""Reasoning request contract: prompt rendering, validation, answer parsing.

Needs the model tokenizer and chat template, but not the GPU and not the server
process.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ensure_chat_import,
    load_processor,
    model_dir,
)

codec = ensure_chat_import()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "apps" / "server"))

from serverlib.openai_protocol import (  # noqa: E402
    OpenAIProtocolError,
    openai_messages_to_hf,
    parse_chat_request,
)

MESSAGES = [{"role": "user", "content": "2+2は？"}]

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TOOL_CALL_TEXT = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\n大阪\n"
    "</parameter>\n</function>\n</tool_call>")


def render(processor, messages, **kwargs):
    return codec.prompt_ids(processor, messages, **kwargs)


def check_prompt_matrix(checker, processor):
    defaults = render(processor, MESSAGES)
    explicit = render(processor, MESSAGES, enable_thinking=True,
                      reasoning_effort="xhigh", preserve_thinking=True)
    direct = processor.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True)
    direct = direct[0] if direct and isinstance(direct[0], list) else direct

    checker.check("u01-defaults-match-xhigh", defaults == explicit,
                  f"{len(defaults)} vs {len(explicit)}")
    checker.check("u01-codec-matches-template", explicit == list(direct),
                  f"codec={len(explicit)} template={len(direct)}")

    open_ids = processor.tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = list(codec.reasoning_close_ids(processor))
    checker.check("boundary-single-token", len(close_ids) == 1, repr(close_ids))
    checker.check("boundary-is-close-tag",
                  close_ids == processor.tokenizer.encode(
                      "</think>", add_special_tokens=False),
                  repr(close_ids))

    tail = explicit[-len(open_ids) - 2:]
    checker.check("u06-prompt-ends-inside-thinking", open_ids[0] in tail,
                  f"tail={tail}")
    checker.check("u06-prompt-has-no-close-token", close_ids[0] not in explicit,
                  f"tail={explicit[-8:]}")

    off = render(processor, MESSAGES, enable_thinking=False)
    checker.check("u03-nothink-prompt-closes-thinking", close_ids[0] in off,
                  f"tail={off[-8:]}")
    checker.check("u03-nothink-drops-effort-instructions",
                  len(explicit) > len(off),
                  f"off={len(off)} on={len(explicit)}")

    xhigh = render(processor, MESSAGES, reasoning_effort="xhigh")
    medium = render(processor, MESSAGES, reasoning_effort="medium")
    low = render(processor, MESSAGES, reasoning_effort="low")
    checker.check("u02-xhigh-differs-from-medium", xhigh != medium,
                  f"{len(xhigh)} vs {len(medium)}")
    checker.check("u02-medium-differs-from-low", medium != low,
                  f"{len(medium)} vs {len(low)}")
    checker.check("u02-repeat-deterministic",
                  render(processor, MESSAGES, reasoning_effort="low") == low, "")

    tools_on = render(processor, MESSAGES, tools=[TOOL])
    tools_ids = processor.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True, tools=[TOOL],
        enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True)
    tools_ids = tools_ids[0] if tools_ids and isinstance(tools_ids[0], list) else tools_ids
    checker.check("u01-tools-match-template", tools_on == list(tools_ids),
                  f"{len(tools_on)} vs {len(tools_ids)}")
    checker.check("u06-tools-prompt-inside-thinking", close_ids[0] not in tools_on, "")


def check_history(checker, processor):
    history = [
        {"role": "user", "content": "問題を解いて"},
        {"role": "assistant", "reasoning_content": "まず条件を整理する。",
         "content": "結果はAです。"},
        {"role": "user", "content": "その根拠は？"},
    ]
    hf = openai_messages_to_hf(history)
    checker.check("u16-history-keeps-reasoning",
                  hf[1].get("reasoning_content") == "まず条件を整理する。", repr(hf))

    preserved = render(processor, hf, enable_thinking=False, preserve_thinking=True)
    dropped = render(processor, hf, enable_thinking=False, preserve_thinking=False)
    checker.check("u16-preserve-thinking-renders",
                  len(preserved) > len(dropped),
                  f"preserved={len(preserved)} dropped={len(dropped)}")
    preserved_text = processor.apply_chat_template(
        hf, tokenize=False, add_generation_prompt=True,
        enable_thinking=False, preserve_thinking=True)
    checker.check("u16-reasoning-text-present", "まず条件を整理する。" in preserved_text,
                  preserved_text[:200])
    dropped_text = processor.apply_chat_template(
        hf, tokenize=False, add_generation_prompt=True,
        enable_thinking=False, preserve_thinking=False)
    checker.check("u16-reasoning-text-absent", "まず条件を整理する。" not in dropped_text,
                  dropped_text[:200])
    checker.check("u16-content-survives", "結果はAです。" in dropped_text, "")


def check_answer_parsing(checker, processor):
    close = codec.reasoning_close_ids(processor)
    reasoning_ids = processor.tokenizer.encode(
        "この依頼はtoolを呼ぶべきだ。" + TOOL_CALL_TEXT, add_special_tokens=False)
    answer_ids = processor.tokenizer.encode("\n\n天気は晴れです。", add_special_tokens=False)
    generated = list(reasoning_ids) + list(close) + list(answer_ids)

    split = codec.split_generation(processor, generated, thinking=True)
    checker.check("u12-tool-tag-stays-in-reasoning", TOOL_CALL_TEXT in split.reasoning,
                  repr(split.reasoning[:120]))
    checker.check("u12-answer-clean", split.answer == "天気は晴れです。", repr(split.answer))
    parsed = codec.parse_answer(processor, split.answer, tools=[TOOL])
    checker.check("u12-no-tool-call-from-reasoning", parsed["tool_calls"] == [],
                  repr(parsed))

    generated = list(close) + processor.tokenizer.encode(
        TOOL_CALL_TEXT, add_special_tokens=False)
    split = codec.split_generation(processor, generated, thinking=True)
    parsed = codec.parse_answer(processor, split.answer, tools=[TOOL])
    calls = parsed["tool_calls"]
    checker.check("u13-tool-call-from-answer", len(calls) == 1, repr(parsed))
    if calls:
        checker.check("u13-tool-call-name",
                      calls[0]["function"]["name"] == "get_weather", repr(calls[0]))
        checker.check("u13-tool-call-arguments",
                      calls[0]["function"]["arguments"].get("city") == "大阪",
                      repr(calls[0]))
    checker.check("u13-content-empty", parsed["content"] == "", repr(parsed["content"]))

    answer_text = "大阪の天気を確認します。\n" + TOOL_CALL_TEXT
    parsed = codec.parse_answer(processor, answer_text, tools=[TOOL])
    checker.check("u13-text-then-tool-call",
                  len(parsed["tool_calls"]) == 1
                  and "大阪の天気" in (parsed["content"] or ""), repr(parsed))

    broken = "<tool_call>\nbroken"
    try:
        codec.parse_answer(processor, broken, tools=[TOOL])
        checker.check("u14-broken-tool-call-rejected", False, "no error raised")
    except codec.ToolParseError as exc:
        checker.check("u14-broken-tool-call-rejected", True, str(exc))

    split = codec.split_generation(processor, processor.tokenizer.encode(
        "天気の話だけです。", add_special_tokens=False), thinking=False)
    checker.check("u20-nothink-answer-only",
                  split.reasoning == "" and split.closed
                  and split.answer == "天気の話だけです。", repr(split))


def check_capability(checker, processor):
    capability = codec.reasoning_capability(processor)
    checker.check("probe-thinking", capability["thinking"], repr(capability))
    checker.check("probe-effort", capability["effort"], repr(capability))
    checker.check("probe-available", codec.reasoning_available(processor), "")


def check_validation(checker):
    base = {"model": "shelf", "messages": MESSAGES}

    request = parse_chat_request(base)
    checker.check("u01-default-enable-thinking", request.enable_thinking is True, "")
    checker.check("u01-default-effort",
                  request.reasoning_effort == codec.DEFAULT_REASONING_EFFORT, "")
    checker.check("u01-default-preserve", request.preserve_thinking is True, "")

    for effort in ("low", "medium", "xhigh"):
        request = parse_chat_request({**base, "reasoning_effort": effort})
        checker.check(f"u02-accept-{effort}", request.reasoning_effort == effort, "")

    for label, payload in (
        ("high", {"reasoning_effort": "high"}),
        ("minimal", {"reasoning_effort": "minimal"}),
        ("number", {"reasoning_effort": 1}),
        ("bool", {"reasoning_effort": True}),
    ):
        try:
            parse_chat_request({**base, **payload})
            checker.check(f"u04-reject-{label}", False, "no error raised")
        except OpenAIProtocolError as exc:
            checker.check(f"u04-reject-{label}",
                          exc.status == 400
                          and exc.param == "reasoning_effort",
                          f"{exc.param} {exc.status}")

    request = parse_chat_request({**base, "reasoning_effort": None})
    checker.check("u04-null-effort-is-default",
                  request.reasoning_effort == "xhigh", "")

    request = parse_chat_request({
        **base, "chat_template_kwargs": {"enable_thinking": False}})
    checker.check("u03-disable-thinking", request.enable_thinking is False, "")

    for label, payload, param in (
        ("unknown-key", {"chat_template_kwargs": {"thinking": True}},
         "chat_template_kwargs.thinking"),
        ("bad-bool", {"chat_template_kwargs": {"enable_thinking": "no"}},
         "chat_template_kwargs.enable_thinking"),
        ("preserve-bad", {"chat_template_kwargs": {"preserve_thinking": 1}},
         "chat_template_kwargs.preserve_thinking"),
        ("not-object", {"chat_template_kwargs": [1]}, "chat_template_kwargs"),
    ):
        try:
            parse_chat_request({**base, **payload})
            checker.check(f"u05-reject-{label}", False, "no error raised")
        except OpenAIProtocolError as exc:
            checker.check(f"u05-reject-{label}",
                          exc.status == 400 and exc.param == param,
                          f"{exc.param} {exc.status}")

    try:
        parse_chat_request({
            **base, "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"enable_thinking": False}})
        checker.check("u05-reject-effort-with-nothink", False, "no error raised")
    except OpenAIProtocolError as exc:
        checker.check("u05-reject-effort-with-nothink",
                      exc.status == 400 and exc.param == "reasoning_effort",
                      f"{exc.param} {exc.status}")

    try:
        parse_chat_request({**base, "reasoning": {"effort": "high"}})
        checker.check("u05-reject-reasoning-object", False, "no error raised")
    except OpenAIProtocolError as exc:
        checker.check("u05-reject-reasoning-object",
                      exc.status == 400 and exc.param == "reasoning", "")


def main() -> int:
    checker = Checker("reasoning-request")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        processor = load_processor()
        check_validation(checker)
        check_capability(checker, processor)
        check_prompt_matrix(checker, processor)
        check_history(checker, processor)
        check_answer_parsing(checker, processor)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
