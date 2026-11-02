#!/usr/bin/env python3
"""Per-request inference budget and thinking budget contract.

Protocol level only: no GPU and no server process.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ensure_chat_import, model_dir  # noqa: E402

codec = ensure_chat_import()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "apps" / "server"))

from serverlib.chat_service import ChatService  # noqa: E402
from serverlib.openai_protocol import (  # noqa: E402
    OpenAIProtocolError,
    answer_reserve_tokens,
    parse_chat_request,
    thinking_budget_tokens,
)

BASE_BUDGET = 32768
LARGE_CONTEXT_TOKENS = 262144
LARGE_BUDGET = 65536
CEILING = max(BASE_BUDGET, LARGE_BUDGET)

MESSAGES = [{"role": "user", "content": "2+2は？"}]


def body(**extra) -> dict:
    return {"model": "shelf", "messages": MESSAGES, **extra}


def parse(checked, name, payload, expected):
    try:
        request = parse_chat_request(
            payload, inference_budget_tokens=CEILING)
    except OpenAIProtocolError as exc:
        checked.check(name, expected is None, f"rejected {exc.code}")
        return
    if expected is None:
        checked.check(name, False, "accepted")
        return
    checked.check(name, request.max_new_tokens == expected,
                  f"{request.max_new_tokens} != {expected}")


def main() -> int:
    checked = Checker("inference-budget")

    checked.check("reserve-32768", answer_reserve_tokens(BASE_BUDGET) == 8192,
                  str(answer_reserve_tokens(BASE_BUDGET)))
    checked.check("thinking-32768", thinking_budget_tokens(BASE_BUDGET) == 24576,
                  str(thinking_budget_tokens(BASE_BUDGET)))
    checked.check("reserve-65536", answer_reserve_tokens(LARGE_BUDGET) == 16384,
                  str(answer_reserve_tokens(LARGE_BUDGET)))
    checked.check("thinking-65536",
                  thinking_budget_tokens(LARGE_BUDGET) == 49152,
                  str(thinking_budget_tokens(LARGE_BUDGET)))
    checked.check("small-budget-no-thinking-cap",
                  thinking_budget_tokens(1023) == 0,
                  str(thinking_budget_tokens(1023)))
    checked.check("thinking-below-budget",
                  thinking_budget_tokens(3000) + answer_reserve_tokens(3000) == 3000,
                  "")

    parse(checked, "omitted-uses-ceiling", body(), CEILING)
    parse(checked, "requested-above-ceiling-clamped",
          body(max_completion_tokens=100000), CEILING)
    parse(checked, "requested-below-ceiling-kept",
          body(max_completion_tokens=1024), 1024)
    parse(checked, "legacy-max_tokens-clamped",
          body(max_tokens=100000), CEILING)
    parse(checked, "non-positive-rejected",
          body(max_completion_tokens=0), None)

    if not model_dir().is_dir():
        return checked.done()

    processor = codec.load_processor(str(model_dir()))
    capability = codec.reasoning_capability(processor)
    checked.check("probe-reports-close-token",
                  isinstance(capability["close_token"], int)
                  and capability["close_token"] >= 0,
                  repr(capability["detail"]))

    service = ChatService(
        processor=processor,
        compute=None,
        model_name="shelf",
        max_seq_len=262144,
        inference_budget_tokens=BASE_BUDGET,
        large_context_tokens=LARGE_CONTEXT_TOKENS,
        large_context_inference_budget_tokens=LARGE_BUDGET,
        reasoning_close_token=capability["close_token"],
    )

    checked.check("budget-under-threshold",
                  service.inference_budget_for(LARGE_CONTEXT_TOKENS) == BASE_BUDGET,
                  "")
    checked.check("budget-over-threshold",
                  service.inference_budget_for(LARGE_CONTEXT_TOKENS + 1) == LARGE_BUDGET,
                  "")
    checked.check("ceiling-is-the-larger-budget",
                  service.inference_budget_ceiling == LARGE_BUDGET, "")

    request = parse_chat_request(body(), inference_budget_tokens=service.inference_budget_ceiling)
    prepared = service.prepare(request)
    checked.check("prepare-small-context-budget",
                  prepared.max_new_tokens == BASE_BUDGET,
                  str(prepared.max_new_tokens))
    checked.check("prepare-small-context-thinking",
                  prepared.reasoning_budget_tokens == 24576,
                  str(prepared.reasoning_budget_tokens))
    checked.check("prepare-close-token-forwarded",
                  prepared.reasoning_close_token == capability["close_token"], "")

    long_messages = [{"role": "user", "content": "語 " * 4000}]
    small_threshold = ChatService(
        processor=processor,
        compute=None,
        model_name="shelf",
        max_seq_len=262144,
        inference_budget_tokens=BASE_BUDGET,
        large_context_tokens=4000,
        large_context_inference_budget_tokens=LARGE_BUDGET,
        reasoning_close_token=capability["close_token"],
    )
    request = parse_chat_request(
        {"model": "shelf", "messages": long_messages},
        inference_budget_tokens=small_threshold.inference_budget_ceiling)
    prepared = small_threshold.prepare(request)
    prompt_tokens = len(prepared.prompt_ids)
    checked.check("long-prompt-over-threshold", prompt_tokens > 4000,
                  str(prompt_tokens))
    checked.check("prepare-large-context-budget",
                  prepared.max_new_tokens == LARGE_BUDGET,
                  str(prepared.max_new_tokens))
    checked.check("prepare-large-context-thinking",
                  prepared.reasoning_budget_tokens == 49152,
                  str(prepared.reasoning_budget_tokens))

    request = parse_chat_request(
        body(chat_template_kwargs={"enable_thinking": False}),
        inference_budget_tokens=service.inference_budget_ceiling)
    prepared = service.prepare(request)
    checked.check("thinking-off-has-no-budget",
                  prepared.reasoning_budget_tokens == 0,
                  str(prepared.reasoning_budget_tokens))

    unknown_close = ChatService(
        processor=processor,
        compute=None,
        model_name="shelf",
        max_seq_len=262144,
        inference_budget_tokens=BASE_BUDGET,
        large_context_tokens=LARGE_CONTEXT_TOKENS,
        large_context_inference_budget_tokens=LARGE_BUDGET,
        reasoning_close_token=-1,
    )
    request = parse_chat_request(body(),
                                 inference_budget_tokens=BASE_BUDGET)
    prepared = unknown_close.prepare(request)
    checked.check("no-close-token-means-no-thinking-budget",
                  prepared.reasoning_budget_tokens == 0,
                  str(prepared.reasoning_budget_tokens))

    capped = ChatService(
        processor=processor,
        compute=None,
        model_name="shelf",
        max_seq_len=prompt_tokens + 512,
        inference_budget_tokens=BASE_BUDGET,
        large_context_tokens=4000,
        large_context_inference_budget_tokens=LARGE_BUDGET,
        reasoning_close_token=capability["close_token"],
    )
    request = parse_chat_request(
        {"model": "shelf", "messages": long_messages},
        inference_budget_tokens=capped.inference_budget_ceiling)
    prepared = capped.prepare(request)
    checked.check("remaining-window-caps-the-budget",
                  prepared.max_new_tokens == 512, str(prepared.max_new_tokens))
    checked.check("small-remaining-has-no-thinking-cap",
                  prepared.reasoning_budget_tokens == 0,
                  str(prepared.reasoning_budget_tokens))

    return checked.done()


if __name__ == "__main__":
    sys.exit(main())
