#!/usr/bin/env python3
"""best-effort tool calling の回帰.

constrained decoding が存在しない状態で、複数の prompt に対して
function name の parse / arguments JSON の parse / multi-turn round trip が
安定して成立するかを確認する。

生成は Transformers の Qwen response parser で解析する best-effort であるため、
100% 保証は目標としない。ただし parse 不能な出力を content として捏造しないこと
は契約であり、その場合は fail として扱う。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json, model_dir  # noqa: E402

ROUNDS = int(os.environ.get("PHASESHIFT_TOOL_REGRESSION_ROUNDS", "100"))

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}

PROMPTS = [
    "大阪の現在の気温をget_weatherを使って確認して",
    "札幌の天気をget_weatherで調べてください",
    "名古屋の現在の天気をget_weatherを使って教えて",
    "福岡の気温をget_weatherで確認してください",
]


def main() -> int:
    checker = Checker("server-tool-regression")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    parse_ok = 0
    name_ok = 0
    args_ok = 0
    roundtrip_attempted = 0
    roundtrip_ok = 0
    markup_leaks = 0
    failures = []

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/chat/completions"

            for index in range(ROUNDS):
                prompt = PROMPTS[index % len(PROMPTS)]
                try:
                    result = http_json(url, {
                        "model": "phaseshift",
                        "messages": [{"role": "user", "content": prompt}],
                        "tools": [WEATHER],
                        "tool_choice": "auto",
                        "temperature": 0,
                        "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False},
                    })
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"round {index}: request failed {exc!r}")
                    continue

                choice = result["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if "<tool_call>" in content:
                    markup_leaks += 1

                calls = message.get("tool_calls") or []
                if choice.get("finish_reason") != "tool_calls" or not calls:
                    failures.append(
                        f"round {index}: no tool call "
                        f"(finish={choice.get('finish_reason')})")
                    continue

                parse_ok += 1
                call = calls[0]
                function = call.get("function") or {}
                if function.get("name") == "get_weather":
                    name_ok += 1
                else:
                    failures.append(f"round {index}: name={function.get('name')!r}")

                try:
                    arguments = json.loads(function.get("arguments") or "")
                except json.JSONDecodeError as exc:
                    failures.append(f"round {index}: arguments not JSON ({exc})")
                    continue
                if not isinstance(arguments, dict) or "city" not in arguments:
                    failures.append(f"round {index}: arguments={arguments!r}")
                    continue
                args_ok += 1

                # multi-turn round trip は generation を倍にするため 10 回に 1 回だけ行う。
                if index % 10 != 0:
                    continue
                roundtrip_attempted += 1

                round_trip = http_json(url, {
                    "model": "phaseshift",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": None, "tool_calls": [call]},
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", "call_0"),
                            "name": "get_weather",
                            "content": json.dumps({"temperature": 22}),
                        },
                    ],
                    "tools": [WEATHER],
                    "temperature": 0,
                    "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
                })
                final = round_trip["choices"][0]
                final_content = (final["message"].get("content") or "").strip()
                if final_content and "<tool_call>" not in final_content:
                    roundtrip_ok += 1
                else:
                    failures.append(
                        f"round {index}: round trip content={final_content!r}")

            print(f"rounds={ROUNDS} parse={parse_ok} name={name_ok} "
                  f"args={args_ok} roundtrip={roundtrip_ok}/{roundtrip_attempted} "
                  f"markup_leaks={markup_leaks}")
            for failure in failures[:20]:
                print(f"[detail] {failure}")

            checker.check("tool-call-parse-rate",
                          parse_ok >= max(1, int(ROUNDS * 0.9)),
                          f"{parse_ok}/{ROUNDS}")
            checker.check("function-name-rate",
                          name_ok >= max(1, int(ROUNDS * 0.9)),
                          f"{name_ok}/{ROUNDS}")
            checker.check("arguments-json-rate",
                          args_ok >= max(1, int(ROUNDS * 0.9)),
                          f"{args_ok}/{ROUNDS}")
            checker.check("roundtrip-rate",
                          roundtrip_ok >= max(1, int(roundtrip_attempted * 0.9)),
                          f"{roundtrip_ok}/{roundtrip_attempted}")
            checker.check("no-tool-markup-leak", markup_leaks == 0,
                          f"leaks={markup_leaks}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
