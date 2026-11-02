#!/usr/bin/env python3
"""Gate 6A: OpenCode agent E2E through PhaseShift Server (Chat Completions)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

PROVIDER_CONFIG = """{{
  "$schema": "https://opencode.ai/config.json",
  "model": "phaseshift/phaseshift",
  "providers": {{
    "phaseshift": {{
      "name": "PhaseShift",
      "env": ["PHASESHIFT_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {{"baseURL": "{base_url}"}},
      "models": {{"phaseshift": {{"name": "PhaseShift",
        "limit": {{"context": {context}, "output": {output}}}}}}}
    }}
  }}
}}
"""

# OpenCode otherwise asks for a 32000-token output budget, which the server
# clamps to the remaining window. Declaring the model limit keeps the requested
# budget inside the served context.
MODEL_CONTEXT = 32768
MODEL_OUTPUT = 8192


def opencode_binary():
    return shutil.which("opencode")


def run_opencode(workspace: Path, message: str, base_url: str, env: dict,
                 timeout: int = 1500) -> subprocess.CompletedProcess:
    (workspace / "opencode.jsonc").write_text(
        PROVIDER_CONFIG.format(base_url=base_url, context=MODEL_CONTEXT,
                               output=MODEL_OUTPUT))
    child_env = dict(env)
    child_env["PWD"] = str(workspace)
    return subprocess.run(
        ["opencode", "run", "--standalone", "--auto",
         "-m", "phaseshift/phaseshift", message],
        cwd=str(workspace), env=child_env, capture_output=True, text=True,
        timeout=timeout)


def read_request_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def agent_env(base: Path, server_env: dict) -> dict:
    env = dict(server_env)
    env["XDG_CONFIG_HOME"] = str(base / "xdg-config")
    env["XDG_DATA_HOME"] = str(base / "xdg-data")
    env["XDG_CACHE_HOME"] = str(base / "xdg-cache")
    env["PHASESHIFT_API_KEY"] = "dummy"
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    for key in ("xdg-config", "xdg-data", "xdg-cache"):
        (base / key).mkdir(parents=True, exist_ok=True)
    return env


def run_task(base: Path, name: str, seed_name: str, seed_text: str,
             message: str, output_name: str, expected: str, server,
             env: dict, attempts: int = 3):
    last = (1, None)
    for attempt in range(1, attempts + 1):
        workspace = base / f"{name}-{attempt}"
        workspace.mkdir()
        (workspace / seed_name).write_text(seed_text)
        process = run_opencode(workspace, message, server.base_url, env)
        target = workspace / output_name
        if (process.returncode == 0 and target.is_file()
                and target.read_text().strip() == expected):
            return True, process, target
        last = (process.returncode, target)
    return False, None, last[1]


def main() -> int:
    checker = Checker("agent-opencode")
    if opencode_binary() is None:
        print("[SKIP] agent-opencode: opencode binary not installed")
        return 77
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    base = Path(tempfile.mkdtemp(prefix="ps-opencode-"))
    request_log = base / "requests.txt"
    token_log = base / "tokens.txt"
    compute_log = base / "compute.log"
    try:
        with ServerHarness(
                max_seq_len=MODEL_CONTEXT, arena_gib=24,
                device=int(os.environ.get("PHASESHIFT_TEST_DEVICE", "0")),
                env={"PHASESHIFT_SERVER_REQUEST_LOG": str(request_log),
                     "PHASESHIFT_SERVER_TOKEN_LOG": str(token_log),
                     "PHASESHIFT_COMPUTE_LOG": str(compute_log)}) as server:
            env = agent_env(base, server.env)

            # Acceptance 1: read + arithmetic + write.
            ok1, proc1, result1 = run_task(
                base, "task1", "input.txt", "41\n",
                "input.txt を読み、数値に1を足し、result.txt にその数値だけを書いてください。"
                "必要なファイル操作toolを使用してください。"
                "作業が完了したらtoolを呼ぶのをやめ、最後に通常の文章で完了を報告してください。",
                "result.txt", "42", server, env)
            checker.check("acceptance1-result", ok1,
                          f"rc={getattr(proc1, 'returncode', None)} "
                          f"value={result1.read_text().strip() if result1 and result1.is_file() else None}")

            # Acceptance 2: read + write transform.
            ok2, proc2, output2 = run_task(
                base, "task2", "source.txt", "PhaseShift\n",
                "source.txt を読み、同じ内容の末尾に \" Server\" を付けて "
                "output.txt を作ってください。"
                "作業が完了したらtoolを呼ぶのをやめ、最後に通常の文章で完了を報告してください。",
                "output.txt", "PhaseShift Server", server, env)
            checker.check("acceptance2-output", ok2,
                          f"rc={getattr(proc2, 'returncode', None)} "
                          f"value={output2.read_text().strip() if output2 and output2.is_file() else None}")

            entries = read_request_log(request_log)
            roles = [m["role"] for e in entries for m in e.get("messages", [])]
            assistant_with_calls = any(
                any(m["role"] == "assistant" and m["has_tool_calls"]
                    for m in e.get("messages", []))
                for e in entries)
            checker.check("tool-loop-tool-result", "tool" in roles, repr(roles[-10:]))
            checker.check("tool-loop-assistant-calls", assistant_with_calls, "")
            checker.check("tool-loop-multiple-requests", len(entries) >= 2,
                          f"requests={len(entries)}")

            generations = 0
            if token_log.is_file():
                generations = sum(1 for line in token_log.read_text().splitlines() if line.strip())
            checker.check("tool-loop-multiple-generations", generations >= 2,
                          f"generations={generations}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
