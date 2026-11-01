#!/usr/bin/env python3
"""OpenAI API の実 E2E 検証（T07）: HTTP → JSONL → GPU → HTTP。

実際に `apps/server/server.py` を子プロセスで起動し、HTTP で次を確認する:
  * 通常応答（finish_reason / usage / seed）
  * SSE（増分の連結が非ストリームと一致・最終 chunk の usage / seed）
  * EOS（短い質問で `finish_reason = "stop"`）
  * エラー（400 / 404 / 415 の envelope）
  * キャンセル（HTTP 切断 → 次の要求がすぐ流れる）
  * 同一プロセスで同じ要求を 2 回 → 本文が一致（決定性の入口）

使い方（要 GPU・要 HIP ビルド）:
  python3 tests/scripts/serve_e2e.py \
      --model models/Qwen3.5-9B-FP8 \
      --engine build/hip_rel/apps/shelf-compute
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

failures: list[str] = []


def check(cond: bool, what: str) -> None:
    if cond:
        print(f"PASS  {what}")
    else:
        print(f"FAIL  {what}")
        failures.append(what)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> bool:
    import aiohttp

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(base + "/healthz") as resp:
                    if resp.status == 200:
                        return True
        except Exception:  # noqa: BLE001 - 起動待ち
            pass
        await asyncio.sleep(0.5)
    return False


async def post_chat(session, base, payload, timeout=300):
    async with session.post(base + "/v1/chat/completions", json=payload,
                            timeout=aiohttp_timeout(timeout)) as resp:
        return resp.status, await resp.json()


def aiohttp_timeout(seconds):
    import aiohttp
    return aiohttp.ClientTimeout(total=seconds)


async def stream_chat(session, base, payload):
    lines = []
    async with session.post(base + "/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            return resp.status, [], await resp.text()
        async for raw in resp.content:
            text = raw.decode("utf-8", "replace").strip()
            if text.startswith("data:"):
                lines.append(text[len("data:"):].strip())
    return 200, lines, ""


def parse_sse(lines):
    assert lines and lines[-1] == "[DONE]", lines[-1:] or "no [DONE]"
    return [json.loads(x) for x in lines[:-1]]


def sse_text(chunks) -> str:
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


def prompt(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


async def run_checks(base: str, model_name: str, thinking: bool) -> None:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # ---- 起動・モデル一覧 ----
        async with session.get(base + "/healthz") as resp:
            check(resp.status == 200, "healthz が 200")
        async with session.get(base + "/v1/models") as resp:
            body = await resp.json()
            check(resp.status == 200 and body["data"][0]["id"] == model_name,
                  "v1/models にモデル名")

        # ---- 通常応答 ----
        body = {"model": model_name, "messages": prompt("1+1は? 日本語で短く答えて"),
                "max_tokens": 32, "temperature": 0.0, "seed": 42,
                "enable_thinking": thinking}
        status, first = await post_chat(session, base, body)
        check(status == 200, "通常応答が 200")
        choice = first["choices"][0]
        check(choice["message"]["role"] == "assistant", "message.role が assistant")
        check(choice["finish_reason"] in ("stop", "length"), "finish_reason が stop/length")
        check(first["seed"] == 42, "seed が要求値")
        check(first["usage"]["prompt_tokens"] > 0, "usage.prompt_tokens > 0")
        check(isinstance(first["usage"]["completion_tokens"], int) and
              first["usage"]["completion_tokens"] > 0,
              "usage.completion_tokens が正の整数")
        check(first["usage"]["total_tokens"] ==
              first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"],
              "usage.total_tokens = prompt + completion")
        text1 = choice["message"]["content"]
        print(f"       content={text1!r} finish={choice['finish_reason']} "
              f"usage={first['usage']}")
        check("\ufffd" not in text1, "本文に U+FFFD が無い")
        check("<|" not in text1, "本文に特殊トークンが漏れない")

        # ---- 決定性（同一プロセスで 2 回）----
        status2, second = await post_chat(session, base, body)
        check(status2 == 200 and second["choices"][0]["message"]["content"] == text1,
              "同一要求 2 回で本文が一致（greedy）")
        check(second["usage"] == first["usage"], "同一要求 2 回で usage が一致")

        # ---- SSE ----
        stream_body = dict(body, stream=True, seed=7)
        status3, lines, err = await stream_chat(session, base, stream_body)
        check(status3 == 200, "SSE が 200")
        chunks = parse_sse(lines)
        check(chunks[0]["choices"][0]["delta"].get("role") == "assistant",
              "SSE の先頭 chunk に role")
        streamed = sse_text(chunks)
        final = chunks[-1]
        check(final["choices"][0]["finish_reason"] in ("stop", "length"),
              "SSE の finish_reason が stop/length")
        check(final["seed"] == 7, "SSE の最終 chunk に seed")
        check(final["usage"] == first["usage"] or final["usage"]["prompt_tokens"] > 0,
              "SSE の最終 chunk に usage")
        check("\ufffd" not in streamed, "SSE の本文に U+FFFD が無い")
        # 同じプロンプト・greedy なので非ストリームと一致する。
        status4, nonstream = await post_chat(session, base, dict(body, seed=7))
        check(status4 == 200 and nonstream["choices"][0]["message"]["content"] == streamed,
              "SSE の連結 = 非ストリームの本文")

        # ---- EOS ----
        status5, eos = await post_chat(session, base, {
            "model": model_name, "messages": prompt("日本の首都はどこ? 一語で"),
            "max_tokens": 64, "enable_thinking": False})
        finish5 = eos["choices"][0]["finish_reason"]
        check(status5 == 200 and finish5 == "stop", f"EOS で stop（実際 {finish5}）")
        check(eos["usage"]["completion_tokens"] < 64, "EOS が max_tokens より先に来た")

        # ---- エラー ----
        status6, err6 = await post_chat(session, base, dict(body, top_k=-1))
        check(status6 == 400 and err6["error"]["code"] == "invalid_argument",
              "不正な top_k は 400")
        check(err6["error"]["param"] == "top_k", "エラーの param が top_k")
        status7, err7 = await post_chat(session, base, dict(body, model="nope"))
        check(status7 == 404 and err7["error"]["code"] == "model_not_found",
              "未知のモデルは 404")
        status8, err8 = await post_chat(session, base, dict(body, stop="END"))
        check(status8 == 400 and err8["error"]["code"] == "unsupported_parameter",
              "未対応の stop は 400")
        async with session.post(base + "/v1/chat/completions", data=b"{}",
                                headers={"Content-Type": "text/plain"}) as resp:
            check(resp.status == 415, "Content-Type 違反は 415")

        # ---- キャンセル（HTTP 切断）----
        long_body = {"model": model_name, "messages": prompt("長い文章を書いて"),
                     "max_tokens": 1024, "stream": True}
        started = time.time()
        async with session.post(base + "/v1/chat/completions", json=long_body) as resp:
            await resp.content.read(200)
            resp.close()
        await asyncio.sleep(0.5)
        t0 = time.time()
        status9, quick = await post_chat(session, base, {
            "model": model_name, "messages": prompt("こんにちは"), "max_tokens": 4})
        elapsed = time.time() - t0
        check(status9 == 200, "切断後の次の要求が 200")
        check(elapsed < 30.0, f"切断後に次の要求がすぐ流れる（{elapsed:.2f}s）")
        del started


async def main_async(args) -> int:
    base = f"http://127.0.0.1:{args.port}"
    log = tempfile.NamedTemporaryFile(prefix="shelf_e2e_", suffix=".log", delete=False)
    cmd = [sys.executable, "server.py",
           "--engine", args.engine, "--model", args.model,
           "--model-name", args.model_name, "--port", str(args.port),
           "--gpu", str(args.gpu), "--pipeline-parallel", str(args.pp),
           "--prefill-rows", str(args.prefill_rows), "--act-quant", args.act_quant]
    if args.offload_weights_mib:
        cmd += ["--offload-weights-mib", str(args.offload_weights_mib)]
    if args.kv_in_host:
        cmd += ["--kv-in-host"]
    if args.online_kv_pages:
        cmd += ["--online-kv-pages", str(args.online_kv_pages)]
    print("起動:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=os.path.join(ROOT, "apps", "server"),
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        ready = await wait_ready(base, proc, args.max_wait)
        if not ready:
            print(f"サーバが起動しない（{args.max_wait}s）。ログ: {log.name}")
            print(open(log.name, encoding="utf-8", errors="replace").read()[-4000:])
            return 1
        await run_checks(base, args.model_name, args.thinking)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    print()
    if failures:
        print(f"{len(failures)} 件失敗。ログ: {log.name}")
        return 1
    print(f"E2E はすべて成功（ログ: {log.name}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI API の実 E2E 検証（T07）")
    parser.add_argument("--model", default="models/Qwen3.5-9B-FP8")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--engine", default="build/hip_rel/apps/shelf-compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--act-quant", default="auto", choices=["auto", "fp4", "fp8"])
    parser.add_argument("--offload-weights-mib", type=int, default=0)
    parser.add_argument("--kv-in-host", action="store_true")
    parser.add_argument("--online-kv-pages", type=int, default=0,
                        help="T14: 直近 N ページを VRAM にミラーする")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-wait", type=float, default=900.0)
    parser.add_argument("--thinking", action="store_true",
                        help="通常応答を思考モード ON で行う")
    args = parser.parse_args(argv)
    if not args.port:
        args.port = free_port()
    args.model = os.path.abspath(args.model)
    args.engine = os.path.abspath(args.engine)
    if not args.model_name:
        args.model_name = os.path.basename(args.model.rstrip("/"))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
