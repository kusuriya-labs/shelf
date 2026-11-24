#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "hy4-light": {
        "first_turn_length": 4096,
        "subsequent_turn_length": 800,
        "turns": 4,
        "max_tokens": 500,
        "concurrencies": [1, 4, 8, 16],
        "config": "hy4_light",
    },
    "hy4-cached": {
        "first_turn_length": 8192,
        "subsequent_turn_length": 800,
        "turns": 5,
        "max_tokens": 500,
        "concurrencies": [1, 4, 8, 16],
        "config": "hy4_cached",
    },
    "single-r9700": {
        "first_turn_length": 4096,
        "subsequent_turn_length": 800,
        "turns": 4,
        "max_tokens": 500,
        "concurrencies": [1, 4, 8],
        "config": "single_r9700",
    },
    "hy4-c32": {
        "first_turn_length": 4096,
        "subsequent_turn_length": 800,
        "turns": 4,
        "max_tokens": 500,
        "concurrencies": [1, 4, 8, 16, 32],
        "config": "hy4_c32",
    },
}

COLUMNS = [
    "config",
    "Conc.",
    "Latency (tps/user)",
    "Output Throughput (tok/min/gpu)",
    "TTFT p50 (ms)",
    "TTFT p95 (ms)",
    "Approx Cache Hit",
    "Decoded Tok/Iter",
]

SYNTHETIC_SNIPPETS = [
    "Analyze the repository structure and locate where tensor residency, KV paging, and request lifecycle management live.",
    "Trace the request state transitions during differential prefill and verify every alignment invariant across block boundaries.",
    "Inspect sliding window attention ring wrap bounds and speculative draft verification slots to guarantee exact token parity.",
    "Summarize the root causes of any cache invalidation races on request cancellation and draft a minimal test plan.",
    "Verify the streaming SSE usage breakdown reports cached prefix tokens and confirm end-to-end length completion behavior.",
]


class FallbackTokenizer:
    def __init__(self) -> None:
        self._pattern = re.compile(r"\S+\s*")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[int] = []
        for match in self._pattern.finditer(text):
            piece = match.group(0).encode("utf-8")
            val = int.from_bytes(hashlib.blake2b(piece, digest_size=4).digest(), "little") % 120000
            tokens.append(val)
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"w{idx}" for idx in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        return self.encode(rendered, add_special_tokens=False)


class TokenizersWrapper:
    def __init__(self, tok_obj: Any, template_str: str | None, bos_token: str, eos_token: str) -> None:
        self._tok = tok_obj
        self._template = None
        if template_str:
            import jinja2

            def raise_exception(msg: str) -> None:
                raise RuntimeError(msg)

            env = jinja2.Environment()
            env.globals["raise_exception"] = raise_exception
            self._template = env.from_string(template_str)
        self._bos = bos_token
        self._eos = eos_token

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        if self._template is not None:
            rendered = self._template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                bos_token=self._bos,
                eos_token=self._eos,
            )
        else:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            rendered = "".join(parts)
        return self.encode(rendered, add_special_tokens=False)


def load_tokenizer(tokenizer_path: str | None, tokenizer_revision: str = "main", allow_fallback: bool = False) -> Any:
    if not tokenizer_path or tokenizer_path == "fallback":
        return FallbackTokenizer()
    path = Path(tokenizer_path)
    if path.is_dir():
        tok_json = path / "tokenizer.json"
        if tok_json.is_file():
            try:
                from tokenizers import Tokenizer

                tok_obj = Tokenizer.from_file(str(tok_json))
                cfg_path = path / "tokenizer_config.json"
                cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
                tpl_path = path / "chat_template.jinja"
                tpl = tpl_path.read_text(encoding="utf-8") if tpl_path.is_file() else cfg.get("chat_template")
                if isinstance(tpl, list):
                    tpl = tpl[0].get("template") if tpl else None
                bos = str(cfg.get("bos_token") or "")
                eos = str(cfg.get("eos_token") or "")
                return TokenizersWrapper(tok_obj, tpl, bos, eos)
            except ImportError:
                pass
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, revision=tokenizer_revision, trust_remote_code=True)
    except Exception:
        if allow_fallback:
            return FallbackTokenizer()
        raise


def prompt_tokens(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def trim_turn(
    messages: list[dict[str, Any]],
    *,
    prefix: list[dict[str, Any]],
    prefix_tokens: int,
    tokenizer: Any,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    remaining = max_prompt_tokens - prefix_tokens
    delta: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "user" or not isinstance(message["content"], str):
            raise ValueError("Source conversations must contain text user messages")
        if remaining <= 0:
            break
        ids = tokenizer.encode(message["content"], add_special_tokens=False)
        item = dict(message)
        if len(ids) > remaining:
            item["content"] = tokenizer.decode(ids[:remaining], skip_special_tokens=False)
        delta.append(item)
        remaining -= min(len(ids), remaining)

    while delta:
        actual = prompt_tokens([*prefix, *delta], tokenizer)
        excess = actual - max_prompt_tokens
        if excess <= 0:
            return delta, actual
        ids = tokenizer.encode(delta[-1]["content"], add_special_tokens=False)
        if len(ids) <= excess:
            delta.pop()
        else:
            content = delta[-1]["content"]
            shortened = tokenizer.decode(ids[:-excess], skip_special_tokens=False)
            delta[-1]["content"] = shortened if len(shortened) < len(content) else content[:-1]
    raise ValueError("Token budget is too small for a nonempty user turn")


def _normalize_source_conversations(raw: Any, turns: int) -> list[list[dict[str, Any]]]:
    if isinstance(raw, dict) and "conversations" in raw:
        items = raw["conversations"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Unsupported source dataset JSON structure")
    normalized: list[list[dict[str, Any]]] = []
    for conv in items:
        if isinstance(conv, dict) and "turns" in conv:
            conv_turns = conv["turns"]
        elif isinstance(conv, dict) and "messages" in conv:
            conv_turns = [{"messages": [m]} for m in conv["messages"] if m.get("role") == "user"]
        elif isinstance(conv, list):
            conv_turns = conv
        else:
            raise ValueError("Unsupported conversation item in source dataset")
        if not conv_turns:
            continue
        padded = list(conv_turns)
        while len(padded) < turns:
            padded.append(conv_turns[len(padded) % len(conv_turns)])
        normalized.append(padded[:turns])
    return normalized


def load_source_dataset(
    source_path: Path | None,
    *,
    num_conversations: int,
    turns: int,
    first_turn_length: int,
    subsequent_turn_length: int,
) -> tuple[list[list[dict[str, Any]]], str]:
    if source_path is not None:
        raw_bytes = source_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        text = raw_bytes.decode("utf-8").strip()
        if source_path.suffix == ".jsonl" or ("\n" in text and not text.startswith("{")):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return _normalize_source_conversations(rows, turns), sha256
        return _normalize_source_conversations(json.loads(text), turns), sha256

    conversations: list[list[dict[str, Any]]] = []
    for idx in range(num_conversations):
        conv: list[dict[str, Any]] = []
        for turn_idx in range(turns):
            budget = first_turn_length if turn_idx == 0 else subsequent_turn_length
            repeat_count = max(32, (budget // 6) + 32)
            body_parts = [f"[task={idx} turn={turn_idx}]"]
            snippet = SYNTHETIC_SNIPPETS[(idx + turn_idx) % len(SYNTHETIC_SNIPPETS)]
            for k in range(repeat_count):
                body_parts.append(f"segment-{idx}-{turn_idx}-{k}: {snippet}")
            conv.append({"messages": [{"role": "user", "content": " ".join(body_parts)}]})
        conversations.append(conv)
    payload = json.dumps({"conversations": conversations}, sort_keys=True).encode("utf-8")
    return conversations, hashlib.sha256(payload).hexdigest()


def build_dataset(
    source_conversations: list[list[dict[str, Any]]],
    *,
    tokenizer: Any,
    model_path: str,
    tokenizer_revision: str,
    source_sha256: str,
    source_revision: str,
    first_turn_length: int,
    subsequent_turn_length: int,
    turns: int,
) -> dict[str, Any]:
    conversations = []
    for conversation in source_conversations:
        prefix: list[dict[str, Any]] = []
        out_turns = []
        previous_tokens = 0
        for turn_index, turn in enumerate(conversation[:turns]):
            target = first_turn_length if turn_index == 0 else previous_tokens + subsequent_turn_length
            msgs = turn["messages"] if isinstance(turn, dict) and "messages" in turn else [turn]
            delta, actual = trim_turn(
                msgs,
                prefix=prefix,
                prefix_tokens=previous_tokens,
                tokenizer=tokenizer,
                max_prompt_tokens=target,
            )
            prefix.extend(delta)
            out_turns.append({"messages": delta, "prompt_tokens": actual})
            previous_tokens = actual
        conversations.append(out_turns)
    first_lengths = [c[0]["prompt_tokens"] for c in conversations]
    return {
        "metadata": {
            "model_path": model_path,
            "tokenizer_revision": tokenizer_revision,
            "source_revision": source_revision,
            "source_sha256": source_sha256,
            "construction": "trim_each_turn_with_model_tokenizer_v1",
            "first_turn_length": first_turn_length,
            "subsequent_turn_length": subsequent_turn_length,
            "turns": turns,
            "num_conversations": len(conversations),
            "first_prompt_tokens_min": min(first_lengths),
            "first_prompt_tokens_max": max(first_lengths),
            "chat_template_included": True,
        },
        "conversations": conversations,
    }


def _meta_url_from_chat_url(chat_url: str) -> str:
    if chat_url.endswith("/v1/chat/completions"):
        return chat_url[: -len("/v1/chat/completions")] + "/v1/meta"
    return chat_url.rstrip("/") + "/v1/meta"


def fetch_server_meta(chat_url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    meta_url = _meta_url_from_chat_url(chat_url)
    try:
        with urllib.request.urlopen(meta_url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _send_turn_sse_sync(
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    extra_args: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    payload.update(extra_args)
    body_bytes = json.dumps(payload).encode("utf-8")
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn_cls: type[http.client.HTTPConnection]
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=600, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(parsed.netloc, timeout=600)

    started = time.perf_counter()
    first_token_t: float | None = None
    last_t: float | None = None
    gaps: list[float] = []
    chunks_text: list[str] = []
    usage: dict[str, Any] = {}
    spec_stats: dict[str, Any] | None = None

    try:
        conn.request("POST", path, body=body_bytes, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {err_body[:300]}")
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            event = json.loads(data_str)
            if "usage" in event and isinstance(event["usage"], dict):
                usage = event["usage"]
            if "speculative_stats" in event and isinstance(event["speculative_stats"], dict):
                spec_stats = event["speculative_stats"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text_piece = delta.get("content") or delta.get("reasoning_content") or ""
            if text_piece:
                now = time.perf_counter()
                if first_token_t is None:
                    first_token_t = now
                    last_t = now
                else:
                    assert last_t is not None
                    gaps.append(now - last_t)
                    last_t = now
                if isinstance(delta.get("content"), str):
                    chunks_text.append(delta["content"])
    finally:
        conn.close()

    ended = time.perf_counter()
    ttft_s = (first_token_t - started) if first_token_t is not None else (ended - started)
    prompt_toks = int(usage.get("prompt_tokens", 0))
    completion_toks = int(usage.get("completion_tokens", 0))
    details = usage.get("prompt_tokens_details") or {}
    cached_toks = int(details.get("cached_tokens", 0))
    return {
        "assistant_text": "".join(chunks_text),
        "ttft_s": ttft_s,
        "elapsed_s": ended - started,
        "gaps": gaps,
        "prompt_tokens": prompt_toks,
        "cached_tokens": cached_toks,
        "completion_tokens": completion_toks,
        "speculative_stats": spec_stats,
    }


def _run_concurrency_builtin(
    *,
    conversations: list[list[dict[str, Any]]],
    url: str,
    model: str,
    parallel: int,
    max_tokens: int,
    extra_args: dict[str, Any],
    run_dir: Path,
    num_gpus: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    server_meta = fetch_server_meta(url)
    if server_meta is not None:
        (run_dir / "server_meta.json").write_text(json.dumps(server_meta, indent=2), encoding="utf-8")

    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    failed = 0

    def run_one(conv: list[dict[str, Any]]) -> None:
        nonlocal failed
        messages: list[dict[str, Any]] = []
        for turn in conv:
            messages.extend(turn["messages"])
            try:
                turn_res = _send_turn_sse_sync(
                    url=url,
                    model=model,
                    messages=list(messages),
                    max_tokens=max_tokens,
                    extra_args=extra_args,
                )
                with lock:
                    results.append(turn_res)
                messages.append({"role": "assistant", "content": turn_res["assistant_text"]})
            except Exception:
                with lock:
                    failed += 1
                break

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(run_one, conversations))
    wall_s = max(time.perf_counter() - wall_start, 1e-9)

    db_path = run_dir / "benchmark_data.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE result ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "completion_tokens INTEGER, "
            "inter_token_latencies TEXT, "
            "prompt_tokens INTEGER, "
            "cached_tokens INTEGER, "
            "ttft_ms REAL, "
            "success INTEGER)"
        )
        for item in results:
            conn.execute(
                "INSERT INTO result (completion_tokens, inter_token_latencies, prompt_tokens, cached_tokens, ttft_ms, success) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (
                    item["completion_tokens"],
                    json.dumps(item["gaps"]),
                    item["prompt_tokens"],
                    item["cached_tokens"],
                    item["ttft_s"] * 1000.0,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_cached = sum(r["cached_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    all_gaps = [g for r in results for g in r["gaps"]]
    if all_gaps:
        avg_tpot_ms = (sum(all_gaps) / len(all_gaps)) * 1000.0
    elif total_completion > len(results) and results:
        total_decode_s = sum(max(r["elapsed_s"] - r["ttft_s"], 1e-6) for r in results)
        avg_tpot_ms = (total_decode_s / max(1, total_completion - len(results))) * 1000.0
    else:
        avg_tpot_ms = 0.0
    ttfts_ms = [r["ttft_s"] * 1000.0 for r in results]
    cache_hit_pct = (100.0 * total_cached / total_prompt) if total_prompt > 0 else 0.0
    output_throughput = total_completion / wall_s
    summary = {
        "Concurrency": parallel,
        "Total Requests": len(results) + failed,
        "Failed Requests": failed,
        "Num GPUs": num_gpus,
        "Avg TPOT (ms)": avg_tpot_ms,
        "TTFT P50 (ms)": _percentile(ttfts_ms, 50.0),
        "TTFT P95 (ms)": _percentile(ttfts_ms, 95.0),
        "Output Throughput (tok/s)": output_throughput,
        "KV Cache Hit Rate (%)": cache_hit_pct,
        "Total Prompt Tokens": total_prompt,
        "Total Cached Tokens": total_cached,
        "Total Completion Tokens": total_completion,
        "Wall Time (s)": wall_s,
    }
    (run_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def decoded_tok_per_iter(run_dir: Path) -> float:
    db_path = run_dir / "benchmark_data.db"
    if not db_path.is_file():
        return -1.0
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT completion_tokens, inter_token_latencies "
            "FROM result WHERE success = 1"
        ).fetchall()
    finally:
        connection.close()
    tokens = iterations = 0
    for completion_tokens, latencies in rows:
        gaps = len(json.loads(latencies))
        if completion_tokens and completion_tokens > 1 and gaps > 0:
            tokens += completion_tokens - 1
            iterations += gaps
    return tokens / iterations if iterations else -1.0


def _infer_ttft_percentiles(run_dir: Path, summary: dict[str, Any]) -> tuple[float, float]:
    if "TTFT P50 (ms)" in summary and "TTFT P95 (ms)" in summary:
        return float(summary["TTFT P50 (ms)"]), float(summary["TTFT P95 (ms)"])
    db_path = run_dir / "benchmark_data.db"
    if db_path.is_file():
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(result)").fetchall()}
            if "ttft_ms" in cols:
                vals = [float(r[0]) for r in conn.execute("SELECT ttft_ms FROM result WHERE success = 1").fetchall()]
                if vals:
                    return _percentile(vals, 50.0), _percentile(vals, 95.0)
            elif "ttft" in cols:
                vals = [float(r[0]) * 1000.0 for r in conn.execute("SELECT ttft FROM result WHERE success = 1").fetchall()]
                if vals:
                    return _percentile(vals, 50.0), _percentile(vals, 95.0)
        finally:
            conn.close()
    avg_ttft = summary.get("Avg TTFT (s)")
    if avg_ttft is not None:
        v = float(avg_ttft) * 1000.0
        return v, v
    avg_ttft_ms = summary.get("Avg TTFT (ms)", summary.get("TTFT (ms)", 0.0))
    return float(avg_ttft_ms), float(avg_ttft_ms)


def collect_sweep(
    sweep_dir: Path,
    *,
    config_name: str,
    expected_concurrencies: list[int],
    num_gpus: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    config_dir = sweep_dir / config_name
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    captured_meta: dict[str, Any] | None = None
    top_meta = sweep_dir / "server_meta.json"
    if top_meta.is_file():
        captured_meta = json.loads(top_meta.read_text(encoding="utf-8"))

    for summary_path in sorted(config_dir.rglob("benchmark_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["Failed Requests"]:
            raise ValueError(f"{summary_path}: benchmark contains failed requests")
        concurrency = int(summary["Concurrency"])
        if concurrency in seen:
            raise ValueError(f"{summary_path}: duplicate concurrency {concurrency}")
        seen.add(concurrency)

        run_meta_path = summary_path.parent / "server_meta.json"
        if run_meta_path.is_file() and captured_meta is None:
            captured_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

        effective_gpus = num_gpus
        if effective_gpus is None:
            if "Num GPUs" in summary:
                effective_gpus = int(summary["Num GPUs"])
            elif captured_meta is not None:
                effective_gpus = (
                    int(captured_meta.get("tensor_parallel", 1))
                    * int(captured_meta.get("expert_parallel", 1))
                    * int(captured_meta.get("pipeline_parallel", 1))
                )
            else:
                effective_gpus = 1
        effective_gpus = max(1, effective_gpus)

        tpot_key = "Avg TPOT (ms)" if "Avg TPOT (ms)" in summary else "TPOT (ms)"
        tpot_ms = float(summary[tpot_key])
        output_throughput = float(summary["Output Throughput (tok/s)"])
        if not math.isfinite(tpot_ms) or tpot_ms <= 0:
            raise ValueError(f"{summary_path}: TPOT must be finite and positive")
        if not math.isfinite(output_throughput) or output_throughput < 0:
            raise ValueError(f"{summary_path}: invalid output throughput")
        ttft_p50, ttft_p95 = _infer_ttft_percentiles(summary_path.parent, summary)
        cache_hit = float(summary.get("KV Cache Hit Rate (%)", 0.0))
        rows.append(
            {
                "config": config_name,
                "Conc.": concurrency,
                "Latency (tps/user)": round(1000.0 / tpot_ms, 2),
                "Output Throughput (tok/min/gpu)": round(output_throughput * 60.0 / effective_gpus, 2),
                "TTFT p50 (ms)": round(ttft_p50, 2),
                "TTFT p95 (ms)": round(ttft_p95, 2),
                "Approx Cache Hit": round(cache_hit, 2),
                "Decoded Tok/Iter": round(decoded_tok_per_iter(summary_path.parent), 4),
            }
        )
    if not rows or seen != set(expected_concurrencies):
        raise ValueError(
            f"Expected concurrencies {sorted(set(expected_concurrencies))}, found {sorted(seen)} in {config_dir}"
        )
    return sorted(rows, key=lambda row: row["Conc."]), captured_meta


def cmd_prepare(args: argparse.Namespace) -> int:
    prof = PROFILES[args.profile]
    first_len = args.first_turn_length or prof["first_turn_length"]
    sub_len = args.subsequent_turn_length or prof["subsequent_turn_length"]
    turns = args.turns or prof["turns"]
    source_path = Path(args.source_path) if args.source_path else None
    source_convs, source_sha256 = load_source_dataset(
        source_path,
        num_conversations=args.synthetic_count,
        turns=turns,
        first_turn_length=first_len,
        subsequent_turn_length=sub_len,
    )
    fingerprint = {
        "model_path": args.model_path,
        "tokenizer_revision": args.tokenizer_revision,
        "source_revision": args.source_revision,
        "source_sha256": source_sha256,
        "construction": "trim_each_turn_with_model_tokenizer_v1",
        "first_turn_length": first_len,
        "subsequent_turn_length": sub_len,
        "turns": turns,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        try:
            meta = json.loads(output_path.read_text(encoding="utf-8")).get("metadata", {})
            if all(meta.get(k) == v for k, v in fingerprint.items()):
                print(json.dumps(meta, indent=2))
                return 0
        except Exception:
            pass
    tokenizer = load_tokenizer(
        args.tokenizer_path or args.model_path,
        args.tokenizer_revision,
        allow_fallback=args.allow_fallback_tokenizer,
    )
    dataset = build_dataset(
        source_convs,
        tokenizer=tokenizer,
        model_path=args.model_path,
        tokenizer_revision=args.tokenizer_revision,
        source_sha256=source_sha256,
        source_revision=args.source_revision,
        first_turn_length=first_len,
        subsequent_turn_length=sub_len,
        turns=turns,
    )
    tmp = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)
    print(json.dumps(dataset["metadata"], indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prof = PROFILES[args.profile]
    concurrencies = list(args.concurrencies or prof["concurrencies"])
    config_name = args.name or prof["config"]
    max_tokens = args.max_tokens or prof["max_tokens"]
    sweep_dir = Path(args.sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    if not args.skip_dataset_check:
        expected_first = args.first_turn_length or prof["first_turn_length"]
        expected_sub = args.subsequent_turn_length or prof["subsequent_turn_length"]
        meta = dataset.get("metadata", {})
        if (
            meta.get("first_turn_length") != expected_first
            or meta.get("subsequent_turn_length") != expected_sub
            or meta.get("chat_template_included") is not True
        ):
            raise SystemExit("Dataset metadata does not match profile token budgets")

    conversations = dataset["conversations"]
    numbers = list(args.numbers) if args.numbers else [min(len(conversations), max(c * 2, 4)) for c in concurrencies]
    if len(numbers) != len(concurrencies):
        raise SystemExit("--numbers length must match --concurrencies")

    meta = fetch_server_meta(args.url)
    if meta is not None:
        (sweep_dir / "server_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    num_gpus = args.num_gpus
    if num_gpus is None and meta is not None:
        num_gpus = (
            int(meta.get("tensor_parallel", 1))
            * int(meta.get("expert_parallel", 1))
            * int(meta.get("pipeline_parallel", 1))
        )
    if num_gpus is None:
        num_gpus = 1

    extra_args = json.loads(args.extra_args) if args.extra_args else {}
    if args.runner == "evalscope":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer-path is required with --runner evalscope")
        merged_extra = {"ignore_eos": True}
        merged_extra.update(extra_args)
        extra_str = json.dumps(merged_extra)
        if args.warmup_number > 0:
            warmup_offset = max(0, len(conversations) - args.warmup_number)
            subprocess.run(
                [
                    args.evalscope_bin,
                    "perf",
                    "--model",
                    args.model,
                    "--tokenizer-path",
                    args.tokenizer_path,
                    "--url",
                    args.url,
                    "--api",
                    "openai",
                    "--dataset",
                    "swe_smith",
                    "--dataset-path",
                    str(args.dataset_path),
                    "--max-tokens",
                    str(max_tokens),
                    "--multi-turn",
                    "--number",
                    str(args.warmup_number),
                    "--parallel",
                    str(min(args.warmup_number, concurrencies[0])),
                    "--dataset-offset",
                    str(warmup_offset),
                    "--extra-args",
                    extra_str,
                    "--outputs-dir",
                    str(sweep_dir / ".warmup"),
                ],
                check=True,
            )
        subprocess.run(
            [
                args.evalscope_bin,
                "perf",
                "--model",
                args.model,
                "--tokenizer-path",
                args.tokenizer_path,
                "--url",
                args.url,
                "--api",
                "openai",
                "--dataset",
                "swe_smith",
                "--dataset-path",
                str(args.dataset_path),
                "--max-tokens",
                str(max_tokens),
                "--multi-turn",
                "--number",
                *[str(n) for n in numbers],
                "--parallel",
                *[str(c) for c in concurrencies],
                "--extra-args",
                extra_str,
                "--name",
                config_name,
                "--outputs-dir",
                str(sweep_dir),
                "--no-timestamp",
            ],
            check=True,
        )
    else:
        if args.warmup_number > 0 and len(conversations) > 0:
            warmup_convs = conversations[-args.warmup_number :]
            _run_concurrency_builtin(
                conversations=warmup_convs,
                url=args.url,
                model=args.model,
                parallel=min(len(warmup_convs), concurrencies[0]),
                max_tokens=max_tokens,
                extra_args=extra_args,
                run_dir=sweep_dir / ".warmup",
                num_gpus=num_gpus,
            )
        for c, n in zip(concurrencies, numbers):
            selected = conversations[:n]
            run_dir = sweep_dir / config_name / f"parallel_{c}_number_{n}"
            _run_concurrency_builtin(
                conversations=selected,
                url=args.url,
                model=args.model,
                parallel=c,
                max_tokens=max_tokens,
                extra_args=extra_args,
                run_dir=run_dir,
                num_gpus=num_gpus,
            )

    rows, captured_meta = collect_sweep(
        sweep_dir,
        config_name=config_name,
        expected_concurrencies=concurrencies,
        num_gpus=num_gpus,
    )
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = sweep_dir / "summary.json"
    json_path.write_text(json.dumps({"rows": rows, "server_meta": captured_meta}, indent=2), encoding="utf-8")
    print(csv_path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    rows, captured_meta = collect_sweep(
        Path(args.sweep_dir),
        config_name=args.config,
        expected_concurrencies=list(args.expected_concurrencies),
        num_gpus=args.num_gpus,
    )
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps({"rows": rows, "server_meta": captured_meta}, indent=2),
            encoding="utf-8",
        )
    if args.output:
        with Path(args.output).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shelf AgentX / EvalScope swe_smith benchmark driver")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--profile", choices=sorted(PROFILES), default="single-r9700")
    prep.add_argument("--source-path", default=None)
    prep.add_argument("--synthetic-count", type=int, default=32)
    prep.add_argument("--output-path", required=True)
    prep.add_argument("--model-path", default="shelf")
    prep.add_argument("--tokenizer-path", default=None)
    prep.add_argument("--tokenizer-revision", default="main")
    prep.add_argument("--source-revision", default="local")
    prep.add_argument("--first-turn-length", type=int, default=None)
    prep.add_argument("--subsequent-turn-length", type=int, default=None)
    prep.add_argument("--turns", type=int, default=None)
    prep.add_argument("--allow-fallback-tokenizer", action="store_true")
    prep.add_argument("--force", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--profile", choices=sorted(PROFILES), default="single-r9700")
    run.add_argument("--runner", choices=("builtin", "evalscope"), default="builtin")
    run.add_argument("--evalscope-bin", default="evalscope")
    run.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    run.add_argument("--model", default="shelf")
    run.add_argument("--tokenizer-path", default=None)
    run.add_argument("--dataset-path", required=True)
    run.add_argument("--sweep-dir", required=True)
    run.add_argument("--name", default=None)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--first-turn-length", type=int, default=None)
    run.add_argument("--subsequent-turn-length", type=int, default=None)
    run.add_argument("--concurrencies", type=int, nargs="+", default=None)
    run.add_argument("--numbers", type=int, nargs="+", default=None)
    run.add_argument("--warmup-number", type=int, default=0)
    run.add_argument("--num-gpus", type=int, default=None)
    run.add_argument("--extra-args", default=None)
    run.add_argument("--skip-dataset-check", action="store_true")

    col = sub.add_parser("collect")
    col.add_argument("sweep_dir")
    col.add_argument("--config", default="single_r9700")
    col.add_argument("--expected-concurrencies", type=int, nargs="+", required=True)
    col.add_argument("--num-gpus", type=int, default=None)
    col.add_argument("-o", "--output", default=None)
    col.add_argument("--json-output", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
