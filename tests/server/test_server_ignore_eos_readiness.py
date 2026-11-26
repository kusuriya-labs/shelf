#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import site
import sys
import unittest
from pathlib import Path

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

from aiohttp.test_utils import TestClient, TestServer

SERVER_DIR = Path(__file__).resolve().parents[2] / "src" / "apps" / "server"
sys.path.insert(0, str(SERVER_DIR))

import shelf_server
from serverlib import chat_service, openai_protocol, responses_protocol
from serverlib.compute_client import ComputeCapabilities, ComputeKVStats


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return [101, 102, 103, 104]

    def decode(self, token_ids, skip_special_tokens=True):
        out = []
        for t in token_ids:
            out.append("</think>" if t == 999 else f"tok{t}")
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [999]
        return [10]


class FakeCompute:
    def __init__(self):
        self.alive = True
        self.calls = []
        self.capabilities = ComputeCapabilities(prefix_cache=True)
        self.inflight = 0
        self.queued = 0
        self.last_kv_stats = None

    def is_alive(self) -> bool:
        return self.alive

    async def kv_stats(self):
        return ComputeKVStats(
            page_tokens=4096,
            total_pages=4,
            used_pages=0,
            free_pages=4,
            capacity_tokens=16384,
            free_tokens=16384,
            active_requests=0,
            queued_requests=0,
            banker_safe=True,
            overcommit_pages=0,
            draft_slots=[{"slot": 0, "draft_tokens": 2}, {"slot": 1, "draft_tokens": 5}],
        )

    async def shutdown(self):
        return None

    async def generate(self, input_ids, max_new_tokens, **kwargs):
        self.calls.append(("generate", list(input_ids), max_new_tokens, dict(kwargs)))
        return {
            "event": "done",
            "generated_ids": [200 + i for i in range(max_new_tokens)],
            "finish_reason": "length" if kwargs.get("ignore_eos") else "stop",
            "restored_tokens": 128,
            "prompt_tokens": len(input_ids),
            "prefill_tokens": len(input_ids),
        }

    async def generate_stream(self, input_ids, max_new_tokens, **kwargs):
        self.calls.append(("stream", list(input_ids), max_new_tokens, dict(kwargs)))
        for i in range(max_new_tokens):
            yield {"token": 200 + i}
        yield {
            "done": {
                "event": "done",
                "generated_ids": [200 + i for i in range(max_new_tokens)],
                "finish_reason": "length" if kwargs.get("ignore_eos") else "stop",
                "restored_tokens": 256,
                "prompt_tokens": len(input_ids),
                "prefill_tokens": len(input_ids),
            }
        }


class IgnoreEosAndReadinessTests(unittest.IsolatedAsyncioTestCase):
    def _make_app(self):
        compute = FakeCompute()
        service = chat_service.ChatService(
            processor=FakeProcessor(),
            compute=compute,
            model_name="hy4-test",
            max_seq_len=4096,
            inference_budget_tokens=1024,
            vocab_size=120832,
        )
        meta = {
            "model": "hy4-test",
            "speculative": {"method": "dfly2", "drafts": 4, "adaptive": True},
            "max_seq_len": 4096,
            "max_concurrent_requests": 4,
            "prefix_cache_strategy": "slot",
        }
        return shelf_server.create_app(service, compute, meta), compute

    def test_parse_ignore_eos(self):
        response = responses_protocol.parse_responses_request(
            {"model": "m", "input": "hello", "ignore_eos": True}, inference_budget_tokens=32)
        self.assertTrue(response.ignore_eos)
        with self.assertRaises(openai_protocol.OpenAIProtocolError):
            responses_protocol.parse_responses_request(
                {"model": "m", "input": "hello", "ignore_eos": 1}, inference_budget_tokens=32)

        req_default = openai_protocol.parse_chat_request({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertFalse(req_default.ignore_eos)

        req_true = openai_protocol.parse_chat_request({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "ignore_eos": True,
        })
        self.assertTrue(req_true.ignore_eos)

        with self.assertRaises(openai_protocol.OpenAIProtocolError):
            openai_protocol.parse_chat_request({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "ignore_eos": "true",
            })

    async def test_readiness_and_meta(self):
        app, compute = self._make_app()
        async with TestClient(TestServer(app)) as client:
            r_ok = await client.get("/readiness")
            self.assertEqual(r_ok.status, 200)
            self.assertEqual(await r_ok.json(), {"ready": True})

            compute.alive = False
            r_down = await client.get("/readiness")
            self.assertEqual(r_down.status, 503)
            self.assertEqual(await r_down.json(), {"ready": False})

            compute.alive = True
            meta = await client.get("/v1/meta")
            self.assertEqual(meta.status, 200)
            payload = await meta.json()
            self.assertEqual(payload["model"], "hy4-test")
            self.assertTrue(payload["prefix_cache"])
            self.assertEqual(payload["speculative"]["drafts"], 4)
            self.assertEqual(payload["speculative"]["slots"],
                             [{"slot": 0, "draft_tokens": 2}, {"slot": 1, "draft_tokens": 5}])
            self.assertFalse(payload["speculative"]["stale"])

    async def test_chat_ignore_eos_and_cached_tokens(self):
        app, compute = self._make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/chat/completions", json={
                "model": "hy4-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "ignore_eos": True,
            })
            self.assertEqual(resp.status, 200)
            data = await resp.json()
            self.assertEqual(data["choices"][0]["finish_reason"], "length")
            self.assertEqual(data["usage"]["completion_tokens"], 5)
            self.assertEqual(data["usage"]["prompt_tokens_details"]["cached_tokens"], 128)
            self.assertEqual(compute.calls[-1][0], "generate")
            self.assertTrue(compute.calls[-1][3]["ignore_eos"])

            sresp = await client.post("/v1/chat/completions", json={
                "model": "hy4-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            })
            self.assertEqual(sresp.status, 200)
            raw = await sresp.text()
            events = []
            for block in raw.strip().split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        events.append(json.loads(line[6:]))
            usage_events = [e for e in events if "usage" in e]
            self.assertTrue(usage_events)
            self.assertEqual(usage_events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 256)
            self.assertEqual(compute.calls[-1][0], "stream")
            self.assertTrue(compute.calls[-1][3]["ignore_eos"])

            bad = await client.post("/v1/chat/completions", json={
                "model": "hy4-test",
                "messages": [{"role": "user", "content": "hello"}],
                "ignore_eos": 1,
            })
            self.assertEqual(bad.status, 400)


if __name__ == "__main__":
    unittest.main()
