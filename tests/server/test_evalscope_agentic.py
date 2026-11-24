#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "bench" / "evalscope_agentic.py"
SPEC = importlib.util.spec_from_file_location("evalscope_agentic", SCRIPT)
agentic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(agentic)


@contextlib.contextmanager
def mock_openai_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/v1/meta":
                body = json.dumps({
                    "model": "hy4-mock",
                    "tensor_parallel": 1,
                    "expert_parallel": 3,
                    "pipeline_parallel": 1,
                    "prefix_cache": True,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.requests.append(req)
            msgs = req.get("messages") or []
            user_turns = sum(1 for m in msgs if m.get("role") == "user")
            prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in msgs) + 10
            cached_tokens = int(prompt_tokens * 0.75) if user_turns > 1 else 0
            max_tokens = int(req.get("max_tokens", 4))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(max_tokens):
                chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.002)
            final = {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                },
            }
            self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    class HighBacklogServer(ThreadingHTTPServer):
        request_queue_size = 128
        allow_reuse_address = True

    srv = HighBacklogServer(("127.0.0.1", 0), Handler)
    srv.lock = threading.Lock()
    srv.requests = []
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", srv
    finally:
        srv.shutdown()
        srv.server_close()


class EvalScopeAgenticTests(unittest.TestCase):
    def test_prepare_synthetic_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_syn = tmp / "syn.json"
            rc = agentic.main([
                "prepare",
                "--profile", "single-r9700",
                "--synthetic-count", "4",
                "--first-turn-length", "64",
                "--subsequent-turn-length", "24",
                "--turns", "3",
                "--tokenizer-path", "fallback",
                "--output-path", str(out_syn),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(out_syn.read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["turns"], 3)
            self.assertEqual(len(data["conversations"]), 4)
            self.assertLessEqual(data["conversations"][0][0]["prompt_tokens"], 64)
            self.assertEqual(len(data["conversations"][0]), 3)

            src_jsonl = tmp / "src.jsonl"
            lines = [
                json.dumps([{"messages": [{"role": "user", "content": "alpha " * 80}]},
                            {"messages": [{"role": "user", "content": "beta " * 40}]}]),
                json.dumps([{"messages": [{"role": "user", "content": "gamma " * 80}]}]),
            ]
            src_jsonl.write_text("\n".join(lines), encoding="utf-8")
            out_jsonl = tmp / "from_jsonl.json"
            rc2 = agentic.main([
                "prepare",
                "--source-path", str(src_jsonl),
                "--first-turn-length", "48",
                "--subsequent-turn-length", "16",
                "--turns", "2",
                "--tokenizer-path", "fallback",
                "--output-path", str(out_jsonl),
            ])
            self.assertEqual(rc2, 0)
            data2 = json.loads(out_jsonl.read_text(encoding="utf-8"))
            self.assertEqual(len(data2["conversations"]), 2)
            self.assertEqual(len(data2["conversations"][1]), 2)

    def test_run_builtin_and_collect(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock_openai_server() as (url, srv):
            tmp = Path(tmpdir)
            ds_path = tmp / "dataset.json"
            agentic.main([
                "prepare",
                "--synthetic-count", "4",
                "--first-turn-length", "48",
                "--subsequent-turn-length", "16",
                "--turns", "3",
                "--tokenizer-path", "fallback",
                "--output-path", str(ds_path),
            ])
            sweep_dir = tmp / "sweep"
            rc = agentic.main([
                "run",
                "--runner", "builtin",
                "--url", url,
                "--model", "hy4-mock",
                "--dataset-path", str(ds_path),
                "--sweep-dir", str(sweep_dir),
                "--name", "single_r9700",
                "--max-tokens", "4",
                "--first-turn-length", "48",
                "--subsequent-turn-length", "16",
                "--concurrencies", "1", "2",
                "--numbers", "2", "3",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((sweep_dir / "summary.csv").is_file())
            self.assertTrue((sweep_dir / "summary.json").is_file())
            for req in srv.requests:
                self.assertTrue(req.get("ignore_eos"))
                self.assertTrue(req.get("stream"))
                self.assertEqual(req.get("stream_options"), {"include_usage": True})
            turn_counts = [len(req["messages"]) for req in srv.requests]
            self.assertEqual(sorted(set(turn_counts)), [1, 3, 5])

            with (sweep_dir / "summary.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([int(r["Conc."]) for r in rows], [1, 2])
            for row in rows:
                self.assertGreater(float(row["Latency (tps/user)"]), 0.0)
                self.assertGreater(float(row["Output Throughput (tok/min/gpu)"]), 0.0)
                self.assertGreater(float(row["Approx Cache Hit"]), 0.0)
                self.assertGreater(float(row["TTFT p50 (ms)"]), 0.0)
                self.assertGreater(float(row["TTFT p95 (ms)"]), 0.0)

            summary_json = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["server_meta"]["expert_parallel"], 3)

    def test_c32_profile_and_cli_concurrencies(self):
        parser = agentic.build_parser()
        args_prof = parser.parse_args([
            "run",
            "--profile", "hy4-c32",
            "--dataset-path", "d.json",
            "--sweep-dir", "s",
        ])
        self.assertEqual(agentic.PROFILES[args_prof.profile]["concurrencies"], [1, 4, 8, 16, 32])
        args_cli = parser.parse_args([
            "run",
            "--profile", "hy4-light",
            "--dataset-path", "d.json",
            "--sweep-dir", "s",
            "--concurrencies", "1", "4", "8", "16", "32",
        ])
        self.assertEqual(args_cli.concurrencies, [1, 4, 8, 16, 32])

        with tempfile.TemporaryDirectory() as tmpdir, mock_openai_server() as (url, srv):
            tmp = Path(tmpdir)
            ds_path = tmp / "ds32.json"
            agentic.main([
                "prepare",
                "--profile", "hy4-c32",
                "--synthetic-count", "32",
                "--first-turn-length", "32",
                "--subsequent-turn-length", "12",
                "--turns", "2",
                "--tokenizer-path", "fallback",
                "--output-path", str(ds_path),
            ])
            sweep_dir = tmp / "sweep32"
            rc = agentic.main([
                "run",
                "--profile", "hy4-c32",
                "--runner", "builtin",
                "--url", url,
                "--model", "hy4-mock",
                "--dataset-path", str(ds_path),
                "--sweep-dir", str(sweep_dir),
                "--max-tokens", "2",
                "--first-turn-length", "32",
                "--subsequent-turn-length", "12",
                "--concurrencies", "1", "32",
                "--numbers", "2", "32",
            ])
            self.assertEqual(rc, 0)
            with (sweep_dir / "summary.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([int(r["Conc."]) for r in rows], [1, 32])

    def test_evalscope_runner_command_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_path = tmp / "ds.json"
            agentic.main([
                "prepare",
                "--profile", "hy4-light",
                "--synthetic-count", "4",
                "--first-turn-length", "32",
                "--subsequent-turn-length", "12",
                "--turns", "2",
                "--tokenizer-path", "fallback",
                "--output-path", str(ds_path),
            ])
            sweep_dir = tmp / "sweep"
            run_dir = sweep_dir / "hy4_light" / "p1"
            run_dir.mkdir(parents=True)
            (run_dir / "benchmark_summary.json").write_text(json.dumps({
                "Concurrency": 1,
                "Failed Requests": 0,
                "Avg TPOT (ms)": 10.0,
                "Output Throughput (tok/s)": 100.0,
                "KV Cache Hit Rate (%)": 50.0,
                "TTFT P50 (ms)": 20.0,
                "TTFT P95 (ms)": 30.0,
                "Num GPUs": 1,
            }), encoding="utf-8")

            calls = []

            def fake_run(cmd, check=False):
                calls.append(cmd)
                return None

            with patch.object(agentic.subprocess, "run", side_effect=fake_run):
                rc = agentic.main([
                    "run",
                    "--profile", "hy4-light",
                    "--runner", "evalscope",
                    "--evalscope-bin", "/usr/bin/evalscope",
                    "--tokenizer-path", "/tmp/tok",
                    "--dataset-path", str(ds_path),
                    "--sweep-dir", str(sweep_dir),
                    "--first-turn-length", "32",
                    "--subsequent-turn-length", "12",
                    "--concurrencies", "1",
                    "--numbers", "4",
                    "--warmup-number", "1",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("--dataset", calls[1])
            self.assertIn("swe_smith", calls[1])
            self.assertIn("--multi-turn", calls[1])
            idx = calls[1].index("--extra-args")
            self.assertTrue(json.loads(calls[1][idx + 1])["ignore_eos"])


if __name__ == "__main__":
    unittest.main()
