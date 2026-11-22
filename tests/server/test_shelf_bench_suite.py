import contextlib
import hashlib
import importlib.util
import json
import pathlib
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('bench_suite', ROOT / 'tools/bench/shelf_bench_suite.py')
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        self.server.request_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        events = [
            {'choices': [{'delta': {'content': 'First result.'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': ' Second result.'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 2}},
        ]
        for event in events:
            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
            self.wfile.flush()
        self.wfile.write(b'data: [DONE]\n\n')


class SuiteTest(unittest.TestCase):
    def test_corpus(self):
        manifest, tasks = SUITE.corpus()
        self.assertEqual(len(manifest['cells']), 30)
        self.assertEqual(len(tasks), 30)
        self.assertEqual({x['language'] for x in tasks}, {'ja', 'en'})
        self.assertTrue(all(x['max_tokens'] >= 2048 and x['thinking'] for x in tasks))
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((SUITE.CORPUS / name).read_bytes()).hexdigest(), expected)
        for tier in ('short_2k', 'mid_8k', 'long_16k'):
            lines = [x for x in (SUITE.CORPUS / (tier + '.txt')).read_text().splitlines() if x.strip()]
            self.assertGreater(len(set(lines)) / len(lines), 0.6)

    def test_loops(self):
        self.assertFalse(SUITE.loop_metrics('これは技術的な説明です。\n所有権と寿命を確認します。\n異常時の経路を検証します。')['loop_flagged'])
        self.assertTrue(SUITE.loop_metrics(('same line\n' * 4))['loop_flagged'])
        self.assertTrue(SUITE.loop_metrics(' '.join([f'token{i}' for i in range(23)] * 4))['loop_flagged'])

    def test_stream_and_missing_macro(self):
        manifest, tasks = SUITE.corpus()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = SUITE.stream(f'http://127.0.0.1:{server.server_port}/v1', 'test', tasks[0], 42)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertTrue(result['ok'])
        self.assertFalse(result['completion_tokens_approximate'])
        self.assertEqual(result['completion_tokens'], 2)
        self.assertEqual(server.request_body['max_tokens'], 2048)
        self.assertIsNone(SUITE.summarize(manifest, tasks[:1], [{'task_id': tasks[0]['id'], **result}])['macro_decode_tok_s'])


if __name__ == '__main__':
    unittest.main()
