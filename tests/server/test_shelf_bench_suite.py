import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/bench/shelf_bench_suite.py'
SPEC = importlib.util.spec_from_file_location('bench_suite', SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


@contextlib.contextmanager
def server(output):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with self.server.lock:
                index = len(self.server.bodies)
                self.server.bodies.append(request)
                self.server.active += 1
                self.server.peak = max(self.server.peak, self.server.active)
            try:
                if not self.path.endswith('/v1/chat/completions'):
                    self.send_error(404)
                    return
                chunks, count, finish = output(self.path, request, index)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for chunk in chunks:
                    self.event({'choices': [{'delta': {'content': chunk}}]})
                    time.sleep(0.002)
                event = {
                    'choices': [{'delta': {}, 'finish_reason': finish}],
                    'speculative_stats': dict(zip(suite.ROUND_FIELDS, (12, 2, 3))),
                }
                if count is not None:
                    event['usage'] = {'completion_tokens': count, 'prompt_tokens': 2015}
                self.event(event)
                if finish:
                    self.wfile.write(b'data: [DONE]\n\n')
            finally:
                with self.server.lock:
                    self.server.active -= 1

        def event(self, value):
            self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
            self.wfile.flush()

    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    srv.bodies = []
    srv.lock = threading.Lock()
    srv.active = srv.peak = 0
    thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.02))
    thread.start()
    try:
        yield srv, 'http://127.0.0.1:' + str(srv.server_port)
    finally:
        srv.shutdown()
        thread.join()
        srv.server_close()


def long_reply(path, body, i):
    return [' alpha' * 512, ' beta' * 512], 1024, 'length'


def run(*argv):
    out = io.StringIO()
    with patch.object(sys, 'argv', [str(SCRIPT), *argv]), contextlib.redirect_stdout(out):
        suite.main()
    value = out.getvalue()
    if value.startswith('#'):
        return value
    return json.loads(value) if value else None


def betterbench(passes=5, warmup=1):
    config = {
        'runs_per_category': passes,
        'warmup': warmup,
        'greedy': False,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 20,
        'seed': None,
        'unique_nonce': True,
    }
    return {
        'config': config,
        'env': {'model': 'target', 'corpus_hash': 'v1-content-ids'},
        'betterbench_version': '1',
        'corpus_version': 'v1',
        'single_stream': {
            c: [
                {
                    'ok': i != 0,
                    'decode_tps': i * 10,
                    'ttft_ms': i * 2,
                    'completion_tokens': 48,
                    'finish_reason': 'length' if i == 1 else 'stop',
                }
                for i in range(passes)
            ]
            for c in suite.CATEGORIES
        },
    }


class SuiteTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)

    def save(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data))
        return str(path)

    def test_corpus(self):
        manifest, tasks = suite.corpus()
        cells = ['domain', 'language', 'length_tier']
        expected = set(
            itertools.product(
                ('code_edit', 'code_review', 'design_debug', 'tech_prose', 'multi_turn'),
                ('en', 'ja'),
                ('short_2k', 'mid_8k', 'long_16k'),
            )
        )
        for objects in (tasks, manifest['cells']):
            self.assertEqual(len(objects), 30)
            self.assertEqual({tuple(t[k] for k in cells) for t in objects}, expected)
        self.assertEqual(len({t['id'] for t in tasks}), 30)
        for t in tasks:
            self.assertGreaterEqual(t['max_tokens'], 2048)
            self.assertGreaterEqual(t['min_expected_tokens'], 1024)
            self.assertTrue(t['thinking'])
            self.assertIn(t['context_file'], manifest['context_tokens'])
            if t['domain'] == 'multi_turn':
                self.assertEqual([m['role'] for m in t['messages']], ['user', 'assistant', 'user'])
                self.assertTrue(all(m['content'] for m in t['messages']))
        self.assertEqual(set(manifest['files']), {p.name for p in suite.CORPUS.iterdir()} - {'manifest.json'})
        for name, expected_hash in manifest['files'].items():
            self.assertEqual(hashlib.sha256((suite.CORPUS / name).read_bytes()).hexdigest(), expected_hash)
        for name, lo, hi in [('short_2k', 1500, 3000), ('mid_8k', 6000, 10000), ('long_16k', 14000, 18000)]:
            filename = name + '.txt'
            sections = (suite.CORPUS / filename).read_text().strip().split('--- src/')
            sources = set()
            for section in sections[1:]:
                filepath, body = section.split(' ---\n', 1)
                self.assertNotIn(filepath, sources)
                sources.add(filepath)
                lines = body.strip().splitlines()
                windows = [tuple(lines[i : i + 16]) for i in range(len(lines) - 15)]
                self.assertGreaterEqual(len(set(windows)) / len(windows), 0.98)
            self.assertGreaterEqual(len(sources), 2)
            self.assertTrue(lo <= manifest['context_tokens'][filename] <= hi)
            self.assertTrue(
                all(lo <= t['reference_prompt_tokens'] <= hi for t in tasks if t['length_tier'] == name)
            )

    def test_loops(self):
        for text in [
            'これは技術的な説明です。\n所有権と寿命を確認します。\n異常時の経路を検証します。',
            'for item in values:\n    if item.valid:\n        result.append(item.key)\nreturn result',
            'if (ready) {\n    emit();\n}\n}\n}\n}',
        ]:
            self.assertFalse(suite.loop_metrics(text)['loop_flagged'])
        for text in ['同じ行です\n' * 4, '確認作業' * 8, ' '.join([f'token{i}' for i in range(23)] * 4)]:
            self.assertTrue(suite.loop_metrics(text)['loop_flagged'])

    def test_track_a_native_and_ab(self):
        binary = self.tmp / 'shelf-bench'
        binary.write_text(
            '#!/usr/bin/env python3\nimport sys\n'
            'if sys.argv[1] == "batch":\n'
            '    print("step_tokens=9 steps=3 wall_ms=80.0")\n    for i in range(4): print(f"GEN[{i}]=1,2 (n=2)")\n'
            'else:\n    print("gpu_tokens_per_sec=120.00\\nwall_tokens_per_sec=100.00")\n'
        )
        binary.chmod(0o755)
        paths = []

        def reply(path, body, i):
            paths.append(path.split('/')[1])
            return long_reply(path, body, i)

        with server(reply) as (srv, url):
            data = run(
                'track-a',
                '--shelf-bench',
                str(binary),
                '--model-dir',
                'fixture',
                '--contexts',
                '2048',
                '--batch-sizes',
                '1,4',
                '--pp-lengths',
                '128',
                '--ab-compare',
                '--spec-endpoint',
                url + '/candidate/v1',
                '--baseline-spec-endpoint',
                url + '/baseline/v1',
                '--repeats',
                '2',
            )
        self.assertEqual(paths, ['baseline', 'candidate', 'candidate', 'baseline'])
        self.assertEqual(data['hash_mismatches'], 0)
        self.assertEqual(data['failures'], 0)
        results = {r['case']: r for r in data['results']}
        self.assertEqual(set(results), {'pp/128', 'tg/2048/1', 'batch/2048/4', 'spec'})
        self.assertEqual(results['tg/2048/1']['gpu_tok_s'], 120)
        self.assertEqual(results['batch/2048/4']['wall_tok_s'], 100)
        self.assertEqual(results['spec']['ms_per_round'], 12)
        self.assertEqual(results['spec']['completion_tokens'], 1024)
        self.assertTrue(all(b['max_tokens'] == 1024 and b['temperature'] == 0 for b in srv.bodies))
        self.assertTrue(
            all(c['sha256_match'] and c['candidate_over_baseline'] > 0 for c in data['comparisons'])
        )

    def test_spec_mismatch_and_short_output(self):
        def reply(path, body, index):
            return [path, ' done'], 1024, 'stop'

        with server(reply) as (_, url):
            data = run(
                'track-a',
                '--ab-compare',
                '--spec-endpoint',
                url + '/b/v1',
                '--baseline-spec-endpoint',
                url + '/a/v1',
                '--repeats',
                '1',
                '--spec-max-tokens',
                '2048',
            )
        self.assertEqual(data['hash_mismatches'], 1)
        self.assertIsNone(data['comparisons'][0]['candidate_over_baseline'])
        for count in (1023, None):
            with server(lambda *args: (['early', ' EOS'], count, 'stop')) as (_, url):
                data = run('track-a', '--spec-endpoint', url + '/v1', '--repeats', '1')
            self.assertEqual(data['failures'], 1)
            self.assertFalse(data['results'][0]['meets_min_tokens'])

    def test_spec_command(self):
        script = self.tmp / 'generate.py'
        script.write_text(
            'import sys,json\nbody=json.load(sys.stdin)\n'
            'print(json.dumps(dict(output_text="測定", completion_tokens=body["max_tokens"], decode_tok_s=10)))\n'
            'print("DFLASH2_ROUNDS=2\\nDFLASH2_ACCEPTED_DRAFTS=6\\nDFLASH2_GENERATED_TOKENS=9\\nDFLASH2_ROUND_MS=34",file=sys.stderr)\n'
        )
        record = run('track-a', '--spec-cmd', f'{sys.executable} {script}', '--repeats', '1')['results'][0]
        self.assertEqual([record[k] for k in suite.ROUND_FIELDS], [17, 3, 4.5])
        self.assertEqual(record['output_sha256'], hashlib.sha256('測定'.encode()).hexdigest())

    def test_betterbench_import_and_logs(self):
        path = self.save('betterbench.json', betterbench())
        log = self.tmp / 'serve.log'
        log.write_text(
            'DFLASH2_ROUNDS=4\nDFLASH2_ACCEPTED_DRAFTS=7\nDFLASH2_GENERATED_TOKENS=10\n'
            'DFLASH2_ROUND_MS=28\nDFLASH2_STAGE round=7.000 draft=4.0\n'
        )
        data = run(
            'track-b', '--import-json', path, '--server-log', 'code=' + str(log), '--server-log', str(log)
        )
        self.assertEqual(data['mode'], 'quick_smoke')
        self.assertEqual(set(data['categories']), set(suite.CATEGORIES))
        code = data['categories']['code']
        self.assertEqual(code['decode_tok_s'], dict(median=25, iqr=15, min=10, max=40))
        self.assertEqual(code['ttft_ms']['median'], 5)
        self.assertEqual([code[k] for k in suite.ROUND_FIELDS], [7, 1.75, 2.5])
        self.assertEqual(code['failure_rate'], 0.2)
        self.assertEqual(code['truncation_rate'], 0.2)
        self.assertIsNone(data['categories']['prose']['ms_per_round'])
        self.assertEqual(data['global_server_stats']['ms_per_round'], 7)
        raw = betterbench()
        del raw['single_stream']['code']
        raw['single_stream']['chat'][3]['completion_tokens'] = None
        data = run('track-b', '--import-json', self.save('less.json', raw))
        self.assertIsNone(data['categories']['code']['failure_rate'])
        self.assertEqual(data['categories']['chat']['missing_usage'], 1)

    def test_betterbench_wrapper(self):
        binary = self.tmp / 'python'
        src = self.save('raw.json', betterbench(20, 3))
        binary.write_text(
            '#!/usr/bin/env python3\nimport pathlib,shutil,sys\n'
            f'shutil.copyfile({src!r}, sys.argv[sys.argv.index("--out")+1])\n'
        )
        binary.chmod(0o755)
        data = run('track-b', '--betterbench-python', str(binary))
        self.assertEqual(data['mode'], 'full')
        cmd = run('--list', 'track-b')['command']
        self.assertEqual(cmd[cmd.index('--passes') + 1], '20')
        self.assertEqual(cmd[cmd.index('--warmup') + 1], '3')
        self.assertIn('--quick', run('track-b', '--quick', '--dry-run')['command'])

    def test_track_c_requests_and_missing(self):
        task = 'multi_turn-ja-short_2k'

        def reply(path, body, i):
            return (['同じ行です\n'] * 4, 1024 if i == 0 else None, 'length')

        with server(reply) as (srv, url):
            data = run('track-c', '--endpoint', url + '/v1', '--task', task, '--repeats', '2')
        messages = srv.bodies[0]['messages']
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
        self.assertIn('admit_request', messages[0]['content'])
        self.assertEqual(data['truncated'], 2)
        self.assertEqual(data['loop_count'], 2)
        self.assertEqual(data['long_completed'], 1)
        self.assertTrue(data['results'][1]['completion_tokens_approximate'])
        self.assertIsNone(data['results'][1]['decode_tok_s'])
        for key in ('all', 'non_loop', 'long_output', 'long_non_loop'):
            self.assertIsNone(data[key]['macro_decode_tok_s'])
        self.assertEqual(sum(c['decode_tok_s'] is not None for c in data['all']['cells']), 1)
        self.assertTrue(all(c['decode_tok_s'] is None for c in data['non_loop']['cells']))

    def test_complete_macro(self):
        manifest, tasks = suite.corpus()
        data = [{'task_id': t['id'], 'ok': True, 'decode_tok_s': i} for i, t in enumerate(tasks)]
        data.extend([dict(data[0], decode_tok_s=10)] * 2)
        self.assertAlmostEqual(suite.summarize(manifest, tasks, data)['macro_decode_tok_s'], (435 + 10) / 30)
        self.assertIsNone(suite.summarize(manifest, tasks, data[2:])['macro_decode_tok_s'])

    def test_serving_mix_and_steady(self):
        def reply(path, body, index):
            long = len(body['messages'][0]['content']) > 20000
            return ['first', ' next', ' last'], 6 if long else 3, 'stop'

        with server(reply) as (srv, url):
            data = run('track-d', '--endpoint', url + '/v1', '--concurrency', '2', '--requests', '6')
        self.assertGreaterEqual(srv.peak, 2)
        self.assertEqual(len({b['messages'][0]['content'].splitlines()[0] for b in srv.bodies}), 12)
        steady, mix = data['levels']
        self.assertTrue(all(r['kind'] == 'short' for r in steady['results']))
        self.assertEqual({r['kind'] for r in mix['results']}, {'long', 'short'})
        self.assertEqual(steady['completion_rate'], 1)
        for level in (steady, mix):
            self.assertEqual(level['errors'], 0)
            self.assertIsNone(level['long_window_tok_s'])
            count = sum(r['completion_tokens'] for r in level['results'])
            self.assertAlmostEqual(level['aggregate_tok_s'], count / level['window_s'])
            self.assertGreater(level['decode_tok_s_median'], 0)
            for key in ('ttft_ms', 'e2e_ms'):
                self.assertGreater(level[key]['p50'], 0)
                self.assertGreaterEqual(level[key]['p95'], level[key]['p50'])
        for r in mix['results']:
            single = r['tokens_per_update'] == 1
            self.assertEqual('itl_ms' in r, single)
            self.assertEqual('update_gap_ms' in r, not single)
        self.assertIsNone(steady['update_gap_ms']['p50'])
        self.assertGreater(mix['update_gap_ms']['p50'], 0)

    def test_serving_disconnected_stream(self):
        with server(lambda *args: (['partial'], None, None)) as (_, url):
            level = run(
                'track-d', '--endpoint', url + '/v1', '--concurrency', '1', '--requests', '2', '--mode', 'mix'
            )['levels'][0]
        self.assertEqual(level['errors'], 2)
        self.assertEqual(level['completion_rate'], 0)
        self.assertIsNone(level['aggregate_tok_s'])
        self.assertIsNone(level['ttft_ms']['p50'])

    def test_render_and_comparison(self):
        conditions = self.save('hardware.json', {k: 'fixture' for k in suite.CONDITIONS})
        b = run(
            'track-b', '--import-json', self.save('bb.json', betterbench(20, 3)), '--conditions', conditions
        )
        with server(long_reply) as (_, url):
            a = run('track-a', '--spec-endpoint', url + '/v1', '--repeats', '1')
            c = run('track-c', '--endpoint', url + '/v1', '--task', 'multi_turn-en-short_2k')
            d = run(
                'track-d',
                '--endpoint',
                url + '/v1',
                '--concurrency',
                '1',
                '--requests',
                '1',
                '--mode',
                'steady',
            )
        paths = [self.save(r['track'] + '.json', r) for r in (a, b, c, d)]
        review = {
            'task_id': c['results'][0]['task_id'],
            'output_sha256': c['results'][0]['output_sha256'],
            'passed': False,
        }
        md = run('render', *paths, '--compare-b', paths[1], '--reviews', self.save('reviews.json', [review]))
        self.assertIn('判定済み 1 件、合格 0 件', md)
        for text in (
            '① 実タスク',
            '② 外部比較',
            '③ 基礎性能',
            '④ サービング',
            'spec candidate',
            'steady / 1',
            '未測定',
            '100.00 | 1.00 |',
        ):
            self.assertIn(text, md)
        b['conditions']['kv'] = 'different'
        md = run('render', *paths, '--compare-b', self.save('other.json', b))
        self.assertIn(
            '未測定 | 未測定 |', next(line for line in md.splitlines() if line.startswith('| code |'))
        )
        self.assertIn('未測定', run('render'))

    def test_cli_preview_lock_and_metadata(self):
        lock = self.tmp / 'gpu.lock'
        dst = self.tmp / 'output.json'
        cmd = [sys.executable, '-S', str(SCRIPT), '--lock-file', str(lock), '--output', str(dst)]
        subprocess.run([*cmd, 'track-a', '--dry-run'], check=True)
        self.assertFalse(lock.exists())
        self.assertTrue(json.loads(dst.read_text()))
        path = self.save('betterbench.json', betterbench())
        subprocess.run([*cmd, 'track-b', '--import-json', path], check=True)
        self.assertTrue(lock.is_file())
        data = json.loads(dst.read_text())
        self.assertEqual(data['schema_version'], 1)
        self.assertTrue(
            {'timestamp_utc', 'git_revision', 'dirty', 'gpu', 'rocm_path', 'arguments'}
            <= data['metadata'].keys()
        )
        bad = subprocess.run(
            [sys.executable, str(SCRIPT), 'track-a', '--spec-max-tokens', '64'], capture_output=True
        )
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
