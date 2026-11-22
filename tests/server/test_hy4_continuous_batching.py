import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import argparse

ROOT = Path(__file__).resolve().parents[2]
class Server:
    def __init__(self, opts, slots, draft=False):
        argv = [opts.compute, 'hy4', 'serve', '--model-dir', opts.model_dir,
                '--engram-dir', opts.engram_dir, '--placement', opts.placement, '--max-seq-len', '512',
                '--chunk', '64']
        if slots != 1:
            argv += ['--max-concurrent-requests', str(slots)]
        if draft:
            argv += ['--dfly-dir', opts.dfly_dir]
        self.log = tempfile.TemporaryFile(mode='w+')
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1, env={**os.environ, 'SHELF_HY4_ENGRAM_POPULATE': '0'})
        self.events = queue.Queue()
        threading.Thread(target=self.read, daemon=True).start()
        self.send({'op': 'ping'})
        assert self.event()['event'] == 'pong'

    def read(self):
        for line in self.proc.stdout:
            self.events.put(json.loads(line))
        self.events.put(None)

    def send(self, event):
        self.proc.stdin.write(json.dumps(event) + '\n')
        self.proc.stdin.flush()

    def event(self):
        try:
            event = self.events.get(timeout=900)
        except queue.Empty:
            raise AssertionError('compute 応答がタイムアウトしました')
        if event is None:
            self.log.seek(0)
            raise AssertionError(f'compute が終了しました: {self.log.read()}')
        if event['event'] == 'error':
            raise AssertionError(event)
        return event

    def close(self):
        if self.proc.poll() is None:
            self.send({'op': 'shutdown'})
            self.proc.wait(timeout=60)
        self.log.close()


def request(i, length, count):
    return {'op': 'generate', 'request_id': i, 'input_ids': [120000] + [100 + (j % 80) for j in range(length - 1)],
            'max_new_tokens': count, 'temperature': 0.7 if i % 2 else 0.0, 'seed': i * 17,
            'top_k': 20 if i % 2 else 0, 'top_p': 0.95 if i % 2 else 1.0}


def run(server, requests, cancel=None, queued_id=None, arrival=None):
    for item in requests:
        server.send(item)
    ids = {item['request_id']: [] for item in requests}
    done = {}
    order = []
    cancelled = False
    while len(done) < len(ids):
        event = server.event()
        if event['event'] == 'token':
            rid = event['request_id']
            if rid == queued_id:
                assert done
            ids[rid].append(event['token_id'])
            order.append(rid)
            if cancel == rid and len(ids[rid]) == 2 and not cancelled:
                server.send({'op': 'cancel', 'request_id': rid})
                cancelled = True
                if arrival:
                    server.send(arrival)
                    ids[arrival['request_id']] = []
        elif event['event'] == 'done':
            rid = event['request_id']
            assert ids[rid] == event['generated_ids']
            done[rid] = event
    return ids, done, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', default=str(ROOT / 'build/shelf-compute'))
    for option in ('model-dir', 'engram-dir', 'placement', 'dfly-dir'):
        parser.add_argument('--' + option, required=True)
    opts = parser.parse_args()
    server = Server(opts, 4)
    try:
        requests = [request(i, n, c) for i, n, c in [(1, 8, 8), (2, 13, 9), (3, 19, 10), (4, 25, 11)]]
        baseline = {}
        for req in requests:
            baseline.update(run(server, [req])[0])
        actual, done, order = run(server, requests)
        assert actual == baseline
        assert len(set(order[:8])) > 1 and len(set(order[4:12])) > 1
        assert all(event['finish_reason'] in ('stop', 'length') for event in done.values())
        print('A PASS', flush=True)
        actual, done, _ = run(server, requests, cancel=1)
        assert done[1]['finish_reason'] == 'cancelled'
        assert all(actual[i] == baseline[i] for i in (2, 3, 4))
        print('C PASS', flush=True)
    finally:
        server.close()
    server = Server(opts, 2)
    try:
        actual, done, _ = run(server, requests[:3], queued_id=3)
        assert all(actual[i] == baseline[i] for i in (1, 2, 3))
        assert all(e['finish_reason'] in ('stop', 'length') for e in done.values())
        print('B PASS', flush=True)
    finally:
        server.close()
    server = Server(opts, 4, draft=True)
    try:
        items = [request(i, 24 + i, 32) for i in (2, 4, 6, 8)]
        items[0].update(temperature=0.7, seed=42, top_k=20, top_p=0.95)
        arrival = request(10, 64, 18)
        refs = {req['request_id']: run(server, [req])[0][req['request_id']] for req in [*items, arrival]}
        server.send({'op': 'stats'})
        stats = server.event()
        before = stats['batch_draft_rounds']
        assert stats['kv']['total_pages'] == stats['kv']['free_pages'] == 4
        assert stats['kv']['capacity_tokens'] == stats['kv']['free_tokens'] == 4 * 512
        actual, done, _ = run(server, items, cancel=2, arrival=arrival)
        assert done[2]['finish_reason'] == 'cancelled'
        assert actual[2] == refs[2][:len(actual[2])]
        assert all(actual[i] == refs[i] for i in (4, 6, 8, 10))
        assert all(done[i]['finish_reason'] == ('stop' if refs[i][-1] == 120025 else 'length') for i in (4, 6, 8, 10))
        server.send({'op': 'stats'})
        assert server.event()['batch_draft_rounds'] > before
        tail = [request(12, 508, 4), request(14, 24, 16)]
        short_refs = {req['request_id']: run(server, [req])[0][req['request_id']] for req in tail}
        assert run(server, tail)[0] == short_refs
        print('E PASS', flush=True)
    finally:
        server.close()
    server = Server(opts, 1, draft=True)
    try:
        first = request(1, 8, 16)
        server.send(first)
        server.send(request(2, 8, 8))
        seen_error = False
        while True:
            try:
                event = server.event()
            except AssertionError as exc:
                assert '1 件' in str(exc)
                seen_error = True
                continue
            if event['event'] == 'done' and event['request_id'] == 1:
                break
        assert seen_error
        print('D PASS', flush=True)
    finally:
        server.close()


if __name__ == '__main__':
    main()
