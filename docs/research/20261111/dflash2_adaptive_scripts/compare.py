import json
import os
import pathlib
import subprocess
import sys
import time

count = int(sys.argv[1])
new_tokens = int(sys.argv[2])
cases_path = pathlib.Path(sys.argv[3])
extra = sys.argv[4:]
cases = json.loads(cases_path.read_text())
model = os.environ['SHELF_MODEL_DIR']
command = [os.environ['SHELF_BENCH_BINARY'], '--model-dir', model, '--serve-stdio',
           '--device', '0', '--max-seq-len', '2048', '--max-concurrent-requests', str(count),
           '--prefix-cache', '0', '--temperature', '0'] + extra
with open(os.environ['ERRF'], 'w') as error:
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=error, text=True, bufsize=1)
    def send(request):
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        proc.stdin.flush()
    def receive():
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f'compute exited before completion: {proc.poll()}')
        return json.loads(line)
    try:
        send({'op': 'ping'})
        while receive().get('event') != 'pong':
            pass
        send({'op': 'generate', 'request_id': 900, 'input_ids': cases[0]['input_ids'],
              'max_new_tokens': 8, 'temperature': 0})
        while True:
            event = receive()
            if event.get('request_id') == 900 and event.get('event') in ('done', 'error'):
                if event.get('event') == 'error':
                    raise RuntimeError(str(event))
                break
        stamps, finished, generated = {}, {}, {}
        start = time.monotonic()
        for i in range(count):
            send({'op': 'generate', 'request_id': i + 1,
                  'input_ids': cases[i % len(cases)]['input_ids'],
                  'max_new_tokens': new_tokens, 'temperature': 0})
        while len(finished) < count:
            event = receive()
            request = event.get('request_id')
            kind = event.get('event')
            if kind == 'token':
                stamps.setdefault(request, []).append(time.monotonic())
            elif kind == 'done':
                finished[request] = time.monotonic()
                generated[request] = event.get('generated_ids')
            elif kind == 'error':
                raise RuntimeError(str(event))
        if len(stamps) != count:
            raise RuntimeError('少なくとも 1 request で token を取得できなかった')
        lo = max(stamps[request][0] for request in stamps)
        hi = min(finished.values())
        observed = sum(sum(lo <= stamp <= hi for stamp in values) for values in stamps.values())
        duration = hi - lo
        result = {'count': count, 'all_active_tps': round(observed / duration, 2) if duration > 0.5 else None,
                  'window_s': round(duration, 2), 'wall_s': round(max(finished.values()) - start, 2),
                  'total_tokens': sum(map(len, stamps.values()))}
        pathlib.Path(os.environ['GENF']).write_text(json.dumps(generated, ensure_ascii=False))
        print(json.dumps(result))
        send({'op': 'shutdown'})
        proc.stdin.close()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
