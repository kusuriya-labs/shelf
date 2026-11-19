#!/usr/bin/env python3
import array
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.hy4-agent'
SOCKET = STATE / 'control.sock'
LOCKS = ['/tmp/r9700.lock', '/tmp/v620.lock', '/tmp/v620-76a08.lock']
DEVICES = 'GPU-a8e9ddefa2d60f55,GPU-08b2ddcbd6e6b36c,GPU-76a08c022586fed6'
PORT = 18123
MODEL = 'Hy4-flash-4bpw'
BASE = Path.home() / 'datapool/rocm-exl3-rdna2'


def emit(message):
    print(message, file=sys.stderr, flush=True)


def vram():
    return {p.parent.parent.name: int(p.read_text()) for p in sorted(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))}


def request_json(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def write_message(sock, value):
    sock.sendall((json.dumps(value, ensure_ascii=False) + '\n').encode())


def read_message(sock):
    data = bytearray()
    while not data.endswith(b'\n'):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError('制御接続が切断されました')
        data.extend(chunk)
        if len(data) > 4_000_000:
            raise ValueError('制御メッセージが大きすぎます')
    return json.loads(data)


def connection():
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(str(SOCKET))
    return sock


def client(action):
    with connection() as sock:
        write_message(sock, {'action': action})
        return read_message(sock)


def alive():
    try:
        return client('status').get('ok', False)
    except (OSError, ValueError, ConnectionError):
        return False


def stop_orphan():
    path = STATE / 'server.pid'
    if not path.exists():
        return
    pid = int(path.read_text().strip())
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(ROOT / 'build-hy4-agent/shelf-server').encode() not in command or os.getpgid(pid) != pid:
            raise RuntimeError('PID の所有者を確認できないため停止できません')
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and Path(f'/proc/{pid}/cmdline').exists():
            if not Path(f'/proc/{pid}/cmdline').read_bytes():
                break
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except FileNotFoundError:
        pass
    path.unlink(missing_ok=True)
    emit(f'管理外サーバーを停止しました。現在の VRAM: {vram()} byte')


class Manager:
    def __init__(self):
        self.server = None
        self.locks = []
        self.baseline = {}
        self.log = None

    def acquire(self):
        for name in LOCKS:
            f = open(name, 'a')
            fcntl.flock(f, fcntl.LOCK_EX)
            self.locks.append(f)
        self.baseline = vram()
        if len(self.baseline) < 3:
            raise RuntimeError('3 台分の VRAM 使用量を取得できません')

    def start(self):
        if self.server and self.server.poll() is None:
            return
        try:
            with socket.create_connection(('127.0.0.1', PORT), timeout=1):
                raise RuntimeError(f'ポート {PORT} は既に使用されています')
        except ConnectionRefusedError:
            pass
        env = os.environ.copy()
        env.update(SHELF_HY4_ENGRAM_SOURCE='rdma', ROCR_VISIBLE_DEVICES=DEVICES)
        env.pop('SHELF_HY4_ENGRAM_MLOCK', None)
        env.pop('SHELF_HY4_ENGRAM_POPULATE', None)
        cmd = [str(Path('/tmp/shelfvenv/bin/python')), str(ROOT / 'build-hy4-agent/shelf-server'),
               '--model-dir', str(BASE / 'models/Hy4-flash-exl3-4bpw-mul24'),
               '--engram-dir', str(Path.home() / 'datapool/hy4-hf-staging-20261119/body'),
               '--placement', str(BASE / 'hy4-flash-pilot/shelf-placement/owners-4bpw.json'),
               '--dfly-dir', str(BASE / 'hy4-flash-pilot/shelf-dfly/Hy4-flash-dfly-4bpw'),
               '--model', MODEL, '--port', str(PORT), '--max-seq-len', '8192', '--max-concurrent-requests', '1']
        self.log = open(STATE / 'server.log', 'ab', buffering=0)
        started = time.monotonic()
        self.server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=tuple(f.fileno() for f in self.locks))
        (STATE / 'server.pid').write_text(str(self.server.pid) + '\n')
        deadline = time.monotonic() + int(os.environ.get('HY4_START_TIMEOUT', '900'))
        try:
            while time.monotonic() < deadline:
                if self.server.poll() is not None:
                    raise RuntimeError(f'サーバーが終了しました: {self.server.returncode}')
                try:
                    models = request_json('/v1/models')
                    if not any(m.get('id') == MODEL for m in models['data']):
                        raise ValueError('モデルが一覧にありません')
                    answer = request_json('/v1/chat/completions', {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 4, 'temperature': 0})
                    choices = answer.get('choices') or []
                    message = choices[0].get('message', {}) if choices else {}
                    if not (message.get('content') or message.get('reasoning_content')):
                        raise ValueError('生成結果が空です')
                    emit(f'Hy4 ready: {time.monotonic()-started:.2f} 秒')
                    return
                except (OSError, ValueError, KeyError, urllib.error.URLError):
                    time.sleep(2)
            raise RuntimeError('サーバーのヘルスチェックがタイムアウトしました')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.server:
            pid = self.server.pid
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                self.server.wait(timeout=15)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.server = None
            (STATE / 'server.pid').unlink(missing_ok=True)
            if self.log:
                self.log.close()
                self.log = None
        deadline = time.monotonic() + 45
        while True:
            usage = vram()
            if all(k in usage and usage[k] <= n + 128 * 1024 * 1024 for k, n in self.baseline.items()):
                emit(f'VRAM 停止後: {usage} byte')
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'VRAM が解放されません: 起動前={self.baseline}, 現在={usage}')
            time.sleep(1)

    def run_command(self, sock, req, fds):
        if len(fds) != 3:
            raise ValueError('標準入出力を受信できません')
        status = 1
        proc = None
        try:
            self.stop()
            env = req['env']
            env['ROCR_VISIBLE_DEVICES'] = DEVICES
            proc = subprocess.Popen(req['argv'], cwd=req['cwd'], env=env, stdin=fds[0], stdout=fds[1], stderr=fds[2],
                                    start_new_session=True, pass_fds=tuple(f.fileno() for f in self.locks))
            started = time.monotonic()
            signaled = None
            while proc.poll() is None:
                if time.monotonic() - started >= req['timeout'] and signaled is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    signaled = (time.monotonic(), 124)
                ready, _, _ = select.select([sock], [], [], 0.2)
                if ready:
                    data = sock.recv(64)
                    if not data:
                        if signaled is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            signaled = (time.monotonic(), 143)
                    elif data[:1] in (b'I', b'T') and signaled is None:
                        sig = signal.SIGINT if data[:1] == b'I' else signal.SIGTERM
                        os.killpg(proc.pid, sig)
                        signaled = (time.monotonic(), 128 + sig)
                if signaled and time.monotonic() - signaled[0] > 10:
                    try: os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
            status = signaled[1] if signaled else (proc.returncode if proc.returncode >= 0 else 128 - proc.returncode)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            for fd in fds:
                os.close(fd)
            try:
                self.stop()
            finally:
                self.start()
        return status

    def serve(self):
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        self.acquire()
        listener = socket.socket(socket.AF_UNIX)
        SOCKET.unlink(missing_ok=True)
        listener.bind(str(SOCKET))
        os.chmod(SOCKET, 0o600)
        listener.listen(8)
        listener.settimeout(1)
        (STATE / 'manager.pid').write_text(str(os.getpid()) + '\n')
        try:
            self.start()
            while True:
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    if self.server.poll() is not None:
                        raise RuntimeError('サーバーが予期せず終了しました')
                    continue
                with conn:
                    fds = []
                    action = ''
                    try:
                        payload, ancillary, _, _ = conn.recvmsg(4_000_000, socket.CMSG_SPACE(3 * array.array('i').itemsize))
                        for level, kind, data in ancillary:
                            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                                a = array.array('i'); a.frombytes(data[:len(data) - len(data) % a.itemsize]); fds.extend(a)
                        while b'\n' not in payload:
                            chunk = conn.recv(65536)
                            if not chunk or len(payload) + len(chunk) > 4_000_000:
                                raise ValueError('制御メッセージが不完全です')
                            payload += chunk
                        message, _, pending = payload.partition(b'\n')
                        req = json.loads(message)
                        action = req['action']
                        if action == 'status':
                            result = {'ok': self.server is not None and self.server.poll() is None, 'pid': self.server.pid if self.server else None, 'vram': vram()}
                        elif action == 'run':
                            if pending[:1] in (b'I', b'T'):
                                code = 130 if pending[:1] == b'I' else 143
                                for fd in fds: os.close(fd)
                            else:
                                code = self.run_command(conn, req, fds)
                            fds = []
                            result = {'ok': True, 'code': code}
                        elif action == 'stop':
                            self.stop(); write_message(conn, {'ok': True}); return
                        else:
                            raise ValueError('不明な操作')
                        write_message(conn, result)
                    except Exception as exc:
                        for fd in fds:
                            try: os.close(fd)
                            except OSError: pass
                        try: write_message(conn, {'ok': False, 'error': str(exc)})
                        except OSError: pass
                        if action == 'run' and (self.server is None or self.server.poll() is not None):
                            return
        finally:
            try: self.stop()
            finally:
                listener.close(); SOCKET.unlink(missing_ok=True)
                (STATE / 'manager.pid').unlink(missing_ok=True)


def start():
    if alive():
        emit('Hy4 は起動済みです'); return 0
    STATE.mkdir(mode=0o700, exist_ok=True)
    with open(STATE / 'startup.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if alive(): return 0
        stop_orphan()
        log = open(STATE / 'manager.log', 'ab', buffering=0)
        proc = subprocess.Popen([sys.executable, __file__, 'manager'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        while True:
            if proc.poll() is not None:
                emit(f'起動失敗。ログ: {STATE / "manager.log"}'); return 1
            if alive():
                emit(f'Hy4 起動完了。ログ: {STATE / "manager.log"}'); return 0
            time.sleep(1)


def run_gpu(args):
    timeout = 86400
    if args[:1] == ['--timeout']:
        timeout = int(args[1]); args = args[2:]
    if args[:1] == ['--']: args = args[1:]
    if not args or timeout <= 0:
        emit('使い方: with-gpu [--timeout 秒] [--] コマンド...'); return 2
    try:
        with connection() as sock:
            req = (json.dumps({'action': 'run', 'argv': args, 'cwd': os.getcwd(), 'env': dict(os.environ), 'timeout': timeout}) + '\n').encode()
            fds = array.array('i', [0, 1, 2])
            sent = sock.sendmsg([req], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
            if sent < len(req):
                sock.sendall(req[sent:])
            interrupted = []
            def forward(sig, frame):
                interrupted.append(sig)
                try: sock.sendall(b'I' if sig == signal.SIGINT else b'T')
                except OSError: pass
            signal.signal(signal.SIGINT, forward); signal.signal(signal.SIGTERM, forward)
            result = read_message(sock)
            if not result.get('ok'):
                emit('Hy4 の再起動または GPU コマンドに失敗: ' + result.get('error', '不明')); return 1
            return 128 + interrupted[0] if interrupted else result['code']
    except (OSError, ValueError, ConnectionError) as exc:
        emit(f'GPU 制御に失敗: {exc}'); return 1


def main():
    STATE.mkdir(mode=0o700, exist_ok=True)
    action = sys.argv[1] if len(sys.argv) > 1 else ''
    if action == 'manager':
        Manager().serve(); return 0
    if action == 'start': return start()
    if action == 'with-gpu': return run_gpu(sys.argv[2:])
    if action == 'restart':
        if alive():
            result = client('stop')
            if not result['ok']: emit(result.get('error')); return 1
        return start()
    if action in ('stop', 'status'):
        try: result = client(action)
        except (OSError, ValueError, ConnectionError):
            if action == 'stop':
                stop_orphan()
            emit('Hy4 は停止しています'); return 0 if action == 'stop' else 1
        emit(json.dumps(result, ensure_ascii=False))
        return 0 if result.get('ok') else 1
    emit('使い方: hy4-llm start|stop|status|restart'); return 2


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        emit(f'Hy4 制御エラー: {exc}'); sys.exit(1)
