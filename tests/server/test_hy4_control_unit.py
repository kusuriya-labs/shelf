import importlib.util
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('hy4_control', ROOT / 'tools/hy4_agent/hy4-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class Hy4ControlTest(unittest.TestCase):
    def test_snapshot_before_stop_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                manager = control.Manager()
                manager.server = Mock(pid=123, returncode=0)
                manager.server.poll.return_value = None
                manager.diagnostics = Path(directory)
                calls = []
                def snapshot(*args, **kwargs):
                    calls.append('snapshot')
                    if failure:
                        raise OSError('失敗')
                    return {'status': 'ok'}
                def kill(*args):
                    calls.append('kill')
                with patch.object(control, 'STATE', Path(directory)), patch.object(control, 'request_json', side_effect=snapshot), \
                     patch.object(control.os, 'killpg', side_effect=kill), patch.object(control, 'vram', return_value={}):
                    manager.stop()
                self.assertEqual(calls[:2], ['snapshot', 'kill'])
                manager_server_exit = (Path(directory) / 'server-exit.txt').read_text()
                self.assertIn('終了コード: 0', manager_server_exit)

    def test_exit_code_and_restart(self):
        self.run_command("print('出力'); raise SystemExit(37)", 10, 37, '出力')

    def test_timeout_and_restart(self):
        self.run_command('import time; time.sleep(10)', 1, 124, '')

    def test_sigterm_and_restart(self):
        manager = control.Manager()
        manager.stop = Mock()
        manager.start = Mock()
        parent, child = socket.socketpair()
        with parent, child, open(os.devnull, 'r+') as devnull:
            fds = [os.dup(devnull.fileno()) for _ in range(3)]
            req = {'argv': [sys.executable, '-c', 'import time; time.sleep(10)'],
                   'cwd': str(ROOT), 'env': dict(os.environ), 'timeout': 20}
            sender = threading.Thread(target=lambda: (time.sleep(0.5), child.sendall(b'T')))
            sender.start()
            self.assertEqual(manager.run_command(parent, req, fds), 143)
            sender.join()
        manager.start.assert_called_once()

    def test_restart_failure_is_reported(self):
        manager = control.Manager()
        manager.stop = Mock()
        manager.start = Mock(side_effect=RuntimeError('再起動失敗'))
        parent, child = socket.socketpair()
        with parent, child, open(os.devnull, 'r+') as devnull:
            fds = [os.dup(devnull.fileno()) for _ in range(3)]
            req = {'argv': [sys.executable, '-c', 'pass'], 'cwd': str(ROOT),
                   'env': dict(os.environ), 'timeout': 10}
            with self.assertRaisesRegex(RuntimeError, '再起動失敗'):
                manager.run_command(parent, req, fds)

    def run_command(self, code, timeout, expected, output):
        manager = control.Manager()
        manager.stop = Mock()
        manager.start = Mock()
        parent, child = socket.socketpair()
        with parent, child, tempfile.TemporaryFile(mode='w+') as f, open(os.devnull, 'r') as stdin:
            fds = [os.dup(stdin.fileno()), os.dup(f.fileno()), os.dup(f.fileno())]
            req = {'argv': [sys.executable, '-c', code], 'cwd': str(ROOT), 'env': dict(os.environ), 'timeout': timeout}
            self.assertEqual(manager.run_command(parent, req, fds), expected)
            f.seek(0)
            self.assertIn(output, f.read())
        self.assertEqual(manager.stop.call_count, 2)
        manager.start.assert_called_once()


if __name__ == '__main__':
    unittest.main()
