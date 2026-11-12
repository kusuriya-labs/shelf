#!/usr/bin/env python3
"""CPU-only safety checks; no ROCm calls or release-marker writes."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import plamo_gpu_guard as guard


def result(rc=0, stdout='', stderr=''):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class GuardTest(unittest.TestCase):
    def run_guard(self, marker=True, blocked=(), commands=()):
        with patch.object(guard.os.path, 'isfile', return_value=marker), \
             patch.object(guard, 'device_is_expected', return_value=True), \
             patch.object(guard, 'blocking_processes', return_value=list(blocked)), \
             patch.object(guard.subprocess, 'run', side_effect=list(commands)) as run, \
             contextlib.redirect_stderr(io.StringIO()):
            try:
                guard.guard()
                return True, run.call_count
            except RuntimeError:
                return False, run.call_count

    def test_no_release_never_queries_gpu(self):
        self.assertEqual(self.run_guard(marker=False), (False, 0))

    def test_build_or_regression_never_queries_gpu(self):
        self.assertEqual(self.run_guard(blocked=[(123, 'ninja')]), (False, 0))
        self.assertEqual(self.run_guard(blocked=[(124, 'python')]), (False, 0))

    def test_device_identity_mismatch_never_queries_gpu(self):
        with patch.object(guard.os.path, 'isfile', return_value=True), \
             patch.object(guard, 'device_is_expected', return_value=False), \
             patch.object(guard.subprocess, 'run') as run, self.assertRaises(RuntimeError):
            guard.guard()
        run.assert_not_called()

    def test_explicit_idle_is_accepted(self):
        self.assertEqual(self.run_guard(commands=[result(stdout='No KFD PIDs currently running'),
                                                   result(rc=1)]), (True, 2))

    def test_kfd_process_or_unknown_response_rejected(self):
        self.assertFalse(self.run_guard(commands=[result(stdout='KFD Processes\n123 shelf-compute 1\n')])[0])
        self.assertFalse(self.run_guard(commands=[result(stdout='KFD Processes')])[0])
        self.assertFalse(self.run_guard(commands=[result(rc=1, stdout='No KFD PIDs currently running')])[0])

    def test_render_process_or_error_rejected(self):
        empty = result(stdout='No KFD PIDs currently running')
        self.assertFalse(self.run_guard(commands=[empty, result(stdout='123')])[0])
        self.assertFalse(self.run_guard(commands=[empty, result(rc=1, stderr='Permission denied')])[0])
        self.assertFalse(self.run_guard(commands=[empty, result(rc=2)])[0])

    def test_release_disappears_before_launch(self):
        with patch.object(guard.os.path, 'isfile', side_effect=[True, False]), \
             patch.object(guard, 'device_is_expected', return_value=True), \
             patch.object(guard, 'blocking_processes', return_value=[]), \
             patch.object(guard.subprocess, 'run', side_effect=[result(stdout='No KFD PIDs currently running'), result(rc=1)]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
            guard.guard()


if __name__ == '__main__':
    unittest.main()
