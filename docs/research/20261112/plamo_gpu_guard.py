#!/usr/bin/env python3
"""Run under flock; fail closed unless the user has released the GPU and it is idle.

Example: flock /tmp/r9700.lock python plamo_gpu_guard.py -- command args...
An exit code of 75 means no GPU workload was started and it is safe to retry.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

RELEASE = '/tmp/agentx/gpu_release'
RENDER = '/dev/dri/renderD130'


def blocking_processes():
    blocked = []
    for path in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            pid = int(path.split('/')[2])
            if pid == os.getpid():
                continue
            with open(path, 'rb') as f:
                args = [x.decode(errors='replace') for x in f.read().split(b'\0') if x]
            if not args:
                continue
            names = [os.path.basename(x) for x in args]
            exe = names[0]
            python_process = exe.startswith('python')
            if (exe in ('ninja', 'shelf-server', 'shelf-compute', 'clang-23', 'clang',
                        'cc1', 'cc1plus', 'ld.lld') or
                    (python_process and ('run_server_regression.py' in names or
                                         'shelf-server' in names))):
                blocked.append((pid, exe))
        except (OSError, ValueError):
            continue
    return blocked


def guard():
    if not os.path.isfile(RELEASE):
        raise RuntimeError('release marker absent')
    blocked = blocking_processes()
    if blocked:
        raise RuntimeError(f'build, regression, or server process active: {blocked}')
    result = subprocess.run(['rocm-smi', '--showpids'], capture_output=True, text=True,
                            timeout=30, check=False)
    print(result.stdout, file=sys.stderr, flush=True)
    rows = [line for line in result.stdout.splitlines() if re.match(r'^\s*\d+\s+\S+', line)]
    if rows:
        raise RuntimeError(f'GPU processes present: {rows}')
    if result.returncode != 0 or 'No KFD PIDs currently running' not in result.stdout:
        raise RuntimeError(f'rocm-smi did not confirm an empty process list: rc={result.returncode}, stderr={result.stderr!r}')
    fuser = subprocess.run(['fuser', RENDER], capture_output=True, text=True,
                           timeout=30, check=False)
    if fuser.returncode not in (0, 1) or 'denied' in fuser.stderr.lower():
        raise RuntimeError(f'fuser failed: rc={fuser.returncode}, stderr={fuser.stderr!r}')
    if fuser.returncode == 0 or fuser.stdout.strip():
        raise RuntimeError(f'R9700 render device in use: {fuser.stdout.strip()} {fuser.stderr.strip()}')
    if not os.path.isfile(RELEASE):
        raise RuntimeError('release marker disappeared')
    blocked = blocking_processes()
    if blocked:
        raise RuntimeError(f'build, regression, or server process appeared: {blocked}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command:
        parser.error('missing command')
    try:
        guard()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'GPU guard: {exc}', file=sys.stderr, flush=True)
        return 75
    os.execvp(command[0], command)


if __name__ == '__main__':
    sys.exit(main())
