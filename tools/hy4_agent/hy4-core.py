#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import re
import subprocess
import sys

pid, executable, destination, started = sys.argv[1:]
destination = Path(destination)
destination.mkdir(parents=True, exist_ok=True, mode=0o700)
started = float(started)


def run(command, output, timeout=60):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        output.write_bytes(result.stdout)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        output.write_text(str(exc))
        return False


if run(['coredumpctl', '--no-pager', 'dump', pid, '--output', str(destination / 'core')], destination / 'core-export.txt', 120):
    cores = [destination / 'core']
else:
    cores = []
    pattern = Path('/proc/sys/kernel/core_pattern').read_text().strip()
    if not pattern.startswith('|'):
        pattern = pattern.replace('%p', pid).replace('%P', pid).replace('%e', Path(executable).name)
        pattern = re.sub(r'%[a-zA-Z%]', '*', pattern)
        for name in glob.glob(pattern):
            path = Path(name)
            if path.is_file() and path.stat().st_mtime >= started:
                cores.append(path)
    else:
        reports = sorted(Path('/var/crash').glob('*' + Path(executable).name + '*.crash'), key=lambda p: p.stat().st_mtime, reverse=True)
        for report in reports:
            if report.stat().st_mtime < started:
                continue
            unpack = destination / 'apport'
            if run(['apport-unpack', str(report), str(unpack)], destination / 'apport-unpack.txt', 120):
                status = unpack / 'ProcStatus'
                core = unpack / 'CoreDump'
                if status.exists() and re.search(r'^Pid:\s*' + re.escape(pid) + r'\s*$', status.read_text(), re.MULTILINE) and core.exists():
                    cores.append(core)
                    break
if cores:
    run(['gdb', '-batch', '-ex', 'bt full', '-ex', 'info sharedlibrary', executable, str(cores[0])], destination / 'gdb.txt', 120)
else:
    (destination / 'core-unavailable.txt').write_text('該当 PID の core を取得できませんでした。core_pattern と coredumpctl の記録を確認してください。\n')
