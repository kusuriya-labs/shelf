#!/usr/bin/env python3
"""Sample R9700 sysfs and host pressure during a command; timestamps are monotonic.

Run the entire invocation under the R9700 flock. The trace includes sampling
latency and failures; sysfs power is an averaged sensor, not instantaneous power.
"""
import argparse
import glob
import json
import os
import subprocess
import threading
import time

GPU = '/sys/class/drm/card2/device'
HWMON = glob.glob(GPU + '/hwmon/hwmon*')[0]
FILES = {
    'gpu_busy_percent': GPU + '/gpu_busy_percent',
    'sclk_hz': HWMON + '/freq1_input',
    'mclk_hz': HWMON + '/freq2_input',
    'power_uw': HWMON + '/power1_average',
    'edge_millic': HWMON + '/temp1_input',
    'junction_millic': HWMON + '/temp2_input',
    'memory_millic': HWMON + '/temp3_input',
    'cpu_pressure': '/proc/pressure/cpu',
    'memory_pressure': '/proc/pressure/memory',
    'vmstat': '/proc/vmstat',
    'cpu_stat': '/proc/stat',
}
VM = {'pgmajfault', 'pswpin', 'pswpout', 'compact_stall', 'compact_success',
      'compact_fail', 'allocstall_normal', 'thp_fault_alloc', 'thp_collapse_alloc',
      'compact_daemon_wake', 'compact_daemon_migrate_scanned', 'compact_daemon_free_scanned',
      'compact_migrate_scanned', 'compact_free_scanned', 'compact_isolated',
      'pgmigrate_success', 'pgmigrate_fail', 'thp_fault_fallback', 'thp_collapse_alloc_failed'}


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as exc:
        return 'ERROR:' + str(exc)


def host():
    vm = dict(line.split() for line in read(FILES['vmstat']).splitlines() if len(line.split()) == 2)
    pids = []
    for path in glob.glob('/proc/[0-9]*/comm'):
        name = read(path)
        if name in ('ninja', 'clang', 'clang-23', 'cc1', 'cc1plus', 'ld.lld', 'kcompactd0', 'kswapd0', 'shelf-compute', 'shelf-server'):
            pid = path.split('/')[2]
            pids.append({'pid': int(pid), 'name': name, 'stat': read('/proc/' + pid + '/stat')})
    return {'vm': {k: int(v) for k, v in vm.items() if k in VM},
            'cpu': read(FILES['cpu_stat']).splitlines()[0],
            'cpu_pressure': read(FILES['cpu_pressure']),
            'memory_pressure': read(FILES['memory_pressure']), 'processes': pids}


def find_kcompactd():
    for path in glob.glob('/proc/[0-9]*/comm'):
        if read(path) == 'kcompactd0':
            return path.split('/')[2]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=float, default=0.05)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == '--' else a.command
    if not command or a.interval <= 0:
        p.error('positive interval and command required')
    stop = threading.Event()
    compact_pid = find_kcompactd()
    def sample():
        next_host = 0.0
        with open(a.output, 'w', buffering=1) as out:
            while not stop.is_set():
                begin = time.monotonic()
                data = {'t': begin, 'gpu': {k: read(v) for k, v in FILES.items()
                                             if k not in ('vmstat', 'cpu_stat', 'cpu_pressure', 'memory_pressure')}}
                data['gpu_end'] = time.monotonic()
                if compact_pid is not None:
                    data['kcompactd_stat'] = read('/proc/' + compact_pid + '/stat')
                if begin >= next_host:
                    data['host'] = host()
                    next_host = begin + 1.0
                data['end'] = time.monotonic()
                out.write(json.dumps(data, separators=(',', ':')) + '\n')
                stop.wait(max(0, a.interval - (time.monotonic() - begin)))
    def process_snapshots():
        with open(a.output + '.pids', 'w', buffering=1) as out:
            for delay in (30, 40):
                if stop.wait(delay):
                    break
                start = time.monotonic()
                try:
                    result = subprocess.run(['rocm-smi', '--showpids'], capture_output=True,
                                            text=True, timeout=15, check=False)
                    users = subprocess.run(['fuser', '/dev/dri/renderD130'], capture_output=True, text=True, timeout=15, check=False)
                    out.write(json.dumps({'t': start, 'rc': result.returncode,
                                          'stdout': result.stdout, 'stderr': result.stderr,
                                          'render_fuser': users.stdout.strip()}) + '\n')
                except (OSError, subprocess.TimeoutExpired) as exc:
                    out.write(json.dumps({'t': start, 'error': str(exc)}) + '\n')
    thread = threading.Thread(target=sample)
    pids_thread = threading.Thread(target=process_snapshots)
    thread.start()
    pids_thread.start()
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        stop.set()
        thread.join()
        pids_thread.join()


if __name__ == '__main__':
    raise SystemExit(main())
