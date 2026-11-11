import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'DFCOST sessions=(\d+) rows=(\d+) wall=([0-9.]+) draft_gpu=([0-9.]+) verify_gpu=([0-9.]+) cpu=([0-9.]+)')
host_pattern = re.compile(r'DFHOST sessions=(\d+) rows=(\d+) wall=([0-9.]+) cpu=([0-9.]+)')
print('| S | K | R | 標本 | wall ms | batch CPU ms | round CPU ms |')
print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for sessions in range(1, 9):
    for drafts in range(1, 8):
        path = root / f's{sessions}k{drafts}.err'
        result = path.with_suffix('.out')
        if not path.exists() or not result.exists() or (not result.read_text().startswith('{') or 'Traceback (most recent call last)' in result.read_text()):
            continue
        rows = sessions * (drafts + 1)
        values = []
        outer = []
        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if match and int(match[1]) == sessions and int(match[2]) == rows:
                values.append((float(match[3]), float(match[6])))
            host = host_pattern.search(line)
            if host and int(host[1]) == sessions and int(host[2]) == rows:
                outer.append((float(host[3]), float(host[4])))
        if values and outer:
            cpu = statistics.median(value[1] for value in values)
            outer_wall = statistics.median(value[0] for value in outer)
            outer_cpu = statistics.median(value[1] for value in outer)
            print(f'| {sessions} | {drafts} | {rows} | {len(values)} | {outer_wall:.3f} | {cpu:.3f} | {outer_cpu:.3f} |')
