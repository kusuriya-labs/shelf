import collections
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'STEPWALL rows=(\d+) pre=([0-9.]+) exec_submit=([0-9.]+) sync=([0-9.]+) post=([0-9.]+) total=([0-9.]+)')
values = collections.defaultdict(list)
for path in root.glob('base_*.err'):
    output = path.with_suffix('.out')
    if not output.exists() or not output.read_text().startswith('{') or 'Traceback (most recent call last)' in output.read_text():
        continue
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            values[int(match[1])].append(tuple(map(float, match.groups()[1:])))
print('| B | 標本 | pre ms | forward ms | sync ms | post ms | total ms |')
print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for batch in range(1, 65):
    data = values[batch]
    if len(data) >= 5:
        medians = [statistics.median(column) for column in zip(*data)]
        print(f'| {batch} | {len(data)} | ' + ' | '.join(f'{x:.3f}' for x in medians) + ' |')
