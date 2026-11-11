import collections
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'STEPWALL rows=(\d+) .* total=([0-9.]+) cpu=([0-9.]+)')
data = collections.defaultdict(list)
for path in root.glob('base_*.err'):
    result = path.with_suffix('.out')
    if not result.exists() or (not result.read_text().startswith('{') or 'Traceback (most recent call last)' in result.read_text()):
        continue
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            data[int(match[1])].append((float(match[2]), float(match[3])))
print('| B | 標本 | wall ms | CPU ms |')
print('| ---: | ---: | ---: | ---: |')
for batch, values in sorted(data.items()):
    if len(values) >= 5:
        print(f'| {batch} | {len(values)} | {statistics.median(v[0] for v in values):.3f} | '
              f'{statistics.median(v[1] for v in values):.3f} |')
