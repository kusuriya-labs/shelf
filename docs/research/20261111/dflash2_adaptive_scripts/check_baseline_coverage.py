import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
counts = collections.Counter()
for path in root.glob('base_*.err'):
    output = path.with_suffix('.out')
    if not output.exists() or not output.read_text().startswith('{') or 'Traceback (most recent call last)' in output.read_text():
        continue
    for line in path.read_text().splitlines():
        match = re.search(r'STEPWALL rows=(\d+) ', line)
        if match:
            counts[int(match[1])] += 1
missing = [batch for batch in range(1, 65) if counts[batch] < 5]
print('missing B:', missing)
raise SystemExit(1 if missing else 0)
