import collections
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
number = r'([0-9.]+)'
spec = re.compile(r'DFCOST sessions=(\d+) rows=(\d+) wall=' + number + r' draft_gpu=' + number + r' verify_gpu=' + number)
base = re.compile(r'STEPWALL rows=(\d+) pre=' + number + r' exec_submit=' + number + r' sync=' + number + r' post=' + number + r' total=' + number)
for path in sorted(root.glob('*.err')):
    groups = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        m = spec.search(line) if path.name.startswith('s') else base.search(line)
        if m:
            values = tuple(map(float, m.groups()))
            key = tuple(map(int, values[:2])) if path.name.startswith('s') else (int(values[0]),)
            groups[key].append(values[len(key):])
    for key, values in sorted(groups.items()):
        if len(values) < 5:
            continue
        medians = [statistics.median(column) for column in zip(*values)]
        print(path.stem, *key, len(values), *(f'{x:.3f}' for x in medians))
