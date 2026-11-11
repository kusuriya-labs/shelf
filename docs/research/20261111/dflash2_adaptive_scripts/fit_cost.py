import collections
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
base_root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else root
base_pattern = re.compile(r'STEPWALL rows=(\d+) .* total=([0-9.]+)')
spec_pattern = re.compile(r'DFHOST sessions=(\d+) rows=(\d+) wall=([0-9.]+) cpu=([0-9.]+)')
base_samples = collections.defaultdict(list)
for path in base_root.glob('base_*.err'):
    result = path.with_suffix('.out')
    if not result.exists() or (not result.read_text().startswith('{') or 'Traceback (most recent call last)' in result.read_text()):
        continue
    for line in path.read_text().splitlines():
        match = base_pattern.search(line)
        if match:
            base_samples[int(match[1])].append(float(match[2]))
base = {batch: statistics.median(samples) for batch, samples in base_samples.items() if len(samples) >= 5}
missing = [batch for batch in range(1, 65) if batch not in base]
if missing:
    raise SystemExit('試料不足の B: ' + ', '.join(map(str, missing)))

points = sorted(base)
def baseline(batch):
    return 0.0 if batch == 0 else base[batch]

spec = []
for sessions in range(1, 9):
    row = []
    for drafts in range(1, 8):
        path = root / f's{sessions}k{drafts}.err'
        samples = []
        if path.exists() and path.with_suffix('.out').exists() and path.with_suffix('.out').read_text().startswith('{') and 'Traceback (most recent call last)' not in path.with_suffix('.out').read_text():
            for line in path.read_text().splitlines():
                match = spec_pattern.search(line)
                if match and int(match[1]) == sessions and int(match[2]) == sessions * (drafts + 1):
                    samples.append(float(match[3]))
        if len(samples) < 5:
            raise SystemExit(f'試料不足: S={sessions} K={drafts} ({len(samples)} samples)')
        row.append(statistics.median(samples))
    spec.append(row)

print('baseline = {')
print(', '.join(f'{baseline(batch):.4f}' for batch in range(65)))
print('}')
print('spec = {')
for row in spec:
    print('    {' + ', '.join(f'{value:.4f}' for value in row) + '},')
print('}')
print('measured B =', ', '.join(map(str, points)), file=sys.stderr)
