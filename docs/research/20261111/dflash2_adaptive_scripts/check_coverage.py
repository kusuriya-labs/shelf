import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'DFCOST sessions=(\d+) rows=(\d+) ')
missing = []
for sessions in range(1, 9):
    for drafts in range(1, 8):
        path = root / f's{sessions}k{drafts}.err'
        count = 0
        if path.exists() and path.with_suffix('.out').exists() and path.with_suffix('.out').read_text().startswith('{') and 'Traceback (most recent call last)' not in path.with_suffix('.out').read_text():
            for line in path.read_text().splitlines():
                match = pattern.search(line)
                if match and int(match[1]) == sessions and int(match[2]) == sessions * (drafts + 1):
                    count += 1
        if count < 5:
            missing.append((sessions, drafts, count))
print(f'covered: {56 - len(missing)}/56')
for sessions, drafts, count in missing:
    print(f'S={sessions} K={drafts} samples={count}')
