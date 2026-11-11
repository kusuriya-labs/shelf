import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(
    r'DFCOST sessions=(\d+) rows=(\d+) wall=([0-9.]+) '
    r'draft_gpu=([0-9.]+) verify_gpu=([0-9.]+)'
)
print('| S | K | R | samples | round ms | draft GPU ms | verify GPU ms | residual ms |')
print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for sessions in range(1, 9):
    for drafts in range(1, 8):
        path = root / f's{sessions}k{drafts}.err'
        if not path.exists() or not path.with_suffix('.out').exists() or not path.with_suffix('.out').read_text().startswith('{') or 'Traceback (most recent call last)' in path.with_suffix('.out').read_text():
            continue
        rows = sessions * (drafts + 1)
        values = []
        for line in path.read_text().splitlines():
            m = pattern.search(line)
            if m and int(m[1]) == sessions and int(m[2]) == rows:
                wall, draft, verify = map(float, m.group(3, 4, 5))
                values.append((wall, draft, verify, wall - draft - verify))
        if not values:
            continue
        medians = [statistics.median(column) for column in zip(*values)]
        print(f'| {sessions} | {drafts} | {rows} | {len(values)} | ' +
              ' | '.join(f'{value:.3f}' for value in medians) + ' |')
