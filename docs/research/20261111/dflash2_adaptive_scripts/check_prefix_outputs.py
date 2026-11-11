import json
import pathlib
import sys

prefix_root, baseline_root = map(pathlib.Path, sys.argv[1:3])
failures = 0
for sessions in range(1, 9):
    fixed_path = prefix_root / f's{sessions}k7.json'
    try:
        fixed = json.loads(fixed_path.read_text())
        if len(fixed) != sessions or not all(isinstance(ids, list) and ids for ids in fixed.values()):
            fixed = None
    except (OSError, ValueError):
        fixed = None
    baseline = None
    for path in sorted(baseline_root.glob(f'base_{sessions}*.json')):
        output = path.with_suffix('.out')
        if not output.exists() or not output.read_text().startswith('{') or 'Traceback (most recent call last)' in output.read_text():
            continue
        try:
            candidate = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if len(candidate) == sessions and all(isinstance(ids, list) and ids for ids in candidate.values()):
            baseline = candidate
            break
    for drafts in range(1, 8):
        path = prefix_root / f's{sessions}k{drafts}.json'
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            result = None
        fixed_ok = fixed is not None and result == fixed
        baseline_ok = baseline is not None and result == baseline
        print(f'S={sessions} K={drafts}: fixed=' + ('一致' if fixed_ok else '未確認または不一致') +
              ' target=' + ('一致' if baseline_ok else '未確認または不一致'))
        failures += not fixed_ok or not baseline_ok
raise SystemExit(1 if failures else 0)
