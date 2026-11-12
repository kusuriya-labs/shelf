#!/usr/bin/env python3
"""Merge separately collected diagnostic runs, keeping the original trial keys."""
import json
from pathlib import Path
import sys

out=Path(sys.argv[1]); more=Path(sys.argv[2]); data=json.loads(out.read_text()); new=json.loads(more.read_text())
def key(t): return (t['c'],t['kind'],t['budget'],t['repeat'],t['diagnostic'])
existing={key(t) for t in data['trials']}
for t in new['trials']:
    if key(t) in existing: raise SystemExit(f'duplicate trial {key(t)}')
    data['trials'].append(t);existing.add(key(t))
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(out)
print('merged',len(new['trials']),'trials; total',len(data['trials']))
