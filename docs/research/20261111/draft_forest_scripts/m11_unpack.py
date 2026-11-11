#!/usr/bin/env python3
"""Recreate per-run JSON files from the checked-in M11 archive."""
import json
import sys
from pathlib import Path
if len(sys.argv) != 3:
    raise SystemExit('usage: m11_unpack.py ARCHIVE.json OUTPUT_DIR')
data=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
for tag, runs in data['runs'].items():
    for run in runs:
        path=out/f'{tag}-{run["prompt_index"]}-{run["mode"]}-{run["rep"]}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(run))
