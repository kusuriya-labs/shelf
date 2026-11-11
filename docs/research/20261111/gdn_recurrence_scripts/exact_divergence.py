#!/usr/bin/env python3
"""Measure exact chunk using the previous experiment's prompts and baseline."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure
ids = json.loads((measure.OUT / 'prompts.json').read_text())['ids']
measure.serve('exact', 1, list(range(len(ids))), ids, tag='-group')
for start in range(0, len(ids), 8):
    measure.serve('exact', 8, list(range(start, min(start + 8, len(ids)))), ids)
