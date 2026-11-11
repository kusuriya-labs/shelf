#!/usr/bin/env python3
"""Run the first batch and three single-request cases early; main run resumes them."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import OUT, serve
ids=json.loads((OUT/'prompts.json').read_text())['ids']
serve('base',8,list(range(8)),ids)
for i in range(3):
 for mode in ('serial','chunk'): serve(mode,1,[i],ids)
