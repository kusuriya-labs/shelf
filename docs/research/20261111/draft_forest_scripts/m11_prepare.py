#!/usr/bin/env python3
"""Prepare non-speculative reference IDs and top-two margins, under flock."""
import os
import sys
from pathlib import Path
os.environ['GDN_RESULTS']=os.environ.get('M11_BASE','/tmp/gdn-divergence-results')
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'docs/research/20261111/gdn_recurrence_scripts'))
from measure import PROMPTS, baseline, prompt_ids
for i, prompt in enumerate(PROMPTS):
    baseline(i,prompt_ids(prompt))
