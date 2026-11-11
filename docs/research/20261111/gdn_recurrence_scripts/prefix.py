#!/usr/bin/env python3
"""Replay each mode up to the common observed prefix; requires measure.py results."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import OUT, PROMPTS, serve
prompts=json.loads((OUT/'prompts.json').read_text()); ids=prompts['ids']
for n in (1,8):
 step=len(ids) if n == 1 else n
 for start in range(0,len(ids),step):
  ix=list(range(start,min(start+step,len(ids))))
  outputs={m:{} for m in ('serial','chunk')}
  for i in ix:
   group_start=(i//n)*n
   for m in outputs:
    outputs[m].update(json.loads((OUT/(f'{m}-1-0-group.json' if n == 1 else f'{m}-{n}-{group_start}.json')).read_text())['outputs'])
  limits={}
  for i in ix:
   baseline=json.loads((OUT/f'base-{i}.json').read_text())['tokens']
   def common(other):
    return next((k for k,(a,b) in enumerate(zip(baseline,other)) if a!=b),min(len(baseline),len(other)))
   limits[i]=min(common(outputs[m][str(i)]) for m in outputs)
  # A zero-length prefix cannot generate a round and is excluded from this replay.
  ix=[i for i in ix if limits[i]>0]
  if not ix: continue
  for m in ('serial','chunk'):
   serve(m,n,ix,ids,limits,tag='-prefix')
   result=json.loads((OUT/f'{m}-{n}-{ix[0]}-prefix.json').read_text())['outputs']
   for i in ix:
    baseline=json.loads((OUT/f'base-{i}.json').read_text())['tokens']
    assert result[str(i)]==baseline[:limits[i]], (m,n,i,limits[i])
