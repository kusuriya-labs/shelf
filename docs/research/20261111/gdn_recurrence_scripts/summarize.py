#!/usr/bin/env python3
"""Summarize saved results, preserving censored and incomplete outputs explicitly."""
import json, os, statistics, re
from pathlib import Path
P=Path(os.environ.get('GDN_RESULTS','/tmp/gdn-divergence-results'))
prompts=json.loads((P/'prompts.json').read_text())['prompts']
def percentile(xs,q):
 xs=sorted(xs); j=(len(xs)-1)*q; a=int(j); b=min(a+1,len(xs)-1); return xs[a]+(xs[b]-xs[a])*(j-a)
base=[json.loads((P/f'base-{i}.json').read_text()) for i in range(len(prompts))]
for mode,n in [('base',8),('serial',1),('chunk',1),('serial',8),('chunk',8),('exact',1),('exact',8)]:
 results={}; rounds=accepted=0
 for start in ([0] if n == 1 else range(0,len(base),n)):
  tag='-group' if n == 1 else ''
  data=json.loads((P/f'{mode}-{n}-{start}{tag}.json').read_text()); results.update(data['outputs']); m=data['metrics']
  if 'DFLASH2_ROUNDS' in m:
   text=(P/f'{mode}-{n}-{start}{tag}.err').read_text()
   counts=[int(x) for x in re.findall(r'DFLASH2_POSITION=\d+-\d+ ROUNDS=(\d+)',text)]
   assert counts, 'missing per-request position rounds'
   rounds+=sum(counts); accepted+=int(m['DFLASH2_ACCEPTED_DRAFTS'])
 lens=[]; margins=[]; full=0; censored=0; early=0
 for i,b in enumerate(base):
  a=b['tokens']; z=results[str(i)]; k=next((j for j,(x,y) in enumerate(zip(a,z)) if x!=y),min(len(a),len(z)))
  lens.append(k)
  if k<min(len(a),len(z)): margins.append(b['margins'][k])
  else:
   if k>=512: full+=1
   elif len(a)==len(z): censored+=1
   else: early+=1
  print(json.dumps({'mode':mode,'n':n,'prompt':i,'first':k,'baseline_length':len(a),'output_length':len(z),'margin':b['margins'][k] if k<min(len(a),len(z)) else None}))
 print(json.dumps({'mode':mode,'n':n,'total':len(base),'median_capped':statistics.median(lens),'p10_capped':percentile(lens,.1),'match_512':full,'same_early_stop':censored,'length_mismatch':early,'margin_median':statistics.median(margins) if margins else None,'margin_max':max(margins) if margins else None,'margin_over_0_1':sum(m > 0.1 for m in margins),'rounds':rounds,'accepted':accepted,'accept_per_round':accepted/rounds if rounds else None}))
for mode in ('serial','chunk'):
 for n in (1,8):
  files=sorted(P.glob(f'{mode}-{n}-*-prefix.json'))
  if not files: continue
  rounds=accepted=total=0
  for f in files:
   d=json.loads(f.read_text()); total+=len(d['outputs']); accepted+=int(d['metrics']['DFLASH2_ACCEPTED_DRAFTS'])
   counts=[int(x) for x in re.findall(r'DFLASH2_POSITION=\d+-\d+ ROUNDS=(\d+)',f.with_suffix('.err').read_text())]
   assert counts
   rounds+=sum(counts)
  print(json.dumps({'mode':mode,'n':n,'prefix_requests':total,'prefix_rounds':rounds,'prefix_accepted':accepted,'prefix_accept_per_round':accepted/rounds if rounds else None}))
