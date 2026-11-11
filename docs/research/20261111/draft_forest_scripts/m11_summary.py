#!/usr/bin/env python3
"""Summarize complete paired M11 measurements and numerical gates."""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
out=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/forest-m11')
modes=['chain','10','12','16']
records={}
for p in out.glob('speed-*.json'):
 d=json.loads(p.read_text()); records[d['prompt_index'],d['mode'],d['rep']]=d
print('complete runs:',len(records),'/ 192')
rows=[]
for i in range(24):
 if any((i,m,r) not in records for m in modes for r in range(2)): continue
 v={m:st.mean(records[i,m,r]['tps'] for r in range(2)) for m in modes}
 rows.append((i,v,{m:v[m]/v['chain'] for m in modes}))
print('\n| prompt | chain tok/s | N10 | N12 | N16 |')
print('|---:|---:|---:|---:|---:|')
for i,v,r in rows: print(f'| {i:02d} | {v["chain"]:.2f} | '+ ' | '.join(f'{v[m]:.2f} ({(r[m]-1)*100:+.2f}%)' for m in modes[1:])+' |')
print('\n| N | arithmetic % | geometric % | median % | p10 % | slower |')
print('|---|---:|---:|---:|---:|---:|')
for m in modes[1:]:
 rs=[r[m] for _,_,r in rows]
 if rs: print(f'| {m} | {(st.mean(rs)-1)*100:+.2f} | {(math.exp(st.mean(map(math.log,rs)))-1)*100:+.2f} | {(st.median(rs)-1)*100:+.2f} | {(np.quantile(rs,.1)-1)*100:+.2f} | {sum(x<1 for x in rs)}/{len(rs)} |')
print('\n| mode | repeats | mean tok/s | sample SD | min | max |')
print('|---|---:|---:|---:|---:|---:|')
for m in modes:
 vals=[json.loads(p.read_text())['tps'] for p in out.glob(f'repeat-2-{m}-*.json')]
 if len(vals)>=2: print(f'| {m} | {len(vals)} | {st.mean(vals):.2f} | {st.stdev(vals):.2f} | {min(vals):.2f} | {max(vals):.2f} |')
print('\n| mode | prompts | exact | mismatches | median margin | max margin |')
print('|---|---:|---:|---:|---:|---:|')
for m in modes:
 ds=[records[i,m,0] for i,_,_ in rows]; margins=[d['margin'] for d in ds if d['margin'] is not None]
 if ds: print(f'| {m} | {len(ds)} | {len(ds)-len(margins)} | {len(margins)} | {st.median(margins) if margins else 0:.5f} | {max(margins,default=0):.5f} |')
print('\n| category | count | N10 | N12 | N16 |')
print('|---|---:|---:|---:|---:|')
cats={'code':[1,5,12,13,21,23],'japanese':[3,4,5,6,7,8,9,10,22,23],'reasoning':[2,14,18,19,20]}
for label,ids in cats.items():
 sel=[r for i,_,r in rows if i in ids]
 if sel: print(f'| {label} | {len(sel)} | '+' | '.join(f'{(st.mean(r[m] for r in sel)-1)*100:+.2f}%' for m in modes[1:])+' |')
