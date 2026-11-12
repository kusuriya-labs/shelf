#!/usr/bin/env python3
"""Print compact tables from summarized results; no GPU access required."""
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parents[4]
path=Path(os.environ.get('FOREST_RESULTS', root/'docs/research/20261111/forest_concurrency_bottleneck_results.json'))
data=json.loads(path.read_text())
groups=defaultdict(list)
for t in data['trials']:
    groups[t['diagnostic'],t['c'],t['kind'],t['budget']].append(t)

for (mode,c,kind,budget), trials in sorted(groups.items(),key=lambda x:(str(x[0][0]),x[0][1:])):
    if mode is False:
        speeds=[next(iter(t['compute_request_logs'].values()))['decode_tok_s'] if c==1 and t.get('compute_request_logs') else t['all_active_tps'] for t in trials]
        print('speed',c,kind,budget,len(trials),*(f'{v:.2f}' for v in speeds),'mean',f'{mean(speeds):.2f}')
    else:
        sums=[t['diagnostic_summary']['common_interval'] for t in trials if t.get('diagnostic_summary')]
        sums=[s for s in sums if s]
        if not sums: continue
        n=sum(s['rounds'] for s in sums)
        values=lambda k:sum(s[k] for s in sums)/n
        phases={k:sum(s['phase_ms'][k] for s in sums)/n for k in sums[0]['phase_ms']}
        print('phase',mode,c,kind,budget,'n',n,'emit',f"{values('emitted'):.3f}",'accepted',f"{values('accepted'):.3f}",
              'wall',f"{values('wall_ms'):.3f}",'host',f"{values('host_ms'):.3f}",
              'draftgpu',f"{values('draft_gpu_ms'):.3f}",'verifygpu',f"{values('verify_gpu_ms'):.3f}",
              'rate',f"{1000*values('emitted')/values('wall_ms'):.2f}",
              'stages',','.join(f'{k}:{v:.3f}' for k,v in phases.items()))
