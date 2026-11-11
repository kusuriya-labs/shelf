#!/usr/bin/env python3
"""Create a compact, portable archive of token lists, margins and round counts."""
import json, os, re
from pathlib import Path
p=Path(os.environ.get('GDN_RESULTS','/tmp/gdn-divergence-results'))
out=Path(os.environ.get('GDN_ARCHIVE',str(p/'archive.json')))
files={}
for f in sorted(p.glob('*.json')):
 if f==out or f.name=='archive.json': continue
 data=json.loads(f.read_text())
 err=f.with_suffix('.err')
 if err.exists():
  counts=[int(x) for x in re.findall(r'DFLASH2_POSITION=\d+-\d+ ROUNDS=(\d+)',err.read_text())]
  if counts: data['request_rounds']=sum(counts)
 files[f.name]=data
out.write_text(json.dumps(files,ensure_ascii=False,separators=(',',':'))+'\n')
print(out, len(files))
