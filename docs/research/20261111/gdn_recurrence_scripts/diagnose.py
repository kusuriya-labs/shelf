#!/usr/bin/env python3
"""Capture fixed-prefix logits for the three large-margin chunk divergences."""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import measure as m
original=m.OUT
m.OUT=Path(os.environ.get('GDN_DIAG_RESULTS','/tmp/gdn-divergence-diagnostic')); m.OUT.mkdir(parents=True,exist_ok=True)
m.BIN=os.environ.get('GDN_DIAG_BINARY','/tmp/gdn-divergence-compute-diagnostic')
ids=json.loads((original/'prompts.json').read_text())['ids']
summary=[]
for i,k in [(9,223),(15,28),(19,34)]:
 baseline=json.loads((original/f'base-{i}.json').read_text()); token_ids=ids[i]; target=len(token_ids)+k
 with tempfile.TemporaryDirectory() as tmp:
  logits=Path(tmp)/'base.bin'
  cmd=['flock','/tmp/r9700.lock',m.BIN,'--model-dir',m.MODEL,'--input-ids-file','/dev/stdin','--max-new-tokens','512','--max-seq-len','1024','--arena-gib','auto','--device','0','--temperature','0','--top-p','1','--top-k','0','--seed','0','--dump-logits',str(logits)]
  run=subprocess.run(cmd,input='\n'.join(map(str,token_ids))+'\n',text=True,capture_output=True,check=True)
  info=dict(line.split('=',1) for line in run.stdout.splitlines() if '=' in line)
  actual=[int(x) for x in info['GENERATED_IDS'].split(',') if x]
  assert actual==baseline['tokens']
  steps=[int(x) for x in info['LOGITS_STEP_ROWS'].split(',')]
  offsets=np.cumsum([0]+steps[:-1])[int(info['PREFILL_CHUNKS']):]
  vocab=int(info['LOGITS_VOCAB']); raw=np.memmap(logits,dtype=np.float32,mode='r').reshape(-1,vocab)
  reference=np.array(raw[offsets[k]],dtype=np.float64)
 vectors={}
 for mode in ('serial','chunk'):
  f=m.OUT/f'{mode}-{i}.bin'; f.unlink(missing_ok=True)
  (m.OUT/f"{mode}-1-{i}-diagnostic.json").unlink(missing_ok=True)
  os.environ['SHELF_GDN_DIAG_LOGITS']=str(f); os.environ['SHELF_GDN_DIAG_POSITION']=str(target)
  m.serve(mode,1,[i],ids,tag='-diagnostic')
  output=json.loads((m.OUT/f'{mode}-1-{i}-diagnostic.json').read_text())['outputs'][str(i)]
  assert output[:k]==baseline['tokens'][:k], (mode,i,'committed prefix differs')
  lines=re.findall(r'GDN_DIAG position=(\d+) target=(\d+) row=(\d+) vocab=(\d+) ids=([\d,]+)',(m.OUT/f'{mode}-1-{i}-diagnostic.err').read_text())
  assert lines and f.exists(),(mode,i,'no rows')
  rows=np.memmap(f,dtype=np.float32,mode='r').reshape(-1,vocab)
  assert len(rows)==len(lines)
  candidates=[]
  for j,(pos,t,row,v,token_string) in enumerate(lines):
   pos=int(pos); t=int(t); row=int(row); tokens=list(map(int,token_string.split(','))); start=pos-len(token_ids)
   assert t==target and int(v)==vocab and len(tokens)==row+1
   if start>=0 and tokens==baseline['tokens'][start:start+len(tokens)]: candidates.append(j)
  assert candidates,(mode,i,lines)
  vectors[mode]=np.array(rows[candidates[-1]],dtype=np.float64)
  del rows
 os.environ.pop('SHELF_GDN_DIAG_LOGITS',None); os.environ.pop('SHELF_GDN_DIAG_POSITION',None)
 def compare(a,b):
  al=a-a.max(); bl=b-b.max(); pa=np.exp(al); pb=np.exp(bl); pa/=pa.sum(); pb/=pb.sum()
  mask=pa>0
  return {'kl':float(np.sum(pa[mask]*((al[mask]-np.log(np.exp(al).sum()))-(bl[mask]-np.log(np.exp(bl).sum()))))), 'max_abs':float(np.max(np.abs(a-b))), 'top1_match':int(np.argmax(a))==int(np.argmax(b)), 'top1_a':int(np.argmax(a)),'top1_b':int(np.argmax(b))}
 result={'prompt':i,'generated_position':k,'baseline_margin':baseline['margins'][k], 'serial_vs_baseline':compare(reference,vectors['serial']),'chunk_vs_baseline':compare(reference,vectors['chunk']),'chunk_vs_serial':compare(vectors['serial'],vectors['chunk'])}
 summary.append(result); print(json.dumps(result),flush=True)
(m.OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
