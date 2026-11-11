#!/usr/bin/env python3
"""Greedy divergence experiment. Run under flock; output files are resumable."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/server'))
from support import prompt_ids
OUT = Path(os.environ.get('GDN_RESULTS', '/tmp/gdn-divergence-results')); OUT.mkdir(parents=True, exist_ok=True)
MODEL=os.environ['SHELF_MODEL_DIR']; DRAFT=os.environ['SHELF_DFLASH2_MODEL_DIR']; BIN=str(ROOT/'build-gfx1201/shelf-compute')
PROMPTS = [
'Write a detailed, long tutorial with many shell command examples for finding, sorting, inspecting, and safely managing large files in a directory.',
'Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests.',
'Explain in detail why the square root of two is irrational, then develop several different proofs and discuss related mathematical examples at length.',
'日本の四季について、それぞれの特徴を詳しく説明してください。', '量子コンピュータの基本原理を平易に、例を挙げて説明して。', 'Pythonでクイックソートを実装して、計算量とテストを詳しく解説して。',
'コーヒーの淹れ方のコツを、道具と工程ごとに詳しく教えてください。', 'ブラックホールとは何か、観測方法も含めて詳しく説明して。', '健康的な朝食のメニューを一週間分提案し、それぞれの栄養を説明して。',
'機械学習の過学習とは何か、具体例と対策を詳しく説明して。', '京都の観光名所を5つ挙げ、歴史と訪問時の見どころを詳しく説明して。',
'Explain the design of a reliable distributed job queue, including failures, retries, ordering, and concrete examples.',
'Write a detailed guide to debugging memory leaks in a C++ application on Linux, with commands and examples.',
'Implement a small CSV parser in Python using a state machine. Include edge cases, tests and explanatory comments.',
'Compare breadth-first and depth-first search using worked examples, pseudocode, complexity and applications.',
'Explain the principles of database transactions and isolation levels with detailed concurrent examples.',
'Write a bash script to back up a directory safely with rsync, verify the result, rotate old backups and explain each step.',
'Explain how to investigate high disk usage on Linux using find, du, sort and related shell commands, with examples.',
'Derive the quadratic formula from completing the square, and solve multiple examples including complex roots.',
'Explain Bayes theorem with a worked medical screening example and discuss common errors in interpretation.',
'A train travels between two cities with changing speeds. Construct and solve several nontrivial average speed problems step by step.',
'Explain how a compiler transforms a program from source code to machine code, using a concrete example at each stage.',
'森林生態系における水と栄養の循環を、具体例を交えながら詳しく説明してください。',
'ある図書館の本を分類するシステムを設計してください。要件、データ構造、検索、テストを順に説明してください。',
]
def save(path,obj):
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False)); tmp.replace(path)
def baseline(i,ids):
 path=OUT/f'base-{i}.json'
 if path.exists(): return
 with tempfile.TemporaryDirectory() as t:
  raw=Path(t)/'logits'; cmd=['flock','/tmp/r9700.lock',BIN,'--model-dir',MODEL,'--input-ids-file','/dev/stdin','--max-new-tokens','512','--max-seq-len','1024','--arena-gib','auto','--device','0','--temperature','0','--top-p','1','--top-k','0','--seed','0','--dump-logits',str(raw)]
  p=subprocess.run(cmd,input='\n'.join(map(str,ids))+'\n',text=True,capture_output=True,check=True)
  d=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
  tokens=[int(x) for x in d['GENERATED_IDS'].split(',') if x.strip()]; steps=[int(x) for x in d['LOGITS_STEP_ROWS'].split(',')]; offsets=np.cumsum([0]+steps[:-1])[int(d['PREFILL_CHUNKS']):]; rows=np.memmap(raw,dtype=np.float32,mode='r').reshape(-1,int(d['LOGITS_VOCAB']))
  assert len(tokens)==len(offsets), (len(tokens),len(offsets))
  margins=[]
  for k,tok in zip(offsets,tokens):
   row=rows[k]; assert int(np.argmax(row))==tok
   v=np.partition(row,-2)[-2:]; margins.append(float(v.max()-v.min()))
  save(path,{'prompt':PROMPTS[i],'input_ids':ids,'tokens':tokens,'margins':margins})
 print('done baseline',i,len(tokens),flush=True)
def serve(mode,n,indices,ids,limits=None,tag=''):
 path=OUT/f'{mode}-{n}-{indices[0]}{tag}.json'
 if path.exists(): return
 env=dict(os.environ); env['SHELF_DFLASH2_SERVE_STATS']='1'; env.pop('SHELF_GDN_VERIFY_CHUNKED',None); env.pop('SHELF_GDN_VERIFY_CHUNKED_EXACT',None); env.pop('SHELF_GDN_RECURRENCE_EXACT',None)
 if mode in ('chunk', 'exact'): env['SHELF_GDN_VERIFY_CHUNKED']='1'
 if mode=='exact': env['SHELF_GDN_VERIFY_CHUNKED_EXACT']='1'
 cmd=['flock','/tmp/r9700.lock',BIN,'--model-dir',MODEL,'--serve-stdio','--device','0','--max-seq-len','1024','--max-concurrent-requests',str(n),'--prefix-cache','0','--kv-cache-dtype','bf16','--temperature','0']
 if mode!='base': cmd+=['--dflash2-model-dir',DRAFT,'--dflash2-drafts','7']
 err=OUT/f'{mode}-{n}-{indices[0]}{tag}.err'
 with err.open('w') as ef:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ef,text=True,env=env,bufsize=1)
  def send(obj): p.stdin.write(json.dumps(obj)+'\n'); p.stdin.flush()
  def recv():
   line=p.stdout.readline()
   if not line: raise RuntimeError(f'server ended: {err}')
   return json.loads(line)
  send({'op':'ping'})
  while recv().get('event')!='pong': pass
  def submit(i): send({'op':'generate','request_id':i+1,'input_ids':ids[i],'max_new_tokens':limits[i] if limits is not None else 512,'temperature':0,'seed':0})
  pending=iter(indices)
  for _ in range(min(n,len(indices))): submit(next(pending))
  outputs={}
  while len(outputs)<len(indices):
   e=recv()
   if e.get('event')=='error': raise RuntimeError(str(e))
   if e.get('event')=='done':
    outputs[str(e['request_id']-1)]=e['generated_ids']
    nxt=next(pending,None)
    if nxt is not None: submit(nxt)
  send({'op':'shutdown'}); p.stdin.close(); p.wait(timeout=60)
  if p.returncode: raise RuntimeError(f'server exit {p.returncode}')
 metrics={}
 for line in err.read_text().splitlines():
  if line.startswith('DFLASH2_') and '=' in line:
   k,v=line.split('=',1); metrics[k]=v
 save(path,{'outputs':outputs,'metrics':metrics})
 print('done',mode,n,indices,[(i,len(outputs[str(i)])) for i in indices],flush=True)
def main():
 ids=[prompt_ids(x) for x in PROMPTS]; save(OUT/'prompts.json',{'prompts':PROMPTS,'ids':ids})
 for i in range(len(ids)): baseline(i,ids[i])
 for mode,n in [('base',8),('serial',1),('chunk',1),('serial',8),('chunk',8)]:
  if n == 1:
   serve(mode,1,list(range(len(ids))),ids,tag='-group')
  else:
   for start in range(0,len(ids),n): serve(mode,n,list(range(start,min(start+n,len(ids)))),ids)
if __name__=='__main__': main()
