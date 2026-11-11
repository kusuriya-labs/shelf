import sys, os, json, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/server'))
from support import prompt_ids
PROMPTS = [
 'Write a detailed, long tutorial with many shell command examples for finding, sorting, inspecting, and safely managing large files in a directory.',
 'Write a complete Python module implementing longest common subsequence, with documentation, several algorithms, validation, command-line examples, and many tests.',
 'Explain in detail why the square root of two is irrational, then develop several different proofs and discuss related mathematical examples at length.',
]
root=Path('/tmp/gdn-exact-round'); root.mkdir(exist_ok=True)
model=os.environ['SHELF_MODEL_DIR']; draft=os.environ['SHELF_DFLASH2_MODEL_DIR']; binary=str(Path.cwd()/'build-gfx1201/shelf-compute')
def worker(mode, i, rep):
 ids=prompt_ids(PROMPTS[i]); env=dict(os.environ); env['SHELF_DFLASH2_SERVE_STATS']='1'
 env.pop('SHELF_GDN_VERIFY_CHUNKED',None); env.pop('SHELF_GDN_VERIFY_CHUNKED_EXACT',None)
 if mode in ('chunk','exact'): env['SHELF_GDN_VERIFY_CHUNKED']='1'
 if mode=='exact': env['SHELF_GDN_VERIFY_CHUNKED_EXACT']='1'
 args=[binary,'--model-dir',model,'--serve-stdio','--device','0','--max-seq-len','512','--max-concurrent-requests','1','--prefix-cache','0','--kv-cache-dtype','bf16','--temperature','0','--dflash2-model-dir',draft,'--dflash2-drafts','7']
 err=root/f'{mode}-{i}-{rep}.err'
 with err.open('w') as ef:
  p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ef,text=True,env=env,bufsize=1)
  def send(x): p.stdin.write(json.dumps(x)+'\n'); p.stdin.flush()
  def recv():
   line=p.stdout.readline()
   if not line: raise RuntimeError('compute ended')
   return json.loads(line)
  send({'op':'ping'})
  while recv().get('event')!='pong': pass
  send({'op':'generate','request_id':1,'input_ids':ids,'max_new_tokens':256,'temperature':0,'seed':0})
  first=None; last=None
  while True:
   e=recv()
   if e.get('event')=='token':
    last=time.monotonic(); first=first or last
   if e.get('event')=='error': raise RuntimeError(str(e))
   if e.get('event')=='done': tokens=e['generated_ids']; break
  send({'op':'shutdown'}); p.stdin.close(); p.wait(timeout=60)
  if p.returncode: raise RuntimeError(f'exit {p.returncode}')
 metrics={}
 for line in err.read_text().splitlines():
  if line.startswith('DFLASH2_') and '=' in line:
   k,v=line.split('=',1); metrics[k]=v
 obj={'tokens':tokens,'metrics':metrics,'decode_tps':(len(tokens)-1)/(last-first) if last and first and last>first else None}
 (root/f'{mode}-{i}-{rep}.json').write_text(json.dumps(obj))
 print('DONE',mode,i,rep,'tokens',len(obj['tokens']),flush=True)
if __name__=='__main__':
 worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
