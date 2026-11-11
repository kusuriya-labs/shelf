#!/usr/bin/env python3
"""M11 paired single-request benchmark; launch via run_m11.sh, resumable JSONL."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/research/20261111/gdn_recurrence_scripts'))
from measure import PROMPTS
BASE=Path(os.environ.get('M11_BASE','/tmp/gdn-divergence-results'))
OUT=Path(os.environ.get('M11_OUTPUT','/tmp/forest-m11')); OUT.mkdir(exist_ok=True,parents=True)
MODEL='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
DRAFT='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
BINARY=ROOT/'build-gfx1201/shelf-compute'
MODES=('chain','10','12','16')
def run(i,mode,rep,tag='speed'):
 path=OUT/f'{tag}-{i}-{mode}-{rep}.json'
 if path.exists(): return
 base=json.loads((BASE/f'base-{i}.json').read_text())
 assert base['prompt']==PROMPTS[i]
 length='256' if tag=='repeat256' else '512'
 seq='512' if tag=='repeat256' else '1024'
 cmd=[str(BINARY),'--model-dir',MODEL,'--input-ids-file','/dev/stdin','--max-new-tokens',length,'--max-seq-len',seq,'--arena-gib','auto','--device','0','--temperature','0','--top-p','1','--top-k','0','--seed','0','--dflash2-model-dir',DRAFT,'--dflash2-drafts','7']
 if mode!='chain': cmd+=['--dflash2-tree-budget',mode]
 env=dict(os.environ,SHELF_GDN_VERIFY_CHUNKED='1',SHELF_GDN_TREE_CHUNKED='1')
 for key in ('SHELF_GDN_VERIFY_CHUNKED_EXACT','SHELF_GDN_RECURRENCE_EXACT'): env.pop(key,None)
 start=time.time(); p=subprocess.Popen(([] if os.environ.get('M11_LOCK_HELD')=='1' else ['flock','/tmp/r9700.lock'])+cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
 stdout,stderr=p.communicate('\n'.join(map(str,base['input_ids']))+'\n')
 if p.returncode: raise RuntimeError(f'{i} {mode} {rep}: {stderr[-4000:]}')
 data=dict(line.split('=',1) for line in stdout.splitlines() if '=' in line)
 toks=[int(x) for x in data['GENERATED_IDS'].split(',') if x.strip()]
 baseline=base['tokens']; mismatch=next((j for j,(x,y) in enumerate(zip(toks,baseline)) if x!=y),None)
 result={'launched_pid':p.pid,'lock_held_by_parent':os.environ.get('M11_LOCK_HELD')=='1','prompt_index':i,'mode':mode,'rep':rep,'started':start,'finished':time.time(),'tps':float(data['DECODE_TOKENS_PER_SEC']),'n_tokens':len(toks),'tokens':toks,'first_mismatch':mismatch,'margin':base['margins'][mismatch] if mismatch is not None else None,'metrics':{k:v for k,v in data.items() if k.startswith('DFLASH2_')}}
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(path)
 print(tag,i,mode,rep,result['tps'],len(toks),flush=True)
def prompt_group(i):
 order=MODES[i%4:]+MODES[:i%4]
 for rep in range(2):
  for mode in (order if rep==0 else order[::-1]): run(i,mode,rep)

def locked_group(kind, number):
 env=dict(os.environ,M11_LOCK_HELD='1')
 subprocess.run(['flock','/tmp/r9700.lock',sys.executable,__file__,kind,str(number)],env=env,check=True)

def main():
 if len(sys.argv)==3 and sys.argv[1]=='--prompt':
  prompt_group(int(sys.argv[2])); return
 if len(sys.argv)==3 and sys.argv[1]=='--repeat256':
  group=int(sys.argv[2])
  for rep in range(group*5,(group+1)*5): run(2,'10',rep,'repeat256')
  return
 if len(sys.argv)==3 and sys.argv[1]=='--repeat':
  rep=int(sys.argv[2]); order=MODES[rep%4:]+MODES[:rep%4]
  for mode in order: run(2,mode,rep,'repeat')
  return
 for i in range(len(PROMPTS)): locked_group('--prompt',i)
 for rep in range(10): locked_group('--repeat',rep)

if __name__ == "__main__": main()
