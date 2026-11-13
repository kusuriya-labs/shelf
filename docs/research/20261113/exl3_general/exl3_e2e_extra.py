import os,json,subprocess,re
from pathlib import Path
root=Path.cwd();d=root/'build-gfx1201/exl3-general';env=os.environ.copy();env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55';env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24';draft='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2';out=[]
for spec in (False,True):
 jobs=[]
 for repeat in (3,4):
  for v in (['candidate','base'] if repeat%2 else ['base','candidate']):jobs.append((v,'code',repeat))
 jobs.extend([('candidate','prose',0),('candidate','ja',0)])
 for variant,prompt,repeat in jobs:
  exe=d/'shelf-compute.base' if variant=='base' else root/'build-gfx1201/shelf-compute'
  cmd=['flock','-w','180','/tmp/r9700.lock','timeout','150',str(exe),'--model-dir',model,'--input-ids-file',str(root/f'docs/research/20261113/exl3_general/prompt_{prompt}.ids'),'--max-new-tokens','128','--max-seq-len','1024','--temperature','0']
  if spec:cmd+=['--dflash2-model-dir',draft,'--dflash2-drafts','7']
  p=subprocess.run(cmd,env=env,text=True,capture_output=True);fields=dict(re.findall(r'^([A-Z_]+)=(.*)$',p.stdout+'\n'+p.stderr,re.M));rec={'variant':variant,'repeat':repeat,'spec':spec,'prompt':prompt,'exit':p.returncode,'fields':fields,'stderr_tail':p.stderr[-2000:]};out.append(rec);(d/'group-e2e-extra.json').write_text(json.dumps(out,indent=2)+'\n');print(variant,spec,prompt,repeat,p.returncode,fields.get('DECODE_TOKENS_PER_SEC'),flush=True)
  if p.returncode:raise SystemExit(p.returncode)
