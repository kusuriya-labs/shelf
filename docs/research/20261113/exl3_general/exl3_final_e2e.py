import os, json, subprocess, re
from pathlib import Path
root=Path.cwd(); d=root/'build-gfx1201/exl3-general'
e=os.environ.copy();e['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55';e['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24';draft='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2';out=[]
for spec in (False,True):
 for repeat in range(3):
  for variant in (['base','candidate'] if repeat%2==0 else ['candidate','base']):
   prompt='code';exe=d/'shelf-compute.base' if variant=='base' else d/'shelf-compute.final'
   cmd=['flock','-w','300','/tmp/r9700.lock','timeout','150',str(exe),'--model-dir',model,'--input-ids-file',str(root / f'docs/research/20261113/exl3_general/prompt_{prompt}.ids'),'--max-new-tokens','128','--max-seq-len','1024','--temperature','0']
   if spec:cmd+=['--dflash2-model-dir',draft,'--dflash2-drafts','7']
   p=subprocess.run(cmd,env=e,text=True,capture_output=True);text=p.stdout+'\n'+p.stderr
   fields=dict(re.findall(r'^([A-Z_]+)=(.*)$',text,re.M));rec={'variant':variant,'repeat':repeat,'spec':spec,'prompt':prompt,'exit':p.returncode,'fields':fields,'stderr_tail':p.stderr[-2000:]};out.append(rec);(d/'final-e2e.json').write_text(json.dumps(out,indent=2)+'\n');print(variant,spec,prompt,p.returncode,fields.get('DECODE_TOKENS_PER_SEC'),flush=True)
   if p.returncode:raise SystemExit(p.returncode)
