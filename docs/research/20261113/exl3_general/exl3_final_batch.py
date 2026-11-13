import json,os,subprocess
from pathlib import Path
root=Path.cwd();d=root/'build-gfx1201/exl3-general'
env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',PATH='/tmp/venv-so/bin:'+env['PATH'])
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24';draft='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
records=[]
for spec in [False,True]:
 for c in [2,4]:
  for repeat in range(3):
   for name in (['base','candidate'] if repeat%2==0 else ['candidate','base']):
    exe=d/'shelf-compute.base' if name=='base' else d/'shelf-compute.final'
    cmd=['flock','-w','300','/tmp/r9700.lock','timeout','-k','10','--signal=INT','150','/tmp/venv-so/bin/python','-I','docs/research/20261113/exl3_general/exl3_batch.py','--compute',str(exe),'--model',model,'--concurrency',str(c),'--runs','1','--tokens','128','--prompts','docs/research/20261113/exl3_general/prompt_code.ids','docs/research/20261113/exl3_general/prompt_prose.ids','docs/research/20261113/exl3_general/prompt_ja.ids']
    if spec: cmd+=['--spec','--drafter',draft]
    p=subprocess.run(cmd,env=env,text=True,capture_output=True);r={'spec':spec,'concurrency':c,'repeat':repeat,'variant':name,'exit':p.returncode,'stderr':p.stderr[-4000:]}
    try:r['result']=json.loads(p.stdout)
    except ValueError:r['stdout']=p.stdout
    records.append(r);(d/'final-batch.json').write_text(json.dumps(records,indent=2)+'\n');print(name,c,spec,repeat,p.returncode,r.get('result',{}).get('runs',[{}])[0].get('tokens_per_s'),flush=True)
    if p.returncode: raise SystemExit(p.returncode)
