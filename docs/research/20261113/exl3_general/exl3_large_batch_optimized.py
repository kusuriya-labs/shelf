import json,os,subprocess
from pathlib import Path
root=Path.cwd();d=root/'build-gfx1201/exl3-general';env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',PATH='/tmp/venv-so/bin:'+env['PATH'])
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24';results=[]
for variant in ['group','tiled']:
 exe=d/('shelf-compute.final' if variant=='group' else 'shelf-compute.optimized')
 cmd=['flock','-w','300','/tmp/r9700.lock','timeout','-k','10','--signal=INT','150','/tmp/venv-so/bin/python','-I','docs/research/20261113/exl3_general/exl3_batch.py','--compute',str(exe),'--model',model,'--concurrency','64','--runs','1','--tokens','32','--trace','--prompts','docs/research/20261113/exl3_general/prompt_code.ids','docs/research/20261113/exl3_general/prompt_prose.ids','docs/research/20261113/exl3_general/prompt_ja.ids']
 r=subprocess.run(cmd,env=env,text=True,capture_output=True);a={'variant':variant,'exit':r.returncode,'stderr':r.stderr[-4000:]}
 try:a['result']=json.loads(r.stdout)
 except ValueError:a['stdout']=r.stdout
 results.append(a);(d/'large-batch-optimized.json').write_text(json.dumps(results,indent=2)+'\n');print(variant,r.returncode,a.get('result',{}).get('runs',[{}])[0].get('tokens_per_s'),flush=True)
 if r.returncode:break
