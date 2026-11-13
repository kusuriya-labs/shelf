import os,subprocess,json
from pathlib import Path
root=Path.cwd();d=root/'build-gfx1201/exl3-general';env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',SHELF_EXL3_STAGE_TRACE='1',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',PATH='/tmp/venv-so/bin:'+env['PATH'])
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24';draft='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2';exe=d/'shelf-compute.trace'
for spec in [False,True]:
 cmd=['flock','-w','120','/tmp/r9700.lock','timeout','150',str(exe),'--model-dir',model,'--input-ids-file',str(root/'docs/research/20261113/exl3_general/prompt_code.ids'),'--max-new-tokens','24','--max-seq-len','512','--temperature','0']
 if spec:cmd+=['--dflash2-model-dir',draft,'--dflash2-drafts','7']
 r=subprocess.run(cmd,env=env,text=True,capture_output=True);(d/f'trace-c1-{int(spec)}.log').write_text(r.stdout+'\n'+r.stderr);print('c1',spec,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
for spec in [False,True]:
 log=d/f'trace-c4-{int(spec)}.log'
 cmd=['flock','-w','120','/tmp/r9700.lock','timeout','-k','10','--signal=INT','150','/tmp/venv-so/bin/python','-I','docs/research/20261113/exl3_general/exl3_batch.py','--compute',str(exe),'--model',model,'--concurrency','4','--runs','1','--tokens','24','--log',str(log),'--prompts','docs/research/20261113/exl3_general/prompt_code.ids','docs/research/20261113/exl3_general/prompt_prose.ids','docs/research/20261113/exl3_general/prompt_ja.ids']
 if spec:cmd+=['--spec','--drafter',draft]
 r=subprocess.run(cmd,env=env,text=True,capture_output=True);(d/f'trace-c4-{int(spec)}.json').write_text(r.stdout);print('c4',spec,r.returncode, r.stderr[-500:],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
