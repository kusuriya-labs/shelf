import os
from pathlib import Path
import re
import subprocess
import sys
root=Path('/home/homelab1/coding-local/shelf/.worktrees/fc-dflash2graph'); out=Path('/tmp/fc_dflash2graph')
ids=(root/'docs/research/20261113/exl3_general/prompt_code.ids').read_text().split()
(out/'context_2000.ids').write_text(' '.join((ids*50)[:2000-len(ids)]+ids)+'\n')
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
draft='/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2'
env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',SHELF_GRAPH_DEBUG='1')
values=[]
for mode in ('off','on'):
 env['SHELF_GRAPH_VERIFY']='0' if mode=='off' else '1'
 cmd=['flock','/tmp/r9700.lock','timeout','-k','5','270',str(root/'build-gfx1201/shelf-compute'),'--model-dir',model,'--input-ids-file',str(out/'context_2000.ids'),'--max-new-tokens','256','--max-seq-len','2304','--temperature','0','--dflash2-model-dir',draft,'--dflash2-drafts','7']
 p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
 (out/f'context_2000_{mode}.stdout').write_text(p.stdout);(out/f'context_2000_{mode}.stderr').write_text(p.stderr)
 m=re.search(r'^GENERATED_IDS=(.*)$',p.stdout,re.M)
 if p.returncode or not m: print('FAIL',mode,p.returncode,p.stderr[-1000:],flush=True);sys.exit(1)
 values.append(m.group(1)); print(mode,'tokens',len(values[-1].split(',')),flush=True)
print('MATCH',values[0]==values[1],flush=True)
if values[0]!=values[1]:sys.exit(2)
