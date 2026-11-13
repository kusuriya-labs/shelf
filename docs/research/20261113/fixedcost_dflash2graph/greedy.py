import os
from pathlib import Path
import re
import subprocess
import sys
root=Path('/home/homelab1/coding-local/shelf/.worktrees/fc-dflash2graph')
out=Path('/tmp/fc_dflash2graph')
env=os.environ.copy(); env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib')
for prompt in ('code','prose','ja'):
 cmd=['flock','/tmp/r9700.lock','timeout','-k','5','270',str(root/'build-gfx1201/shelf-compute'),'--model-dir','/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24','--input-ids-file',str(root/f'docs/research/20261113/exl3_general/prompt_{prompt}.ids'),'--max-new-tokens','256','--max-seq-len','1024','--temperature','0']
 p=subprocess.run(cmd,env=env,cwd=root,capture_output=True,text=True)
 (out/f'greedy_{prompt}.stdout').write_text(p.stdout); (out/f'greedy_{prompt}.stderr').write_text(p.stderr)
 actual=re.search(r'^GENERATED_IDS=(.*)$',p.stdout,re.M)
 ref=re.search(r'^GENERATED_IDS=(.*)$',(out/f'bf16_{prompt}_on.stdout').read_text(),re.M)
 if p.returncode or not actual or not ref or actual.group(1)!=ref.group(1):
  print('FAIL',prompt,p.returncode,flush=True); sys.exit(1)
 print('MATCH greedy',prompt,len(actual.group(1).split(',')),flush=True)
