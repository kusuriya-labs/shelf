import os, subprocess, pathlib, json, re, sys
root=pathlib.Path('/home/homelab1/coding-local/shelf/.worktrees/fc-dflash2graph')
out=pathlib.Path('/tmp/fc_dflash2graph')
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'
drafters={'bf16':'/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2','mul24':'/home/homelab1/datapool/rocm-exl3-rdna2/models/dflash2-drafter-exl3-4bpw-mul24'}
env=os.environ.copy(); env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',SHELF_GRAPH_DEBUG='1')
for draft,path in drafters.items():
 for prompt in ['code','prose','ja']:
  ids=[]
  for mode in ['off','on']:
   env['SHELF_GRAPH_VERIFY']='0' if mode=='off' else '1'
   cmd=['flock','/tmp/r9700.lock','timeout','-k','5','270',str(root/'build-gfx1201/shelf-compute'),'--model-dir',model,'--input-ids-file',str(root/f'docs/research/20261113/exl3_general/prompt_{prompt}.ids'),'--max-new-tokens','256','--max-seq-len','1024','--temperature','0','--dflash2-model-dir',path,'--dflash2-drafts','7']
   p=subprocess.run(cmd,env=env,cwd=root,capture_output=True,text=True)
   (out/f'{draft}_{prompt}_{mode}.stdout').write_text(p.stdout); (out/f'{draft}_{prompt}_{mode}.stderr').write_text(p.stderr)
   match=re.search(r'^GENERATED_IDS=(.*)$',p.stdout,re.M)
   if p.returncode or not match: print('FAIL',draft,prompt,mode,p.returncode,p.stderr[-2000:],flush=True); sys.exit(1)
   ids.append(match.group(1)); print(draft,prompt,mode,'tokens',len(ids[-1].split(',')),'capture',len(re.findall(r'capture ok.*N=(?:8|10)',p.stderr)),'replay',len(re.findall(r'replay.*N=(?:8|10)',p.stderr)),flush=True)
  if ids[0]!=ids[1]: print('MISMATCH',draft,prompt,flush=True); sys.exit(2)
  print('MATCH',draft,prompt,flush=True)
