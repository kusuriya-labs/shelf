import json, os, subprocess, sys
from pathlib import Path
root=Path.cwd()
folder=root/'build-gfx1201/exl3-general'
env=os.environ.copy()
env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55'
env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
shapes=[(1024,5120,4),(6144,5120,4),(10240,5120,4),(5120,6144,4),(17408,5120,4),(5120,17408,4),(248320,5120,6)]
records=[]
for n,k,b in shapes:
    for rows in [3,4,5,6,7]:
        for repeat in range(5):
            for variant in ['base']:
                exe=folder/'shelf-bench.base' if variant=='base' else root/'build-gfx1201/shelf-bench'
                cmd=['flock','-w','60','/tmp/r9700.lock','timeout','60',str(exe),'exl3-int8','--n',str(n),'--k',str(k),'--bits',str(b),'--rows',str(rows),'--scratch-rows','1024','--working-mib','256','--iterations','16']
                r=subprocess.run(cmd,env=env,text=True,capture_output=True)
                rec={'n':n,'k':k,'bits':b,'rows':rows,'repeat':repeat,'variant':variant,'exit':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
                records.append(rec)
                (folder/'micro-extra.json').write_text(json.dumps(records,indent=2)+'\n')
                if r.returncode: print(json.dumps(rec),flush=True);sys.exit(1)
        print(n,k,b,rows,flush=True)
