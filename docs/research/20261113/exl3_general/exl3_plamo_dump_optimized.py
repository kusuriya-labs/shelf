import hashlib,json,os,subprocess
from pathlib import Path
p=Path.cwd();d=p/'build-gfx1201/exl3-general';env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',LD_LIBRARY_PATH='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib',PATH='/tmp/venv-plamo/bin:'+env['PATH'])
model='/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24';ref=Path('/tmp/plamo-cumulative-final-quality-1');results=[]
for m in [1,64]:
 for v in ['base','candidate']:
  exe=d/'shelf-bench.base' if v=='base' else d/'shelf-bench.optimized';out=d/f'plamo-optimized-{m}-{v}.dump'
  cmd=['flock','-w','300','/tmp/r9700.lock','timeout','120',str(exe),'quality','--model-dir',model,'--tokens-file',str(ref/'general-corpus-0.tokens'),'--context','1','--batch-size',str(m),'--compare',str(ref/'general-corpus-0.baseline'),'--dump',str(out)]
  r=subprocess.run(cmd,env=env,text=True,capture_output=True);x={'batch':m,'variant':v,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-1500:]};
  if r.returncode==0:x['bytes']=out.stat().st_size;x['sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
  results.append(x);(d/'plamo-dump-optimized.json').write_text(json.dumps(results,indent=2)+'\n');print(m,v,r.returncode,x.get('sha256'),flush=True)
  if r.returncode:raise SystemExit(r.returncode)
