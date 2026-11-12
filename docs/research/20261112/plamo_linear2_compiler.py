import argparse
import concurrent.futures
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

import yaml


def compile_one(root,build,output,name,extra):
    source=root/f'src/shelf/kernels/optimized/linear/exl3_family/mul24/{name}.hip'
    commands=json.loads((build/'compile_commands.json').read_text())
    base=shlex.split(next(c['command'] for c in commands if c['file']==str(source)))
    directory=output/name
    directory.mkdir(parents=True,exist_ok=True)
    obj=directory/(name+'.o')
    command=['nice','-n','10',*base[:base.index('-o')],'--offload-device-only','-save-temps=obj',
        '-Rpass-analysis=kernel-resource-usage',*extra,'-c',str(source),'-o',str(obj)]
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (directory/'compile.log').write_text(result.stdout+result.stderr)
    record={'source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'command':command,'exit_code':result.returncode}
    if result.returncode:return record
    llvm=Path('/opt/rocm/core-7.14/lib/llvm/bin')
    device=directory/(name+'-hip-amdgcn-amd-amdhsa-gfx1201.out')
    if not device.exists():device=directory/(name+'-hip-amdgcn-amd-amdhsa-gfx1201.o')
    notes=subprocess.check_output([str(llvm/'llvm-readobj'),'--notes',str(device)],text=True)
    (directory/'notes.txt').write_text(notes)
    parts=re.findall(r'---\n.*?\n\.\.\.',notes,re.S)
    metadata=next((yaml.safe_load(x) for x in parts if 'amdhsa.kernels:' in x),{})
    isa=subprocess.check_output([str(llvm/'llvm-objdump'),'-d','--demangle',str(device)],text=True)
    (directory/'isa.txt').write_text(isa)
    record['kernels']=metadata.get('amdhsa.kernels',[])
    record['object_sha256']=hashlib.sha256(device.read_bytes()).hexdigest()
    remarks=[]
    for chunk in re.split(r'(?=^remark:.*?Function Name:)',result.stderr,flags=re.M):
        m=re.search(r'Function Name:\s*(\S+)',chunk)
        if not m:continue
        values={key:int(value) for key,value in re.findall(r'(TotalSGPRs|SGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s*(\d+)',chunk)}
        remarks.append({'function':m[1],**values})
    record['remarks']=remarks
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='baseline')
    parser.add_argument('--sources',nargs='+',default=['decode','prefill_wmma','grouped','projection'])
    parser.add_argument('--flag',action='append',default=[])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    build=root/'build-gfx1201'
    output=build/'plamo-linear2/compiler'/args.variant
    output.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda name:compile_one(root,build,output,name,args.flag),args.sources))
    data={'variant':args.variant,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'flags':args.flag,'translation_units':results}
    (output/'resources.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for r in results:print(json.dumps({'source':r['source'],'exit_code':r['exit_code'],'kernels':len(r.get('kernels',[])),'remarks':len(r.get('remarks',[]))}),flush=True)
    if any(r['exit_code'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    main()
