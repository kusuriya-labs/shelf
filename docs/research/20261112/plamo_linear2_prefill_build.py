import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);parser.add_argument('--define',action='append',default=[]);parser.add_argument('--flag',action='append',default=[]);parser.add_argument('--rows',nargs='+',type=int,default=[320,512]);parser.add_argument('--compile-only',action='store_true');parser.add_argument('--shapes',nargs='+',default=['512,2560,4','512,2560,5','3072,2560,4','3072,2560,5','2560,3072,4','10240,2560,4','2560,10240,4']);args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];build=root/'build-gfx1201';out=build/'plamo-linear2/prefill'/args.variant;out.mkdir(parents=True,exist_ok=True)
    source=out/'micro.hip';source.write_text(Path(__file__).with_name('plamo_linear2_prefill_kernel.hip').read_text()+Path(__file__).with_name('plamo_linear2_prefill_micro.hip').read_text())
    commands=json.loads((build/'compile_commands.json').read_text());base=shlex.split(next(c['command'] for c in commands if c['file'].endswith('/exl3_dense_gemm.hip')));base=base[:base.index('-o')]
    command=['nice','-n','10',*base,*['-D'+d for d in args.define],*args.flag,'-Rpass-analysis=kernel-resource-usage','-save-temps=obj','-x','hip',str(source),'-o',str(out/'micro'),'-x','none',str(build/'libshelf_inference_execution.a'),str(build/'libshelf_kernels_optimized.a'),'-L/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib','-L/opt/rocm/core-7.14/lib','-lhipblaslt','-lamdhip64','-lc++','-lc++abi']
    with (out/'compile.log').open('w') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps({'variant':args.variant,'defines':args.define,'flags':args.flag,'command':command},indent=2)+'\n')
    if args.compile_only:return
    env=os.environ.copy();env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55';env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    with (out/'micro.jsonl').open('w') as log,(out/'checks.log').open('w') as errors:
        for m in args.rows:
            for shape in args.shapes:
                n,k,bits=map(int,shape.split(','));print(json.dumps({'variant':args.variant,'shape':[m,n,k,bits]}),flush=True)
                subprocess.run(['flock','/tmp/r9700.lock','timeout','175',str(out/'micro'),str(m),str(n),str(k),str(bits),'1'],env=env,stdout=log,stderr=errors,check=True);log.flush()


if __name__=='__main__':main()
