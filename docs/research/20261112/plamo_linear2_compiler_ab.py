import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--occ',type=int)
    parser.add_argument('--flag',action='append',default=[])
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--shape',action='append')
    parser.add_argument('--nontemporal',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];build=root/'build-gfx1201';output=build/'plamo-linear2/compiler'/args.variant;output.mkdir(parents=True,exist_ok=True)
    source=(root/'src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip').read_text()
    if args.occ is not None:
        old='(B == 4 && MT >= 3) ? 8 : 1'
        if old not in source:raise ValueError('occupancy anchor missing')
        source=source.replace(old,f'(B == 4 && MT >= 3) ? {args.occ} : 1')
    if args.nontemporal:
        for index in ('i0','i1','i2'):
            source=source.replace('tile['+index+']','__builtin_nontemporal_load(tile+'+index+')')
    source=source.replace('namespace shelf::kernel {','namespace plamo_candidate {\nusing namespace shelf::kernel;',1)
    candidate=output/'candidate.hip';candidate.write_text(source)
    commands=json.loads((build/'compile_commands.json').read_text());base=shlex.split(next(c['command'] for c in commands if c['file'].endswith('/test_plamo_exl3_projection.hip')));base=base[:base.index('-o')]
    obj=output/'candidate.o';command=['nice','-n','10',*base,*args.flag,'-Rpass-analysis=kernel-resource-usage','-save-temps=obj','-x','hip','-c',str(candidate),'-o',str(obj)]
    with (output/'compile.log').open('w') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=output/'micro';main_source=Path(__file__).with_name('plamo_linear2_compiler_micro.hip')
    subprocess.run([*base,'-x','hip',str(main_source),'-o',str(binary),'-x','none',str(obj),str(build/'libshelf_kernels_optimized.a'),
        '-L/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib','-L/opt/rocm/core-7.14/lib','-lamdhip64','-lc++','-lc++abi'],cwd=root,check=True)
    (output/'build.json').write_text(json.dumps({'variant':args.variant,'occ':args.occ,'flags':args.flag,'nontemporal':args.nontemporal,'shapes':args.shape,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'command':command},indent=2)+'\n')
    if args.compile_only:return
    env=os.environ.copy();env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55';env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    with (output/'micro.jsonl').open('w') as log,(output/'checks.log').open('w') as errors:
        shapes=[tuple(map(int,shape.split(','))) for shape in args.shape] if args.shape else [(192,3072,2560,4,3),(192,10240,2560,4,2),(192,2560,10240,4,1),(192,512,2560,5,1),(64,10240,2560,4,2)]
        for shape in shapes:
            m,n,k,bits,count,*splits=shape
            print(json.dumps({'variant':args.variant,'shape':list(shape)}),flush=True)
            subprocess.run(['flock','/tmp/r9700.lock','timeout','175',str(binary),str(m),str(n),str(k),str(bits),str(count),*map(str,splits)],env=env,stdout=log,stderr=errors,check=True);log.flush()


if __name__=='__main__':main()
