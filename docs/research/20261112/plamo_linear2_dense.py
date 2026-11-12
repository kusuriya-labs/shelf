import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,nargs='+',default=[320,512])
    parser.add_argument('--output',type=Path,default=Path('build-gfx1201/plamo-linear2/dense.jsonl'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    build=root/'build-gfx1201'
    commands=json.loads((build/'compile_commands.json').read_text())
    command=shlex.split(next(c['command'] for c in commands if c['file'].endswith('/exl3_dense_gemm.hip')))
    command=command[:command.index('-o')]+['-x','hip',str(Path(__file__).with_suffix('.hip')),'-o',str(build/'plamo-linear2/dense'),
        '-L/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib','-L/opt/rocm/core-7.14/lib',
        '-lhipblaslt','-lamdhip64','-lc++','-lc++abi']
    subprocess.run(command,cwd=root,check=True)
    env=os.environ.copy()
    env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55'
    env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    with args.output.open('w') as output:
        for m in args.rows:
            for n,k in [(512,2560),(3072,2560),(2560,3072),(10240,2560),(2560,10240)]:
                print(json.dumps({'start':[m,n,k]}),flush=True)
                subprocess.run(['flock','/tmp/r9700.lock','timeout','175',str(build/'plamo-linear2/dense'),str(m),str(n),str(k)],env=env,stdout=output,check=True)
                output.flush()


if __name__=='__main__':
    main()
