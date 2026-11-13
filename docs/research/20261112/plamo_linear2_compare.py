import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shapes',nargs='+',default=['512,2560,4','512,2560,5','2560,3072,4','3072,2560,4','3072,2560,5','10240,2560,4','2560,10240,4','107520,2560,8'])
    args=parser.parse_args()
    env=os.environ.copy()
    env['ROCR_VISIBLE_DEVICES']='GPU-a8e9ddefa2d60f55'
    env['LD_LIBRARY_PATH']='/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as output,args.output.with_suffix('.checks').open('w') as errors:
        for m in args.rows:
            for shape in args.shapes:
                n,k,b=map(int,shape.split(','))
                print(json.dumps({'start':[m,n,k,b]}),flush=True)
                subprocess.run(['flock','/tmp/r9700.lock','timeout','175',str(args.binary.resolve()),str(m),str(n),str(k),str(b),'7'],env=env,stdout=output,stderr=errors,check=True)
                output.flush()


if __name__=='__main__':
    main()
