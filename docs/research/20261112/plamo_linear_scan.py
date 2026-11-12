import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=Path('build-gfx1201'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--extra', action='store_true')
    parser.add_argument('--selected', action='store_true')
    parser.add_argument('--groups', action='store_true')
    parser.add_argument('--mean20', action='store_true')
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 64])
    args = parser.parse_args()
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    shapes = [(3072,2560,4), (3072,2560,5), (512,2560,4), (512,2560,5),
              (2560,3072,4), (2560,3072,5), (10240,2560,4), (2560,10240,4),
              (107520,2560,8)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output, args.output.with_suffix('.checks').open('w') as errors:
        for m in args.rows:
            for n,k,bits in shapes:
                if args.groups and (n not in (3072,10240) or k != 2560):
                    continue
                print(json.dumps({'start': [m,n,k,bits], 'sweep': args.sweep}), flush=True)
                command = ['flock','/tmp/r9700.lock',str(args.build_dir.resolve()/'plamo-linear/micro'),
                           str(m),str(n),str(k),str(bits),str(4 if args.selected else 3 if args.groups else 2 if args.extra else int(args.sweep)),str(int(args.mean20))]
                result = subprocess.run(command,env=env,text=True,stdout=output,stderr=errors,check=True)
                output.flush()


if __name__ == '__main__':
    main()
