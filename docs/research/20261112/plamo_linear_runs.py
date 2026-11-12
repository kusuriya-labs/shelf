import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 4, 16, 64, 128])
    parser.add_argument('--duration', type=float, default=90)
    parser.add_argument('--warmup', type=float, default=20)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55',
               SHELF_COMPUTE=str(args.compute.resolve()),
               PATH='/tmp/venv-plamo/bin:' + env['PATH'])
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
    args.output.mkdir(parents=True, exist_ok=True)
    for index, c in enumerate(args.concurrency):
        stem = f'{args.tag}-{index}-c{c}'
        command = ['flock', '/tmp/r9700.lock', '/tmp/venv-plamo/bin/python',
                   str(Path(__file__).with_name('plamo_closed_loop.py')),
                   '--server', str(root / 'build-gfx1201/shelf-server'), '--model', model,
                   '--concurrency', str(c), '--warmup', str(args.warmup),
                   '--duration', str(args.duration), '--port', '18472',
                   '--log', str(args.output / f'{stem}.server.log')]
        print(json.dumps({'start': stem}), flush=True)
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        data = json.loads(result.stdout)
        (args.output / f'{stem}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        lengths = [record['output_tokens'] for record in data['records']]
        summary = {k: v for k, v in data.items() if k != 'records'}
        summary.update(tag=args.tag, output_tokens_min=min(lengths),
                       output_tokens_median=statistics.median(lengths), output_tokens_max=max(lengths))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
