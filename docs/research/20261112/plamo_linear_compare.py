import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[128,64,16,4,1])
    args = parser.parse_args()
    runner = Path(__file__).with_name('plamo_linear_runs.py')
    for index, c in enumerate(args.concurrency):
        variants = [('baseline', args.baseline), ('final', args.candidate)]
        if index % 2 == 0:
            variants.reverse()
        for name, binary in variants:
            subprocess.run(['python3', str(runner), '--compute', str(binary), '--output', str(args.output),
                            '--tag', f'{name}-{args.tag}', '--concurrency', str(c)], check=True)


if __name__ == '__main__':
    main()
