import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = 'GPU-a8e9ddefa2d60f55'
    env['PATH'] = '/tmp/venv-plamo/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    data = {p.stem: json.loads(p.read_text()) for p in args.output.glob('*-c*.json')}
    for repeat in range(args.repeats):
        variants = [('baseline',args.baseline),('final',args.candidate)]
        if repeat:
            variants.reverse()
        for name, binary in variants:
            env['SHELF_COMPUTE'] = str(binary.resolve())
            for c in ((1,64) if repeat < 2 else (64,)):
                stem = f'{name}-{repeat}-c{c}'
                if stem in data:
                    continue
                command = ['flock','/tmp/r9700.lock','/tmp/venv-plamo/bin/python',
                    str(Path(__file__).with_name('plamo_batch.py')), '--server', str(root/'build-gfx1201/shelf-server'),
                    '--model','/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24',
                    '--concurrency',str(c),'--skip-initial-warmup','--port','18473',
                    '--log',str(args.output/f'{stem}.server.log')]
                result = subprocess.run(command,env=env,text=True,capture_output=True,check=True)
                data[stem] = json.loads(result.stdout)
                (args.output/f'{stem}.json').write_text(json.dumps(data[stem],ensure_ascii=False,indent=2)+'\n')
    summary = {}
    for name in ('baseline','final'):
        one = data[f'{name}-0-c1']['outputs']
        repetitions = []
        for repeat in range(args.repeats):
            many = data[f'{name}-{repeat}-c64']['outputs']
            repetitions.append({'repeat': repeat, 'differ_from_c1': [i for i,text in enumerate(many) if text != one[i%8]]})
        summary[name] = {'cold64_vs_c1': repetitions,
            'c1_differ_between_repeats': sum(a!=b for a,b in zip(one,data[f'{name}-1-c1']['outputs'])),
            'c64_differ_between_repeats': sum(a!=b for a,b in zip(data[f'{name}-0-c64']['outputs'],data[f'{name}-1-c64']['outputs']))}
        pairs = []
        for first in range(args.repeats):
            for second in range(first+1,args.repeats):
                count = sum(a!=b for a,b in zip(data[f'{name}-{first}-c64']['outputs'],data[f'{name}-{second}-c64']['outputs']))
                pairs.append({'first':first,'second':second,'differing_requests':count})
        values = [pair['differing_requests'] for pair in pairs]
        summary[name].update(pairwise=pairs,pairwise_mean=statistics.mean(values),
            pairwise_median=statistics.median(values),pairwise_max=max(values))
    summary['final_vs_baseline_c1'] = sum(a!=b for a,b in zip(data['baseline-0-c1']['outputs'],data['final-0-c1']['outputs']))
    (args.output/'cold.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
