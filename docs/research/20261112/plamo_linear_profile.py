import argparse
import collections
import csv
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', SHELF_QWEN35_PERF_TIMING='1',
               SHELF_BATCH_STEPWALL='1', PATH='/tmp/venv-plamo/bin:' + env['PATH'])
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    output = []
    for repeat in range(args.repeats):
        variants = [('baseline','0'),('final','1')]
        if repeat % 2:
            variants.reverse()
        for name, enabled in variants:
            for c in (1,64):
                stem = f'{name}-{repeat}-c{c}'
                trace = args.output.resolve() / f'{stem}.csv'
                env.update(SHELF_PLAMO_EXL3_LINEAR=enabled,
                           SHELF_PLAMO_LINEAR_TRACE=str(trace), SHELF_PLAMO_LINEAR_TRACE_ROWS=str(c))
                command = ['flock','/tmp/r9700.lock','/tmp/venv-plamo/bin/python',
                    str(Path(__file__).with_name('plamo_decode_batch.py')), '--compute',str(args.compute.resolve()),
                    '--model',model,'--tokenizer',model,'--concurrency',str(c),'--tokens','64',
                    '--case-index','7','--warmup-batch','--log',str(args.output/f'{stem}.log')]
                run = subprocess.run(command,env=env,text=True,capture_output=True,check=True)
                totals = collections.defaultdict(lambda: {'event_us':0.0,'dispatches':0,'logical_consumed':0})
                with trace.open() as source:
                    for row in csv.DictReader(source):
                        total = totals[row['kernel_id']]
                        total['event_us'] += float(row['duration_us'])
                        total['dispatches'] += 1
                        total['logical_consumed'] += int(row['logical_consumed'])
                record = {'variant':name,'repeat':repeat,'concurrency':c,'events':dict(totals),
                          'event_total_us':sum(v['event_us'] for v in totals.values()),
                          'diagnostic_decode':json.loads(run.stdout)}
                output.append(record)
                (args.output/'profile.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps({'done':stem,'linear':totals['LINEAR_EXL3_INT8']}),flush=True)


if __name__ == '__main__':
    main()
