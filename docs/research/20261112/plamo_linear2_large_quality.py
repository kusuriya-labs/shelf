import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows', type=int, default=320)
    parser.add_argument('--prefix', action='append')
    parser.add_argument('--score-row', type=int)
    parser.add_argument('--variants', nargs='+', choices=['before','candidate'], default=['before','candidate'])
    parser.add_argument('--reference', type=Path, default=Path('/tmp/plamo-expanded-reference.pt'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PATH='/tmp/venv-plamo/bin:/tmp/venv-so/bin:' + env['PATH'])
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    env['SHELF_PLAMO_EXL3_PREFILL_ROWS'] = '65'
    prefixes = args.prefix or ['heldout-ja-en-', 'heldout-en-ja-']
    score_row = args.score_row if args.score_row is not None else args.rows - 1
    result = {'rows': args.rows, 'score_row': score_row, 'variants': {}}
    script = Path(__file__).with_name('plamo_quality.py')
    for name, mode in [('before', '0'), ('candidate', '1')]:
        if name not in args.variants:
            continue
        folder = args.output / name
        wrapper = args.output.resolve() / f'bench-{name}'
        wrapper.write_text('#!/bin/sh\nexec flock /tmp/r9700.lock timeout 175 ' + shlex.quote(str(args.bench.resolve())) +
                           ' "$@" --score-row ' + str(score_row) + '\n')
        wrapper.chmod(0o755)
        env['SHELF_PLAMO_EXL3_PREFILL'] = mode
        if not (folder / 'summary.json').exists():
            with (args.output / f'{name}.log').open('w') as log:
                subprocess.run(['/tmp/venv-plamo/bin/python', str(script), '--reference', str(args.reference),
                                '--model', model, '--bench', str(wrapper), '--workdir', str(folder),
                                '--batch-size', str(args.rows), *[arg for prefix in prefixes for arg in ('--prefix', prefix)]], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result['variants'][name] = {'summary': json.loads((folder / 'summary.json').read_text()),
                                    'cases': json.loads((folder / 'results.json').read_text())}
        print(json.dumps({'variant': name, 'summary': result['variants'][name]['summary']}), flush=True)
    if len(result['variants']) < 2:
        (args.output / 'quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return
    import torch
    cases = torch.load(args.reference, map_location='cpu', weights_only=True)
    selected = [case for case in cases if (args.output/'candidate'/(case['name']+'.candidate')).exists()]
    paired_reference = args.output/'paired-reference.pt'
    torch.save(selected, paired_reference)
    paired = subprocess.run(['/tmp/venv-plamo/bin/python', str(script.with_name('plamo_quality_kl_paired.py')),
                             '--reference', str(paired_reference), '--baseline', str(args.output / 'before'),
                             '--candidate', str(args.output / 'candidate')], env=env, text=True, capture_output=True)
    if paired.returncode:
        raise RuntimeError(paired.stderr)
    result['paired_kl'] = json.loads(paired.stdout)
    result['bit_equal'] = {p.name: p.read_bytes() == (args.output/'before'/p.name).read_bytes()
                           for p in (args.output/'candidate').glob('*.candidate')}
    (args.output / 'quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'paired_kl': result['paired_kl']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
