import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('/tmp/plamo-expanded-reference.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-1', type=Path, default=Path('/tmp/plamo-cumulative-final-quality-1'))
    parser.add_argument('--baseline-64', type=Path, default=Path('/tmp/plamo-cumulative-final-quality-64'))
    parser.add_argument('--original-64', type=Path, default=Path('/tmp/plamo-next-final-quality-64'))
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1,64])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wrapper = args.output.resolve() / 'bench-locked'
    wrapper.write_text('#!/bin/sh\nexec flock /tmp/r9700.lock ' + shlex.quote(str(args.bench.resolve())) + ' "$@"\n')
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env.update(ROCR_VISIBLE_DEVICES='GPU-a8e9ddefa2d60f55', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PATH='/tmp/venv-plamo/bin:' + env['PATH'])
    env['LD_LIBRARY_PATH'] = '/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib'
    model = '/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24'
    output = {}
    for batch in args.batch_sizes:
        work = args.output / f'batch-{batch}'
        command = ['/tmp/venv-plamo/bin/python', str(Path(__file__).with_name('plamo_quality.py')),
                   '--reference', str(args.reference), '--model', model, '--bench', str(wrapper),
                   '--workdir', str(work), '--batch-size', str(batch), '--prefix', 'heldout-ja-en-',
                   '--prefix', 'heldout-en-ja-', '--prefix', 'general-corpus-']
        with (args.output / f'batch-{batch}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        baseline = args.baseline_1 if batch == 1 else args.baseline_64
        equality = {}
        for candidate in sorted(work.glob('*.candidate')):
            before = hashlib.sha256((baseline / candidate.name).read_bytes()).hexdigest()
            after = hashlib.sha256(candidate.read_bytes()).hexdigest()
            equality[candidate.name] = {'baseline_sha256': before, 'candidate_sha256': after,
                                        'bit_equal': before == after}
        paired = {}
        for name, folder in [('incumbent', baseline), ('original', args.original_64)] if batch == 64 else [('incumbent', baseline)]:
            result = subprocess.run(['/tmp/venv-plamo/bin/python',
                str(Path(__file__).with_name('plamo_quality_kl_paired.py')),
                '--reference', str(args.reference), '--baseline', str(folder), '--candidate', str(work)],
                env=env, text=True, capture_output=True, check=True)
            paired[name] = json.loads(result.stdout)
        output[str(batch)] = {'summary': json.loads((work/'summary.json').read_text()),
                              'logits': equality, 'paired_kl': paired,
                              'all_logits_bit_equal': all(v['bit_equal'] for v in equality.values())}
        (args.output / 'quality.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'batch': batch, 'summary': output[str(batch)]['summary'],
                          'bit_equal': output[str(batch)]['all_logits_bit_equal']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
