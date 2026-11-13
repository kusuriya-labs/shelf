import argparse
import hashlib
import json
from pathlib import Path

import torch
from plamo_quality_kl_paired import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=Path('/tmp/plamo-expanded-reference.pt'))
    parser.add_argument('--workdir', type=Path, default=Path('build-gfx1201/plamo-linear'))
    parser.add_argument('--quality-tag', default='quality-final')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = torch.load(args.reference, map_location='cpu', weights_only=True)
    expected = json.loads(Path(__file__).with_name('plamo3_expanded_logits_sha256.json').read_text())['64']
    output = {}
    for batch in (1,64):
        candidate = args.workdir/args.quality_tag/f'batch-{batch}'
        baselines = {'incumbent':Path(f'/tmp/plamo-cumulative-final-quality-{batch}')}
        if batch == 64:
            baselines['original'] = Path('/tmp/plamo-next-final-quality-64')
        output[str(batch)] = {}
        for name, folder in baselines.items():
            rows = []
            for case in cases:
                filename = case['name'] + '.candidate'
                if not (candidate/filename).is_file():
                    continue
                if name == 'original':
                    actual = hashlib.sha256((folder/filename).read_bytes()).hexdigest()
                    if actual != expected[filename]:
                        raise ValueError(f'original baseline hash mismatch: {filename}')
                values = compare(case,folder/filename,candidate/filename)
                rows.append({'case':case['name'],'dump_positions':list(range(2,len(values)+2)),
                             'delta_kl':values.tolist()})
            output[str(batch)][name] = rows
    (args.output if args.output else args.workdir/'paired_positions.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({batch:{name:sum(len(row['delta_kl']) for row in rows)
                            for name,rows in groups.items()} for batch,groups in output.items()}))


if __name__ == '__main__':
    main()
