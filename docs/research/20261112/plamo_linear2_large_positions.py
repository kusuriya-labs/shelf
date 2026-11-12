import argparse
import json
from pathlib import Path

import torch
from plamo_quality_kl_paired import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path, default=Path('/tmp/plamo-expanded-reference.pt'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = torch.load(args.reference, map_location='cpu', weights_only=True)
    result = {}
    for name, baseline in [('before320', args.quality / 'before'),
                           ('incumbent64', Path('/tmp/plamo-cumulative-final-quality-64'))]:
        rows = []
        for case in cases:
            filename = case['name'] + '.candidate'
            candidate = args.quality / 'candidate' / filename
            if not candidate.exists():
                continue
            values = compare(case, baseline / filename, candidate)
            rows.append({'case': case['name'], 'dump_positions': list(range(2, len(values) + 2)),
                         'delta_kl': values.tolist()})
        result[name] = rows
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: sum(len(row['delta_kl']) for row in rows) for name, rows in result.items()}))


if __name__ == '__main__':
    main()
