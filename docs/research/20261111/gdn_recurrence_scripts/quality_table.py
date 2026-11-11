import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print('| corpus | 経路 | mean KL | p99 KL | top-1 一致率 | 最大 logit 差 | ΔPPL |')
print('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
for corpus in ('prose', 'code', 'japanese', 'reasoning'):
    for mode in ('batch8', 'serial', 'chunk', 'exact', 'serial8', 'exact8'):
        path = root / 'reports' / f'{corpus}-{mode}.json'
        if not path.exists():
            continue
        x = json.loads(path.read_text())
        print(f'| {corpus} | {mode} | {x["mean_kl"]:.8g} | {x["p99_kl"]:.8g} | {x["top1"]:.4%} | {x["max_logit_diff"]:.6g} | {x["delta_ppl"]:+.6g} |')
