import json
import sys
from pathlib import Path

results = {}
for name in sys.argv[1:]:
    for corpus, modes in json.loads(Path(name).read_text()).items():
        results.setdefault(corpus, {}).update(modes)
print('| corpus | 経路 | mean KL 全位置 → 上位5除外 | top-1 全位置 → 上位5除外 | 上位5のKL寄与 |')
print('| --- | --- | ---: | ---: | ---: |')
for corpus, modes in results.items():
    for mode, v in modes.items():
        print(f"| {corpus} | {mode} | {v['mean_kl']:.8g} → {v['trimmed_mean_kl']:.8g} | {v['top1']*100:.3f}% → {v['trimmed_top1']*100:.3f}% | {v['top5_share']*100:.2f}% |")
print('\n| corpus | 経路 | KL > 0.05: 位置 / mod 8 / Δ2048 / 入力 mod 16 / 入力 mod 512 / KL |')
print('| --- | --- | --- |')
for corpus, modes in results.items():
    for mode, v in modes.items():
        out = ', '.join(f"{x['position']}/{x['row_mod8']}/{x['position']-2048}/{(x['position']-1)%16}/{(x['position']-1)%512}/{x['kl']:.5f}" for x in v['outliers']) or 'なし'
        print(f'| {corpus} | {mode} | {out} |')
