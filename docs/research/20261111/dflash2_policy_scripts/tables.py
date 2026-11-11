#!/usr/bin/env python3
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
comparison = pathlib.Path(sys.argv[2])
print('### 損益分岐（tok/s、各 2 回）\n')
print('| prompt | 同時数 | 非投機 | 固定 K=7 |')
print('| --- | ---: | --- | --- |')
for name in ('benchmark', 'japanese'):
    for count in (2, 4, 8, 12, 16, 20, 24, 32):
        fields = []
        for mode in ('off', 'fixed'):
            values = []
            for rep in (1, 2):
                path = root / f'{name}_{count}_r{rep}_{mode}.out'
                try:
                    value = json.loads(path.read_text().splitlines()[0])['all_active_tps']
                    values.append(f'{value:.2f}' if value is not None else '—')
                except (OSError, ValueError, IndexError, KeyError):
                    values.append('—')
            fields.append(' / '.join(values))
        print(f'| {name} | {count} | ' + ' | '.join(fields) + ' |')
print('\n### 比較（tok/s、各 1 回）\n')
print('| prompt | 同時数 | 非投機 | 固定 K=7 | ポリシー | 判定 | greedy 一致 |')
print('| --- | ---: | ---: | ---: | ---: | --- | --- |')
for name in ('benchmark', 'japanese', 'shell', 'code', 'reasoning', 'mixed'):
    for count in (1, 2, 4, 8, 16, 32):
        if name == 'mixed' and count == 1:
            continue
        values, tokens = [], []
        for mode in ('off', 'fixed', 'adaptive'):
            tag = f'{name}_{count}_{mode}'
            try:
                value = json.loads((comparison / f'{tag}.out').read_text().splitlines()[0])['all_active_tps']
                ids = json.loads((comparison / f'{tag}.json').read_text())
                values.append(value)
                tokens.append(ids if len(ids) == count and all(ids.values()) else None)
            except (OSError, ValueError, IndexError, KeyError):
                values.append(None)
                tokens.append(None)
        verdict = ('通過' if values[2] >= .97 * max(values[:2]) else '未達') if all(v is not None for v in values) else '未測定'
        match = ('一致' if tokens[0] == tokens[1] == tokens[2] else '不一致') if all(v is not None for v in tokens) else '未確認'
        numbers = [f'{v:.2f}' if v is not None else '—' for v in values]
        print(f'| {name} | {count} | ' + ' | '.join(numbers) + f' | {verdict} | {match} |')
