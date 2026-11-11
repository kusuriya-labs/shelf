import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print('| prompt | 同時数 | 投機なし | 固定 K=7 | adaptive | 判定 | greedy 一致 |')
print('| --- | ---: | ---: | ---: | ---: | --- | --- |')
for prompt in ('benchmark', 'japanese', 'shell', 'code', 'reasoning', 'mixed'):
    for count in (1, 2, 4, 8, 16, 32):
        results, generations = [], []
        for mode in ('off', 'fixed', 'adaptive'):
            tag = f'{prompt}_{count}_{mode}'
            output = root / f'{tag}.out'
            generation = root / f'{tag}.json'
            try:
                results.append(json.loads(output.read_text().splitlines()[0])['all_active_tps'])
                tokens = json.loads(generation.read_text())
                generations.append(tokens if len(tokens) == count and
                                   all(isinstance(ids, list) and ids for ids in tokens.values()) else None)
            except (OSError, ValueError, KeyError, IndexError):
                results.append(None)
                generations.append(None)
        if all(value is not None for value in results):
            best = max(results[:2])
            verdict = '通過' if results[2] >= 0.97 * best else '未達'
            if prompt == 'mixed' and count == 1:
                verdict = '混在不可'
            elif prompt == 'mixed' and verdict == '通過':
                verdict = '同等以上' if results[2] >= best else '同等範囲（3%）'
        else:
            verdict = '未測定'
        matches = ('一致' if generations[0] == generations[1] == generations[2] else '不一致') \
            if all(value is not None for value in generations) else '未確認'
        numbers = ['—' if value is None else f'{value:.2f}' for value in results]
        print(f'| {prompt} | {count} | ' + ' | '.join(numbers) + f' | {verdict} | {matches} |')
