import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[4]
work = root / 'build-gfx1201/exl3-general'
doc = Path(__file__).with_name('exl3_general.md')
data = json.loads(Path(__file__).with_name('exl3_general.json').read_text())
if data['missing'] or data['incomplete']:
    raise RuntimeError('集計対象が未完了: ' + str((data['missing'], data['incomplete'])))


def metric(record):
    if record['concurrency'] == 1:
        return float(record['fields']['DECODE_TOKENS_PER_SEC'])
    return record['result']['runs'][0]['tokens_per_s']


def values(records, mode, concurrency, variant):
    return [metric(x) for x in records if x['mode'] == mode and x['concurrency'] == concurrency
            and x['variant'] == variant and x['prompt'] in ('code', 'mixed')]


def fmt(samples):
    if len(samples) < 2:
        raise ValueError('反復数が不足')
    return f'{statistics.mean(samples):.2f} ± {statistics.stdev(samples):.2f}'


lines = ['初期 revision と最終 commit を交互に各 3 回。表示は平均 ± run 間標本 SD、単位 tok/s。'
         'c=1 は `shelf-compute` の decode 指標、c=2/4 は stdio protocol の同時 request で、'
         'warmup 後の 128 token batch の prefill と出力送受信を含む。HTTP transport は含まない。'
         'c=1 の投機は draft 7・既定 tree、c=2/4 の投機は draft 7・tree budget 0。', '',
         '| 経路 | c | 初期 | 最終 | 平均比 |', '| --- | ---: | ---: | ---: | ---: |']
for mode, label in [('none', '非投機'), ('bf16', 'DFlash2 BF16'), ('mul24', 'DFlash2 mul24')]:
    records = data['mul24-initial-vs-optimized'] if mode == 'mul24' else data['initial-vs-optimized']
    for concurrency in [1, 2, 4]:
        before, after = (values(records, mode, concurrency, variant) for variant in ['group', 'candidate'])
        if len(before) != 3 or len(after) != 3:
            raise ValueError(f'{mode} c={concurrency}: 反復数不正')
        delta = 100 * (statistics.mean(after) / statistics.mean(before) - 1)
        lines.append(f'| {label} | {concurrency} | {fmt(before)} | {fmt(after)} | {delta:+.2f}% |')
lines += ['', '連続統合のみの版と、そこに列分割・tile を加えた最終版の比較。'
          'c=1 非投機は外れ値を含め追加 5 組も合算し、各 8 run とした。', '',
          '| 経路 | c | 統合のみ | 全候補 | 平均比 |', '| --- | ---: | ---: | ---: | ---: |']
for mode, label in [('none', '非投機'), ('bf16', 'DFlash2 BF16'), ('mul24', 'DFlash2 mul24')]:
    for concurrency in [1, 2, 4]:
        records = list(data['optimized-vs-group'])
        if mode == 'none' and concurrency == 1:
            records += data['optimized-c1-recheck']
        before, after = (values(records, mode, concurrency, variant) for variant in ['group', 'candidate'])
        if len(before) != len(after) or len(before) < 3:
            raise ValueError(f'{mode} c={concurrency}: 反復数不正')
        delta = 100 * (statistics.mean(after) / statistics.mean(before) - 1)
        lines.append(f'| {label} | {concurrency} | {fmt(before)} | {fmt(after)} | {delta:+.2f}% |')
bf16_before, bf16_after = (values(data['optimized-vs-group'], 'bf16', 4, v) for v in ['group', 'candidate'])
mul_before, mul_after = (values(data['optimized-vs-group'], 'mul24', 4, v) for v in ['group', 'candidate'])
lines += ['', f'c=4 投機の平均は下振れしたが、いずれも統合のみの標本 SD 内で、中央値は BF16 が '
          f'{statistics.median(bf16_before):.2f}→{statistics.median(bf16_after):.2f}、mul24 が '
          f'{statistics.median(mul_before):.2f}→{statistics.median(mul_after):.2f} tok/s。'
          '各 3 組で負方向が再現しておらず、更新後の採否基準では維持した。'
          '小差が測定誤差と区別できないことと、常に高速化すると証明したことは区別する。']
large = data['large-batch-repeat']
a = [x['result']['runs'][0]['tokens_per_s'] for x in large if x['variant'] == 'group']
b = [x['result']['runs'][0]['tokens_per_s'] for x in large if x['variant'] == 'tiled']
lines += ['', f'c=64 の非投機は 32 token・prefill を含む 3 組で {fmt(a)} → {fmt(b)} tok/s。'
          '64 request すべての token が一致した。別の初回 1 組は 318.63→311.50 tok/s で、'
          'こちらも JSON に残す。', '',
          'PLaMo 専用経路の短窓参考値。最初の完了後 5 秒 warmup、30 秒窓、各 2 run。'
          '短窓なので細かな差の有意性を主張しない。', '',
          '| c | 初期 | 最終 |', '| ---: | ---: | ---: |']
for concurrency in [1, 64]:
    before, after = ([x['result']['completion_tokens_per_s'] for x in data['plamo-speed']
                      if x['concurrency'] == concurrency and x['variant'] == variant]
                     for variant in ['base', 'candidate'])
    lines.append(f'| {concurrency} | {fmt(before)} | {fmt(after)} |')
lines += ['', '固定 prompt 3 種、c=1・2・4、非投機と BF16 / mul24 drafter の投機で'
          '対応する生成 token ID はすべて一致した。batch の ID は固定幅 int32 で byte 比較し、'
          'c=1 の ID 列は出力 byte 列を比較した。単体の BF16 出力 70 条件もファイルを byte 比較し、'
          'PLaMo の c=1/64 dump も一致した。全結果と SHA256 は JSON に保存する。', '']
text = doc.read_text()
heading = '## End-to-end と一致性\n'
start = text.index(heading) + len(heading)
end = text.index('\n## PLaMo とテスト', start)
doc.write_text(text[:start] + '\n' + '\n'.join(lines) + text[end:])
print('updated', doc)
