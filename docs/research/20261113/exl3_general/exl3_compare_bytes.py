import json
from pathlib import Path

root = Path.cwd() / 'build-gfx1201/exl3-general'
for source, output, extra in [('tile-exact.json', 'tile-byte.json', False),
                              ('tile-exact-extra.json', 'tile-byte-extra.json', True)]:
    records = json.loads((root / source).read_text())
    compared = []
    for record in records:
        if extra:
            stem = f"extra-exact-{record['bits']}-{record['n']}-{record['k']}-{record['rows']}-{record['splits']}"
        else:
            stem = f"exact-{record['bits']}-{record['n']}-{record['rows']}-{record['splits']}"
        before = (root / f'{stem}-base.bin').read_bytes()
        after = (root / f'{stem}-candidate.bin').read_bytes()
        compared.append({**record, 'bytes': len(before), 'byte_equal': before == after})
    (root / output).write_text(json.dumps(compared, indent=2) + '\n')
    if any(not item['byte_equal'] for item in compared):
        raise RuntimeError(output + ': byte 不一致')

plamo = []
for variant in ['final', 'full', 'optimized']:
    for batch in [1, 64]:
        before = (root / f'plamo-{variant}-{batch}-base.dump').read_bytes()
        after = (root / f'plamo-{variant}-{batch}-candidate.dump').read_bytes()
        plamo.append({'variant': variant, 'batch': batch, 'bytes': len(before),
                      'byte_equal': before == after})
(root / 'plamo-byte.json').write_text(json.dumps(plamo, indent=2) + '\n')
if any(not item['byte_equal'] for item in plamo):
    raise RuntimeError('PLaMo dump の byte 不一致')
print(json.dumps({'tile': 42, 'tile_extra': 28, 'plamo': len(plamo)}))
