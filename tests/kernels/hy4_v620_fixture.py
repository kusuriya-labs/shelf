#!/usr/bin/env python3
"""Extract one EXL3 expert to the bench's read-only raw fixture format."""
import argparse
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('model', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--layer', type=int, default=1)
p.add_argument('--expert', type=int, default=0)
a = p.parse_args()
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
a.output.mkdir(parents=True, exist_ok=True)
for number, projection in enumerate(('gate', 'up', 'down')):
    with (a.output / f'{number}.bin').open('wb') as out:
        for field in ('suh', 'svh', 'trellis'):
            key = f'model.layers.{a.layer}.mlp.experts.{a.expert}.{projection}_proj.{field}'
            with (a.model / index[key]).open('rb') as src:
                size = struct.unpack('<Q', src.read(8))[0]
                header = json.loads(src.read(size))
                entry = header[key]
                expected = [1536] if field != 'trellis' else [96, 96, 64]
                if entry['shape'] != expected or entry['dtype'] != ('F16' if field != 'trellis' else 'I16'):
                    raise ValueError(f'{key}: unexpected shape or dtype')
                start, end = entry['data_offsets']
                src.seek(8 + size + start)
                data = src.read(end - start)
                if len(data) != end - start:
                    raise ValueError(f'{key}: truncated tensor')
                out.write(data)
