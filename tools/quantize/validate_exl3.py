import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys


MUL24 = 0x00DCD12D
SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
         'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8,
         'F8_E4M3': 1, 'F8_E5M2': 1}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'重複した JSON キー: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def safe_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'不正な shard パス: {name}')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'shard が存在しないか root 外です: {name}')
    return path


def shard_tensors(path):
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f'不正なヘッダー: {path}')
        size = struct.unpack('<Q', raw)[0]
        if size > min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f'不正なヘッダー長: {path}')
        header = json.loads(f.read(size), object_pairs_hook=unique)
        if not isinstance(header, dict):
            raise ValueError(f'不正なヘッダー: {path}')
        base = 8 + size
        spans = []
        result = {}
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if not isinstance(spec, dict):
                raise ValueError(f'不正な tensor: {name}')
            dtype, shape, offsets = spec.get('dtype'), spec.get('shape'), spec.get('data_offsets')
            if dtype not in SIZES or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError(f'不正な dtype または shape: {name}')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError(f'不正な offset: {name}')
            start, end = offsets
            if start < 0 or end < start or base + end > path.stat().st_size or end - start != math.prod(shape) * SIZES[dtype]:
                raise ValueError(f'不正な tensor 範囲: {name}')
            spans.append((start, end))
            result[name] = (dtype, shape, path, base + start)
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError(f'tensor が重複しています: {path}')
        return result


def validate(root):
    root = Path(root).resolve(strict=True)
    config = read_json(safe_path(root, 'config.json'))
    if not isinstance(config, dict):
        raise ValueError('config.json が object ではありません')
    qc = config.get('quantization_config')
    if not isinstance(qc, dict):
        text = config.get('text_config', {})
        qc = text.get('quantization_config') if isinstance(text, dict) else None
    if not isinstance(qc, dict) or qc.get('quant_method') != 'exl3':
        raise ValueError('config.json の quant_method が exl3 ではありません')
    index = read_json(safe_path(root, 'model.safetensors.index.json'))
    weight_map = index.get('weight_map') if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('weight_map が空か不正です')
    for path in root.rglob('*.safetensors'):
        if any(name.endswith('.s0') for name in shard_tensors(safe_path(root, str(path.relative_to(root))))):
            raise ValueError(f'.s0 は使用できません: {path}')
    tensors = {}
    for shard in set(weight_map.values()):
        path = safe_path(root, shard)
        for name, spec in shard_tensors(path).items():
            if name in tensors:
                raise ValueError(f'重複 tensor: {name}')
            if weight_map.get(name) != shard:
                raise ValueError(f'index と shard が一致しません: {name}')
            tensors[name] = spec
    if set(tensors) != set(weight_map):
        raise ValueError('index と tensor 名が一致しません')
    if any(name.endswith('.s0') for name in tensors):
        raise ValueError('.s0 は使用できません')
    suffixes = ('.trellis', '.suh', '.svh', '.mul1')
    prefixes = {name[:-len(s)] for name in tensors for s in suffixes if name.endswith(s)}
    if not prefixes:
        raise ValueError('EXL3 tensor がありません')
    for prefix in sorted(prefixes):
        if any(prefix + s not in tensors for s in suffixes):
            raise ValueError(f'EXL3 tensor が不足しています: {prefix}')
        dtype, shape, _, _ = tensors[prefix + '.trellis']
        if dtype != 'I16' or len(shape) != 3:
            raise ValueError(f'trellis の dtype または shape が不正です: {prefix}')
        kt, nt, words = shape
        ka, rem = divmod(words, 16)
        if not kt or not nt or kt % 8 or nt % 8 or rem not in (0, 8) or not 1 <= ka <= 8 or (rem == 8 and ka > 3):
            raise ValueError(f'trellis のサイズまたは bit rate が不正です: {prefix}')
        for suffix, dim in (('.suh', kt * 16), ('.svh', nt * 16)):
            if tensors[prefix + suffix][:2] != ('F16', [dim]):
                raise ValueError(f'{suffix} の dtype または shape が不正です: {prefix}')
        dtype, shape, path, pos = tensors[prefix + '.mul1']
        if dtype != 'I32' or shape:
            raise ValueError(f'mul1 の dtype または shape が不正です: {prefix}')
        with path.open('rb') as f:
            f.seek(pos)
            if struct.unpack('<I', f.read(4))[0] != MUL24:
                raise ValueError(f'mul1 multiplier が不正です: {prefix}')
    if (root / 'ngram_manifest.json').exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name('ngram_quantize.py')), 'verify', str(root)], check=True)
    return len(prefixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    try:
        count = validate(args.bundle)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as e:
        print(f'検証失敗: {e}', file=sys.stderr)
        return 1
    print(f'検証成功: {count} 個の EXL3 tensor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
