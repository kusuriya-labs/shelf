import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def library():
    paths = [Path(os.environ['SHELF_NGRAM_LIB'])] if os.environ.get('SHELF_NGRAM_LIB') else sorted(ROOT.glob('build*/libshelf_ngram_quant_c.so'))
    if not paths:
        raise ValueError('libshelf_ngram_quant_c.so が見つかりません。SHELF_NGRAM_LIB を指定してください')
    lib = ct.CDLL(str(paths[0].resolve()))
    p, z, u, err = ct.c_void_p, ct.c_size_t, ct.c_uint, ct.c_char_p
    lib.shelf_ngram_train_codebook_f32.argtypes = [p, z, u, p, err, z]
    lib.shelf_ngram_quantize_rows_bf16.argtypes = [p, z, z, u, p, p, p, u, err, z]
    lib.shelf_ngram_decode_rows_f32.argtypes = [p, p, p, z, z, u, p, err, z]
    lib.shelf_ngram_verify_bundle.argtypes = [err, err, z]
    return lib


def call(fn, *args):
    error = ct.create_string_buffer(1024)
    if fn(*args, error, len(error)):
        raise ValueError(error.value.decode(errors='replace'))


def ptr(a):
    return a.ctypes.data_as(ct.c_void_p)


def verify(lib, root):
    call(lib.shelf_ngram_verify_bundle, os.fsencode(root))
    manifest = json.loads((root / 'ngram_manifest.json').read_text())
    print('verify: PASS')
    for t in manifest['tables']:
        print(f"  {t['name']}: {t['rows']} x {t['dim']}, {t['bits']} bit")


def sources(root):
    found = {}
    for path in sorted(root.glob('*.safetensors')):
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f'不正な safetensors: {path}')
            size = struct.unpack('<Q', raw)[0]
            header = json.loads(f.read(size))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if name in found:
                raise ValueError(f'重複 tensor: {name}')
            found[name] = (path, 8 + size, spec)
    return found


def automatic_specs(root, src, family):
    config = json.loads((root / 'config.json').read_text())
    config = config.get('text_config', config)
    hy_pattern = re.compile(r'model\.layers\.(\d+)\.engram_layer\.multi_head_embedding\.embedding\.weight\Z')
    qw_pattern = re.compile(r'model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight\Z')
    hy = [(name, hy_pattern.fullmatch(name)) for name in src]
    hy = [(name, m) for name, m in hy if m]
    qw = [(name, qw_pattern.fullmatch(name)) for name in src]
    qw = [(name, m) for name, m in qw if m]
    if family == 'auto':
        if bool(hy) == bool(qw):
            raise ValueError('モデル形式を一意に判定できません')
        family = 'hy4' if hy else 'qwen4exp'
    specs = []
    if family == 'hy4':
        by_layer = {int(m[1]): name for name, m in hy}
        layers = [int(x) - 1 for x in config['engram_layer_ids']]
        if set(by_layer) != set(layers):
            raise ValueError('Hy4 の layer 構成が不正です')
        for layer in layers:
            name = by_layer[layer]
            sizes = [n for group in config['engram_layer_head_vocab_sizes'][str(layer + 1)] for n in group]
            if len(sizes) != 16 or any(not isinstance(n, int) or n <= 0 for n in sizes) or sum(sizes) != src[name][2]['shape'][0]:
                raise ValueError('Hy4 の head 行数が不正です')
            start = 0
            for head, size in enumerate(sizes):
                specs.append({'name': name, 'layer': layer, 'ngram_order': 2 + head // 8, 'head': head, 'segments': [(name, start, size)]})
                start += size
    else:
        layers = [int(x) - 1 for x in config['ple_layer_ids']]
        parts = config['split_ngram_parts']
        if layers != [1] or parts != 128 or config['heads_per_ngram'] != 8 or config['ngram_size'] != 3:
            raise ValueError('Qwen PLE の構成が不正です')
        shards = {int(m[2]): name for name, m in qw if int(m[1]) == 1}
        if len(qw) != parts or set(shards) != set(range(parts)):
            raise ValueError('Qwen PLE の shard 構成が不正です')
        shapes = [src[shards[i]][2]['shape'] for i in range(parts)]
        if any(shape != shapes[0] for shape in shapes) or len(shapes[0]) != 2 or shapes[0][1] != 160:
            raise ValueError('Qwen PLE の shard shape が不正です')
        def prime(n):
            if n < 2:
                return False
            for d in range(2, int(n ** 0.5) + 1):
                if n % d == 0:
                    return False
            return True
        sizes, n = [], config['ngram_vocab_size_base']
        while len(sizes) < 16:
            if prime(n):
                sizes.append(n)
            n += 1
        shard_rows = shapes[0][0]
        divisor = config['make_ngram_vocab_size_divisible_by']
        if ((sum(sizes) + divisor - 1) // divisor) * divisor != shard_rows * parts:
            raise ValueError('Qwen PLE の行数が config と一致しません')
        start = 0
        for head, size in enumerate(sizes):
            segments, pos, end = [], start, start + size
            while pos < end:
                shard = pos // shard_rows
                local = pos % shard_rows
                count = min(end - pos, shard_rows - local)
                segments.append((shards[shard], local, count))
                pos += count
            specs.append({'name': f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_{head}', 'layer': 1, 'ngram_order': 2 + head // 8, 'head': head, 'segments': segments})
            start = end
    return specs


def source_rows(src, segments, start, count):
    result = np.empty((count, src[segments[0][0]][2]['shape'][1]), dtype='<u2')
    end, pos = start + count, 0
    for name, local, length in segments:
        lo, hi = max(start, pos), min(end, pos + length)
        if lo < hi:
            path, base, spec = src[name]
            if spec['dtype'] != 'BF16' or len(spec['shape']) != 2:
                raise ValueError(f'BF16 行列ではありません: {name}')
            begin, finish = spec['data_offsets']
            if finish - begin != spec['shape'][0] * spec['shape'][1] * 2 or base + finish > path.stat().st_size or local + length > spec['shape'][0]:
                raise ValueError(f'不正な tensor 範囲: {name}')
            source = np.memmap(path, dtype='<u2', mode='r', offset=base + begin, shape=tuple(spec['shape']))
            result[lo-start:hi-start] = source[local + lo-pos:local + hi-pos]
        pos += length
    if end > pos:
        raise ValueError('行範囲が不正です')
    return result

def convert(args, lib):
    overall_start = time.perf_counter()
    src = sources(args.input)
    specs = json.loads(args.tensor_spec.read_text()) if args.tensor_spec else []
    if not specs:
        specs = automatic_specs(args.input, src, args.model_family)
    if not specs:
        raise ValueError('対象 tensor が見つかりません')
    bits_map = json.loads(args.bits_map.read_text()) if args.bits_map else {}
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('出力ディレクトリは空にしてください')
    args.output.mkdir(parents=True, exist_ok=True)
    tables = []
    for i, spec in enumerate(specs):
        start = time.perf_counter()
        name = spec['name']
        segments = spec.get('segments')
        if segments is None:
            shape = src[name][2].get('shape', [])
            if src[name][2].get('dtype') != 'BF16' or len(shape) != 2:
                raise ValueError(f'BF16 行列ではありません: {name}')
            segments = [(name, 0, shape[0])]
        rows = sum(segment[2] for segment in segments)
        dim = src[segments[0][0]][2]['shape'][1]
        rows = min(rows, args.max_rows) if args.max_rows else rows
        if rows <= 0 or dim <= 0:
            raise ValueError(f'不正な tensor 範囲: {name}')
        bits = bits_map.get(name, spec.get('bits', 4 if args.profile != 'ngram5' else 5))
        if bits not in (4, 5) or (args.profile != 'mixed' and bits != (4 if args.profile == 'ngram4' else 5)):
            raise ValueError(f'不正な bit 幅: {name}')
        count = min(rows, args.sample_rows)
        indexes = np.linspace(0, rows - 1, count, dtype=np.int64)
        sample_raw = np.empty((count, dim), dtype='<u2')
        for j, index in enumerate(indexes):
            sample_raw[j] = source_rows(src, segments, int(index), 1)[0]
        sample = (sample_raw.astype(np.uint32) << 16).view(np.float32)
        book = np.empty(1 << bits, dtype='<u2')
        call(lib.shelf_ngram_train_codebook_f32, ptr(sample), sample.size, bits, ptr(book))
        key = f'table_{i:04d}'
        shard = f'{key}.safetensors'
        width = (dim * bits + 7) // 8
        csize, ssize, bsize = rows * width, rows * 2, book.nbytes
        header = {key + '.codes': {'dtype': 'U8', 'shape': [rows, width], 'data_offsets': [0, csize]}, key + '.scale': {'dtype': 'F16', 'shape': [rows], 'data_offsets': [csize, csize + ssize]}, key + '.codebook': {'dtype': 'F16', 'shape': [1 << bits], 'data_offsets': [csize + ssize, csize + ssize + bsize]}}
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        target = args.output / shard
        with target.open('wb') as f:
            f.write(struct.pack('<Q', len(raw)))
            f.write(raw)
            f.truncate(8 + len(raw) + csize + ssize + bsize)
        base = 8 + len(raw)
        codes = np.memmap(target, dtype='u1', mode='r+', offset=base, shape=(rows, width))
        scales = np.memmap(target, dtype='<u2', mode='r+', offset=base + csize, shape=(rows,))
        cbout = np.memmap(target, dtype='<u2', mode='r+', offset=base + csize + ssize, shape=book.shape)
        cbout[:] = book
        for pos in range(0, rows, args.chunk_rows):
            stop = min(pos + args.chunk_rows, rows)
            chunk = source_rows(src, segments, pos, stop-pos)
            call(lib.shelf_ngram_quantize_rows_bf16, ptr(chunk), stop-pos, dim, bits, ptr(book), ptr(codes[pos:stop]), ptr(scales[pos:stop]), args.threads)
        codes.flush(); scales.flush(); cbout.flush()
        sample_codes = np.ascontiguousarray(codes[indexes]); sample_scales = np.ascontiguousarray(scales[indexes])
        decoded = np.empty(sample.shape, dtype=np.float32)
        call(lib.shelf_ngram_decode_rows_f32, ptr(sample_codes), ptr(sample_scales), ptr(book), count, dim, bits, ptr(decoded))
        norm = np.linalg.norm(sample.astype(np.float64))
        rel = np.linalg.norm((sample - decoded).astype(np.float64)) / norm if norm else 0.0
        elapsed = time.perf_counter() - start
        table = {'id': key, 'name': name, 'layer': spec.get('layer', 0), 'ngram_order': spec.get('ngram_order', 1), 'head': spec.get('head', 0), 'rows': rows, 'dim': dim, 'bits': bits, 'shard': shard, 'codes_tensor': key+'.codes', 'scale_tensor': key+'.scale', 'codebook_tensor': key+'.codebook'}
        tables.append(table)
        print(f'{name}: rows={rows} dim={dim} bits={bits} theoretical_bytes={csize+ssize+bsize} file_bytes={target.stat().st_size} seconds={elapsed:.3f} rows/s={rows/elapsed:.1f} relative_l2={rel:.6f}', flush=True)
        del codes, scales, cbout
    manifest = {'format': 'ngram-q', 'version': 1, 'profile': args.profile, 'codec': 'lloyd-row-f16', 'source_model': args.input.name, 'tables': tables}
    (args.output / 'ngram_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(lib, args.output)
    elapsed = time.perf_counter() - overall_start
    total = sum(t['rows'] for t in tables)
    print(f"total: rows={total} seconds={elapsed:.3f} rows/s={total/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('convert')
    c.add_argument('input', type=Path); c.add_argument('output', type=Path)
    c.add_argument('--tensor-spec', type=Path)
    c.add_argument('--model-family', choices=['auto', 'hy4', 'qwen4exp'], default='auto')
    c.add_argument('--profile', choices=['ngram4', 'ngram5', 'mixed'], default='ngram4')
    c.add_argument('--bits-map', type=Path)
    c.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    c.add_argument('--sample-rows', type=int, default=1024)
    c.add_argument('--chunk-rows', type=int, default=65536)
    c.add_argument('--max-rows', type=int)
    v = sub.add_parser('verify'); v.add_argument('input', type=Path)
    args = parser.parse_args()
    if args.command == 'convert' and (min(args.threads, args.sample_rows, args.chunk_rows) < 1 or (args.max_rows is not None and args.max_rows < 1)):
        parser.error('行数とスレッド数は正の整数にしてください')
    try:
        lib = library()
        convert(args, lib) if args.command == 'convert' else verify(lib, args.input)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(f'エラー: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
