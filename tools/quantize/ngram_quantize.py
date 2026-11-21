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


def convert(args, lib):
    overall_start = time.perf_counter()
    src = sources(args.input)
    specs = json.loads(args.tensor_spec.read_text()) if args.tensor_spec else []
    if not specs:
        for name, (_, _, s) in src.items():
            low = name.lower()
            hy4 = 'engram' in low and 'embedding.weight' in low
            qwen = ('ple' in low or 'ngram' in low) and ('embed' in low or 'table' in low)
            if s.get('dtype') == 'BF16' and len(s.get('shape', [])) == 2 and ((args.model_family in ('auto', 'hy4') and hy4) or (args.model_family in ('auto', 'qwen4exp') and qwen)):
                layer = re.search(r'layers\.(\d+)', name)
                specs.append({'name': name, 'layer': int(layer.group(1)) if layer else 0, 'ngram_order': 1, 'head': 0})
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
        path, offset, s = src[name]
        if s['dtype'] != 'BF16' or len(s['shape']) != 2:
            raise ValueError(f'BF16 行列ではありません: {name}')
        rows, dim = s['shape']
        rows = min(rows, args.max_rows) if args.max_rows else rows
        begin, end = s['data_offsets']
        if rows <= 0 or dim <= 0 or end - begin != s['shape'][0] * dim * 2 or offset + end > path.stat().st_size:
            raise ValueError(f'不正な tensor 範囲: {name}')
        source = np.memmap(path, dtype='<u2', mode='r', offset=offset + begin, shape=(s['shape'][0], dim))
        bits = bits_map.get(name, spec.get('bits', 4 if args.profile != 'ngram5' else 5))
        if bits not in (4, 5) or (args.profile != 'mixed' and bits != (4 if args.profile == 'ngram4' else 5)):
            raise ValueError(f'不正な bit 幅: {name}')
        count = min(rows, args.sample_rows)
        indexes = np.linspace(0, rows - 1, count, dtype=np.int64)
        sample_raw = np.ascontiguousarray(source[indexes])
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
            chunk = np.ascontiguousarray(source[pos:stop])
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
        del source, codes, scales, cbout
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
