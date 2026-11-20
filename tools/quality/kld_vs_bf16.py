#!/usr/bin/env python3
"""cases-v1 の全位置を BF16 基準と比較する。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / 'datapool/qwen38-kld-20260918'
MODEL = Path.home() / 'datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    start = time.monotonic()
    if os.environ.get('SHELF_EXL3_MUL24_FUSED_DECODE') == '1':
        if b'SHELF_EXL3_MUL24_FUSED_DECODE' not in args.bench.read_bytes():
            raise ValueError('指定された shelf-bench は融合 decode の環境変数に対応していません')
    inputs_path = args.data / 'cases-v1.json'
    vocab_path = args.data / 'vocab-v1.json'
    refdir = args.data / 'results/llama-bf16-fp16'
    cases = json.loads(inputs_path.read_text())['cases']
    refs = {x['id']: x for x in json.loads((refdir / 'manifest.json').read_text())['cases']}
    vocab = np.asarray(json.loads(vocab_path.read_text())['valid_token_ids'], dtype=np.int64)
    if len(vocab) != 248077 or len(np.unique(vocab)) != len(vocab) or np.any(vocab[1:] <= vocab[:-1]):
        raise ValueError('有効語彙が想定と異なります')
    out = {'configuration': args.name, 'definition': 'KL(softmax(reference)||softmax(candidate)), temperature=1, FP64, nats',
           'reference': str(refdir), 'reference_kv': 'FP16', 'candidate_kv': 'BF16',
           'corpus_sha256': sha(inputs_path), 'vocab_sha256': sha(vocab_path),
           'executable_sha256': sha(args.bench), 'model': str(args.model.resolve()),
           'git_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
           'environment': {k: v for k, v in sorted(os.environ.items()) if k.startswith(('SHELF_', 'HIPBLASLT_', 'ROCR_', 'HIP_VISIBLE_', 'LD_'))},
           'cases': []}
    all_kl, matches = [], []
    with tempfile.TemporaryDirectory(prefix='shelf-bf16-kld-', dir=args.tmpdir) as tmp:
        tmp = Path(tmp)
        for case in cases:
            name = case['id']
            if name.endswith('_repeat'):
                continue
            ids = case['token_ids']
            ref = refs[name]
            expected = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            binary_hash = hashlib.sha256(struct.pack('<' + 'i' * len(ids), *ids)).hexdigest()
            if ref['input_token_ids_sha256'].removeprefix('sha256:') not in (expected, binary_hash) or ref['positions'] != list(range(len(ids))):
                raise ValueError(f'{name}: 基準 token または位置が不一致')
            shape = ref['shape']
            refpath = refdir / ref['logits_file']
            if shape[0] != len(ids) or shape[1] <= vocab[-1] or refpath.stat().st_size != np.prod(shape) * 4:
                raise ValueError(f'{name}: 基準 shape が不正')
            refdata = np.memmap(refpath, dtype='<f4', mode='r', shape=tuple(shape))
            tokenfile, dump = tmp / 'tokens', tmp / 'dump'
            tokenfile.write_bytes(b'PSKLDTOK' + struct.pack('<IQ', 1, len(ids) + 1) + struct.pack('<' + 'i' * (len(ids)+1), *(ids + [ids[-1]])))
            context = min(args.context, len(ids) - 1)
            cmd = [str(args.bench), 'quality', '--model-dir', str(args.model), '--tokens-file', str(tokenfile),
                   '--context', str(context), '--positions', str(len(ids)-context), '--prefill-chunk', str(args.context),
                   '--arena-gib', 'auto', '--device', '0', '--kv-cache-dtype', 'bf16', '--dump-prefill', '--dump', str(dump)]
            print(f'{name}: {len(ids)} 位置、prefill {context}', flush=True)
            env = os.environ.copy()
            fused = env.get('SHELF_EXL3_MUL24_FUSED_DECODE') == '1'
            if fused:
                env['SHELF_EXL3_MUL24_FUSED_DECODE_TRACE'] = '1'
            result = subprocess.run(cmd, env=env, stderr=subprocess.PIPE, text=True)
            sys.stderr.write(result.stderr)
            result.check_returncode()
            if fused and 'SHELF_FUSED_DECODE_LAUNCHED' not in result.stderr:
                raise RuntimeError(f'{name}: 融合 decode の実行を確認できません')
            with dump.open('rb') as f:
                magic, ver, width, rows, kv, first = struct.unpack('<8sIIQII', f.read(32))
                if magic != b'PSQUAL01' or ver != 1 or rows != len(ids) or width <= vocab[-1] or kv != 0 or first != 1:
                    raise ValueError(f'{name}: dump header 不正')
                if dump.stat().st_size != 32 + rows * (8 + 4 * width):
                    raise ValueError(f'{name}: dump サイズ不正')
                kls, tops = [], []
                for i in range(len(ids)):
                    target, pos = struct.unpack('<iI', f.read(8))
                    if pos != i + 1 or target != (ids[i+1] if i+1 < len(ids) else ids[-1]):
                        raise ValueError(f'{name}: 位置 {i} の token が不一致')
                    q = np.frombuffer(f.read(width * 4), dtype='<f4')[vocab].astype(np.float64)
                    p = refdata[i, vocab].astype(np.float64)
                    if not np.isfinite(p).all() or not np.isfinite(q).all():
                        raise ValueError(f'{name}: 非有限 logits')
                    top = int(np.argmax(p) == np.argmax(q))
                    p -= p.max(); q -= q.max()
                    p -= np.log(np.exp(p).sum()); q -= np.log(np.exp(q).sum())
                    kl = float(np.sum(np.exp(p) * (p - q)))
                    if kl < -1e-10 or not np.isfinite(kl):
                        raise ValueError(f'{name}: KLD 不正')
                    kls.append(max(0., kl)); tops.append(top)
            item = {'id': name, 'positions': len(ids), 'prefill_positions': context,
                    'mean_kld': float(np.mean(kls)), 'top1_agreement': float(np.mean(tops)),
                    'position_kld': kls, 'position_top1_agreement': tops,
                    'candidate_dump_sha256': sha(dump), 'reference_logits_sha256': sha(refpath)}
            out['cases'].append(item); all_kl.extend(kls); matches.extend(tops)
    if len(all_kl) != 2632 or len(out['cases']) != 8:
        raise ValueError('主 case の数または位置数が不正')
    out.update(positions=len(all_kl), mean_kld=float(np.mean(all_kl)), top1_agreement=float(np.mean(matches)), elapsed_seconds=time.monotonic()-start)
    output = ROOT / 'docs/perf/quality-bf16' / (args.name + '.json')
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    reports = []
    for path in sorted(output.parent.glob('*.json')):
        obj = json.loads(path.read_text())
        reports.append(f"| {obj['configuration']} | {obj['mean_kld']:.8f} | {100*obj['top1_agreement']:.4f}% | {obj['elapsed_seconds']:.1f} |")
    (output.parent / 'results.md').write_text('# BF16 基準の測定結果\n\n| 構成 | 平均 KLD (nats) | top-1 一致率 | 所要秒 |\n| --- | ---: | ---: | ---: |\n' + '\n'.join(reports) + '\n')
    print(json.dumps({k: out[k] for k in ('positions', 'mean_kld', 'top1_agreement', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='R9700 の B 構成は約196秒（実測1回）。実測値を results.md に記録します。GPU lock はシェルラッパで取得します。')
    p.add_argument('name', help='結果ファイルに使う構成名（英数字、ハイフン、下線）')
    p.add_argument('--bench', type=Path, default=ROOT/'build-bf16-kld/shelf-bench')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--context', type=int, default=256, help='prefill 最大 token 数（既定 256）')
    p.add_argument('--tmpdir', type=Path, default=None)
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or args.context < 1:
        p.error('構成名または context が不正')
    run(args)


if __name__ == '__main__':
    main()
